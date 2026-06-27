"""
core/llm.py
-----------
Layer 4 — Steered Response Generation (Mistral / Qwen)

Loads the chosen LLM on LLM_DEVICE (config.py) ONCE at startup, keeps full
conversation history across turns, and on each turn:

    1. Builds a prompt = history + current user message + style instructions
       (style instructions come from DecisionEngine's StyleContract)
    2. If decision.vector != "none":
           effective_alpha = ALPHA * decision.vector_intensity
           registers an EmotionSteerer that injects the emotion direction
           vector into layers 11-20 (MLP + attention) for this turn only
    3. Generates the response
    4. Removes the steerer immediately (fresh per turn)
    5. Cleans the response (strip emojis/unicode junk) for history + TTS

GPU safety
----------
LLM_DEVICE is an explicit single device string ("cuda:1", "cuda:0", or "cpu")
from config.py. We NEVER use device_map="auto" — that splits layers across
GPUs and causes device-mismatch crashes in steering hooks.
device_map={"": LLM_DEVICE} pins the entire model to one device.
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from config import (
    LLM_BACKEND,
    LLM_PATHS,
    ALPHA,
    LAYERS,
    LAST_K,
    SCALE,
    DIRECTIONS_DIR,
    LLM_DTYPE,
    LLM_DEVICE,
    HF_TOKEN,
    MAX_HISTORY_TURNS,
)
from core.decision_engine import DecisionOutput, StyleContract

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================================
# HELPERS
# ============================================================================

def parse_layers(layer_arg: str) -> List[int]:
    if "-" in layer_arg:
        a, b = layer_arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in layer_arg.split(",") if x.strip()]


def extract_emotion_from_vector(vector_str: str) -> str:
    """
    "mirror_surprise"  -> "surprise"
    "inject_happiness" -> "happiness"
    "none"             -> "none"
    """
    if vector_str == "none":
        return "none"
    return vector_str.split("_", 1)[1]


def normalize_for_history(text: str) -> str:
    """Strip emojis/unicode junk and collapse repeated punctuation."""
    text = ''.join(
        c for c in text
        if unicodedata.category(c) not in {'So', 'Cs', 'Co', 'Cn'}
        and ord(c) < 0x2FFFF
    )
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[*]{2,}', '', text)
    text = re.sub(r'[-]{2,}', '-', text)
    text = re.sub(r'[#]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "I understand."


def build_style_prompt_instructions(style_contract: StyleContract) -> str:
    bullets_inst   = "- use bullet points for listing items" if style_contract.allow_bullets \
                     else "- do not use numbered lists"
    questions_inst = "- ask questions if appropriate" if style_contract.allow_questions \
                     else "- do not ask any questions"
    commands_inst  = "- use suggestions" if style_contract.allow_commands \
                     else "- do not give commands"
    profanity_inst = "- Mild profanity is permitted if natural" if style_contract.profanity \
                     else "- do not use any profanity"
    return (
        f"Response instructions:\n"
        f"                - Maximum length: {style_contract.max_words} words\n"
        f"                - Maximum sentences: {style_contract.max_sentences}\n"
        f"                {bullets_inst}\n"
        f"                {questions_inst}\n"
        f"                {commands_inst}\n"
        f"                {profanity_inst}\n"
    )


def load_directions(directions_dir: Path, direction_type: str) -> dict:
    """Load all emotion direction vectors for given type (mlp / attention)."""
    directions_file = directions_dir / f"emo_directions_{direction_type}.pt"
    if not directions_file.exists():
        raise FileNotFoundError(f"Direction file not found: {directions_file}")

    obj  = torch.load(directions_file, map_location="cpu", weights_only=False)
    dirs = obj["dirs"]

    for e in dirs:
        if not isinstance(dirs[e], np.ndarray):
            dirs[e] = np.array(dirs[e], dtype=np.float32)
        else:
            dirs[e] = dirs[e].astype(np.float32)

    return dirs


def clear_gpu_memory():
    """Clear memory cache on every visible GPU."""
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# EMOTION STEERER
# ============================================================================

class EmotionSteerer:
    """
    Injects a single emotion direction vector during one forward pass.
    effective_alpha = ALPHA x vector_intensity (from decision engine).
    Created fresh each turn, removed immediately after generation.

    All vectors are moved to the layer's own device at hook-registration time,
    so there is no cross-device tensor operation regardless of which GPU the
    model is on.
    """

    def __init__(self, model, direction_vector: np.ndarray, layer_ids: List[int],
                 effective_alpha: float, last_k: int = 1, scale_mode: str = "rms"):

        self.model      = model
        self.layer_ids  = list(layer_ids)
        self.last_k     = last_k
        self.scale_mode = scale_mode
        self.is_active  = True

        language_model = self._find_language_model(model)

        if hasattr(language_model, "layers"):
            self.layers = language_model.layers
        elif hasattr(language_model, "decoder") and hasattr(language_model.decoder, "layers"):
            self.layers = language_model.decoder.layers
        else:
            self.layers = None
            for name, module in language_model.named_children():
                if "layer" in name.lower() and hasattr(module, "__len__"):
                    self.layers = module
                    break

        if self.layers is None:
            raise ValueError("Could not find transformer layers in the model")

        # Prepare per-layer vectors — move each to the layer's own device
        self.v = {}
        for l in self.layer_ids:
            if l < len(self.layers):
                device        = next(self.layers[l].parameters()).device
                raw_vec       = torch.from_numpy(direction_vector[l]).to(device)
                self.v[l]     = effective_alpha * raw_vec

        # Register hooks
        self.handles = []
        for l in self.layer_ids:
            if l in self.v:
                h = self.layers[l].register_forward_hook(self._make_hook(l))
                self.handles.append(h)

    @staticmethod
    def _find_language_model(model):
        """Locate the inner decoder stack across Mistral3 / Qwen2 architectures."""
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model   # Mistral3ForConditionalGeneration
        if hasattr(model, "model"):
            return model.model                  # Qwen2ForCausalLM / MistralForCausalLM
        return model

    def _make_hook(self, layer_id: int):
        v          = self.v[layer_id]
        last_k     = self.last_k
        scale_mode = self.scale_mode

        def hook(module, inputs, output):
            if not self.is_active:
                return output

            if isinstance(output, (tuple, list)):
                if len(output) > 0:
                    hs      = output[0].clone()
                    B, T, H = hs.shape
                    start   = max(0, T - last_k)
                    if scale_mode == "rms":
                        seg   = hs[:, start:T, :]
                        rms   = torch.sqrt((seg ** 2).mean(dim=-1, keepdim=True) + 1e-12)
                        delta = v.view(1, 1, H) * rms
                    else:
                        delta = v.view(1, 1, H)
                    hs[:, start:T, :] = hs[:, start:T, :] + delta
                    return (hs,) + output[1:]
                return output
            else:
                hs      = output.clone()
                B, T, H = hs.shape
                start   = max(0, T - last_k)
                if scale_mode == "rms":
                    seg   = hs[:, start:T, :]
                    rms   = torch.sqrt((seg ** 2).mean(dim=-1, keepdim=True) + 1e-12)
                    delta = v.view(1, 1, H) * rms
                else:
                    delta = v.view(1, 1, H)
                hs[:, start:T, :] = hs[:, start:T, :] + delta
                return hs

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ============================================================================
# STEERED LLM
# ============================================================================

class SteeredLLM:
    """
    Loads Mistral or Qwen ONCE on LLM_DEVICE, keeps full conversation history,
    and generates a steered + style-constrained response each turn.

    Usage
    -----
        llm = SteeredLLM()                  # uses config.LLM_BACKEND
        llm = SteeredLLM(backend="qwen")    # explicit override

        response = llm.generate(user_message, decision_output)
        llm.reset()                          # start a new conversation
    """

    BASE_GENERATION_KWARGS = {
        "do_sample":              False,
        "use_cache":              True,
        "min_new_tokens":         10,
        "max_new_tokens":         100,
        "repetition_penalty":     1.05,
        "no_repeat_ngram_size":   3,
    }

    def __init__(self, backend: str = LLM_BACKEND, direction_type: str = "mlp"):
        self.backend        = backend
        self.direction_type = direction_type
        self.layer_ids      = parse_layers(LAYERS)

        model_path = LLM_PATHS[backend]
        print(f"   [LLM] loading '{backend}' from {model_path} on {LLM_DEVICE} ...")

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            token=HF_TOKEN if HF_TOKEN else True,
            trust_remote_code=True,
        )
        # Mistral3 tokenizer regex fix (prevents slow-tokenizer warnings)
        if hasattr(self.tokenizer, "fix_mistral_regex"):
            self.tokenizer.fix_mistral_regex = True
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = self._load_model(model_path)
        self.model.eval()
        self.model.config.use_cache = True
        clear_gpu_memory()

        self.BASE_GENERATION_KWARGS["eos_token_id"] = self.tokenizer.eos_token_id
        self.BASE_GENERATION_KWARGS["pad_token_id"] = self.tokenizer.pad_token_id

        # ── Direction vectors ─────────────────────────────────────────────────
        directions_dir = DIRECTIONS_DIR[backend]
        self.direction_vectors = {}
        for dtype_ in ["mlp", "attention"]:
            try:
                self.direction_vectors[dtype_] = load_directions(directions_dir, dtype_)
                print(f"   [LLM] loaded '{dtype_}' directions: "
                      f"{list(self.direction_vectors[dtype_].keys())}")
            except Exception as e:
                print(f"   [LLM] WARNING: could not load '{dtype_}' directions: {e}")
                self.direction_vectors[dtype_] = None

        # ── Conversation state ────────────────────────────────────────────────
        self.messages: List[dict] = []

    def _load_model(self, model_path: str):
        """
        Load the LLM onto a single explicit device (LLM_DEVICE).

        CRITICAL: device_map={"": LLM_DEVICE} pins the WHOLE model to ONE device.
        Never use device_map="auto" — that splits layers and breaks steering hooks.

        Loader priority for Ministral-3:
            1. Mistral3ForConditionalGeneration  (correct class for this checkpoint)
            2. MistralForCausalLM                (fallback if class not found)
            3. AutoModelForCausalLM              (generic fallback for Qwen etc.)
            4. CPU float32                        (last resort)
        """
        is_ministral3 = (
            "Ministral-3" in model_path
            or "ministral-3" in model_path.lower()
            or self.backend == "mistral"
        )

        if is_ministral3:
            # ── Try Mistral3ForConditionalGeneration first ─────────────────────
            try:
                from transformers.models.mistral3 import Mistral3ForConditionalGeneration
                clear_gpu_memory()
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=LLM_DTYPE,
                    device_map={"": LLM_DEVICE},   # whole model on one device
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                print(f"   [LLM] ✅ Ministral3ForConditionalGeneration on {LLM_DEVICE}")
                self._log_gpu_memory()
                return model
            except Exception as e:
                print(f"   [LLM] Mistral3ForConditionalGeneration failed: {e}")

            # ── Fallback: MistralForCausalLM ──────────────────────────────────
            try:
                from transformers import MistralForCausalLM
                clear_gpu_memory()
                model = MistralForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=LLM_DTYPE,
                    device_map={"": LLM_DEVICE},
                    token=HF_TOKEN if HF_TOKEN else True,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                print(f"   [LLM] ✅ MistralForCausalLM (fallback) on {LLM_DEVICE}")
                self._log_gpu_memory()
                return model
            except Exception as e:
                print(f"   [LLM] MistralForCausalLM fallback failed: {e}")

        # ── Generic: AutoModelForCausalLM (Qwen or final fallback) ────────────
        try:
            from transformers import AutoModelForCausalLM
            clear_gpu_memory()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=LLM_DTYPE,
                device_map={"": LLM_DEVICE},
                token=HF_TOKEN if HF_TOKEN else True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print(f"   [LLM] ✅ AutoModelForCausalLM on {LLM_DEVICE}")
            self._log_gpu_memory()
            return model
        except Exception as e:
            print(f"   [LLM] GPU loading failed: {e}. Falling back to CPU ...")

        # ── Last resort: CPU ───────────────────────────────────────────────────
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            token=HF_TOKEN if HF_TOKEN else True,
            trust_remote_code=True,
        )
        print("   [LLM] ⚠️ Loaded on CPU (very slow)")
        return model

    def _log_gpu_memory(self):
        if torch.cuda.is_available():
            for dev in set([LLM_DEVICE, "cuda:0"]):
                try:
                    idx   = int(dev.split(":")[-1]) if ":" in dev else 0
                    alloc = torch.cuda.memory_allocated(idx) / 1024**3
                    print(f"   [LLM] {dev} memory: {alloc:.2f} GB allocated")
                except Exception:
                    pass

    # ── conversation history management ───────────────────────────────────────

    def reset(self):
        """Start a fresh conversation (call alongside StateMemory.reset())."""
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self.messages.append({"role": "assistant", "content": message})
        if len(self.messages) > MAX_HISTORY_TURNS * 2:
            self.messages = self.messages[-MAX_HISTORY_TURNS * 2:]

    def _build_prompt(self, user_message: str, style_contract: Optional[StyleContract]) -> str:
        history_prompt = ""
        for msg in self.messages:
            if msg["role"] == "user":
                history_prompt += f"User: {msg['content']}\n"
            else:
                history_prompt += f"{msg['content']}\n"

        prompt = history_prompt + f"User: {user_message}\n"

        if style_contract is not None:
            prompt += build_style_prompt_instructions(style_contract)

        return prompt

    # ── main generation entrypoint ─────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, user_message: str, decision: DecisionOutput) -> str:
        """
        Generate a steered, style-constrained response for one turn.

        Parameters
        ----------
        user_message : the ASR transcript for this turn
        decision     : DecisionOutput from core.decision_engine.DecisionEngine

        Returns
        -------
        clean_response : str — emoji/unicode-stripped, ready for TTS + history
        """
        prompt = self._build_prompt(user_message, decision.style_contract)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(LLM_DEVICE)

        # ── Setup steerer if vector != "none" ──────────────────────────────────
        vector_emotion = extract_emotion_from_vector(decision.vector)
        steerer = None

        dir_table = self.direction_vectors.get(self.direction_type)
        if vector_emotion != "none" and dir_table is not None and vector_emotion in dir_table:
            effective_alpha = ALPHA * decision.vector_intensity
            steerer = EmotionSteerer(
                model            = self.model,
                direction_vector = dir_table[vector_emotion],
                layer_ids        = self.layer_ids,
                effective_alpha  = effective_alpha,
                last_k           = LAST_K,
                scale_mode       = SCALE,
            )
            print(f"   [LLM] steering: {decision.vector} | effective_alpha={effective_alpha:.3f}")
        else:
            print("   [LLM] no steering (neutral baseline)")

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            gen = self.model.generate(**inputs, **self.BASE_GENERATION_KWARGS)
        finally:
            if steerer is not None:
                steerer.remove()

        out_ids        = gen[0][inputs.input_ids.shape[1]:]
        response       = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        clean_response = normalize_for_history(response)

        # ── Update history ─────────────────────────────────────────────────────
        self.add_user_message(user_message)
        self.add_assistant_message(clean_response)

        return clean_response