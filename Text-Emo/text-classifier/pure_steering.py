"""
pure_steering.py
---------------------
Pathway 1 — Re-inject emotion vector every single turn.

Pipeline per turn:
    user_sentence
        → Classifier       (emotion, target, user_intensity)
        → StateMemory      (ai_state with blended intensity)
        → DecisionEngine   (vector, vector_intensity, mode)
        → EmotionSteerer   (inject vector with effective_alpha = ALPHA × vector_intensity)
        → generate response
        → remove steerer
        → next turn repeats

vector = "none" → skip steerer entirely (neutral baseline turn).

Input JSON:
    /data/kaushal/Various-Model/classification/testing-scenario/{emotion}.json

Output:
    /data/kaushal/Various-Model/classification/output/{model_name}/multiturn_test/{TARGET_EMOTION}/run_{timestamp}/
"""

import os
import re
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    RobertaTokenizer,
    RobertaModel,
)
from huggingface_hub import login
from datetime import datetime
import pytz
import torch.nn as nn
import unicodedata
# ── Import our modules ────────────────────────────────────────────────────────
from state_memory import StateMemory
from decision_engine import DecisionEngine

# ============================================================================
# CONFIGURATION
# ============================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"]    = "1,2,3"

# ── Mistral model ─────────────────────────────────────────────────────────────
MODEL_NAME = "ministral3_3b"
MODEL_PATH = "mistralai/Ministral-3-3B-Instruct-2512"

# ── Target emotion (change to run different emotion files) ────────────────────
TARGET_EMOTION = "anger"

# ── Steering parameters ───────────────────────────────────────────────────────
ALPHA  = 8.0        # base steering strength — scaled by vector_intensity each turn
LAYERS = "11-20"
LAST_K = 1
SCALE  = "rms"

# ── Classifier ────────────────────────────────────────────────────────────────
CLASSIFIER_DIR = Path("/data/kaushal/Various-Model/classification/finetune/checkpoints-2")
CLASSIFIER_MAX_LENGTH = 128

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH       = Path("/data/kaushal/Various-Model")
DIRECTIONS_DIR  = BASE_PATH / "outputs" / MODEL_NAME / "02_emotion_directions"
SCENARIOS_DIR   = BASE_PATH / "classification" / "testing-scenario"
OUTPUT_BASE     = BASE_PATH / "classification" / "output" / MODEL_NAME / "multiturn_test"

DTYPE   = torch.float16
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN:
    login(token=HF_TOKEN)

# ============================================================================
# CLASSIFIER — RoBERTa multi-head (from checkpoints-2)
# ============================================================================

EMOTIONS   = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]
TARGETS_CL = ["you", "other", "self", "situation"]

EMOTION2ID = {e: i for i, e in enumerate(EMOTIONS)}
TARGET2ID  = {t: i for i, t in enumerate(TARGETS_CL)}
ID2EMOTION = {i: e for e, i in EMOTION2ID.items()}
ID2TARGET  = {i: t for t, i in TARGET2ID.items()}


class EmotionMultiHeadModel(nn.Module):
    """
    Exact architecture from classifier training — must match checkpoints-2.
    RoBERTa + 3 heads: emotion (6), target (4), intensity (1).
    """
    def __init__(self, model_name: str = "roberta-base", dropout: float = 0.1):
        super().__init__()
        self.roberta        = RobertaModel.from_pretrained(model_name)
        hidden_size         = self.roberta.config.hidden_size
        self.dropout        = nn.Dropout(dropout)
        self.emotion_head   = nn.Linear(hidden_size, len(EMOTIONS))
        self.target_head    = nn.Linear(hidden_size, len(TARGETS_CL))
        self.intensity_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls     = self.dropout(outputs.last_hidden_state[:, 0, :])
        emotion_logits  = self.emotion_head(cls)
        target_logits   = self.target_head(cls)
        intensity_preds = self.intensity_head(cls).squeeze(-1)
        intensity_preds = torch.clamp(intensity_preds, min=0.0, max=1.0)
        return emotion_logits, target_logits, intensity_preds


class Classifier:
    """
    Wraps EmotionMultiHeadModel for single-sentence inference.
    Outputs: (emotion: str, target: str, intensity: float)
    All outputs are already normalised lowercase.
    """

    def __init__(self, checkpoint_dir: Path, device: str):
        self.device = torch.device(device)

        self.tokenizer = RobertaTokenizer.from_pretrained(str(checkpoint_dir))

        self.model = EmotionMultiHeadModel("roberta-base", dropout=0.1)
        self.model.to(self.device)

        checkpoint_path = checkpoint_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"   ✅ Classifier loaded from {checkpoint_dir}")

    @torch.no_grad()
    def classify(self, sentence: str):
        """
        Parameters
        ----------
        sentence : raw user sentence

        Returns
        -------
        emotion   : str  — e.g. "surprise"
        target    : str  — e.g. "you"
        intensity : float — in [0.0, 1.0]
        """
        encoding = self.tokenizer(
            sentence,
            max_length=CLASSIFIER_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        emotion_logits, target_logits, intensity_preds = self.model(
            input_ids, attention_mask
        )

        emotion_id  = emotion_logits.argmax(dim=1).item()
        target_id   = target_logits.argmax(dim=1).item()
        intensity   = float(intensity_preds.item())

        return ID2EMOTION[emotion_id], ID2TARGET[target_id], intensity

# ============================================================================
# GPU UTILITIES (unchanged from original)
# ============================================================================

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_info():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved  = torch.cuda.memory_reserved(i)  / 1024**3
            total     = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")

def print_gpu_devices():
    if torch.cuda.is_available():
        print("\n💻 GPU Configuration:")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Number of visible GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("\n⚠️  No GPU available, using CPU")

# ============================================================================
# HELPERS
# ============================================================================

def parse_layers(layer_arg: str) -> List[int]:
    if "-" in layer_arg:
        a, b = layer_arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in layer_arg.split(",") if x.strip()]


def load_directions(direction_type: str) -> dict:
    """Load all emotion direction vectors for given type (mlp / attention)."""
    directions_file = DIRECTIONS_DIR / f"emo_directions_{direction_type}.pt"
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


def extract_emotion_from_vector(vector_str: str) -> str:
    """
    "mirror_surprise"  → "surprise"
    "inject_happiness" → "happiness"
    "none"             → "none"
    """
    if vector_str == "none":
        return "none"
    # format is always "mirror_<emotion>" or "inject_<emotion>"
    return vector_str.split("_", 1)[1]


def load_scenario(emotion: str) -> dict:
    """Load test scenario JSON for the given emotion."""
    path = SCENARIOS_DIR / f"{emotion}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_for_history(text: str) -> str:
    """
    Clean response before adding to conversation history.
    Removes emojis, unicode symbols, broken characters.
    Keeps only readable ASCII + basic punctuation.
    """
    # remove emojis and unicode symbols
    text = ''.join(
        c for c in text
        if unicodedata.category(c) not in {
            'So',  # other symbols (most emojis)
            'Cs',  # surrogates
            'Co',  # private use
            'Cn',  # unassigned
        }
        and ord(c) < 0x2FFFF
    )

    # collapse excessive punctuation like !! ** -- ***
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[*]{2,}', '', text)
    text = re.sub(r'[-]{2,}', '-', text)
    text = re.sub(r'[#]+', '', text)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text if text else "I understand."

# ============================================================================
# EMOTION STEERER (per-turn, created fresh each turn)
# ============================================================================

class EmotionSteerer:
    """
    Injects a single emotion direction vector during one forward pass.
    effective_alpha = ALPHA × vector_intensity (from decision engine).
    Created fresh each turn, removed after generation.
    """

    def __init__(self, model, direction_vector: np.ndarray, layer_ids: List[int],
                 effective_alpha: float, last_k: int = 1, scale_mode: str = "rms"):

        self.model       = model
        self.layer_ids   = list(layer_ids)
        self.last_k      = last_k
        self.scale_mode  = scale_mode
        self.is_active   = True

        language_model = self.model.model.language_model

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

        # Prepare per-layer vectors scaled by effective_alpha
        self.v = {}
        for l in self.layer_ids:
            if l < len(self.layers):
                device    = next(self.layers[l].parameters()).device
                raw_vec   = torch.from_numpy(direction_vector[l]).to(device)
                # scale vector by effective_alpha here so hook is simple
                self.v[l] = effective_alpha * raw_vec

        # Register hooks
        self.handles = []
        for l in self.layer_ids:
            if l in self.v:
                h = self.layers[l].register_forward_hook(self._make_hook(l))
                self.handles.append(h)

    def _make_hook(self, layer_id: int):
        v          = self.v[layer_id]
        last_k     = self.last_k
        scale_mode = self.scale_mode

        def hook(module, inputs, output):
            if not self.is_active:
                return output

            if isinstance(output, (tuple, list)):
                if len(output) > 0:
                    hs    = output[0].clone()
                    B, T, H = hs.shape
                    start = max(0, T - last_k)
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
                hs    = output.clone()
                B, T, H = hs.shape
                start = max(0, T - last_k)
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
# MULTI-TURN CONVERSATION MANAGER
# ============================================================================

class MultiTurnConversation:
    """Full conversation history preserved across all turns. No system prompt."""

    def __init__(self, model, tokenizer):
        self.model    = model
        self.tokenizer = tokenizer
        self.messages = []
        self.turn_count = 0

    def add_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self.messages.append({"role": "assistant", "content": message})

    @torch.no_grad()
    def generate_response(self, max_new_tokens: int = 60) -> str:
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            min_new_tokens=20,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

        out_ids  = gen[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        
        clean_response = normalize_for_history(response)        
        self.add_assistant_message(clean_response)
        self.turn_count += 1
        
        return response, clean_response

# ============================================================================
# PATHWAY 1 — MAIN TEST FUNCTION
# ============================================================================

def run_pathway1(
    model,
    tokenizer,
    classifier:       Classifier,
    all_directions:   dict,        # {emotion: direction_array} for all emotions
    scenario:         dict,
    direction_type:   str,
    layer_ids:        List[int],
    output_dir:       Path,
) -> dict:
    """
    Runs a full multi-turn conversation with vector re-injection every turn.

    Per turn:
        1. Classifier  → emotion, target, user_intensity
        2. StateMemory → ai_state (blended intensity, decay)
        3. DecisionEngine → vector, vector_intensity, mode
        4. If vector != "none":
               effective_alpha = ALPHA × vector_intensity
               EmotionSteerer injected during generation
           Else:
               generate without steering (neutral baseline)
        5. Steerer removed, next turn starts fresh
    """

    emotion_name = scenario["emotion"].lower()
    turns_data   = scenario["turns"]
    total_turns  = len(turns_data)

    print(f"\n   📋 Pathway 1 | {direction_type.upper()} | {emotion_name} | {total_turns} turns")

    # Fresh state memory and decision engine per conversation
    memory = StateMemory()
    engine = DecisionEngine()
    conv   = MultiTurnConversation(model, tokenizer)

    conversation_id = str(int(time.time())) + "_" + str(np.random.randint(1000, 9999))

    result = {
        "conversation_id": conversation_id,
        "pathway": "pathway1_reinject_every_turn",
        "experiment": {
            "target_emotion":   emotion_name,
            "direction_type":   direction_type,
            "steering_parameters": {
                "base_alpha": ALPHA,
                "layers":     layer_ids,
                "last_k":     LAST_K,
                "scale":      SCALE,
                "note":       "effective_alpha = base_alpha × vector_intensity each turn",
            },
            "system_prompt_used":  False,
            "context_handling":    "full_conversation_history_preserved",
        },
        "scenario": {
            "emotion":       emotion_name,
            "context":       scenario.get("context", ""),
            "total_turns":   total_turns,
        },
        "turns":     [],
        "timestamp": time.time(),
    }

    try:
        for turn_idx, turn_entry in enumerate(turns_data):
            turn_num      = turn_idx + 1
            user_sentence = turn_entry["user_sentence"]

            print(f"      Turn {turn_num:02d}/{total_turns}: classifying...")

            # ── Step 1: Classifier ────────────────────────────────────────────
            cl_emotion, cl_target, cl_intensity = classifier.classify(user_sentence)

            # ── Step 2: State Memory ──────────────────────────────────────────
            ai_state = memory.update(
                user_emotion   = cl_emotion,
                user_target    = cl_target,
                user_intensity = cl_intensity,
            )

            # ── Step 3: Decision Engine ───────────────────────────────────────
            decision = engine.decide(
                user_emotion = cl_emotion,
                user_target  = cl_target,
                user_intensity = cl_intensity,
                ai_state     = ai_state,
            )

            # ── Step 4: Steer + Generate ──────────────────────────────────────
            conv.add_user_message(user_sentence)

            vector_emotion = extract_emotion_from_vector(decision.vector)
            steerer        = None

            if vector_emotion != "none" and vector_emotion in all_directions:
                effective_alpha = ALPHA * decision.vector_intensity
                steerer = EmotionSteerer(
                    model           = model,
                    direction_vector = all_directions[vector_emotion],
                    layer_ids       = layer_ids,
                    effective_alpha = effective_alpha,
                    last_k          = LAST_K,
                    scale_mode      = SCALE,
                )
                print(f"         Steering: {decision.vector} | "
                      f"effective_alpha={effective_alpha:.3f} | "
                      f"vector_intensity={decision.vector_intensity}")
            else:
                print(f"         No steering (neutral baseline)")

            response, clean_response = conv.generate_response()

            # Remove steerer immediately after generation
            if steerer is not None:
                steerer.remove()

            # ── Step 5: Record turn ───────────────────────────────────────────
            turn_record = {
                "turn_number": turn_num,

                # user input
                "user_sentence": user_sentence,

                # classifier output
                "classifier": {
                    "emotion":    cl_emotion,
                    "target":     cl_target,
                    "intensity":  round(cl_intensity, 4),
                },

                # state memory output
                "state_memory": {
                    "ai_emotion":       ai_state.emotion,
                    "ai_intensity":     ai_state.ai_intensity,
                    "user_intensity":   ai_state.user_intensity,
                    "anchor": {
                        "trigger_emotion":   ai_state.anchor.trigger_emotion,
                        "trigger_target":    ai_state.anchor.trigger_target,
                        "trigger_intensity": ai_state.anchor.trigger_intensity,
                        "turn_triggered":    ai_state.anchor.turn_triggered,
                    },
                },

                # decision engine output
                "decision": {
                    "mode":             decision.mode,
                    "vector":           decision.vector,
                    "vector_intensity": decision.vector_intensity,
                    "effective_alpha":  round(ALPHA * decision.vector_intensity, 4),
                },

                # model output
                "model_response":   response,
                "model_response_clean":  clean_response,
                "steering_applied": vector_emotion != "none",
                "timestamp":        time.time(),
            }
            result["turns"].append(turn_record)

            print(f"         Response: {response[:150]}{'...' if len(response) > 150 else ''}")
            time.sleep(0.5)

            if turn_num % 5 == 0:
                clear_gpu_memory()

    except Exception as e:
        print(f"   ❌ Error at turn {turn_num}: {e}")
        result["error"] = str(e)

    # ── Save output ───────────────────────────────────────────────────────────
    out_file = output_dir / f"{emotion_name}_{direction_type}_{conversation_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    combined_file = output_dir / f"all_conversations_{direction_type}.jsonl"
    with open(combined_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"      ✅ Saved → {out_file.name}")
    return result

# ============================================================================
# MAIN
# ============================================================================
# ============================================================================
# MAIN
# ============================================================================

def main():
    # ── All emotions to process ────────────────────────────────────────────────────
    ALL_EMOTIONS = ["happiness", "surprise"]
    
    print("=" * 80)
    print("PATHWAY 1 — MULTI-TURN EMOTION STEERING (RE-INJECT EVERY TURN)")
    print("=" * 80)
    print(f"Model          : {MODEL_PATH}")
    print(f"Base Alpha     : {ALPHA}  (scaled by vector_intensity each turn)")
    print(f"Layers         : {LAYERS}")
    print(f"Emotions       : {ALL_EMOTIONS}")
    print("=" * 80)

    print_gpu_devices()

    # ── Create parent output directory for this run (contains all emotions) ───
    ist       = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y%m%d_%H%M%S")
    parent_output_dir = OUTPUT_BASE / f"run_all_emotions_{timestamp}"
    parent_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/6] Parent output directory: {parent_output_dir}")

    # ── Load classifier (once, shared across all emotions) ────────────────────
    print("\n[2/6] Loading classifier...")
    classifier = Classifier(CLASSIFIER_DIR, DEVICE)

    # ── Load Mistral (once, shared across all emotions) ───────────────────────
    clear_gpu_memory()
    print("\n[3/6] Loading Mistral model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, use_fast=True,
        token=HF_TOKEN if HF_TOKEN else True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        clear_gpu_memory()
        model = Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=DTYPE,
            device_map="auto",
            token=HF_TOKEN if HF_TOKEN else True,
            low_cpu_mem_usage=True,
        )
        print(f"   ✅ Model loaded with device_map: {model.hf_device_map}")
    except Exception as e:
        print(f"   ⚠️  auto device map failed ({e}), trying cuda:0...")
        clear_gpu_memory()
        try:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=DTYPE,
                device_map="cuda:0",
                token=HF_TOKEN if HF_TOKEN else True,
                low_cpu_mem_usage=True,
            )
        except Exception as e2:
            print(f"   ⚠️  cuda:0 failed ({e2}), falling back to CPU...")
            model = Mistral3ForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=DTYPE,
                device_map="cpu",
                token=HF_TOKEN if HF_TOKEN else True,
            )

    model.eval()
    model.config.use_cache = True
    clear_gpu_memory()
    print("\n   💾 Memory after model load:")
    get_gpu_memory_info()

    layer_ids = parse_layers(LAYERS)
    print(f"\n   ✅ Steering layers: {layer_ids}")

    # ── Load direction vectors once (shared across all emotions) ──────────────
    print("\n[4/6] Loading direction vectors...")
    direction_vectors = {}
    for direction_type in ["mlp", "attention"]:
        try:
            direction_vectors[direction_type] = load_directions(direction_type)
            print(f"   ✅ Loaded {direction_type} directions for: {list(direction_vectors[direction_type].keys())}")
        except Exception as e:
            print(f"   ❌ Failed to load {direction_type} directions: {e}")
            direction_vectors[direction_type] = None

    # ── Save master metadata for the entire run ───────────────────────────────
    master_metadata = {
        "experiment_id":   timestamp,
        "pathway":         "pathway1_reinject_every_turn",
        "model":           MODEL_NAME,
        "all_emotions":    ALL_EMOTIONS,
        "steering_parameters": {
            "base_alpha": ALPHA,
            "layers":     LAYERS,
            "last_k":     LAST_K,
            "scale":      SCALE,
            "note":       "effective_alpha = base_alpha × vector_intensity per turn",
        },
        "classifier_dir":   str(CLASSIFIER_DIR),
        "direction_types":  ["mlp", "attention"],
        "system_prompt":    False,
        "context_handling": "full_history_preserved",
        "output_directory": str(parent_output_dir),
        "timestamp":        timestamp,
        "gpu_info": {
            "cuda_available": torch.cuda.is_available(),
            "num_gpus":       torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "visible_gpus":   os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        },
    }
    with open(parent_output_dir / "master_metadata.json", "w") as f:
        json.dump(master_metadata, f, indent=2, ensure_ascii=False)

    # ── Process each emotion one by one ───────────────────────────────────────
    print("\n[5/6] Starting emotion-by-emotion processing...")
    
    results_summary = {}
    
    for emotion_idx, target_emotion in enumerate(ALL_EMOTIONS, 1):
        print("\n" + "=" * 70)
        print(f"🎯 [{emotion_idx}/{len(ALL_EMOTIONS)}] PROCESSING EMOTION: {target_emotion.upper()}")
        print("=" * 70)
        
        # Load scenario for this emotion
        try:
            scenario = load_scenario(target_emotion)
            print(f"   ✅ Loaded {len(scenario['turns'])} turns")
        except FileNotFoundError as e:
            print(f"   ❌ Skipping {target_emotion}: {e}")
            results_summary[target_emotion] = {"status": "failed", "error": str(e)}
            continue
        
        # Create emotion-specific subdirectory under parent output dir
        emotion_output_dir = parent_output_dir / target_emotion
        emotion_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   📁 Output: {emotion_output_dir}")
        
        # Save emotion-specific metadata
        emotion_metadata = {
            "experiment_id":   timestamp,
            "target_emotion":  target_emotion,
            "total_turns":     len(scenario["turns"]),
            "scenario_context": scenario.get("context", ""),
            "timestamp":       time.time(),
        }
        with open(emotion_output_dir / "emotion_metadata.json", "w") as f:
            json.dump(emotion_metadata, f, indent=2, ensure_ascii=False)
        
        # Run both direction types for this emotion
        for direction_type in ["mlp", "attention"]:
            if direction_vectors.get(direction_type) is None:
                print(f"   ⚠️  Skipping {direction_type} — vectors not loaded")
                continue
                
            print(f"\n   ── {direction_type.upper()} directions ──")
            clear_gpu_memory()
            
            try:
                run_pathway1(
                    model           = model,
                    tokenizer       = tokenizer,
                    classifier      = classifier,
                    all_directions  = direction_vectors[direction_type],
                    scenario        = scenario,
                    direction_type  = direction_type,
                    layer_ids       = layer_ids,
                    output_dir      = emotion_output_dir,
                )
            except Exception as e:
                print(f"   ❌ Error running {direction_type} for {target_emotion}: {e}")
                results_summary[target_emotion] = {"status": "partial_fail", "error": str(e)}
        
        results_summary[target_emotion] = {"status": "completed", "output_dir": str(emotion_output_dir)}
        print(f"\n   ✅ Completed {target_emotion.upper()}")
        
        # Clear memory between emotions
        clear_gpu_memory()
        time.sleep(2)

    # ── Save results summary ──────────────────────────────────────────────────
    print("\n[6/6] Saving final summary...")
    summary = {
        "experiment_id": timestamp,
        "total_emotions": len(ALL_EMOTIONS),
        "emotions_processed": results_summary,
        "output_directory": str(parent_output_dir),
        "completion_time": time.time(),
    }
    with open(parent_output_dir / "all_emotions_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ ALL EMOTIONS COMPLETE!")
    print(f"📁 Master output directory: {parent_output_dir}")
    print("=" * 80)
    
    # Print summary table
    print("\n📊 SUMMARY:")
    print("-" * 50)
    for emotion, status in results_summary.items():
        status_str = status.get("status", "unknown")
        if status_str == "completed":
            print(f"   ✅ {emotion.upper()}: {status_str}")
        else:
            print(f"   ❌ {emotion.upper()}: {status_str}")
    print("-" * 50)


if __name__ == "__main__":
    main()