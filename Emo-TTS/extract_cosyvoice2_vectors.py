#!/usr/bin/env python3
"""
extract_cosyvoice2_vectors_FIXED.py
====================================
Fixes applied:
  FIX 1 — Full recursive float32 conversion (covers Qwen2 LLM weights)
  FIX 2 — Disable autocast at the context manager level, not just forward()
           The LLM runs in a thread (llm_job), so patching forward() alone
           does NOT propagate. We patch model.llm.llm_context instead.
  FIX 3 — Hook attachment point consistency (norm1 pre-hook everywhere)
  FIX 4 — NaN/Inf guard on raw difference norm
"""

import os
import sys
import json
import random
import traceback
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR             = "/workspace/audio-em"
DATASET_BASE_DIR     = os.path.join(BASE_DIR, "dataset")
STEERING_JSON_DIR    = os.path.join(BASE_DIR, "emo-tts", "data", "used", "steering")
OUTPUT_BASE_DIR      = os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector", "cosyvoice2")

COSYVOICE2_LOCAL_DIR = "/workspace/audio-em/emo-tts/models/cosyvoice2"
COSYVOICE2_HF_REPO   = "FunAudioLLM/CosyVoice2-0.5B"
COSYVOICE_REPO_DIR   = "/workspace/audio-em/emo-tts/models/CosyVoice"

EMOTIONS        = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]
NEUTRAL_EMOTION = "neutral"

TOTAL_BLOCKS  = 56
HIDDEN_DIM    = 512
TARGET_LAYERS = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51]

MAX_NEUTRAL_SAMPLES   = 1000
MAX_EMOTIONAL_SAMPLES = 1000
SAVE_CHECKPOINT_EVERY_N_SAMPLES = 50
CLEAR_CACHE_EVERY_N_SAMPLES = 10

NEUTRAL_TEXTS = [
    "The sun rises in the east and sets in the west.",
    "She walked along the quiet path near the old library.",
    "The conference begins at nine o'clock in the morning.",
    "Please place your order before the kitchen closes.",
    "The children played in the yard until the evening.",
    "He carefully arranged the books on the shelf.",
    "The train arrived at the station right on time.",
    "We need to finish the report before the deadline.",
    "The weather forecast shows rain for the weekend.",
    "She read the letter twice before putting it away.",
]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(f"   [GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved]")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 + FIX 2: MODEL LOADING WITH PROPER DTYPE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def _inject_syspath():
    repo   = COSYVOICE_REPO_DIR
    matcha = os.path.join(repo, "third_party", "Matcha-TTS")
    for p in [repo, matcha]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def force_entire_model_float32(model):
    """
    FIX 1: Recursively convert EVERY parameter and buffer to float32.
    CosyVoice2 is a plain Python wrapper (not nn.Module), so we must
    discover and convert each nn.Module it holds by inspecting its attrs.
    """
    print("  Converting ALL parameters/buffers to float32 (recursive)...")
    converted = 0

    _FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float64}

    def convert_nn_module(module: torch.nn.Module, prefix: str = ""):
        nonlocal converted
        for name, param in module.named_parameters():
            if param.dtype in _FLOAT_DTYPES:
                param.data = param.data.float()
                converted += 1
        for name, buf in module.named_buffers():
            if buf.dtype in _FLOAT_DTYPES:
                try:
                    buf.data = buf.data.float()
                    converted += 1
                except Exception:
                    pass

    # Walk every attribute of the wrapper and convert nn.Modules found
    visited = set()

    def walk(obj, depth=0):
        if depth > 10 or id(obj) in visited:
            return
        visited.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            convert_nn_module(obj)
            return  # named_parameters/buffers already recurses children
        # For plain Python objects, inspect their attributes
        try:
            attrs = vars(obj)
        except TypeError:
            return
        for attr_name, attr_val in attrs.items():
            if attr_name.startswith("_"):
                continue
            walk(attr_val, depth + 1)

    walk(model)
    print(f"  Converted {converted} tensors to float32.")
    return model


def disable_autocast_on_model(model):
    """
    FIX 2: The LLM inference runs inside a thread (llm_job in model.py).
    Patching the forward() method doesn't help because the autocast context
    is set at a higher level via model.llm.llm_context.

    Strategy: replace llm_context with nullcontext() so no autocast is
    applied, then the float32 weights work correctly.
    """
    print("  Disabling autocast context on LLM...")

    # Primary target: model.llm.llm_context (used in cli/model.py llm_job)
    patched = False
    if hasattr(model, 'llm'):
        llm_obj = model.llm
        if hasattr(llm_obj, 'llm_context'):
            original = llm_obj.llm_context
            print(f"    Found llm_context: {type(original).__name__} — replacing with nullcontext")
            llm_obj.llm_context = nullcontext()
            patched = True

        # Also patch fp16 flag to prevent any autocast being re-created
        if hasattr(llm_obj, 'fp16'):
            llm_obj.fp16 = False
        if hasattr(llm_obj, 'llm') and hasattr(llm_obj.llm, 'fp16'):
            llm_obj.llm.fp16 = False

    # Secondary: model.model level
    if hasattr(model, 'model'):
        m = model.model
        if hasattr(m, 'llm_context'):
            m.llm_context = nullcontext()
            patched = True
        if hasattr(m, 'fp16'):
            m.fp16 = False

    if patched:
        print("  ✓ autocast disabled via nullcontext replacement")
    else:
        print("  [WARN] llm_context not found — trying fp16=False only")

    # Patch the flow decoder autocast too (second autocast in the traceback)
    if hasattr(model, 'model') and hasattr(model.model, 'flow'):
        flow = model.model.flow
        if hasattr(flow, 'fp16'):
            flow.fp16 = False
        # The flow forward uses: with torch.cuda.amp.autocast(self.fp16)
        # Setting fp16=False means autocast(False) = no-op
        print("  ✓ flow.fp16 set to False")

    return model


def load_model():
    print("Loading CosyVoice2 model...")
    _inject_syspath()

    yaml_path = os.path.join(COSYVOICE2_LOCAL_DIR, "cosyvoice2.yaml")
    if not os.path.exists(yaml_path):
        from huggingface_hub import snapshot_download
        print(f"  Downloading from HuggingFace: {COSYVOICE2_HF_REPO} …")
        snapshot_download(repo_id=COSYVOICE2_HF_REPO, local_dir=COSYVOICE2_LOCAL_DIR)

    from cosyvoice.cli.cosyvoice import CosyVoice2

    # Load with fp16=False so it doesn't set up BFloat16 from the start
    model = CosyVoice2(
        model_dir=COSYVOICE2_LOCAL_DIR,
        load_jit=False, load_trt=False, fp16=False,
    )

    # FIX 2: disable autocast BEFORE float32 conversion to avoid any
    # race with threads that might start during conversion
    model = disable_autocast_on_model(model)

    # FIX 1: convert everything to float32 (catches Qwen2 BFloat16 weights)
    model = force_entire_model_float32(model)

    # Freeze and eval all submodules
    inner  = model.model
    frozen = 0
    for attr_name in vars(inner):
        try:
            attr = getattr(inner, attr_name)
        except Exception:
            continue
        if isinstance(attr, torch.nn.Module):
            attr.eval()
            for p in attr.parameters():
                p.requires_grad_(False)
            frozen += 1

    print(f"  Model loaded. ({frozen} sub-modules frozen)")
    print_gpu_memory()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_transformer_blocks(model) -> List[object]:
    est    = model.model.flow.decoder.estimator
    blocks = []
    for b in est.down_blocks[0][1]:
        blocks.append(b)
    for i in range(len(est.mid_blocks)):
        for b in est.mid_blocks[i][1]:
            blocks.append(b)
    for b in est.up_blocks[0][1]:
        blocks.append(b)
    print(f"  Collected {len(blocks)} BasicTransformerBlocks")
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: ACTIVATION CAPTURE (norm1 pre-hook, consistent with steering)
# ─────────────────────────────────────────────────────────────────────────────

class ActivationCapture:
    def __init__(self, blocks: List[object], target_indices: List[int],
                 avg_seq_len: int):
        self.blocks         = blocks
        self.target_indices = target_indices
        self.avg_seq_len    = avg_seq_len
        self.running_sum: Dict[int, Optional[torch.Tensor]] = {i: None for i in target_indices}
        self.count:       Dict[int, int]                    = {i: 0    for i in target_indices}
        self._hooks: list = []
        self._current_activation: Dict[int, Optional[torch.Tensor]] = {}

    def _make_hook(self, block_idx: int):
        def hook(module, args):
            x = args[0] if isinstance(args, tuple) else args
            if x.dim() == 3:
                x = x[0]
            elif x.dim() == 2:
                pass
            else:
                return
            self._current_activation[block_idx] = x.detach().cpu().float()
        return hook

    def register(self):
        self._hooks = []
        registered  = 0
        for idx in self.target_indices:
            if idx >= len(self.blocks):
                print(f"  [WARN] block index {idx} out of range, skipped")
                continue
            block = self.blocks[idx]
            if hasattr(block, "norm1"):
                h = block.norm1.register_forward_pre_hook(self._make_hook(idx))
            else:
                h = block.register_forward_pre_hook(self._make_hook(idx))
                print(f"  [WARN] Block {idx} has no norm1; hooked block input instead")
            self._hooks.append(h)
            registered += 1
        print(f"  Registered {registered} norm1 pre-hooks on target blocks.")

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def add_current_sample(self):
        for idx in self.target_indices:
            if idx not in self._current_activation or self._current_activation[idx] is None:
                continue
            a = self._current_activation[idx]
            if a.shape[0] != self.avg_seq_len:
                a = a.unsqueeze(0).permute(0, 2, 1).float()
                a = F.interpolate(a, size=self.avg_seq_len, mode="nearest")
                a = a.permute(0, 2, 1).squeeze(0)
            if self.running_sum[idx] is None:
                self.running_sum[idx] = a.clone()
            else:
                self.running_sum[idx] += a
            self.count[idx] += 1
            self._current_activation[idx] = None

    def reset(self):
        for idx in self.target_indices:
            self.running_sum[idx] = None
            self.count[idx]       = 0
        self._current_activation = {}

    def get_mean(self) -> Dict[int, torch.Tensor]:
        means = {}
        for idx in self.target_indices:
            if self.count[idx] > 0 and self.running_sum[idx] is not None:
                means[idx] = self.running_sum[idx] / self.count[idx]
            else:
                means[idx] = torch.zeros(self.avg_seq_len, 256)
        return means


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_WAV_KWARG: Optional[str] = None


def _resolve_prompt_kwarg(model) -> str:
    global _PROMPT_WAV_KWARG
    if _PROMPT_WAV_KWARG is not None:
        return _PROMPT_WAV_KWARG
    import inspect
    sig = inspect.signature(model.inference_zero_shot)
    for cand in ["prompt_wav", "prompt_speech_16k", "prompt_audio_16k"]:
        if cand in sig.parameters:
            _PROMPT_WAV_KWARG = cand
            return cand
    _PROMPT_WAV_KWARG = "prompt_wav"
    return _PROMPT_WAV_KWARG


def run_forward(model, gen_text: str, ref_audio_path: str,
                ref_text: str) -> bool:
    if not ref_text or not ref_text.strip():
        ref_text = "This is a reference audio sample."
    kwarg = _resolve_prompt_kwarg(model)
    try:
        with torch.no_grad():
            gen = model.inference_zero_shot(
                tts_text=gen_text,
                prompt_text=ref_text,
                **{kwarg: ref_audio_path},
                stream=False,
            )
            for _ in gen:
                pass
        return True
    except Exception as e:
        if not hasattr(run_forward, '_count'):
            run_forward._count = 0
        run_forward._count += 1
        if run_forward._count <= 5 or run_forward._count % 50 == 0:
            print(f"  [WARN] Forward failed ({run_forward._count}): {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(emotion: str, max_n: int) -> List[Dict]:
    json_path = os.path.join(STEERING_JSON_DIR, f"{emotion}_steering.json")
    if not os.path.exists(json_path):
        print(f"  [WARN] Not found: {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.get("files", raw.get("samples", []))
    valid = []
    for item in items:
        audio_path = os.path.join(DATASET_BASE_DIR, item.get("audio_path", ""))
        text       = (item.get("transcription") or item.get("transcript") or "").strip()
        if os.path.isfile(audio_path) and text and len(valid) < max_n:
            valid.append({"full_audio_path": audio_path, "transcription": text})
    print(f"  Loaded {len(valid)} '{emotion}' samples")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE LENGTH PROBE
# ─────────────────────────────────────────────────────────────────────────────

def probe_seq_length(model, blocks: List[object], samples: List[Dict]) -> int:
    probe_idx = TARGET_LAYERS[0]
    cap       = ActivationCapture(blocks, [probe_idx], 256)
    cap.register()
    lengths = []
    for s in samples[:min(20, len(samples))]:
        cap.reset()
        if run_forward(model, random.choice(NEUTRAL_TEXTS),
                       s["full_audio_path"], s["transcription"]):
            act = cap._current_activation.get(probe_idx)
            if act is not None:
                lengths.append(act.shape[0])
        cap._current_activation = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cap.remove()
    if not lengths:
        default = 256
        print(f"  Using default seq length: {default}")
        return default
    avg = int(np.mean(lengths))
    print(f"  Probed avg sequence length: {avg}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_vectors(
    model, blocks, neutral_samples, emotional_samples,
    target_indices, avg_seq_len, emotion,
) -> Dict[int, torch.Tensor]:

    print(f"\n  Processing {len(neutral_samples)} neutral samples...")
    neutral_cap = ActivationCapture(blocks, target_indices, avg_seq_len)
    neutral_cap.register()

    success_neutral = 0
    for idx, sample in enumerate(tqdm(neutral_samples, desc="    Neutral")):
        if run_forward(model, random.choice(NEUTRAL_TEXTS),
                       sample["full_audio_path"], sample["transcription"]):
            neutral_cap.add_current_sample()
            success_neutral += 1
        if (idx + 1) % CLEAR_CACHE_EVERY_N_SAMPLES == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    neutral_cap.remove()
    print(f"  ✓ Neutral: {success_neutral}/{len(neutral_samples)} succeeded")

    if success_neutral == 0:
        raise RuntimeError("All neutral forward passes failed")

    neutral_means = neutral_cap.get_mean()

    checkpoint_dir = os.path.join(OUTPUT_BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({k: v.cpu() for k, v in neutral_means.items()},
               os.path.join(checkpoint_dir, f"{emotion}_neutral_means.pt"))

    print(f"\n  Processing {len(emotional_samples)} emotional samples...")
    emotional_cap = ActivationCapture(blocks, target_indices, avg_seq_len)
    emotional_cap.register()

    success_emotional = 0
    for idx, sample in enumerate(tqdm(emotional_samples, desc="    Emotional")):
        if run_forward(model, random.choice(NEUTRAL_TEXTS),
                       sample["full_audio_path"], sample["transcription"]):
            emotional_cap.add_current_sample()
            success_emotional += 1
        if (idx + 1) % SAVE_CHECKPOINT_EVERY_N_SAMPLES == 0:
            print(f"\n    [Checkpoint] {success_emotional} emotional samples processed")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    emotional_cap.remove()
    print(f"  ✓ Emotional: {success_emotional}/{len(emotional_samples)} succeeded")

    if success_emotional == 0:
        raise RuntimeError("All emotional forward passes failed")

    emotional_means = emotional_cap.get_mean()

    print("  Computing steering vectors...")
    vectors = {}
    for idx in target_indices:
        diff      = emotional_means[idx] - neutral_means[idx]
        collapsed = diff.mean(dim=0)
        raw_norm  = collapsed.norm().item()

        # FIX 4: NaN/Inf guard
        if not torch.isfinite(collapsed).all():
            raise RuntimeError(
                f"Block {idx} difference contains NaN/Inf before normalisation."
            )

        print(f"    Block {idx:2d}: raw Δ norm = {raw_norm:.6f}")

        if raw_norm < 1e-5:
            raise RuntimeError(
                f"Block {idx}: raw difference norm is {raw_norm:.2e} — nearly zero.\n"
                "Emotional and neutral activations are identical.\n"
                "Check that emotional audio files are truly expressive."
            )

        normed = collapsed / (collapsed.norm() + 1e-8)

        if not torch.isfinite(normed).all():
            raise RuntimeError(
                f"Block {idx} steering vector contains NaN/Inf "
                f"(raw_norm={raw_norm:.6f})."
            )

        vectors[idx] = normed

    for f_name in os.listdir(checkpoint_dir):
        if f_name.startswith(emotion) and f_name.endswith(".pt"):
            os.remove(os.path.join(checkpoint_dir, f_name))

    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_vectors(emotion: str, vectors: Dict[int, torch.Tensor], avg_seq_len: int):
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    payload = {
        "emotion":        emotion,
        "model":          "cosyvoice2",
        "avg_seq_len":    avg_seq_len,
        "target_layers":  TARGET_LAYERS,
        "vectors":        {str(k): v.cpu() for k, v in vectors.items()},
        "hook_point":     "norm1_pre_hook",
        "extraction_fix": "v2_norm1_emotional_text",
    }
    pt_path = os.path.join(OUTPUT_BASE_DIR, f"cosyvoice2_{emotion}_steering.pt")
    torch.save(payload, pt_path)
    print(f"  ✓ Saved: {pt_path}")

    import json as _json
    json_data = {k: ({str(kk): vv.cpu().tolist() for kk, vv in v.items()}
                      if k == "vectors" else v)
                 for k, v in payload.items()}
    json_path = pt_path.replace(".pt", ".json")
    with open(json_path, "w") as f:
        _json.dump(json_data, f, indent=2)
    print(f"  ✓ Saved: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check():
    print("\n" + "=" * 70)
    print("POST-EXTRACTION SANITY CHECK")
    print("=" * 70)

    loaded = {}
    for emo in EMOTIONS:
        pt = os.path.join(OUTPUT_BASE_DIR, f"cosyvoice2_{emo}_steering.pt")
        if not os.path.exists(pt):
            print(f"  [MISSING] {emo}")
            continue
        data = torch.load(pt, map_location="cpu")
        vecs = data.get("vectors", {})
        loaded[emo] = {int(k): v.float() for k, v in vecs.items()}

    if len(loaded) < 2:
        return

    all_ok = True
    pairs  = [(e1, e2) for i, e1 in enumerate(list(loaded.keys()))
                        for e2 in list(loaded.keys())[i+1:]]
    for l in TARGET_LAYERS:
        for e1, e2 in pairs:
            if l not in loaded.get(e1, {}) or l not in loaded.get(e2, {}):
                continue
            cos = F.cosine_similarity(
                loaded[e1][l].unsqueeze(0),
                loaded[e2][l].unsqueeze(0)
            ).item()
            if abs(cos) > 0.85:
                print(f"  [HIGH] L{l:2d} {e1:10s} vs {e2:10s}: cosine={cos:.4f}")
                all_ok = False

    if all_ok:
        print("  ✅ All inter-emotion cosine similarities < 0.85 — vectors look distinct!")
    else:
        print("\n  ⚠ Some vectors are still too similar. Check audio file quality.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CosyVoice2 Steering Vector Extraction  [FIXED v3]")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Target layers: {TARGET_LAYERS}")
    print(f"Neutral samples: {MAX_NEUTRAL_SAMPLES}")
    print(f"Emotional samples: {MAX_EMOTIONAL_SAMPLES}")
    print_gpu_memory()

    print("\n📁 Loading model...")
    model = load_model()

    print("\n📁 Collecting transformer blocks...")
    blocks = collect_transformer_blocks(model)

    print("\n📁 Loading neutral samples...")
    neutral_samples = load_samples(NEUTRAL_EMOTION, MAX_NEUTRAL_SAMPLES)
    if not neutral_samples:
        print("ERROR: No neutral samples found.")
        return

    print("\n📏 Probing sequence length...")
    avg_seq_len = probe_seq_length(model, blocks, neutral_samples[:50])

    for emotion in EMOTIONS:
        output_path = os.path.join(OUTPUT_BASE_DIR, f"cosyvoice2_{emotion}_steering.pt")
        if os.path.exists(output_path):
            data = torch.load(output_path, map_location="cpu")
            if data.get("extraction_fix") == "v2_norm1_emotional_text":
                print(f"\n⏭️  Skipping {emotion} (already fixed version)")
                continue
            else:
                print(f"\n♻️  Re-extracting {emotion} (old version — overwriting)")

        print(f"\n{'─' * 60}")
        print(f"🎭 Processing: {emotion.upper()}")
        print(f"{'─' * 60}")

        emotional_samples = load_samples(emotion, MAX_EMOTIONAL_SAMPLES)
        if not emotional_samples:
            print(f"  ⚠ No samples for {emotion}")
            continue

        try:
            vectors = extract_vectors(
                model=model,
                blocks=blocks,
                neutral_samples=neutral_samples,
                emotional_samples=emotional_samples,
                target_indices=TARGET_LAYERS,
                avg_seq_len=avg_seq_len,
                emotion=emotion,
            )
            save_vectors(emotion, vectors, avg_seq_len)
            print(f"  ✓ Completed {emotion}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory()

    print("\n" + "=" * 70)
    print("✅ Done!")
    print("=" * 70)

    sanity_check()


if __name__ == "__main__":
    main()