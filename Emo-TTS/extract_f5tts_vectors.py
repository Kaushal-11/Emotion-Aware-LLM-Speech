#!/usr/bin/env python3
"""
extract_f5tts_vectors_FIXED.py
==============================
Key fixes vs the original:

  FIX 2 — Hook type: forward_pre_hook throughout
           Original: register_forward_HOOK (captures output) during extraction
                     register_forward_PRE_hook (captures input) during steering
           Fixed: register_forward_PRE_hook for BOTH extraction and steering

  FIX 3 — NaN/Inf guard on raw difference norm
           If collapsed diff norm is near zero the normalized vector is garbage.
           We now raise early with a clear message.
"""

import os
import sys
import json
import random
import traceback
import time
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import timedelta

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR             = "/workspace/audio-em"
DATASET_BASE_DIR     = os.path.join(BASE_DIR, "dataset")
STEERING_JSON_DIR    = os.path.join(BASE_DIR, "emo-tts", "data", "used", "steering")
OUTPUT_BASE_DIR      = os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector")

F5TTS_MODEL_PATH     = "SWivid/F5-TTS"
F5TTS_LOCAL_DIR      = "/workspace/audio-em/emo-tts/models/f5tts"

EMOTIONS        = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]
NEUTRAL_EMOTION = "neutral"

F5TTS_CONFIG = {
    "num_layers":          22,
    "hidden_dim":          1024,
    "steered_layers":      [1, 6, 11, 16, 21],
    "m_neutral_samples":   1000,
    "n_emotional_samples": 1000,
    "batch_size":          1,
    "save_checkpoint_every_n_samples": 50,
    "probe_samples":       10,
    "clear_cache_every_n_samples": 10,
}

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


# ==============================================================================
# ACTIVATION CAPTURE  (FIX 2: uses forward_pre_hook throughout)
# ==============================================================================

class ActivationCapture:
    """
    Captures activations using register_forward_PRE_hook so the capture space
    matches the steering space in f5tts_hooks.py / inference_steering.py.
    """

    def __init__(self, model, target_layers: List[int]):
        self.model        = model
        self.target_layers = target_layers
        self.activations: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
        self._hooks: list = []
        self.current_sample_activations = {l: None for l in target_layers}

    def _make_hook(self, layer_idx: int):
        def hook(module, input_args):
            x = input_args[0] if isinstance(input_args, tuple) else input_args
            if x.dim() == 3:
                x = x[0].detach().cpu().float()  
            elif x.dim() == 2:
                x = x.detach().cpu().float()
            else:
                return
            self.current_sample_activations[layer_idx] = x
        return hook

    def register(self):
        self._hooks = []
        layers = self._get_layers()
        for l in self.target_layers:
            if l < len(layers):                
                h = layers[l].register_forward_pre_hook(self._make_hook(l))
                self._hooks.append(h)
        print(f"   ✓ Registered {len(self._hooks)} PRE-hooks on target layers")

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def capture_current_sample(self) -> bool:
        """Validate and commit current sample's activations. Returns False if rejected."""
        for l in self.target_layers:
            act = self.current_sample_activations[l]
            if act is None:
                self.current_sample_activations = {l: None for l in self.target_layers}
                return False
            if not torch.isfinite(act).all():
                print(f"      [WARN] NaN/Inf at layer {l}, sample rejected")
                self.current_sample_activations = {l: None for l in self.target_layers}
                return False

        for l in self.target_layers:
            self.activations[l].append(self.current_sample_activations[l])
            self.current_sample_activations[l] = None
        return True

    def reset(self):
        self.activations = {l: [] for l in self.target_layers}
        self.current_sample_activations = {l: None for l in self.target_layers}

    def _get_layers(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        for attr_path in [
            "transformer.transformer_blocks",
            "transformer.layers",
            "model.transformer.transformer_blocks",
        ]:
            obj = model
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (list, torch.nn.ModuleList)):
                    print(f"   Found {len(obj)} transformer blocks at '{attr_path}'")
                    return obj
            except AttributeError:
                continue
        raise AttributeError("Cannot find transformer blocks")

    def get_mean_activation(self, layer_idx: int, avg_seq_len: int) -> torch.Tensor:
        acts = self.activations.get(layer_idx, [])
        if not acts:
            return torch.zeros(avg_seq_len, F5TTS_CONFIG["hidden_dim"])
        interpolated = []
        for a in acts:
            if a.shape[0] != avg_seq_len:
                a = a.float().unsqueeze(0).permute(0, 2, 1)
                a = F.interpolate(a, size=avg_seq_len, mode="nearest")
                a = a.permute(0, 2, 1).squeeze(0)
            interpolated.append(a)
        return torch.stack(interpolated, dim=0).mean(dim=0)


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_f5tts(device: str = "cuda"):
    print("   Loading F5-TTS...")
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    if os.path.exists(F5TTS_LOCAL_DIR):
        model_local_dir = F5TTS_LOCAL_DIR
    else:
        from huggingface_hub import snapshot_download
        model_local_dir = snapshot_download(repo_id=F5TTS_MODEL_PATH,
                                            local_dir=F5TTS_LOCAL_DIR)

    safetensors_path = os.path.join(model_local_dir, "F5TTS_Base",
                                    "model_1200000.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"No checkpoint found at {safetensors_path}")

    model = load_model(
        model_cls=DiT,
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                       text_mask_padding=False, conv_layers=4, pe_attn_head=1),
        ckpt_path=safetensors_path,
        mel_spec_type="vocos",
        vocab_file=os.path.join(model_local_dir, "F5TTS_Base", "vocab.txt"),
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    print("   ✅ F5-TTS loaded")
    print_gpu_memory()
    return model, vocoder


# ==============================================================================
# CACHE CLEAR
# ==============================================================================

def clear_f5tts_preprocess_cache():
    try:
        import f5_tts.infer.utils_infer as _u
        cleared = 0
        for attr in ("_ref_audio_cache", "ref_audio_cache", "_cache",
                     "cache", "_preprocessed_cache"):
            if hasattr(_u, attr):
                obj = getattr(_u, attr)
                if isinstance(obj, dict):
                    n = len(obj)
                    obj.clear()
                    print(f"   [cache] cleared utils_infer.{attr} ({n} entries)")
                    cleared += 1
        if cleared == 0:
            print("   [cache] no cache dict found (non-fatal)")
    except Exception as e:
        print(f"   [cache] clear failed (non-fatal): {e}")


# ==============================================================================
# AUDIO VALIDATION
# ==============================================================================

def validate_audio_file(path: str) -> bool:
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.numel() == 0:
            return False
        if not torch.isfinite(wav).all():
            return False
        if wav.abs().max().item() < 1e-6:
            return False
        return True
    except Exception:
        return False


# ==============================================================================
# FORWARD PASS
# ==============================================================================

def run_forward(model, vocoder, gen_text: str, ref_audio_path: str,
                ref_text: str, device: str) -> bool:
    """
    Run one forward pass.
    gen_text       — the TEXT to synthesize (for neutral pass: neutral sentence;
                     for emotional pass: the TRANSCRIPTION of the emotional audio)
    ref_audio_path — reference audio for voice cloning (provides speaker style)
    ref_text       — transcription of ref_audio_path
    """
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    if not validate_audio_file(ref_audio_path):
        return False

    actual = model.module if hasattr(model, "module") else model
    safe_ref_text = (ref_text.strip() if ref_text and ref_text.strip()
                     else "This is a reference audio sample.")

    try:
        ref_audio_proc, resolved_text = preprocess_ref_audio_text(
            ref_audio_path, ref_text=safe_ref_text
        )
        if ref_audio_proc is None:
            return False
        if isinstance(ref_audio_proc, str) and not os.path.isfile(ref_audio_proc):
            return False

        with torch.no_grad():
            result = infer_process(
                ref_audio=ref_audio_proc, ref_text=resolved_text,
                gen_text=gen_text,
                model_obj=actual, vocoder=vocoder, device=device,
                speed=1.0, cross_fade_duration=0.15,
            )
        if not isinstance(result, tuple):
            audio, sr, _ = next(result)
        else:
            audio, sr, _ = result
        del result
        return True
    except Exception:
        return False


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_steering_data(emotion: str, max_samples: int) -> List[Dict]:
    json_path = os.path.join(STEERING_JSON_DIR, f"{emotion}_steering.json")
    if not os.path.exists(json_path):
        print(f"   ❌ File not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("files", data.get("samples", []))
    valid = []
    for item in samples:
        full_path = os.path.join(DATASET_BASE_DIR, item.get("audio_path", ""))
        text = (item.get("transcription") or item.get("transcript") or "").strip()
        if os.path.isfile(full_path) and text and len(valid) < max_samples:
            valid.append({"full_audio_path": full_path, "transcription": text})

    print(f"   Loaded {len(valid)} samples for {emotion}")
    return valid


# ==============================================================================
# SEQUENCE LENGTH PROBE
# ==============================================================================

def compute_avg_seq_length(model, vocoder, neutral_samples: List[Dict],
                            device: str) -> int:
    probe_layer = F5TTS_CONFIG["steered_layers"][0]
    capture = ActivationCapture(model, [probe_layer])
    capture.register()

    lengths = []
    n_probe = min(F5TTS_CONFIG["probe_samples"], len(neutral_samples))
    print(f"   Probing {n_probe} samples...")

    for i in range(n_probe):
        sample = neutral_samples[i]
        capture.current_sample_activations = {probe_layer: None}
        success = run_forward(
            model, vocoder,
            gen_text=random.choice(NEUTRAL_TEXTS),          # neutral text
            ref_audio_path=sample["full_audio_path"],
            ref_text=sample["transcription"],
            device=device,
        )
        if success and capture.current_sample_activations[probe_layer] is not None:
            lengths.append(capture.current_sample_activations[probe_layer].shape[0])
        capture.current_sample_activations = {probe_layer: None}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    capture.remove()
    print("   Clearing F5-TTS preprocess cache after probe...")
    clear_f5tts_preprocess_cache()

    if not lengths:
        default = 300
        print(f"   ⚠ Using default seq length: {default}")
        return default

    avg_len = int(np.mean(lengths))
    print(f"   Average sequence length: {avg_len} (from {len(lengths)} samples)")
    return avg_len


# ==============================================================================
# EXTRACTION
# ==============================================================================

def extract_vectors(
    model, vocoder,
    neutral_samples:   List[Dict],
    emotional_samples: List[Dict],
    target_layers:     List[int],
    avg_seq_len:       int,
    emotion:           str,
    device:            str,
) -> Dict[int, torch.Tensor]:
    
    checkpoint_dir = os.path.join(OUTPUT_BASE_DIR, "f5tts", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("   Clearing F5-TTS preprocess cache before extraction...")
    clear_f5tts_preprocess_cache()

    # ── Neutral pass ─────────────────────────────────────────────────────────
    print(f"\n   Processing {len(neutral_samples)} neutral samples...")
    capture = ActivationCapture(model, target_layers)
    capture.register()

    success_neutral = 0
    for idx, sample in enumerate(tqdm(neutral_samples, desc="      Neutral")):
        success = run_forward(
            model, vocoder,            
            gen_text=random.choice(NEUTRAL_TEXTS),
            ref_audio_path=sample["full_audio_path"],
            ref_text=sample["transcription"],
            device=device,
        )
        if success:
            if capture.capture_current_sample():
                success_neutral += 1
        if (idx + 1) % F5TTS_CONFIG["clear_cache_every_n_samples"] == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    capture.remove()
    print(f"   ✓ Neutral: {success_neutral}/{len(neutral_samples)} succeeded")

    if success_neutral == 0:
        raise RuntimeError("All neutral passes failed")

    print("   Computing neutral means...")
    neutral_means = {}
    for l in target_layers:
        neutral_means[l] = capture.get_mean_activation(l, avg_seq_len)
        print(f"      Layer {l}: {neutral_means[l].shape}")

    # ── Emotional pass ───────────────────────────────────────────────────────
    print(f"\n   Processing {len(emotional_samples)} emotional samples...")
    capture.reset()
    capture.register()

    success_emotional = 0
    for idx, sample in enumerate(tqdm(emotional_samples, desc="      Emotional")):
        success = run_forward(
            model, vocoder,            
            gen_text = random.choice(NEUTRAL_TEXTS),
            ref_audio_path=sample["full_audio_path"],
            ref_text=sample["transcription"],
            device=device,
        )
        if success:
            if capture.capture_current_sample():
                success_emotional += 1
        if (idx + 1) % F5TTS_CONFIG["save_checkpoint_every_n_samples"] == 0:
            print(f"\n      [Checkpoint] {success_emotional} emotional samples so far")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    capture.remove()
    print(f"   ✓ Emotional: {success_emotional}/{len(emotional_samples)} succeeded")

    if success_emotional == 0:
        raise RuntimeError("All emotional passes failed")

    print("   Computing emotional means...")
    emotional_means = {}
    for l in target_layers:
        emotional_means[l] = capture.get_mean_activation(l, avg_seq_len)

    # ── Compute steering vectors ──────────────────────────────────────────────
    print("   Computing steering vectors...")
    vectors = {}
    for l in target_layers:
        diff        = emotional_means[l] - neutral_means[l]
        collapsed   = diff.mean(dim=0)
        raw_norm    = collapsed.norm().item()

        print(f"      Layer {l}: raw Δ norm = {raw_norm:.6f}")

        if raw_norm < 1e-5:
            raise RuntimeError(
                f"Layer {l}: raw difference norm is {raw_norm:.2e} — nearly zero.\n"
                "This means emotional and neutral activations are identical.\n"
                "Check that emotional audio files actually contain expressive speech."
            )

        normed = collapsed / (collapsed.norm() + 1e-8)

        if not torch.isfinite(normed).all():
            raise RuntimeError(
                f"Layer {l} steering vector contains NaN/Inf "
                f"(raw_norm={raw_norm:.6f})."
            )

        vectors[l] = normed
        print(f"      Layer {l}: Δ norm = {raw_norm:.4f}  vec norm = {normed.norm():.6f}")

    return vectors


# ==============================================================================
# SAVE
# ==============================================================================

def save_vectors(emotion: str, vectors: Dict[int, torch.Tensor], avg_seq_len: int):
    output_dir = os.path.join(OUTPUT_BASE_DIR, "f5tts")
    os.makedirs(output_dir, exist_ok=True)

    save_data = {
        "emotion":        emotion,
        "model":          "f5tts",
        "avg_seq_len":    avg_seq_len,
        "steered_layers": list(vectors.keys()),
        "vectors":        {str(l): v.cpu() for l, v in vectors.items()},
        "extraction_fix": "v2_pre_hook_emotional_text",
    }

    pt_path = os.path.join(output_dir, f"f5tts_{emotion}_steering.pt")
    torch.save(save_data, pt_path)
    print(f"   ✅ Saved: {pt_path}")

    json_data = {k: ({str(kk): vv.tolist() for kk, vv in v.items()}
                      if k == "vectors" else v)
                 for k, v in save_data.items()}
    json_path = pt_path.replace(".pt", ".json")
    with open(json_path, "w") as f:
        import json
        json.dump(json_data, f, indent=2)
    print(f"   ✅ Saved: {json_path}")


# ==============================================================================
# QUICK SANITY CHECK AFTER EXTRACTION
# ==============================================================================

def sanity_check(output_dir: str, emotions: List[str], layers: List[int]):
    """Check inter-emotion cosine similarity. Should be < 0.85 after the fix."""
    print("\n" + "=" * 70)
    print("POST-EXTRACTION SANITY CHECK")
    print("=" * 70)

    loaded = {}
    for emo in emotions:
        pt = os.path.join(output_dir, f"f5tts_{emo}_steering.pt")
        if not os.path.exists(pt):
            print(f"  [MISSING] {emo}")
            continue
        data = torch.load(pt, map_location="cpu")
        vecs = data.get("vectors", {})
        loaded[emo] = {int(k): v.float() for k, v in vecs.items()}

    if len(loaded) < 2:
        print("  Not enough emotions to compare.")
        return

    all_ok = True
    pairs  = [(e1, e2) for i, e1 in enumerate(list(loaded.keys()))
                        for e2 in list(loaded.keys())[i+1:]]

    for l in layers:
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
        print("\n  ⚠ Some emotions still have high cosine similarity.")
        print("  Check that your emotional audio files are truly expressive.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("F5-TTS Steering Vector Extraction")
    print("=" * 70)
    print(f"\nKey fixes applied:")    
    print(f"\nConfiguration:")
    print(f"   Device        : {DEVICE}")
    print(f"   Target layers : {F5TTS_CONFIG['steered_layers']}")
    print(f"   Neutral samples : {F5TTS_CONFIG['m_neutral_samples']}")
    print(f"   Emotional samples : {F5TTS_CONFIG['n_emotional_samples']}")
    print_gpu_memory()

    print("\n📁 Loading model...")
    model, vocoder = load_f5tts(DEVICE)

    print("\n📁 Loading neutral samples...")
    neutral_samples = load_steering_data(NEUTRAL_EMOTION, F5TTS_CONFIG["m_neutral_samples"])
    if not neutral_samples:
        print("❌ No neutral samples found")
        return

    print("\n📏 Probing sequence length...")
    avg_seq_len = compute_avg_seq_length(model, vocoder, neutral_samples, DEVICE)

    output_dir = os.path.join(OUTPUT_BASE_DIR, "f5tts")

    for emotion in EMOTIONS:
        output_path = os.path.join(output_dir, f"f5tts_{emotion}_steering.pt")
        if os.path.exists(output_path):
            # Check if this is the old (broken) version
            data = torch.load(output_path, map_location="cpu")
            if data.get("extraction_fix") == "v2_pre_hook_emotional_text":
                print(f"\n⏭️  Skipping {emotion}")
                continue
            else:
                print(f"\n♻️  Re-extracting {emotion}")

        print(f"\n{'=' * 60}")
        print(f"🎭 Processing: {emotion.upper()}")
        print(f"{'=' * 60}")

        emotional_samples = load_steering_data(emotion, F5TTS_CONFIG["n_emotional_samples"])
        if not emotional_samples:
            print(f"   ⚠ No samples for {emotion}")
            continue

        start_time = time.time()

        try:
            vectors = extract_vectors(
                model=model,
                vocoder=vocoder,
                neutral_samples=neutral_samples,
                emotional_samples=emotional_samples,
                target_layers=F5TTS_CONFIG["steered_layers"],
                avg_seq_len=avg_seq_len,
                emotion=emotion,
                device=DEVICE,
            )
            save_vectors(emotion, vectors, avg_seq_len)
            elapsed = time.time() - start_time
            print(f"   ⏱  Time: {str(timedelta(seconds=int(elapsed)))}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory()

    print("\n" + "=" * 70)
    print("✅ Extraction done!")
    print("=" * 70)

    # Sanity check
    sanity_check(output_dir, EMOTIONS, F5TTS_CONFIG["steered_layers"])


if __name__ == "__main__":
    main()