"""
cosyvoice2_hooks.py  —  Emotion steering for CosyVoice2
========================================================
Fixes applied:
  1. BFloat16/Float32 mismatch in Qwen2 LLM (runs in a thread, so weight
     conversion at load-time is not enough — we patch via a forward pre-hook
     that casts inputs on every call).
  2. Ref audio passed as a 16 kHz WAV file path (frontend.py calls
     torchaudio.load() internally, so it must be a path, not a tensor).
  3. Unicode punctuation sanitized before passing to the text frontend.
"""

import os, sys, tempfile
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import soundfile as sf
import librosa
from contextlib import nullcontext

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
STEERED_LAYERS = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51]
ALPHA_DEFAULT  = 6.0
ALPHA_MAX      = 7.0

COSYVOICE_REPO = "/workspace/audio-em/emo-tts/models/CosyVoice"
MODEL_DIR      = "/workspace/audio-em/emo-tts/models/cosyvoice2"
VECTORS_DIR    = "/workspace/audio-em/emo-tts/results/activation_vector_old/cosyvoice2"
OUTPUT_DIR     = "/workspace/audio-em/emo-tts/results/generated/cosyvoice2"

EMOTIONS = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]

REF_TEXT = (
    "We should only dream about becoming IAS officers, doctors and engineers. "
    "In this nation of Mahatma Gandhi and vocational education, why cannot I dream a nation?"
)
CLONE_REF_AUDIO = "/workspace/audio-em/emo-tts/ref-speech-clip.mp3"

# Plain ASCII only — no em-dashes, no curly quotes (they break the text frontend)
CUSTOM_TEXT = (
    "It sounds like you are going through a tough moment - being kind to yourself is so important! "
    "It is completely normal to say or do things we later regret, especially when emotions run high."
)

# ─── TEXT SANITIZER ──────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    """Replace Unicode punctuation that breaks CosyVoice2's text normalizer."""
    return (
        text
        .replace("\u2014", " - ")   # em dash
        .replace("\u2013", " - ")   # en dash
        .replace("\u2018", "'")     # left single quote
        .replace("\u2019", "'")     # right single quote
        .replace("\u201c", '"')     # left double quote
        .replace("\u201d", '"')     # right double quote
        .replace("\u2026", "...")   # ellipsis
    )

# ─── REF AUDIO → TEMP 16 kHz WAV ─────────────────────────────────────────────

def make_ref_wav(src_path: str) -> str:
    """
    Load any audio file, resample to 16 kHz mono, write to a temp WAV and
    return the path.  CosyVoice2's frontend.py calls torchaudio.load() on
    whatever you pass as prompt_wav, so it MUST be a file path.
    """
    audio, sr = librosa.load(src_path, sr=None, mono=True)
    if sr != 16000:
        print(f"  Resampling ref audio {sr} Hz → 16000 Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, 16000, subtype="PCM_16")
    tmp.close()
    print(f"  Ref WAV: {tmp.name}  ({len(audio)/16000:.2f}s)")
    return tmp.name

# ─── MODEL LOADING ───────────────────────────────────────────────────────────

def _cast_input_hook(module, args):
    """
    Forward pre-hook: cast all float-tensor inputs to match the first
    weight's dtype so Float32 inputs don't crash BFloat16 layers.
    This runs inside the LLM thread, so it is the only reliable fix.
    """
    target_dtype = next(
        (p.dtype for p in module.parameters() if p.is_floating_point()), None
    )
    if target_dtype is None:
        return args
    new_args = tuple(
        a.to(target_dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
        for a in args
    )
    return new_args


def load_cosyvoice2():
    matcha = os.path.join(COSYVOICE_REPO, "third_party", "Matcha-TTS")
    for p in [COSYVOICE_REPO, matcha]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    from cosyvoice.cli.cosyvoice import CosyVoice2

    print(f"  Loading from {MODEL_DIR} ...")
    cv2 = CosyVoice2(model_dir=MODEL_DIR, load_jit=False, load_trt=False, fp16=False)

    # ── Disable fp16 flags ────────────────────────────────────────────────────
    for obj in [cv2, getattr(cv2, 'llm', None), getattr(cv2, 'flow', None),
                getattr(getattr(cv2, 'model', None), 'flow', None)]:
        if obj is not None and hasattr(obj, 'fp16'):
            obj.fp16 = False

    if hasattr(cv2, 'llm') and hasattr(cv2.llm, 'llm_context'):
        cv2.llm.llm_context = nullcontext()

    # ── Convert flow / hift weights to float32 ────────────────────────────────
    _HALF = {torch.float16, torch.bfloat16}

    def to_f32(mod):
        if mod is None:
            return
        for p in mod.parameters():
            if p.dtype in _HALF:
                p.data = p.data.float()
        for b in mod.buffers():
            if b.dtype in _HALF:
                b.data = b.data.float()
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)

    to_f32(getattr(cv2, 'flow', None))
    if hasattr(cv2, 'model'):
        for name in vars(cv2.model):
            try:
                attr = getattr(cv2.model, name)
                if isinstance(attr, torch.nn.Module):
                    to_f32(attr)
            except Exception:
                pass

    # ── Patch Qwen2 LLM with a dtype-cast hook ────────────────────────────────
    # The LLM (Qwen2) stays in BFloat16 — that's fine.  We just add a
    # pre-hook on its Attention layers so incoming float32 activations
    # are cast to bfloat16 before the matmul.
    llm_module = None
    if hasattr(cv2, 'llm') and hasattr(cv2.llm, 'llm'):
        llm_module = cv2.llm.llm          # Qwen2ForCausalLM
    elif hasattr(cv2, 'llm') and hasattr(cv2.llm, 'model'):
        llm_module = cv2.llm.model

    if llm_module is not None:
        hooked = 0
        for mod in llm_module.modules():
            cls = type(mod).__name__
            if "Attention" in cls or "MLP" in cls or "Linear" in cls:
                mod.register_forward_pre_hook(_cast_input_hook)
                hooked += 1
        print(f"  Registered dtype-cast hooks on {hooked} LLM sub-modules")

    print("  ✅ CosyVoice2 ready")
    return cv2

# ─── VECTOR LOADING ──────────────────────────────────────────────────────────

def load_vectors(device=DEVICE):
    vectors = {}
    if not os.path.exists(VECTORS_DIR):
        print(f"  ⚠ Vectors dir not found: {VECTORS_DIR}")
        return vectors

    for emotion in EMOTIONS:
        candidates = [
            os.path.join(VECTORS_DIR, f"cosyvoice2_{emotion}_steervec.pt"),
            os.path.join(VECTORS_DIR, f"cosyvoice2_{emotion}_steering.pt"),
        ]
        loaded = False
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                data = torch.load(path, map_location=device)
                raw  = (data.get("steering_vectors") or data.get("vectors") or data
                        ) if isinstance(data, dict) else data
                vectors[emotion] = {
                    (int(k) if isinstance(k, str) else k): v.to(device).float()
                    for k, v in raw.items()
                }
                print(f"  ✓ {emotion}: {len(vectors[emotion])} layers")
                loaded = True
                break
            except Exception as e:
                print(f"  ⚠ Failed to load {path}: {e}")
        if not loaded:
            print(f"  ⚠ No vector file found for {emotion}")

    return vectors

# ─── BLOCK COLLECTION ────────────────────────────────────────────────────────

def collect_blocks(cv2):
    try:
        if hasattr(cv2, 'model') and hasattr(cv2.model, 'flow'):
            est = cv2.model.flow.decoder.estimator
        elif hasattr(cv2, 'flow') and hasattr(cv2.flow, 'decoder'):
            est = cv2.flow.decoder.estimator
        else:
            print("  ⚠ Cannot locate flow.decoder.estimator")
            return []

        blocks = []
        if hasattr(est, 'down_blocks') and est.down_blocks:
            for b in est.down_blocks[0][1]:
                blocks.append(b)
        if hasattr(est, 'mid_blocks'):
            for grp in est.mid_blocks:
                for b in grp[1]:
                    blocks.append(b)
        if hasattr(est, 'up_blocks') and est.up_blocks:
            for b in est.up_blocks[0][1]:
                blocks.append(b)

        print(f"  ✓ {len(blocks)} transformer blocks collected")
        return blocks
    except Exception as e:
        print(f"  ⚠ collect_blocks: {e}")
        return []

# ─── INFERENCE ───────────────────────────────────────────────────────────────

def voice_clone(cv2, text: str, ref_wav_path: str, ref_text: str):
    text     = sanitize(text)
    ref_text = sanitize(ref_text)

    import inspect
    params = list(inspect.signature(cv2.inference_zero_shot).parameters.keys())
    wav_kwarg = (
        "prompt_speech_16k" if "prompt_speech_16k" in params else
        "prompt_wav"        if "prompt_wav"        in params else
        params[2]
    )

    chunks = []
    try:
        with torch.no_grad():
            for result in cv2.inference_zero_shot(
                tts_text=text, prompt_text=ref_text,
                **{wav_kwarg: ref_wav_path}, stream=False
            ):
                if isinstance(result, dict):
                    for key in ("tts_speech", "speech"):
                        if key in result:
                            chunks.append(result[key]); break
                elif isinstance(result, torch.Tensor):
                    chunks.append(result)
    except Exception as e:
        print(f"    ❌ inference error: {e}")
        import traceback; traceback.print_exc()
        return None, None

    if not chunks:
        print("    ❌ model returned no audio")
        return None, None

    audio = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
    audio = audio.squeeze().cpu().numpy()

    if np.abs(audio).max() < 0.01:
        print("    ❌ output is near-silent")
        return None, None

    sr = getattr(cv2, 'sample_rate', None) or getattr(cv2, 'sr', None) or 22050
    print(f"    ✅ {len(audio)/sr:.2f}s @ {sr} Hz  amp={np.abs(audio).max():.3f}")
    return audio, sr

# ─── STEERING HOOKS ──────────────────────────────────────────────────────────

def _bcast(vec, B, T, dtype, device):
    v = (vec / (vec.norm() + 1e-8)).to(device=device, dtype=dtype)
    return v.unsqueeze(0).unsqueeze(0).expand(B, T, -1).contiguous()

def _steer_hook(block_idx, vecs, alpha):
    def hook(module, args, kwargs):
        if block_idx not in vecs:
            return args, kwargs
        x = args[0]
        if x.dim() == 3:
            x = x + alpha * _bcast(vecs[block_idx], x.size(0), x.size(1), x.dtype, x.device)
            return (x,) + args[1:], kwargs
        return args, kwargs
    return hook

class HookManager:
    def __init__(self, blocks, hooks):
        self.handles = []
        for idx, fn in hooks.items():
            if idx >= len(blocks):
                continue
            blk = blocks[idx]
            tgt = getattr(blk, 'norm1', None) or getattr(blk, 'attn1', None)
            if tgt is not None:
                self.handles.append(
                    tgt.register_forward_pre_hook(fn, with_kwargs=True)
                )

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

# ─── STEERER ─────────────────────────────────────────────────────────────────

class CosyVoice2Steerer:
    def __init__(self, cv2, vectors=None, device=DEVICE):
        self.cv2    = cv2
        self.blocks = collect_blocks(cv2)
        self.vecs   = vectors if vectors is not None else load_vectors(device)

    def baseline(self, text, ref_wav, ref_text):
        return voice_clone(self.cv2, text, ref_wav, ref_text)

    def with_emotion(self, text, ref_wav, emotion, alpha=ALPHA_DEFAULT, ref_text=""):
        ref_text = ref_text or REF_TEXT
        if not self.vecs.get(emotion):
            print(f"    ⚠ no vectors for {emotion} → baseline")
            return self.baseline(text, ref_wav, ref_text)

        alpha = float(np.clip(alpha, -ALPHA_MAX, ALPHA_MAX))
        hooks = {
            l: _steer_hook(l, self.vecs[emotion], alpha)
            for l in STEERED_LAYERS if l in self.vecs[emotion]
        }
        if not hooks:
            return self.baseline(text, ref_wav, ref_text)

        mgr = HookManager(self.blocks, hooks)
        try:
            return voice_clone(self.cv2, text, ref_wav, ref_text)
        finally:
            mgr.remove()

    def available(self):
        return [e for e in self.vecs if self.vecs[e]]

# ─── MAIN ────────────────────────────────────────────────────────────────────

def save(audio, sr, path):
    if audio is None:
        print(f"      ❌ skipped"); return
    sf.write(path, audio, sr)
    print(f"      ✅ {os.path.basename(path)}  ({len(audio)/sr:.2f}s)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("CosyVoice2 Emotion Steering")
    print("=" * 70)

    if not os.path.exists(CLONE_REF_AUDIO):
        print(f"❌ Ref audio not found: {CLONE_REF_AUDIO}"); return

    print("\n🔄 Preparing reference audio...")
    ref_wav = make_ref_wav(CLONE_REF_AUDIO)

    print("\n📦 Loading CosyVoice2...")
    try:
        cv2 = load_cosyvoice2()
    except Exception as e:
        import traceback; traceback.print_exc()
        os.unlink(ref_wav); return

    print("\n📦 Loading emotion vectors...")
    steerer = CosyVoice2Steerer(cv2)
    if not steerer.available():
        print(f"❌ No vectors in {VECTORS_DIR}"); os.unlink(ref_wav); return

    print(f"\n✅ Emotions available: {steerer.available()}")
    print(f"   Text: {CUSTOM_TEXT[:80]}...")

    # Baseline
    print("\n" + "=" * 70)
    print("🎵 BASELINE")
    print("=" * 70)
    audio, sr = steerer.baseline(CUSTOM_TEXT, ref_wav, REF_TEXT)
    save(audio, sr, os.path.join(OUTPUT_DIR, "baseline_neutral.wav"))

    # All emotions
    for emotion in EMOTIONS:
        print(f"\n{'=' * 70}\n🎵 {emotion.upper()}\n{'=' * 70}")

        for alpha in [4.0, 5.0, 6.0, 7.0]:
            print(f"\n  α={alpha}")
            audio, sr = steerer.with_emotion(CUSTOM_TEXT, ref_wav, emotion,
                                             alpha=alpha, ref_text=REF_TEXT)
            save(audio, sr, os.path.join(OUTPUT_DIR, f"{emotion}_alpha{alpha}.wav"))

        print(f"\n  α={ALPHA_DEFAULT} (default)")
        audio, sr = steerer.with_emotion(CUSTOM_TEXT, ref_wav, emotion,
                                         alpha=ALPHA_DEFAULT, ref_text=REF_TEXT)
        save(audio, sr, os.path.join(OUTPUT_DIR, f"{emotion}_default.wav"))

    try:
        os.unlink(ref_wav)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("✅ DONE —", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()