"""
f5tts_hooks.py  —  Emotion steering for F5-TTS with multiple alpha values
==========================================================================
Generate speech with different emotion intensities.
"""

import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import soundfile as sf

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS      = 22
STEERED_LAYERS  = [1, 6, 11, 16, 21]
ALPHA_DEFAULT   = 5.0
ALPHA_MAX       = 4.5

F5TTS_LOCAL_DIR = "/workspace/audio-em/emo-tts/models/f5tts"
VECTORS_DIR     = "/workspace/audio-em/emo-tts/results/activation_vector/f5tts/final"
OUTPUT_DIR      = "/workspace/audio-em/emo-tts/results/generated/f5tts"

# Only use emotions you have vectors for
EMOTIONS        = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]

# ─── YOUR REFERENCE AUDIO ────────────────────────────────────────────────────
REF_TEXT = "We should only dream about becoming IAS officers, doctors and engineers. In this nation of Mahatma Gandhi and vocational education, why cannot I dream a nation?"

CLONE_REF_AUDIO = "/workspace/audio-em/emo-tts/ref-speech-clip.mp3"

# ─── YOUR CUSTOM TEXT TO SPEAK ──────────────────────────────────────────────
CUSTOM_TEXT = """It sounds like you're going through a tough moment—being kind to yourself is so important! It's completely normal to say or do things we later regret, especially when emotions run high."""

# ─── MODEL LOADING ────────────────────────────────────────────────────────────

def load_f5tts(device=DEVICE):
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT

    safetensors_path = os.path.join(F5TTS_LOCAL_DIR, "F5TTS_Base", "model_1200000.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Model not found: {safetensors_path}")

    print(f"  Loading F5-TTS from {F5TTS_LOCAL_DIR}...")
    model = load_model(
        model_cls=DiT,
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                       text_mask_padding=False, conv_layers=4, pe_attn_head=1),
        ckpt_path=safetensors_path,
        mel_spec_type="vocos",
        vocab_file=os.path.join(F5TTS_LOCAL_DIR, "F5TTS_Base", "vocab.txt"),
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    print("  ✅ F5-TTS loaded successfully")
    return model, vocoder

# ─── VECTORS ─────────────────────────────────────────────────────────────────

def load_vectors(device=DEVICE) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load emotion steering vectors for F5-TTS."""
    vectors = {}
    
    if not os.path.exists(VECTORS_DIR):
        print(f"  ⚠ Vectors directory not found: {VECTORS_DIR}")
        return vectors
    
    for emotion in EMOTIONS:
        # Look for vector files
        vector_paths = [
            os.path.join(VECTORS_DIR, f"f5tts_{emotion}_steering.pt"),
            os.path.join(VECTORS_DIR, f"f5tts_{emotion}_steervec.pt"),
        ]
        
        found = False
        for path in vector_paths:
            if os.path.exists(path):
                try:
                    data = torch.load(path, map_location=device)
                    
                    # Handle different save formats
                    if isinstance(data, dict):
                        if "vectors" in data:
                            raw = data["vectors"]
                        elif "steering_vectors" in data:
                            raw = data["steering_vectors"]
                        else:
                            raw = data
                    else:
                        raw = data
                    
                    # Convert to proper format
                    vectors[emotion] = {}
                    for k, v in raw.items():
                        layer_idx = int(k) if isinstance(k, str) else k
                        vectors[emotion][layer_idx] = v.to(device).float()
                    
                    print(f"  ✓ {emotion}: {len(vectors[emotion])} layers")
                    found = True
                    break
                    
                except Exception as e:
                    print(f"  ⚠ Failed to load {emotion}: {e}")
        
        if not found:
            print(f"  ⚠ No vectors found for {emotion}")
    
    return vectors

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _bcast(vec: torch.Tensor, batch: int, seq_len: int, dtype, device) -> torch.Tensor:
    """
    [H] → [batch, seq_len, H] — contiguous, matches actual tensor batch size.
    """
    v = (vec / (vec.norm() + 1e-8)).to(device=device, dtype=dtype)
    return v.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1).contiguous()

def _renorm(x_orig: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
    """
    Restore original per-item L2 norm (paper's fr function).
    Works for any batch size — computes one scale factor per batch item.
    """
    n_orig = x_orig.norm(p=2, dim=(1, 2), keepdim=True)
    n_new  = x_new.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return x_new * (n_orig / n_new)

# ─── LAYER ACCESS ─────────────────────────────────────────────────────────────

def _get_layers(model):
    model = model.module if hasattr(model, "module") else model
    for path in ["transformer.transformer_blocks", "transformer.layers",
                 "dit.transformer_blocks", "net.transformer_blocks"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)):
                return obj
        except AttributeError:
            continue
    raise AttributeError("Cannot find DiT transformer blocks")

# ─── HOOK FACTORIES ───────────────────────────────────────────────────────────

def _make_steer_hook(layer_idx: int, vecs: Dict[int, torch.Tensor], alpha: float):
    """Add α·s to the hidden state at this layer (Eq. 8), then renorm."""
    def hook(module, args, kwargs):
        if layer_idx not in vecs:
            return args, kwargs
        x   = args[0]                               # [B, T, H] B=2 during CFM
        act = _bcast(vecs[layer_idx], x.size(0), x.size(1), x.dtype, x.device)
        x_new = _renorm(x, x + alpha * act)
        return (x_new,) + args[1:], kwargs
    return hook

# ─── HOOK MANAGER ─────────────────────────────────────────────────────────────

class _HookManager:
    def __init__(self, model, hook_per_layer: Dict[int, callable]):
        layers = _get_layers(model)
        self._handles = []
        for idx, fn in hook_per_layer.items():
            if idx < len(layers):
                self._handles.append(
                    layers[idx].register_forward_pre_hook(fn, with_kwargs=True)
                )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

# ─── INFERENCE ────────────────────────────────────────────────────────────────

def _infer(model, vocoder, text: str, ref_audio: str, ref_text: str, device=DEVICE):
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    actual = model.module if hasattr(model, "module") else model
    
    # Use the provided reference text
    ref_audio_proc, _ = preprocess_ref_audio_text(ref_audio, ref_text=ref_text)
    
    with torch.no_grad():
        result = infer_process(
            ref_audio=ref_audio_proc, ref_text=ref_text, gen_text=text,
            model_obj=actual, vocoder=vocoder, device=device,
            speed=1.0, cross_fade_duration=0.15,
        )
    if not isinstance(result, tuple):
        audio, sr, _ = next(result)
    else:
        audio, sr, _ = result
    return audio, sr

# ─── PUBLIC API ───────────────────────────────────────────────────────────────

class F5TTSSteerer:
    """
    Emotion steerer for F5-TTS with multiple alpha intensities.
    """

    def __init__(self, model, vocoder,
                 vectors: Optional[Dict] = None,
                 device: str = DEVICE):
        self.model   = model
        self.vocoder = vocoder
        self.vecs    = vectors if vectors is not None else load_vectors(device)
        self.device  = device

    def baseline(self, text: str, ref_audio: str, ref_text: str) -> Tuple[np.ndarray, int]:
        """Generate with no steering (pure voice clone)."""
        return _infer(self.model, self.vocoder, text, ref_audio, ref_text, self.device)

    def steer(self, text: str, ref_audio: str, ref_text: str, emotion: str,
              alpha: float = ALPHA_DEFAULT) -> Tuple[np.ndarray, int]:
        """
        Generate with emotion steering.
          emotion : one of the loaded emotions
          alpha   : intensity (0.0 = no effect, 1.5 = good, 2.0 = max)
        """
        if emotion not in self.vecs:
            raise KeyError(f"Unknown emotion '{emotion}'. Available: {list(self.vecs.keys())}")
        
        alpha = float(np.clip(alpha, -ALPHA_MAX, ALPHA_MAX))
        hooks = {l: _make_steer_hook(l, self.vecs[emotion], alpha) for l in STEERED_LAYERS if l in self.vecs[emotion]}
        
        mgr = _HookManager(self.model, hooks)
        try:
            return _infer(self.model, self.vocoder, text, ref_audio, ref_text, self.device)
        finally:
            mgr.remove()

    def available_emotions(self) -> List[str]:
        return list(self.vecs.keys())

# ─── MAIN GENERATION WITH MULTIPLE ALPHA VALUES ─────────────────────────────

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("F5-TTS Emotion Steering - Your Custom Text with Multiple Intensities")
    print("="*70)
    
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"🎤 Reference audio: {CLONE_REF_AUDIO}")
    print(f"📝 Reference text: {REF_TEXT[:80]}...")
    print(f"🎭 Emotions: {EMOTIONS}")
    
    # Show the text to be spoken
    print(f"\n📄 Your text to speak:")
    print("-" * 70)
    print(CUSTOM_TEXT)
    print("-" * 70)
    
    # Load model
    print("\n📦 Loading F5-TTS...")
    try:
        model, vocoder = load_f5tts(DEVICE)
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return
    
    # Load emotion vectors
    print("\n📦 Loading emotion vectors...")
    steerer = F5TTSSteerer(model, vocoder)
    
    if not steerer.available_emotions():
        print("  ❌ No emotion vectors found! Run extraction first.")
        print(f"  Looking in: {VECTORS_DIR}")
        return
    
    print(f"\n✅ Ready! Available emotions: {steerer.available_emotions()}")
    
    # Generate baseline (no emotion)
    print("\n" + "="*70)
    print("🎵 GENERATING BASELINE (neutral/no emotion)")
    print("="*70)
    
    print(f"\n  Generating neutral version...")
    audio, sr = steerer.baseline(CUSTOM_TEXT, CLONE_REF_AUDIO, REF_TEXT)
    
    if audio is not None:
        filename = "baseline_neutral.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        sf.write(filepath, audio, sr)
        print(f"    ✅ Saved: {filename} ({len(audio)/sr:.2f}s)")
        print(f"    Amplitude: {np.abs(audio).max():.3f}")
    else:
        print(f"    ❌ Failed to generate baseline")
    
    # Generate for each emotion with different intensities
    for emotion in EMOTIONS:
        print("\n" + "="*70)
        print(f"🎵 GENERATING {emotion.upper()} VERSIONS")
        print("="*70)
        
        # Test different alpha intensities
        print(f"\n  Testing different intensities for {emotion}...")
        
        alphas = [4.0, 5.0, 6.0, 7.0]
        for alpha in alphas:
            print(f"\n    α={alpha}...")
            try:
                audio, sr = steerer.steer(
                    CUSTOM_TEXT, CLONE_REF_AUDIO, REF_TEXT, emotion, 
                    alpha=alpha
                )
                
                if audio is not None:
                    filename = f"{emotion}_alpha{alpha}.wav"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    sf.write(filepath, audio, sr)
                    print(f"      ✅ Saved: {filename} ({len(audio)/sr:.2f}s)")
                else:
                    print(f"      ❌ Failed")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Also generate default intensity version with clear name
        print(f"\n  Generating {emotion} at default intensity (α={ALPHA_DEFAULT})...")
        try:
            audio, sr = steerer.steer(
                CUSTOM_TEXT, CLONE_REF_AUDIO, REF_TEXT, emotion, 
                alpha=ALPHA_DEFAULT
            )
            
            if audio is not None:
                filename = f"{emotion}_default.wav"
                filepath = os.path.join(OUTPUT_DIR, filename)
                sf.write(filepath, audio, sr)
                print(f"    ✅ Saved: {filename} ({len(audio)/sr:.2f}s)")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\n📁 All files saved in: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - baseline_neutral.wav              (no emotion)")
    for emotion in EMOTIONS:
        print(f"  - {emotion}_default.wav             ({emotion} at α={ALPHA_DEFAULT})")
        print(f"  - {emotion}_alpha5.0.wav            ({emotion} low intensity)")
        print(f"  - {emotion}_alpha5.0.wav            ({emotion} medium-low intensity)")
        print(f"  - {emotion}_alpha6.0.wav            ({emotion} medium-high intensity)")
        print(f"  - {emotion}_alpha7.0.wav            ({emotion} high intensity)")
    
    print("\n🎧 Listen to compare:")
    print("   - baseline_neutral.wav (no emotion)")
    for emotion in EMOTIONS:
        print(f"   - {emotion}_default.wav ({emotion} version)")
        print(f"   - {emotion}_alpha5.0.wav (strong {emotion})")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()