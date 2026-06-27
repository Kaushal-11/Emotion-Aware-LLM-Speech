"""
inference_steering.py  —  Batch Emotion Steering for F5-TTS and CosyVoice2
==========================================================================
Generates emotion-steered speech from 300 neutral reference samples.
"""

import os
import json
import gc
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_TO_USE = "cosyvoice2"  # "f5tts" or "cosyvoice2"

BASE_DIR = "/workspace/audio-em"
DATASET_BASE_DIR = os.path.join(BASE_DIR, "dataset")
NEUTRAL_JSON = os.path.join(BASE_DIR, "emo-tts", "data", "used", "test", "neutral_test.json")

OUTPUT_BASE = os.path.join(BASE_DIR, "emo-tts", "results", "inference-test")
OUTPUT_DIR = {
    "f5tts": os.path.join(OUTPUT_BASE, "f5tts"),
    "cosyvoice2": os.path.join(OUTPUT_BASE, "cosyvoice2"),
}

F5TTS_MODEL_DIR = "/workspace/audio-em/emo-tts/models/f5tts"
COSYVOICE2_MODEL_DIR = "/workspace/audio-em/emo-tts/models/cosyvoice2"
VECTORS_DIR = {
    "f5tts": os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector", "f5tts", "final"),
    "cosyvoice2": os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector", "cosyvoice2", "final"),
}

EMOTIONS = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]
ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0]
MAX_SAMPLES = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FALLBACK_TEXTS = [
    "The weather today is pleasant and sunny.",
    "I would like to order a coffee with milk please.",
]

# ==============================================================================
# LOAD NEUTRAL SAMPLES
# ==============================================================================

def load_neutral_samples(json_path: str, max_samples: int = 300) -> List[Dict]:
    """Load neutral test samples from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        if "samples" in data:
            samples = data["samples"]
        elif "files" in data:
            samples = data["files"]
        else:
            # Get first key that contains a list
            for key, value in data.items():
                if isinstance(value, list):
                    samples = value
                    break
            else:
                raise ValueError(f"Unknown JSON structure. Keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unexpected JSON type: {type(data)}")
    
    valid_samples = []
    for idx, item in enumerate(samples[:max_samples]):
        audio_path = os.path.join(DATASET_BASE_DIR, item["audio_path"])
        if os.path.exists(audio_path):
            valid_samples.append({
                "audio_path": audio_path,
                "transcription": item.get("transcription", ""),
                "audio_name": item.get("audio_name", f"sample_{idx}"),
                "speaker_id": item.get("speaker_id", "unknown"),
                "dataset": item.get("dataset", "unknown"),
            })
    
    print(f"  Loaded {len(valid_samples)} valid neutral samples")
    return valid_samples

# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_f5tts(device: str = DEVICE):
    """Load F5-TTS model and vocoder."""
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    
    safetensors_path = os.path.join(F5TTS_MODEL_DIR, "F5TTS_Base", "model_1200000.safetensors")
    
    model = load_model(
        model_cls=DiT,
        model_cfg=dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            text_mask_padding=False, conv_layers=4, pe_attn_head=1
        ),
        ckpt_path=safetensors_path,
        mel_spec_type="vocos",
        vocab_file=os.path.join(F5TTS_MODEL_DIR, "F5TTS_Base", "vocab.txt"),
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    print("  ✅ F5-TTS loaded")
    return model, vocoder

def load_cosyvoice2(device: str = DEVICE):
    """Load CosyVoice2 model."""
    import sys
    sys.path.insert(0, "/workspace/audio-em/emo-tts/models/CosyVoice")
    from cosyvoice.cli.cosyvoice import CosyVoice2
    
    model = CosyVoice2(
        model_dir=COSYVOICE2_MODEL_DIR,
        load_jit=False,
        load_trt=False,
        fp16=False,
    )
    print("  ✅ CosyVoice2 loaded")
    return model, None

# ==============================================================================
# STEERING VECTOR LOADING
# ==============================================================================

def load_steering_vectors(model_name: str, device: str = DEVICE) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load steering vectors for all emotions."""
    vectors = {}
    vectors_dir = VECTORS_DIR[model_name]
    
    for emotion in EMOTIONS:
        vec_path = os.path.join(vectors_dir, f"{model_name}_{emotion}_steervec.pt")
        if os.path.exists(vec_path):
            data = torch.load(vec_path, map_location=device)
            
            if isinstance(data, dict):
                if "steering_vectors" in data:
                    raw = data["steering_vectors"]
                elif "vectors" in data:
                    raw = data["vectors"]
                else:
                    raw = data
            else:
                raw = data
            
            vectors[emotion] = {}
            for k, v in raw.items():
                layer_idx = int(k) if isinstance(k, str) else k
                vectors[emotion][layer_idx] = v.to(device).float()
            
            print(f"  ✓ {emotion}: {len(vectors[emotion])} layers")
        else:
            print(f"  ⚠ No vectors found for {emotion}")
    
    return vectors

# ==============================================================================
# F5-TTS STEERING IMPLEMENTATION (FIXED)
# ==============================================================================

class F5TTSSteerer:
    """F5-TTS with emotion steering hooks."""
    
    def __init__(self, model, vocoder, emotion_vectors: Dict, device: str = DEVICE):
        self.model = model
        self.vocoder = vocoder
        self.emotion_vectors = emotion_vectors
        self.device = device
        self.steered_layers = [1, 6, 11, 16, 21]
        self.hooks = []
    
    def _get_layers(self):
        """Get transformer blocks for hooking."""
        m = self.model.module if hasattr(self.model, "module") else self.model
        for path in ["transformer.transformer_blocks", "transformer.layers"]:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (list, torch.nn.ModuleList)):
                    return obj
            except AttributeError:
                continue
        raise AttributeError("Cannot find transformer layers")
    
    def _make_hook(self, layer_idx: int, vec: torch.Tensor, alpha: float):
        """Create forward pre-hook for steering."""
        def hook(module, args, kwargs):
            x = args[0]
            if x.dim() == 3:
                v = (vec / (vec.norm() + 1e-8)).to(x.device, x.dtype)
                v = v.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
                x = x + alpha * v
                return (x,) + args[1:], kwargs
            return args, kwargs
        return hook
    
    def _register_hooks(self, emotion: str, alpha: float):
        """Register hooks for a specific emotion and alpha."""
        self._remove_hooks()
        
        if emotion not in self.emotion_vectors:
            return False
        
        layers = self._get_layers()
        vecs = self.emotion_vectors[emotion]
        
        for layer_idx in self.steered_layers:
            if layer_idx in vecs and layer_idx < len(layers):
                hook_fn = self._make_hook(layer_idx, vecs[layer_idx], alpha)
                handle = layers[layer_idx].register_forward_pre_hook(hook_fn, with_kwargs=True)
                self.hooks.append(handle)
        
        return len(self.hooks) > 0
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def generate(self, text: str, ref_audio: str, ref_text: str, 
                 emotion: Optional[str] = None, alpha: float = 0.0) -> Tuple[np.ndarray, int]:
        """Generate speech with optional emotion steering."""
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
        
        actual = self.model.module if hasattr(self.model, "module") else self.model
        
        # Register hooks if emotion specified
        if emotion and emotion in self.emotion_vectors and alpha != 0:
            self._register_hooks(emotion, alpha)
        
        try:
            # Preprocess reference audio with its transcription
            ref_audio_proc, _ = preprocess_ref_audio_text(ref_audio, ref_text=ref_text)
            
            with torch.no_grad():
                result = infer_process(
                    ref_audio=ref_audio_proc, ref_text=ref_text, gen_text=text,
                    model_obj=actual, vocoder=self.vocoder, device=self.device,
                    speed=1.0, cross_fade_duration=0.15,
                )
            
            # Handle different return types
            if isinstance(result, tuple):
                audio, sr, _ = result
            else:
                # result is a generator
                audio, sr, _ = next(result)
            
            # Convert to numpy array if it's a torch tensor
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            # If it's already numpy, just use it as is
            elif isinstance(audio, np.ndarray):
                pass
            else:
                # Try to convert to numpy
                audio = np.array(audio)
            
            return audio, sr
            
        finally:
            self._remove_hooks()

# ==============================================================================
# COSYVOICE2 STEERING IMPLEMENTATION
# ==============================================================================

class CosyVoice2Steerer:
    """CosyVoice2 with emotion steering hooks."""
    
    def __init__(self, model, emotion_vectors: Dict, device: str = DEVICE):
        self.model = model
        self.emotion_vectors = emotion_vectors
        self.device = device
        self.steered_layers = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51]
        self.hooks = []
        self.blocks = None
    
    def _collect_blocks(self):
        """Collect transformer blocks for hooking."""
        if self.blocks is not None:
            return self.blocks
        
        est = self.model.model.flow.decoder.estimator
        blocks = []
        
        for b in est.down_blocks[0][1]:
            blocks.append(b)
        for i in range(len(est.mid_blocks)):
            for b in est.mid_blocks[i][1]:
                blocks.append(b)
        for b in est.up_blocks[0][1]:
            blocks.append(b)
        
        self.blocks = blocks
        return blocks
    
    def _make_hook(self, layer_idx: int, vec: torch.Tensor, alpha: float):
        """Create forward pre-hook for steering."""
        def hook(module, args, kwargs):
            x = args[0]
            if x.dim() == 3:
                v = (vec / (vec.norm() + 1e-8)).to(x.device, x.dtype)
                v = v.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
                x = x + alpha * v
                return (x,) + args[1:], kwargs
            return args, kwargs
        return hook
    
    def _register_hooks(self, emotion: str, alpha: float):
        """Register hooks for a specific emotion and alpha."""
        self._remove_hooks()
        
        if emotion not in self.emotion_vectors:
            return False
        
        blocks = self._collect_blocks()
        vecs = self.emotion_vectors[emotion]
        
        for layer_idx in self.steered_layers:
            if layer_idx in vecs and layer_idx < len(blocks):
                if hasattr(blocks[layer_idx], 'attn1'):
                    hook_fn = self._make_hook(layer_idx, vecs[layer_idx], alpha)
                    handle = blocks[layer_idx].attn1.register_forward_pre_hook(hook_fn, with_kwargs=True)
                    self.hooks.append(handle)
        
        return len(self.hooks) > 0
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def generate(self, text: str, ref_audio: str, ref_text: str,
                 emotion: Optional[str] = None, alpha: float = 0.0) -> Tuple[np.ndarray, int]:
        """Generate speech with optional emotion steering."""
        import inspect
        
        if emotion and emotion in self.emotion_vectors and alpha != 0:
            self._register_hooks(emotion, alpha)
        
        try:
            sig = inspect.signature(self.model.inference_zero_shot)
            audio_param = "prompt_wav" if "prompt_wav" in sig.parameters else "prompt_speech_16k"
            
            audio_chunks = []
            with torch.no_grad():
                for result in self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=ref_text,
                    **{audio_param: ref_audio},
                    stream=False
                ):
                    if "tts_speech" in result:
                        audio_chunks.append(result["tts_speech"])
            
            if not audio_chunks:
                return None, None
            
            audio = torch.cat(audio_chunks, dim=-1).squeeze().cpu().numpy()
            return audio, 22050
            
        finally:
            self._remove_hooks()

# ==============================================================================
# MAIN GENERATION LOOP
# ==============================================================================

def generate_for_model(model_name: str):
    """Run generation for a specific model."""
    print(f"\n{'='*70}")
    print(f"🎭 Generating with {model_name.upper()}")
    print(f"{'='*70}")
    
    # Create output directories
    output_root = OUTPUT_DIR[model_name]
    for alpha in ALPHA_VALUES:
        for emotion in EMOTIONS:
            os.makedirs(os.path.join(output_root, f"alpha-{alpha}", emotion), exist_ok=True)
    
    # Load samples
    print("\n📁 Loading neutral test samples...")
    samples = load_neutral_samples(NEUTRAL_JSON, MAX_SAMPLES)
    
    if not samples:
        print("  ❌ No samples loaded!")
        return
    
    # Load model
    print(f"\n📦 Loading {model_name} model...")
    if model_name == "f5tts":
        model, vocoder = load_f5tts(DEVICE)
        steerer = F5TTSSteerer(model, vocoder, emotion_vectors, DEVICE)
    else:
        model, vocoder = load_cosyvoice2(DEVICE)
        steerer = CosyVoice2Steerer(model, emotion_vectors, DEVICE)
    
    total_generations = len(samples) * len(EMOTIONS) * len(ALPHA_VALUES)
    print(f"\n🎬 Starting generation: {len(samples)} samples × {len(EMOTIONS)} emotions × {len(ALPHA_VALUES)} alphas = {total_generations} files")
    
    success_count = 0
    fail_count = 0
    
    for sample_idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        ref_audio = sample["audio_path"]
        ref_text = sample["transcription"]
        sample_name = sample["audio_name"].replace(".wav", "")
        
        # Use transcription as text to generate
        gen_text = ref_text if ref_text and len(ref_text) > 5 else FALLBACK_TEXTS[0]
        
        # Generate baseline (neutral, no steering)
        try:
            audio_baseline, sr = steerer.generate(gen_text, ref_audio, ref_text, emotion=None, alpha=0)
            if audio_baseline is not None:
                baseline_dir = os.path.join(output_root, "baseline")
                os.makedirs(baseline_dir, exist_ok=True)
                sf.write(os.path.join(baseline_dir, f"{sample_name}.wav"), audio_baseline, sr)
        except Exception as e:
            if sample_idx < 3:
                print(f"\n  ⚠ Baseline failed: {e}")
        
        # Generate for each emotion and alpha
        for emotion in EMOTIONS:
            for alpha in ALPHA_VALUES:
                try:
                    audio, sr = steerer.generate(
                        gen_text, ref_audio, ref_text, 
                        emotion=emotion, alpha=alpha
                    )
                    
                    if audio is not None:
                        out_path = os.path.join(output_root, f"alpha-{alpha}", emotion, f"{sample_name}.wav")
                        sf.write(out_path, audio, sr)
                        success_count += 1
                    else:
                        fail_count += 1
                        
                except Exception as e:
                    fail_count += 1
                    if fail_count <= 5:
                        print(f"\n  ⚠ Failed {emotion} α={alpha}: {e}")
        
        # Periodic cleanup
        if (sample_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n  📊 Progress: {sample_idx+1}/{len(samples)} samples, {success_count} successes, {fail_count} failures")
    
    print("\n" + "="*70)
    print(f"✅ GENERATION COMPLETE for {model_name.upper()}")
    print(f"   Success: {success_count} files")
    print(f"   Failed: {fail_count} files")
    print(f"   Output: {output_root}")
    print("="*70)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EMOTION STEERING INFERENCE - BATCH GENERATION")
    print("="*70)
    print(f"Model: {MODEL_TO_USE}")
    print(f"Device: {DEVICE}")
    print(f"Emotions: {EMOTIONS}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print(f"Max samples: {MAX_SAMPLES}")
    print("="*70)
    
    # Load steering vectors
    print("\n📦 Loading steering vectors...")
    emotion_vectors = load_steering_vectors(MODEL_TO_USE, DEVICE)
    
    if not emotion_vectors:
        print("  ❌ No steering vectors found!")
        exit(1)
    
    # Run generation
    generate_for_model(MODEL_TO_USE)
    
    print("\n🎉 All done!")