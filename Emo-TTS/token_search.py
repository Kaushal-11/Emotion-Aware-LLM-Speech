#!/usr/bin/env python3
"""
Script 03: Top-k Emotional Token Search & Steering Vector Construction
=======================================================================
- Scorer : emotion2vec_plus_large (soft probability scores, suitable for ranking)
- Models : F5-TTS  or  CosyVoice2  (switch via MODEL_TO_USE)
- Output : one .pt + .json per emotion, saved to the model's own final/ directory

Output paths
────────────
  F5-TTS   : results/activation_vector/f5tts/final/f5tts_{emotion}_steervec.pt
  CosyVoice: results/activation_vector/cosyvoice2/final/cosyvoice2_{emotion}_steervec.pt
"""

import os, sys, json, random, gc, uuid
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION  — only change these two lines to switch models
# ==============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TO_USE = "cosyvoice2"      # "f5tts"  |  "cosyvoice2"

BASE_DIR = "/workspace/audio-em"

# Input dirs (raw activation-difference vectors from script 02)
_RAW = {
    "f5tts":     os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector", "f5tts"),
    "cosyvoice2":os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector", "cosyvoice2"),
}

# Output dirs (final steering vectors) — each model gets its own final/ subdir
_FINAL = {
    "f5tts":     os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector", "f5tts",      "final"),
    "cosyvoice2":os.path.join(BASE_DIR, "emo-tts", "results", "activation_vector", "cosyvoice2", "final"),
}

EMOTIONS      = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]
TOP_K          = 200
N_NEUTRAL_REFS = 10

RANDOM_TEXTS = [
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

SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Create output dir for the selected model
os.makedirs(_FINAL[MODEL_TO_USE], exist_ok=True)

print(f"{'='*70}")
print(f"  Top-k Emotional Token Search")
print(f"{'='*70}")
print(f"  Model  : {MODEL_TO_USE}")
print(f"  Device : {DEVICE}")
print(f"  Top-k  : {TOP_K}")
print(f"  Output : {_FINAL[MODEL_TO_USE]}")
print(f"{'='*70}\n")


# ==============================================================================
# EMOTION2VEC SCORER
# ==============================================================================

# emotion2vec_plus_large label vocabulary → Ekman names
_LABEL_MAP: Dict[str, str] = {
    "ang": "anger",   "angry": "anger",      "anger": "anger",
    "angry/disgusted": "anger",
    "hap": "happiness", "happy": "happiness", "happiness": "happiness",
    "excited": "happiness", "joy": "happiness",
    "sad": "sadness",  "sadness": "sadness",
    "dis": "disgust",  "disgust": "disgust",  "disgusted": "disgust",
    "fea": "fear",     "fear": "fear",        "fearful": "fear",   "scared": "fear",
    "sur": "surprise", "surprise": "surprise","surprised": "surprise",
    "neu": "neutral",  "neutral": "neutral",  "calm": "neutral",   "other": "neutral",
}

def _to_ekman(raw: str) -> str:
    s = raw.strip().lower()
    if s in _LABEL_MAP:
        return _LABEL_MAP[s]
    for k, v in _LABEL_MAP.items():
        if k in s:
            return v
    return "unknown"


class Emotion2VecScorer:
    """
    Wraps emotion2vec_plus_large.
    generate() returns:
      [{'key': '...', 'scores': [0.82, 0.03, ...], 'labels': ['angry/disgusted', 'happy', ...]}]
    score() returns the softmax probability for target_emotion (0.0–1.0).
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model  = None
        self._debug_done = False
        self._load()

    def _load(self):
        try:
            from funasr import AutoModel
            self.model = AutoModel(model="iic/emotion2vec_plus_large", device=self.device)
            print("[SER] ✅ emotion2vec_plus_large loaded")
        except Exception as e:
            print(f"[SER] ❌ Failed to load emotion2vec: {e}")
            self.model = None

    def score_file(self, audio_path: str, target_emotion: str) -> float:
        """Score a wav file. Returns probability for target_emotion."""
        if self.model is None:
            return 0.0
        try:
            result = self.model.generate(input=audio_path, granularity="utterance")
            if not self._debug_done:
                print(f"   [DBG emotion2vec] {result}")
                self._debug_done = True
            if not result:
                return 0.0
            item = result[0]
            # Primary format: scores + labels
            if "scores" in item and "labels" in item:
                for raw_label, prob in zip(item["labels"], item["scores"]):
                    if _to_ekman(raw_label) == target_emotion:
                        return float(prob)
            # Fallback: pred + conf
            if "pred" in item and _to_ekman(item["pred"]) == target_emotion:
                return float(item.get("conf", 0.5))
            return 0.0
        except Exception as e:
            print(f"   [SER error] {e}")
            return 0.0

    def score(self, audio, sr: int, target_emotion: str,
              tmp_dir: str = "/tmp") -> float:
        """Score from audio tensor or numpy array."""
        try:
            import soundfile as sf
            if isinstance(audio, torch.Tensor):
                audio_np = audio.squeeze().cpu().float().numpy()
            elif isinstance(audio, np.ndarray):
                audio_np = audio.squeeze().astype(np.float32)
            else:
                audio_np = np.array(audio, dtype=np.float32).squeeze()
            p = os.path.join(tmp_dir, f"e2v_{uuid.uuid4().hex}.wav")
            sf.write(p, audio_np, sr)
            s = self.score_file(p, target_emotion)
            os.remove(p)
            return s
        except Exception as e:
            print(f"   [SER score error] {e}")
            return 0.0


# ==============================================================================
# SINGLE-TOKEN STEERING HOOK
# ==============================================================================

class SingleTokenSteeringHook:
    """
    Applies the steering vector ONLY at token position `active_idx`.
    All other positions get zero perturbation, so each iteration is unique.
    """

    def __init__(self, model, target_layers: List[int], model_type: str):
        self.model         = model
        self.target_layers = target_layers
        self.model_type    = model_type
        self._hooks: list  = []
        self._vec:   Optional[torch.Tensor] = None   # [hidden_dim]
        self._idx:   int   = 0
        self._ref_len: int = 0

    def set_state(self, vec: torch.Tensor, active_idx: int, ref_len: int):
        self._vec     = vec
        self._idx     = active_idx
        self._ref_len = ref_len

    # ── F5-TTS hook ───────────────────────────────────────────────────────
    def _hook_f5tts(self, _layer_idx: int):
        def hook(module, input_args):
            if self._vec is None:
                return input_args
            x = input_args[0] if isinstance(input_args, tuple) else input_args
            if x.dim() == 3:
                v   = self._vec.to(x.device, x.dtype)
                v   = v / (v.norm() + 1e-8)
                idx = min(self._idx, x.size(1) - 1)
                delta = torch.zeros_like(x)
                delta[:, idx, :] = 5.0 * v.unsqueeze(0)
                
                x = x + delta
                
                if isinstance(input_args, tuple):
                    return (x,) + input_args[1:]
            return input_args
        return hook

    # ── CosyVoice2 hook - FIXED: using norm1 instead of attn1 ─────────────────
    def _hook_cosyvoice2(self, _layer_idx: int):
        def hook(module, input_args):
            if self._vec is None:
                return input_args
            x = input_args[0] if isinstance(input_args, tuple) else input_args
            if x.dim() == 3:
                v   = self._vec.to(x.device, x.dtype)
                v   = v / (v.norm() + 1e-8)
                idx = min(self._idx, x.size(1) - 1)
                delta = torch.zeros_like(x)
                delta[:, idx, :] = 5.0 * v.unsqueeze(0)

                orig_norm = x.norm(p=2, dim=(1, 2), keepdim=True)
                x = x + delta
                x = x * (orig_norm / (x.norm(p=2, dim=(1, 2), keepdim=True) + 1e-8))

                if isinstance(input_args, tuple):
                    return (x,) + input_args[1:]
            return input_args
        return hook

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def register(self):
        if self.model_type == "cosyvoice2":
            blocks = self._collect_cosyvoice2_blocks()
            fn = self._hook_cosyvoice2
            for l in self.target_layers:
                if l < len(blocks):
                    block = blocks[l]
                    # FIX: Use norm1 instead of attn1 to match extraction script
                    if hasattr(block, "norm1"):
                        self._hooks.append(block.norm1.register_forward_pre_hook(fn(l)))
                    elif hasattr(block, "attn1"):
                        self._hooks.append(block.attn1.register_forward_pre_hook(fn(l)))
                    else:
                        self._hooks.append(block.register_forward_pre_hook(fn(l)))
                else:
                    print(f"  [WARN] target layer {l} out of range ({len(blocks)} blocks)")
        else:
            layers = self._get_layers_f5tts()
            fn = self._hook_f5tts
            for l in self.target_layers:
                if l < len(layers):
                    self._hooks.append(layers[l].register_forward_pre_hook(fn(l)))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _collect_cosyvoice2_blocks(self) -> list:
        """
        Collect all BasicTransformerBlocks in the same traversal order as
        the extraction script: down_blocks[0][1] → mid_blocks[i][1] → up_blocks[0][1].
        This makes layer index N in target_layers refer to the exact same block
        that was hooked during extraction.
        """
        m = self.model.module if hasattr(self.model, "module") else self.model
        # CosyVoice2 CLI wraps the real model under .model
        inner = m.model if hasattr(m, "model") else m
        est = inner.flow.decoder.estimator

        blocks = []
        for b in est.down_blocks[0][1]:
            blocks.append(b)
        for i in range(len(est.mid_blocks)):
            for b in est.mid_blocks[i][1]:
                blocks.append(b)
        for b in est.up_blocks[0][1]:
            blocks.append(b)

        print(f"  [CosyVoice2] {len(blocks)} BasicTransformerBlocks collected for hooking")
        return blocks

    def _get_layers_f5tts(self):
        """Layer-finding logic for F5-TTS only."""
        m = self.model.module if hasattr(self.model, "module") else self.model
        for path in [
            "transformer.transformer_blocks",
            "transformer.layers",
            "dit.layers",
        ]:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (list, torch.nn.ModuleList)):
                    return obj
            except AttributeError:
                continue
        raise AttributeError("Cannot find transformer layers for f5tts")


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_f5tts(device: str):
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    mp = "/workspace/audio-em/emo-tts/models/f5tts"
    model = load_model(
        model_cls=DiT,
        model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                       text_mask_padding=False, conv_layers=4, pe_attn_head=1),
        ckpt_path=os.path.join(mp, "F5TTS_Base", "model_1200000.safetensors"),
        mel_spec_type="vocos",
        vocab_file=os.path.join(mp, "F5TTS_Base", "vocab.txt"),
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
    return model, vocoder


def load_cosyvoice2(device: str):
    from contextlib import nullcontext
    sys.path.insert(0, "/workspace/audio-em/emo-tts/models/CosyVoice")
    from cosyvoice.cli.cosyvoice import CosyVoice2

    model = CosyVoice2(
        model_dir="/workspace/audio-em/emo-tts/models/cosyvoice2",
        load_jit=False, load_trt=False, fp16=False,
    )

    # Disable autocast (LLM runs in thread, must patch llm_context)
    if hasattr(model, 'llm') and hasattr(model.llm, 'llm_context'):
        model.llm.llm_context = nullcontext()
    if hasattr(model, 'llm') and hasattr(model.llm, 'fp16'):
        model.llm.fp16 = False
    if hasattr(model, 'model') and hasattr(model.model, 'flow'):
        model.model.flow.fp16 = False

    # Convert all weights to float32
    _FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float64}
    for attr_name in vars(model.model):
        try:
            attr = getattr(model.model, attr_name)
            if isinstance(attr, torch.nn.Module):
                for param in attr.parameters():
                    if param.dtype in _FLOAT_DTYPES:
                        param.data = param.data.float()
                for buf in attr.buffers():
                    if buf.dtype in _FLOAT_DTYPES:
                        buf.data = buf.data.float()
                attr.eval()
                for p in attr.parameters():
                    p.requires_grad_(False)
        except Exception:
            pass

    return model


# ==============================================================================
# DATA LOADING - FIXED: returns (path, text) tuples
# ==============================================================================

def load_raw_vectors(model_name: str, emotion: str) -> Dict:
    path = os.path.join(_RAW[model_name], f"{model_name}_{emotion}_steering.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw vectors not found: {path}")
    print(f"   Raw vectors: {path}")
    return torch.load(path, map_location="cpu")


def load_neutral_references(n: int = 10) -> List[Tuple[str, str]]:
    """
    Load neutral reference audio files with their transcriptions.
    Returns list of (audio_path, transcription) tuples.
    """
    j = "/workspace/audio-em/emo-tts/data/used/steering/neutral_steering.json"
    refs = []
    if os.path.exists(j):
        with open(j, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("files", data.get("samples", [])):
            p = os.path.join("/workspace/audio-em/dataset", item.get("audio_path", ""))
            text = item.get("transcription", "").strip()
            if os.path.exists(p) and text and len(refs) < n:
                refs.append((p, text))
    print(f"   Neutral refs: {len(refs)} loaded (with transcriptions)")
    return refs


# ==============================================================================
# FORWARD PASS - FIXED: uses ref_text for CosyVoice2
# ==============================================================================

def synthesize_f5tts(model, vocoder, text: str, ref_path: str, ref_text: str, device: str):
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    actual = model.module if hasattr(model, "module") else model
    ref_audio, ref_text = preprocess_ref_audio_text(ref_path, ref_text=ref_text)
    with torch.no_grad():
        result = infer_process(
            ref_audio=ref_audio, ref_text=ref_text, gen_text=text,
            model_obj=actual, vocoder=vocoder, device=device,
            speed=1.0, cross_fade_duration=0.15,
        )
    if isinstance(result, tuple):
        return result[0], result[1]
    audio, sr, _ = next(result)
    return audio, sr


def synthesize_cosyvoice2(model, text: str, ref_path: str, ref_text: str):
    """
    FIXED: Now uses the actual transcription from the reference audio
    instead of a hardcoded placeholder text.
    """
    import inspect
    sig = inspect.signature(model.inference_zero_shot)
    kwarg = "prompt_wav" if "prompt_wav" in sig.parameters else "prompt_speech_16k"
    with torch.no_grad():
        chunks = []
        for chunk in model.inference_zero_shot(
            tts_text=text,
            prompt_text=ref_text,  # FIXED: use actual transcription
            **{kwarg: ref_path},
            stream=False,
        ):
            if isinstance(chunk, dict):
                chunks.append(chunk.get("tts_speech", chunk.get("speech", torch.zeros(1,1))))
            elif isinstance(chunk, torch.Tensor):
                chunks.append(chunk)
    if chunks:
        return torch.cat(chunks, dim=-1), 22050
    return None, None


def estimate_ref_len(model_type: str, ref_path: str) -> int:
    import librosa
    if model_type == "f5tts":
        a, _ = librosa.load(ref_path, sr=24000)
        return max(int(len(a) / 256), 32)
    else:
        a, _ = librosa.load(ref_path, sr=16000)
        return max(int(len(a) / 320), 50)


# ==============================================================================
# WEIGHTED VECTOR CONSTRUCTION
# ==============================================================================

def build_weighted_vectors(
    raw_vectors:   Dict[int, torch.Tensor],
    token_scores:  np.ndarray,
    top_k_indices: np.ndarray,
) -> Dict[int, torch.Tensor]:
    """
    s_hat^l = normalize( Σ_i  softmax(p_i) * u^l[i] )

    If raw_vectors[l] is shape [T, D]: each token position has its own vector
    → proper weighted sum across positions.
    If raw_vectors[l] is shape [D]: one global direction shared by all positions
    → weighted sum collapses to normalize(u^l), but structure stays consistent.
    """
    top_scores = token_scores[top_k_indices]
    weights    = F.softmax(torch.tensor(top_scores, dtype=torch.float32), dim=0)

    result = {}
    for l, u in raw_vectors.items():
        if u.dim() == 2:                        # [T, D] — per-position vectors
            T     = u.size(0)
            s_hat = torch.zeros(u.size(1), dtype=torch.float32)
            for rank, (tok_idx, w) in enumerate(zip(top_k_indices, weights)):
                if int(tok_idx) < T:
                    s_hat = s_hat + w.item() * u[int(tok_idx)].float()
        else:                                   # [D] — single global vector
            s_hat = u.float().clone()

        result[l] = F.normalize(s_hat.unsqueeze(0), dim=1).squeeze(0)
    return result


# ==============================================================================
# TOP-K TOKEN SEARCH - UPDATED to use (path, text) tuples
# ==============================================================================

def search_top_k_tokens(
    model_name:        str,
    model,
    vocoder,
    emotion:           str,
    raw_vectors:       Dict[int, torch.Tensor],
    neutral_refs:      List[Tuple[str, str]],  # FIXED: list of (path, text)
    scorer:            Emotion2VecScorer,
    avg_seq_len:       int,
    top_k:             int = TOP_K,
    device:            str = DEVICE,
) -> Tuple[Dict[int, torch.Tensor], np.ndarray, np.ndarray]:
    """
    For each token position i (0 … avg_seq_len-1):
      1. Steer ONLY position i using the mid-depth layer's vector
      2. Synthesize audio
      3. Score with emotion2vec → token_scores[i]

    Returns:
      weighted_vectors  — final steering vectors per layer
      top_k_indices     — top-k positions ranked by score
      token_scores      — full score array (all positions)
    """
    target_layers  = list(raw_vectors.keys())
    primary_layer  = target_layers[len(target_layers) // 2]   # mid-depth layer
    hook           = SingleTokenSteeringHook(model, target_layers, model_type=model_name)

    # Get first reference for length estimation
    ref_len = estimate_ref_len(model_name, neutral_refs[0][0])
    token_scores   = np.zeros(avg_seq_len)

    print(f"   Scanning {avg_seq_len} positions  |  ref_len={ref_len}  |  layers={target_layers}")

    for i in tqdm(range(avg_seq_len), desc=f"   {emotion}"):
        u = raw_vectors[primary_layer]
        # Per-position vector if available, else global vector
        vec = u[i].float() if (u.dim() == 2 and i < u.size(0)) else u.float().clone()

        hook.set_state(vec, active_idx=i, ref_len=ref_len)
        hook.register()

        ref_path, ref_text = random.choice(neutral_refs)  # FIXED: get both path and text
        text = random.choice(RANDOM_TEXTS)
        try:
            if model_name == "f5tts":
                audio, sr = synthesize_f5tts(model, vocoder, text, ref_path, ref_text, device)
            else:
                # FIXED: pass ref_text to synthesis function
                audio, sr = synthesize_cosyvoice2(model, text, ref_path, ref_text)

            if audio is not None:
                token_scores[i] = scorer.score(audio, sr, emotion)
        except Exception as ex:
            print(f"   [warn pos {i}] {ex}")
        finally:
            hook.remove()

        if (i + 1) % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"   [{i+1}/{avg_seq_len}] mean={token_scores[:i+1].mean():.4f}  "
                  f"max={token_scores[:i+1].max():.4f}")

    top_k_indices  = np.argsort(token_scores)[::-1][:top_k]
    top5_scores    = token_scores[top_k_indices[:5]]
    print(f"\n   Top-5 positions : {top_k_indices[:5].tolist()}")
    print(f"   Top-5 scores    : {top5_scores.tolist()}")
    print(f"   Mean top-{top_k} score : {token_scores[top_k_indices].mean():.4f}")

    weighted_vectors = build_weighted_vectors(raw_vectors, token_scores, top_k_indices)
    return weighted_vectors, top_k_indices, token_scores


# ==============================================================================
# SAVE
# ==============================================================================

def save_results(
    model_name:       str,
    emotion:          str,
    weighted_vectors: Dict[int, torch.Tensor],
    top_k_indices:    np.ndarray,
    token_scores:     np.ndarray,
    avg_seq_len:      int,
):
    out_dir = _FINAL[model_name]
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{model_name}_{emotion}_steervec"

    save_data = {
        "emotion":          emotion,
        "model":            model_name,
        "ser_model":        "emotion2vec_plus_large",
        "avg_seq_len":      avg_seq_len,
        "top_k":            TOP_K,
        "steering_vectors": {str(l): v.cpu() for l, v in weighted_vectors.items()},
        "top_k_indices":    top_k_indices.tolist(),
        "token_scores":     token_scores.tolist(),
        "stats": {
            "mean":  float(token_scores.mean()),
            "max":   float(token_scores.max()),
            "top5_positions": top_k_indices[:5].tolist(),
            "top5_scores":    token_scores[top_k_indices[:5]].tolist(),
        },
    }

    pt_path   = os.path.join(out_dir, f"{stem}.pt")
    json_path = os.path.join(out_dir, f"{stem}.json")

    torch.save(save_data, pt_path)

    # JSON: convert tensors → lists
    json_data = {}
    for k, v in save_data.items():
        if isinstance(v, dict) and k == "steering_vectors":
            json_data[k] = {kk: vv.tolist() for kk, vv in v.items()}
        else:
            json_data[k] = v
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"   ✅ {pt_path}")
    print(f"   ✅ {json_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # ── Scorer ────────────────────────────────────────────────────────────────
    scorer = Emotion2VecScorer(DEVICE)
    if scorer.model is None:
        print("❌ emotion2vec failed to load — cannot proceed.")
        return

    # ── TTS model ─────────────────────────────────────────────────────────────
    print(f"\n📁 Loading {MODEL_TO_USE}...")
    if MODEL_TO_USE == "f5tts":
        model, vocoder = load_f5tts(DEVICE)
    else:
        model, vocoder = load_cosyvoice2(DEVICE), None

    # ── Neutral references (now returns path + text) ──────────────────────────
    print("\n📁 Loading neutral references...")
    neutral_refs = load_neutral_references(N_NEUTRAL_REFS)
    if not neutral_refs:
        print("❌ No neutral references found — cannot proceed.")
        return

    # ── Process each emotion ──────────────────────────────────────────────────
    for emotion in EMOTIONS:
        out_pt = os.path.join(_FINAL[MODEL_TO_USE], f"{MODEL_TO_USE}_{emotion}_steervec.pt")
        if os.path.exists(out_pt):
            print(f"\n⏭️  Skipping {emotion} (already exists: {out_pt})")
            continue

        print(f"\n{'='*60}")
        print(f"  🎭  {emotion.upper()}  ({MODEL_TO_USE})")
        print(f"{'='*60}")

        try:
            raw_data    = load_raw_vectors(MODEL_TO_USE, emotion)
        except FileNotFoundError as e:
            print(f"   ❌ {e}")
            continue

        raw_vectors = raw_data.get("vectors", raw_data.get("steering_vectors", {}))
        raw_vectors = {int(k): v.float() for k, v in raw_vectors.items()}
        avg_seq_len = raw_data.get("avg_seq_len", 300)

        try:
            weighted_vecs, top_k_idx, scores = search_top_k_tokens(
                model_name=MODEL_TO_USE, model=model, vocoder=vocoder,
                emotion=emotion, raw_vectors=raw_vectors,
                neutral_refs=neutral_refs, scorer=scorer,
                avg_seq_len=avg_seq_len, top_k=TOP_K, device=DEVICE,
            )
            save_results(MODEL_TO_USE, emotion, weighted_vecs,
                         top_k_idx, scores, avg_seq_len)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback; traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*70}")
    print(f"  ✅ Done!  Results in: {_FINAL[MODEL_TO_USE]}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()