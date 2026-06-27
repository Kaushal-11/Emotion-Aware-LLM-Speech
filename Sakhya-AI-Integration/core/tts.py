"""
core/tts.py
-----------
Layer 5 — Emotional Text-to-Speech

Converts the LLM's steered text response into emotional speech using:
    - a PRESET speaker reference (NOT cloned from the user's voice —
      user picks one of config.PRESET_SPEAKERS in the UI)
    - emotion direction-vector injection into the TTS model's transformer
      blocks, scaled by `alpha = <BACKEND>_DEFAULT_ALPHA * vector_intensity`
      (vector_intensity comes from the same DecisionOutput used for LLM steering)

Two backends are supported, selected via config.TTS_BACKEND:
    - "f5tts"      (DEFAULT) — F5-TTS, DiT-based flow-matching TTS
    - "cosyvoice2" (OPTION)  — CosyVoice2

Only the selected backend is loaded at startup.
`TTS.synthesize()` has an identical signature regardless of backend.
`pipeline.switch_tts_backend()` can hot-swap between them (heavy reload).
"""

import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from config import (
    TTS_BACKEND,
    TTS_EMOTIONS,
    PRESET_SPEAKERS,
    DEFAULT_SPEAKER,
    DEVICE,

    # F5-TTS
    F5TTS_MODEL_NAME,
    F5TTS_CKPT_PATH,
    F5TTS_VOCAB_PATH,
    F5TTS_VECTORS_DIR,
    F5TTS_STEERED_LAYERS,
    F5TTS_DEFAULT_ALPHA,
    F5TTS_SAMPLE_RATE,
    F5TTS_NFE_STEP,
    F5TTS_CFG_STRENGTH,
    F5TTS_SPEED,

    # CosyVoice2
    COSYVOICE_MODEL_DIR,
    COSYVOICE_REPO_PATH,
    COSYVOICE_VECTORS_DIR,
    COSYVOICE_STEERED_LAYERS,
    COSYVOICE_DEFAULT_ALPHA,
    COSYVOICE_SAMPLE_RATE,
)
from core.llm import extract_emotion_from_vector
from core.decision_engine import DecisionOutput


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _load_steering_vectors(vectors_dir, filename_prefix: str, device: str = DEVICE) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Load emotion steering vectors for all TTS_EMOTIONS.

    Expects one file per emotion: {vectors_dir}/{filename_prefix}_{emotion}_steervec.pt
    containing either a raw {layer_idx: tensor} dict, or a dict with key
    "steering_vectors" / "vectors" holding that mapping.
    """
    vectors: Dict[str, Dict[int, torch.Tensor]] = {}

    for emotion in TTS_EMOTIONS:
        vec_path = vectors_dir / f"{filename_prefix}_{emotion}_steervec.pt"
        if vec_path.exists():
            data = torch.load(vec_path, map_location=device)

            if isinstance(data, dict):
                raw = data.get("steering_vectors", data.get("vectors", data))
            else:
                raw = data

            vectors[emotion] = {}
            for k, v in raw.items():
                layer_idx = int(k) if isinstance(k, str) else k
                vectors[emotion][layer_idx] = v.to(device).float()

            print(f"   [TTS] loaded steering vectors for '{emotion}': "
                  f"{len(vectors[emotion])} layers")
        else:
            print(f"   [TTS] WARNING: no steering vectors found for '{emotion}' ({vec_path})")

    return vectors


# ============================================================================
# F5-TTS STEERER  (default backend)
# ============================================================================

class F5TTSSteerer:
    """
    F5-TTS (DiT-based flow-matching TTS) with emotion-steering forward-pre-hooks
    on the DiT transformer blocks' self-attention input.

    NOTE: F5-TTS internals vary slightly between versions. This assumes the
    standard `f5_tts` package layout:
        f5tts_api.ema_model.transformer.transformer_blocks[i].attn
    If your installed version differs, adjust `_collect_blocks()` accordingly.
    """

    def __init__(self, f5tts_api, emotion_vectors: Dict, device: str = DEVICE):
        self.api             = f5tts_api
        self.emotion_vectors = emotion_vectors
        self.device          = device
        self.steered_layers  = F5TTS_STEERED_LAYERS
        self.hooks           = []
        self.blocks          = None

    def _collect_blocks(self):
        if self.blocks is not None:
            return self.blocks

        # f5_tts.api.F5TTS exposes the underlying DiT at .ema_model (EMA-wrapped)
        model = getattr(self.api, "ema_model", None) or getattr(self.api, "model", None)
        if model is None:
            raise RuntimeError("Could not locate underlying F5-TTS model on API object")

        transformer = getattr(model, "transformer", model)
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError(
                "Could not find 'transformer_blocks' on F5-TTS model — "
                "check your f5_tts version and adjust F5TTSSteerer._collect_blocks()"
            )

        self.blocks = list(blocks)
        return self.blocks

    @staticmethod
    def _make_hook(vec: torch.Tensor, alpha: float):
        def hook(module, args, kwargs):
            # F5-TTS calls self.attn(x=norm, mask=mask, rope=rope) — all kwargs,
            # so args is an empty tuple. Read/write x from kwargs["x"].
            # Fall back to args[0] if x is positional (future-proofing).
            x_in_kwargs = "x" in kwargs
            if x_in_kwargs:
                x = kwargs["x"]
            elif len(args) > 0:
                x = args[0]
            else:
                return args, kwargs  # nothing to inject into

            if x.dim() == 3:
                v = (vec / (vec.norm() + 1e-8)).to(x.device, x.dtype)
                v = v.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
                x = x + alpha * v
                if x_in_kwargs:
                    kwargs["x"] = x
                    return args, kwargs
                else:
                    return (x,) + args[1:], kwargs

            return args, kwargs
        return hook

    def _register_hooks(self, emotion: str, alpha: float) -> bool:
        self._remove_hooks()

        if emotion not in self.emotion_vectors:
            return False

        try:
            blocks = self._collect_blocks()
        except RuntimeError as e:
            print(f"   [TTS/F5] {e}")
            return False

        vecs = self.emotion_vectors[emotion]

        for layer_idx in self.steered_layers:
            if layer_idx in vecs and layer_idx < len(blocks):
                block = blocks[layer_idx]
                attn  = getattr(block, "attn", None)
                if attn is not None:
                    hook_fn = self._make_hook(vecs[layer_idx], alpha)
                    handle  = attn.register_forward_pre_hook(hook_fn, with_kwargs=True)
                    self.hooks.append(handle)

        return len(self.hooks) > 0

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(self, text: str, ref_audio: str, ref_text: str,
                  emotion: Optional[str] = None, alpha: float = 0.0
                  ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Generate speech with optional emotion steering.

        ref_audio / ref_text come from a PRESET speaker (config.PRESET_SPEAKERS),
        never from the user's input audio — no voice cloning.
        """
        if emotion and emotion in self.emotion_vectors and alpha != 0:
            self._register_hooks(emotion, alpha)

        try:
            with torch.no_grad():
                wav, sr, _spec = self.api.infer(
                    ref_file=ref_audio,
                    ref_text=ref_text,
                    gen_text=text,
                    nfe_step=F5TTS_NFE_STEP,
                    cfg_strength=F5TTS_CFG_STRENGTH,
                    speed=F5TTS_SPEED,
                    remove_silence=True,
                )

            if wav is None:
                return None, None

            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().numpy()
            else:
                wav = np.asarray(wav).squeeze()

            return wav.astype(np.float32), int(sr) if sr else F5TTS_SAMPLE_RATE

        finally:
            self._remove_hooks()


def _load_f5tts(device: str):
    from f5_tts.api import F5TTS

    print(f"   [TTS] loading F5-TTS ('{F5TTS_MODEL_NAME}') ...")

    kwargs = dict(model=F5TTS_MODEL_NAME, device=device)

    if F5TTS_CKPT_PATH and str(F5TTS_CKPT_PATH) not in ("", "None") and F5TTS_CKPT_PATH.exists():
        kwargs["ckpt_file"] = str(F5TTS_CKPT_PATH)
    if F5TTS_VOCAB_PATH and str(F5TTS_VOCAB_PATH) not in ("", "None") and F5TTS_VOCAB_PATH.exists():
        kwargs["vocab_file"] = str(F5TTS_VOCAB_PATH)

    api = F5TTS(**kwargs)
    print("   [TTS] F5-TTS loaded")

    emotion_vectors = _load_steering_vectors(F5TTS_VECTORS_DIR, "f5tts", device)
    steerer = F5TTSSteerer(api, emotion_vectors, device)

    return steerer, F5TTS_SAMPLE_RATE, F5TTS_DEFAULT_ALPHA


# ============================================================================
# COSYVOICE2 STEERER  (toggle option)
# ============================================================================

class CosyVoice2Steerer:
    """
    CosyVoice2 with emotion-steering forward-pre-hooks on the flow-decoder
    transformer blocks (attn1 of each block).
    """

    def __init__(self, model, emotion_vectors: Dict, device: str = DEVICE):
        self.model           = model
        self.emotion_vectors = emotion_vectors
        self.device          = device
        self.steered_layers  = COSYVOICE_STEERED_LAYERS
        self.hooks           = []
        self.blocks          = None

    def _collect_blocks(self):
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

    @staticmethod
    def _make_hook(vec: torch.Tensor, alpha: float):
        def hook(module, args, kwargs):
            x = args[0]
            if x.dim() == 3:
                v = (vec / (vec.norm() + 1e-8)).to(x.device, x.dtype)
                v = v.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
                x = x + alpha * v
                return (x,) + args[1:], kwargs
            return args, kwargs
        return hook

    def _register_hooks(self, emotion: str, alpha: float) -> bool:
        self._remove_hooks()

        if emotion not in self.emotion_vectors:
            return False

        blocks = self._collect_blocks()
        vecs   = self.emotion_vectors[emotion]

        for layer_idx in self.steered_layers:
            if layer_idx in vecs and layer_idx < len(blocks):
                if hasattr(blocks[layer_idx], "attn1"):
                    hook_fn = self._make_hook(vecs[layer_idx], alpha)
                    handle  = blocks[layer_idx].attn1.register_forward_pre_hook(hook_fn, with_kwargs=True)
                    self.hooks.append(handle)

        return len(self.hooks) > 0

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(self, text: str, ref_audio: str, ref_text: str,
                  emotion: Optional[str] = None, alpha: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Generate speech with optional emotion steering.

        ref_audio / ref_text come from a PRESET speaker (config.PRESET_SPEAKERS),
        never from the user's input audio — no voice cloning.
        """
        import inspect

        if emotion and emotion in self.emotion_vectors and alpha != 0:
            self._register_hooks(emotion, alpha)

        try:
            sig         = inspect.signature(self.model.inference_zero_shot)
            audio_param = "prompt_wav" if "prompt_wav" in sig.parameters else "prompt_speech_16k"

            audio_chunks = []
            with torch.no_grad():
                for result in self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=ref_text,
                    **{audio_param: ref_audio},
                    stream=False,
                ):
                    if "tts_speech" in result:
                        audio_chunks.append(result["tts_speech"])

            if not audio_chunks:
                return None, None

            audio = torch.cat(audio_chunks, dim=-1).squeeze().cpu().numpy()
            return audio, COSYVOICE_SAMPLE_RATE

        finally:
            self._remove_hooks()


def _load_cosyvoice2(device: str):
    sys.path.insert(0, COSYVOICE_REPO_PATH)
    from cosyvoice.cli.cosyvoice import CosyVoice2

    print(f"   [TTS] loading CosyVoice2 from {COSYVOICE_MODEL_DIR} ...")
    model = CosyVoice2(
        model_dir=str(COSYVOICE_MODEL_DIR),
        load_jit=False,
        load_trt=False,
        fp16=False,
    )
    print("   [TTS] CosyVoice2 loaded")

    emotion_vectors = _load_steering_vectors(COSYVOICE_VECTORS_DIR, "cosyvoice2", device)
    steerer = CosyVoice2Steerer(model, emotion_vectors, device)

    return steerer, COSYVOICE_SAMPLE_RATE, COSYVOICE_DEFAULT_ALPHA


# ============================================================================
# UNIFIED TTS WRAPPER
# ============================================================================

_BACKEND_LOADERS = {
    "f5tts":      _load_f5tts,
    "cosyvoice2": _load_cosyvoice2,
}


class TTS:
    """
    Usage
    -----
        tts = TTS()                          # uses config.TTS_BACKEND (default "f5tts")
        tts = TTS(backend="cosyvoice2")       # explicit override

        audio, sr = tts.synthesize(
            text       = llm_response_text,
            speaker_id = "speaker_2",
            decision   = decision_output,    # for emotion + vector_intensity
        )
    """

    def __init__(self, backend: str = TTS_BACKEND, device: str = DEVICE):
        if backend not in _BACKEND_LOADERS:
            raise ValueError(f"Unknown TTS backend: '{backend}' (expected 'f5tts' or 'cosyvoice2')")

        self.backend       = backend
        self.device        = device
        self.sample_rate   = None
        self.default_alpha = None
        self.steerer       = None

        self._load(backend)

    def _load(self, backend: str):
        loader = _BACKEND_LOADERS[backend]
        self.steerer, self.sample_rate, self.default_alpha = loader(self.device)
        self.backend = backend

    def synthesize(
        self,
        text: str,
        speaker_id: str,
        decision: Optional[DecisionOutput] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Parameters
        ----------
        text       : LLM response text to speak
        speaker_id : key into config.PRESET_SPEAKERS (user-selected in UI)
        decision   : DecisionOutput from this turn — supplies emotion + vector_intensity
                      for TTS steering. If None, generates neutral speech.

        Returns
        -------
        audio : np.ndarray (float32, mono) or None
        sr    : int sample rate or None
        """
        speaker   = PRESET_SPEAKERS.get(speaker_id, PRESET_SPEAKERS[DEFAULT_SPEAKER])
        ref_audio = speaker["ref_audio"]
        ref_text  = speaker["ref_text"]

        emotion = None
        alpha   = 0.0

        if decision is not None:
            vector_emotion = extract_emotion_from_vector(decision.vector)
            if vector_emotion != "none" and vector_emotion in TTS_EMOTIONS:
                emotion = vector_emotion
                alpha   = self.default_alpha * decision.vector_intensity

        if emotion:
            print(f"   [TTS/{self.backend}] steering speech with emotion='{emotion}', alpha={alpha:.3f}")
        else:
            print(f"   [TTS/{self.backend}] neutral speech (no steering)")

        return self.steerer.generate(text, ref_audio, ref_text, emotion=emotion, alpha=alpha)