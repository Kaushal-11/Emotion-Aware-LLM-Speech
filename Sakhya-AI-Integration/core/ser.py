"""
core/ser.py
-----------
Layer 0b — Speech Emotion Recognition (SER)

Converts the SenseVoice / WavLM test-evaluation scripts into a live,
single-utterance inference wrapper.

Exposes one class, `SER`, which loads either backend based on
config.SER_BACKEND ("sensevoice" default, or "wavlm") and provides:

    predict(audio_path_or_array) -> (emotion: str, confidence: float)

SER output is used as a VALIDATOR / confidence signal in the fusion layer
(core/fusion.py) — the text classifier remains the primary decider for
target and intensity.
"""

import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from config import (
    SER_BACKEND,
    SER_EMOTIONS,
    SER_TARGET_SR,
    SER_MAX_DURATION,
    DEVICE,
    SENSEVOICE_MODEL_DIR,
    SENSEVOICE_FUNASR_FALLBACK,
    SENSEVOICE_LORA_R,
    SENSEVOICE_LORA_ALPHA,
    SENSEVOICE_LORA_DROPOUT,
    SENSEVOICE_LORA_TARGETS,
    SENSEVOICE_NUM_PREFIX_TOKENS,
    SENSEVOICE_FIXED_LID_TOKEN,
    SENSEVOICE_FIXED_TEXTNORM_TOKEN,
    WAVLM_MODEL_DIR,
    WAVLM_LORA_R,
    WAVLM_LORA_ALPHA,
    WAVLM_LORA_DROPOUT,
    WAVLM_LORA_TARGET,
)

NUM_CLASSES = len(SER_EMOTIONS)


# ============================================================================
# AUDIO LOADING (shared by both backends)
# ============================================================================

def load_waveform(audio: Union[str, np.ndarray], sample_rate: int = 16000):
    """
    Returns (waveform: torch.FloatTensor (T,), length: int, padded_to_max: torch.FloatTensor)
    - resampled to SER_TARGET_SR
    - mono
    - peak-normalised
    - padded/truncated to SER_MAX_DURATION seconds
    """
    if isinstance(audio, (str, Path)):
        wav, sr = torchaudio.load(str(audio))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SER_TARGET_SR:
            wav = T.Resample(sr, SER_TARGET_SR)(wav)
        wav = wav.squeeze(0)
    else:
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if sample_rate != SER_TARGET_SR:
            wav = T.Resample(sample_rate, SER_TARGET_SR)(wav.unsqueeze(0)).squeeze(0)

    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak

    max_samples = int(SER_MAX_DURATION * SER_TARGET_SR)
    true_len = min(wav.shape[0], max_samples)

    if wav.shape[0] < max_samples:
        wav = F.pad(wav, (0, max_samples - wav.shape[0]))
    else:
        wav = wav[:max_samples]

    return wav.float(), true_len


# ============================================================================
# SENSEVOICE  (default backend)
# ============================================================================

class LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with frozen base + trainable low-rank ΔW."""

    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features
        self.weight  = linear.weight
        self.bias    = linear.bias
        self.lora_A  = nn.Parameter(torch.empty(r, in_features))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, r))
        self.scale   = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)
        return base + lora * self.scale


class SenseVoiceEmotionClassifier(nn.Module):
    """
    Must be architecturally identical to the training script so that
    strict=True state_dict loading works.

    Uses the model's own WavFrontend (LFR lfr_m=7 -> 560-dim) so encode()
    receives the correct input shape.
    """

    def __init__(self, model_dir: str, strategy: str):
        super().__init__()
        self.strategy = strategy
        self.sv_model, self.frontend, self.encoder_dim = self._load_full_model(model_dir)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )

        if strategy == "frozen":
            for p in self.sv_model.parameters():
                p.requires_grad = False
        elif strategy == "lora":
            self._apply_lora()

    def _load_full_model(self, model_dir: str):
        from funasr import AutoModel
        auto_model = AutoModel(
            model=model_dir,
            device="cpu",
            hub="hf",
            disable_update=True,
        )
        sv_model = auto_model.model
        frontend = auto_model.kwargs.get("frontend", None)

        if frontend is None:
            raise RuntimeError(
                "WavFrontend not found in auto_model.kwargs['frontend']. "
                "Available keys: " + str(list(auto_model.kwargs.keys()))
            )

        encoder     = sv_model.encoder
        encoder_dim = None
        if hasattr(encoder, "output_size"):
            v = encoder.output_size
            encoder_dim = v() if callable(v) else v
        if encoder_dim is None and hasattr(encoder, "_output_size"):
            encoder_dim = encoder._output_size
        if encoder_dim is None:
            for m in encoder.modules():
                if isinstance(m, nn.Linear):
                    encoder_dim = m.out_features
                    break
        if encoder_dim is None:
            encoder_dim = 512

        print(f"   [SenseVoice] encoder_dim={encoder_dim}")
        return sv_model, frontend, encoder_dim

    def _apply_lora(self):
        for p in self.sv_model.parameters():
            p.requires_grad = False
        injected = 0
        for module in self.sv_model.encoder.modules():
            for attr in SENSEVOICE_LORA_TARGETS:
                original = getattr(module, attr, None)
                if isinstance(original, nn.Linear):
                    setattr(module, attr, LoRALinear(
                        original, SENSEVOICE_LORA_R, SENSEVOICE_LORA_ALPHA, SENSEVOICE_LORA_DROPOUT
                    ))
                    injected += 1
        if injected == 0:
            print("   [SenseVoice] WARNING: No LoRA targets found - model stays frozen")

    def _extract_frontend_features(self, waveform: torch.Tensor, lengths: torch.Tensor):
        """Run WavFrontend on CPU (kaldi.fbank requirement), return on original device."""
        device  = waveform.device
        wav_cpu = waveform.float().cpu()
        len_cpu = lengths.cpu()
        with torch.no_grad():
            feats, feat_lengths = self.frontend(wav_cpu, len_cpu)
        return feats.to(device), feat_lengths.to(device)

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        waveform : (B, T)  raw 16kHz, float32, peak-normalised
        lengths  : (B,)    true sample counts before padding
        returns  : (B, NUM_CLASSES) logits
        """
        B      = waveform.shape[0]
        device = waveform.device

        feats, feat_lengths = self._extract_frontend_features(waveform, lengths)

        dummy_text       = torch.zeros(B, 4, dtype=torch.long, device=device)
        dummy_text[:, 0] = SENSEVOICE_FIXED_LID_TOKEN
        dummy_text[:, 3] = SENSEVOICE_FIXED_TEXTNORM_TOKEN

        enc_out, _ = self.sv_model.encode(feats, feat_lengths, dummy_text)

        speech_enc = enc_out[:, SENSEVOICE_NUM_PREFIX_TOKENS:, :]
        pooled     = speech_enc.mean(dim=1)

        return self.classifier(pooled)


def _load_sensevoice(device: torch.device):
    import json
    model_dir = Path(SENSEVOICE_MODEL_DIR)
    cfg_path  = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    strategy   = cfg.get("strategy", "lora")
    funasr_dir = cfg.get("model_dir", SENSEVOICE_FUNASR_FALLBACK)

    ckpt = model_dir / "weights" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = SenseVoiceEmotionClassifier(funasr_dir, strategy)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    print(f"   [SenseVoice] loaded ({strategy}) from {model_dir}")
    return model


# ============================================================================
# WAVLM  (toggle backend)
# ============================================================================

class WavLMEmotionClassifier(nn.Module):
    """Must match the WavLM training script exactly."""

    def __init__(self, strategy: str = "lora"):
        super().__init__()
        from transformers import WavLMModel
        from peft import LoraConfig, get_peft_model

        self.strategy = strategy
        self.wavlm    = WavLMModel.from_pretrained("microsoft/wavlm-large")
        hidden        = self.wavlm.config.hidden_size  # 1024

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_CLASSES),
        )

        if strategy == "frozen":
            for p in self.wavlm.parameters():
                p.requires_grad = False
        elif strategy == "lora":
            cfg = LoraConfig(
                r=WAVLM_LORA_R, lora_alpha=WAVLM_LORA_ALPHA,
                lora_dropout=WAVLM_LORA_DROPOUT,
                target_modules=WAVLM_LORA_TARGET,
                bias="none",
            )
            self.wavlm = get_peft_model(self.wavlm, cfg)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self.wavlm(input_values=input_values).last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


def _load_wavlm(device: torch.device):
    import json
    model_dir = Path(WAVLM_MODEL_DIR)
    cfg_path  = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    strategy = cfg.get("strategy", "lora")

    ckpt_path = model_dir / "weights" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model      = WavLMEmotionClassifier(strategy=strategy)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(f"   [WavLM] loaded ({strategy}) from {model_dir}")
    return model


# ============================================================================
# UNIFIED SER WRAPPER
# ============================================================================

class SER:
    """
    Usage
    -----
        ser = SER()                       # uses config.SER_BACKEND
        ser = SER(backend="wavlm")        # explicit override

        emotion, confidence = ser.predict("turn_001.wav")
    """

    def __init__(self, backend: str = SER_BACKEND, device: str = DEVICE):
        self.backend = backend
        self.device  = torch.device(device)

        if backend == "sensevoice":
            self.model = _load_sensevoice(self.device)
        elif backend == "wavlm":
            self.model = _load_wavlm(self.device)
        else:
            raise ValueError(f"Unknown SER backend: '{backend}' (expected 'sensevoice' or 'wavlm')")

    @torch.no_grad()
    def predict(self, audio: Union[str, np.ndarray], sample_rate: int = 16000) -> tuple[str, float]:
        """
        Parameters
        ----------
        audio       : path to wav file, OR 1-D float32 numpy array
        sample_rate : required if `audio` is a numpy array

        Returns
        -------
        emotion    : str   — one of SER_EMOTIONS
        confidence : float — softmax probability of predicted class, in [0, 1]
        """
        waveform, length = load_waveform(audio, sample_rate)
        waveform = waveform.unsqueeze(0).to(self.device)   # (1, T)

        if self.backend == "sensevoice":
            lengths = torch.tensor([length], dtype=torch.long, device=self.device)
            logits  = self.model(waveform, lengths)
        else:  # wavlm
            logits = self.model(waveform)

        probs = torch.softmax(logits, dim=-1)
        pred_id    = int(probs.argmax(dim=-1).item())
        confidence = float(probs[0, pred_id].item())

        return SER_EMOTIONS[pred_id], confidence