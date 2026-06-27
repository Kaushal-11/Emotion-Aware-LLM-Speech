"""
Emotional AI – SenseVoice Small: Train
=======================================
Model:  FunAudioLLM/SenseVoiceSmall

Root-cause of the crash
────────────────────────
SenseVoiceSmall uses a WavFrontend that:
  1. Computes 80-dim log-Mel filterbank (via torchaudio.compliance.kaldi.fbank)
  2. Applies LFR (Low Frame Rate) with lfr_m=7, lfr_n=1:
       stacks 7 consecutive frames  →  80×7 = 560-dim features
  3. Applies CMVN normalisation (from the model's am.mvn file)

SenseVoiceSmall.embed is nn.Embedding(N, input_size=560), so the
style/language/event query tokens are 560-dim.

SenseVoiceSmall.encode() does:
    style_query = self.embed(styles)        # (B, 1, 560)
    speech = torch.cat((style_query, speech), dim=1)   # needs speech to be 560-dim!

If you pass raw 80-dim fbank → crash:
    "Expected size 560 but got size 80 for tensor number 1 in the list"

Fix: use the model's own WavFrontend to produce 560-dim features, then call encode().

Launch (single GPU):
    python train_sensevoice.py

Launch (multi-GPU):
    accelerate launch --num_processes=2 --multi_gpu --mixed_precision=bf16 train_sensevoice.py
"""

import json
import math
import os
import random
import time
import pytz
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T

from accelerate import Accelerator, DistributedDataParallelKwargs

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

os.environ.setdefault("NCCL_TIMEOUT", "1800")
os.environ.setdefault("NCCL_DEBUG", "WARN")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_DIR        = "FunAudioLLM/SenseVoiceSmall"
MODEL_SHORT_NAME = "sensevoice-small"
SPLITS_DIR       = "/workspace/audio-em/dataset/splits"
DATASET_ROOT     = "/workspace/audio-em/dataset"
BASE_RESULTS_DIR = "/workspace/audio-em/finetune-results"

STRATEGY = "lora"   # "frozen" or "lora"

TARGET_SR    = 16_000
MAX_DURATION = 10.0   # seconds

BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 8

MAX_EPOCHS   = 30
PATIENCE     = 7
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 1e-4

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_LORA = 3e-5

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ["linear_q_k_v", "linear_out"]  # nn.Linear attr names inside MultiHeadedAttentionSANM

FOCAL_GAMMA     = 2.0
LABEL_SMOOTHING = 0.1

NUM_WORKERS   = 2
USE_GRAD_CKPT = False

USE_TENSORBOARD = True
SAVE_CURVES     = True

EMOTIONS    = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# SenseVoice encode() prepends 4 special tokens:
#   [language_query(1), event_query(1), emo_query(1), style_query(1)] = 4 total
# We pool encoder output from index 4 onwards (the actual speech frames).
NUM_PREFIX_TOKENS = 4

# Fixed token IDs for the dummy text tensor passed to encode():
#   text[:, 0] = language ID.  0 maps to lid_int_dict key for "auto"
#   text[:, 3] = textnorm ID.  25016 maps to textnorm_int_dict → index 14 ("withitn")
FIXED_LID_TOKEN      = 0
FIXED_TEXTNORM_TOKEN = 25016


# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits, targets):
        ce           = F.cross_entropy(logits, targets, weight=self.weight,
                                       label_smoothing=self.ls, reduction="none")
        focal_weight = (1.0 - torch.exp(-ce)) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class SenseVoiceEmotionDataset(Dataset):
    def __init__(self, split_json_path: str, dataset_root: str,
                 max_samples: int, augment: bool = False):
        with open(split_json_path, encoding="utf-8") as f:
            all_records = json.load(f)

        self.root        = Path(dataset_root).resolve()
        self.max_samples = max_samples
        self.augment     = augment
        self.noise_std   = 0.005

        self.records = []
        missing = 0
        for r in all_records:
            if r.get("emotion") not in EMOTIONS:
                continue
            ap = self._resolve_path(r["audio_path"])
            if not ap.exists():
                missing += 1
                continue
            self.records.append({
                "audio_path": str(ap),
                "emotion":    r["emotion"],
                "emotion_id": r["emotion_id"],
                "dataset":    r.get("dataset", ""),
            })

        if missing > 0:
            print(f"    WARNING: {missing} audio files not found and skipped")

    def _resolve_path(self, audio_path: str) -> Path:
        p = Path(audio_path)
        return p if p.is_absolute() else self.root / p

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        waveform, length = self._load_audio(r["audio_path"])
        if self.augment:
            waveform = self._augment(waveform)
        return {
            "waveform":   waveform,             # (max_samples,) float32
            "length":     length,               # actual samples before padding (int)
            "emotion_id": torch.tensor(r["emotion_id"], dtype=torch.long),
        }

    def _load_audio(self, path: str):
        """Returns (waveform, true_length) where waveform is padded/clipped to max_samples."""
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                wav = T.Resample(sr, TARGET_SR)(wav)
            wav = wav.squeeze(0)   # (T,)
            peak = wav.abs().max()
            if peak > 1e-6:
                wav = wav / peak
            true_len = min(wav.shape[0], self.max_samples)
            if wav.shape[0] < self.max_samples:
                wav = F.pad(wav, (0, self.max_samples - wav.shape[0]))
            else:
                wav = wav[:self.max_samples]
            return wav.float(), true_len
        except Exception:
            return torch.zeros(self.max_samples), self.max_samples

    def _augment(self, w: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.50:
            w = w + torch.randn_like(w) * self.noise_std
        if random.random() < 0.40:
            shift = int(random.uniform(-0.10, 0.10) * w.shape[0])
            w = torch.roll(w, shift)
        if random.random() < 0.40:
            w = w * random.uniform(0.7, 1.3)
        peak = w.abs().max()
        if peak > 1e-6:
            w = w / peak
        return w


def build_weighted_sampler(dataset: SenseVoiceEmotionDataset) -> WeightedRandomSampler:
    counts = defaultdict(int)
    for r in dataset.records:
        counts[r["emotion_id"]] += 1
    cw = {eid: 1.0 / cnt for eid, cnt in counts.items()}
    weights = [cw[r["emotion_id"]] for r in dataset.records]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def compute_class_weights(dataset: SenseVoiceEmotionDataset) -> torch.Tensor:
    counts = torch.zeros(NUM_CLASSES)
    for r in dataset.records:
        counts[r["emotion_id"]] += 1
    w = 1.0 / (counts + 1e-6)
    return w / w.sum() * NUM_CLASSES


# ─────────────────────────────────────────────
# MANUAL LoRA
# ─────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank ΔW = B·A path.

    During training:  output = x @ W.T + bias + (x @ A.T) @ B.T * scale
    A is initialised with kaiming_uniform, B with zeros → ΔW = 0 at step 0.
    Only A and B are trainable; the original weight/bias stay frozen.
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features

        # Keep the frozen original
        self.weight = linear.weight   # shape (out, in)
        self.bias   = linear.bias     # shape (out,) or None

        # LoRA matrices
        self.lora_A   = nn.Parameter(torch.empty(r, in_features))
        self.lora_B   = nn.Parameter(torch.zeros(out_features, r))
        self.scale    = alpha / r
        self.dropout  = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.lora_A)   # (*, r)
        lora = F.linear(lora, self.lora_B)               # (*, out)
        return base + lora * self.scale


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class SenseVoiceEmotionClassifier(nn.Module):
    """
    Wraps the full SenseVoiceSmall model and adds a classification head.

    Why encode() needs 560-dim input
    ─────────────────────────────────
    SenseVoiceSmall uses a WavFrontend that applies LFR (lfr_m=7) to 80-dim
    fbank, producing 80*7=560-dim features.  The model's token embedding
    (self.embed = nn.Embedding(N, input_size=560)) therefore outputs 560-dim
    vectors.  encode() does:

        style_query = self.embed(styles)           # (B, 1, 560)
        speech = torch.cat((style_query, speech))  # requires speech at dim-2 = 560

    Passing raw 80-dim fbank → RuntimeError: Expected size 560 but got size 80.

    Fix: call the model's own WavFrontend (stored at auto_model.kwargs["frontend"])
    which correctly applies kaldi fbank → LFR(7,1) → CMVN → 560-dim output,
    then pass that to encode().

    Encoder output shape: (B, 4+T'', encoder_dim)
    We pool only indices [4:] (the speech frames, skipping the 4 prefix tokens).
    """

    def __init__(self, model_dir: str, strategy: str,
                 use_grad_ckpt: bool = False):
        super().__init__()
        self.strategy = strategy

        print(f"  Loading SenseVoice from: {model_dir}")
        self.sv_model, self.frontend, self.encoder_dim = self._load_full_model(model_dir)

        # Classification head
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
            print("  [SenseVoice] Model FROZEN — training head only")
            total  = sum(p.numel() for p in self.parameters())
            head_p = sum(p.numel() for p in self.classifier.parameters())
            print(f"  Trainable: {head_p:,} / {total:,}")

        elif strategy == "lora":
            self._apply_lora()

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_full_model(self, model_dir: str):
        """Load SenseVoiceSmall + its WavFrontend."""
        try:
            from funasr import AutoModel
            auto_model = AutoModel(
                model=model_dir,
                device="cpu",
                hub="hf",
                disable_update=True,
            )
            sv_model = auto_model.model   # SenseVoiceSmall instance
            frontend = auto_model.kwargs.get("frontend", None)

            if frontend is None:
                raise RuntimeError(
                    "WavFrontend not found in auto_model.kwargs['frontend']. "
                    "FunASR may have changed its API. Check auto_model.kwargs keys: "
                    + str(list(auto_model.kwargs.keys()))
                )

            # Determine encoder output dim
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
                encoder_dim = 512   # SenseVoiceSmall default

            print(f"  [SenseVoice] Model loaded, encoder_dim={encoder_dim}")
            print(f"  [SenseVoice] Frontend: {type(frontend).__name__}, "
                  f"output_size={frontend.output_size() if hasattr(frontend, 'output_size') else '?'}")
            return sv_model, frontend, encoder_dim

        except ImportError:
            raise ImportError("FunASR not installed. Run: pip install funasr")
        except Exception as e:
            raise RuntimeError(f"Failed to load SenseVoice model: {e}")

    # ------------------------------------------------------------------ #
    # LoRA  (manual injection — no PEFT wrapper to avoid input_ids clash)
    # ------------------------------------------------------------------ #

    def _apply_lora(self):
        """
        Manually replace target nn.Linear layers inside every
        MultiHeadedAttentionSANM with a LoRALinear wrapper.

        Why not get_peft_model():
          PEFT wraps the entire encoder module and routes calls through
          a HuggingFace-style forward that injects `input_ids` as a kwarg.
          SenseVoiceEncoderSmall.forward(xs_pad, ilens) does not accept
          `input_ids`, causing:
              TypeError: forward() got an unexpected keyword argument 'input_ids'

        Manual injection replaces only the specific nn.Linear objects
        (linear_q_k_v, linear_out) inside each SANM attention block,
        leaving the encoder's own forward() completely untouched.
        """
        # First freeze everything
        for p in self.sv_model.parameters():
            p.requires_grad = False

        injected = 0
        for module in self.sv_model.encoder.modules():
            # Target: MultiHeadedAttentionSANM blocks (identified by having both attrs)
            for attr in LORA_TARGETS:
                original = getattr(module, attr, None)
                if isinstance(original, nn.Linear):
                    setattr(module, attr, LoRALinear(original, LORA_R, LORA_ALPHA, LORA_DROPOUT))
                    injected += 1

        if injected == 0:
            print("  [SenseVoice] WARNING: No LoRA targets found — model stays frozen")
            return

        total     = sum(p.numel() for p in self.sv_model.parameters())
        trainable = sum(p.numel() for p in self.sv_model.parameters() if p.requires_grad)
        print(f"  [SenseVoice] LoRA injected into {injected} linear layers")
        print(f"  [SenseVoice] Encoder trainable: {trainable:,} / {total:,} "
              f"({100*trainable/max(total,1):.2f}%)")

    # ------------------------------------------------------------------ #
    # Frontend: raw waveform → 560-dim LFR fbank
    # ------------------------------------------------------------------ #

    def _extract_frontend_features(
        self,
        waveform: torch.Tensor,    # (B, T)  float32, peak-normalised
        lengths: torch.Tensor,     # (B,)    int, true sample counts
    ):
        """
        Run the model's own WavFrontend on a batch.

        WavFrontend.forward() expects:
          input          (B, T)   — padded waveforms in the range the model expects
          input_lengths  (B,)     — true lengths in *samples*

        It applies kaldi.fbank → LFR(7,1) → CMVN and returns:
          feats          (B, T'', 560)
          feat_lengths   (B,)

        The frontend internally scales waveforms by (1<<15) if upsacle_samples is True
        (the SenseVoice config sets this to True for 16-bit PCM equivalence).
        We do NOT pre-scale here — the frontend handles it.

        Note: WavFrontend uses torchaudio.compliance.kaldi.fbank which requires
        CPU tensors. We move to CPU, run the frontend, then move back.
        """
        device = waveform.device

        # kaldi.fbank requires CPU
        wav_cpu = waveform.float().cpu()
        len_cpu = lengths.cpu()

        with torch.no_grad() if not self.sv_model.training else torch.enable_grad():
            # Frontend is not a gradient path we care about (frozen weights),
            # but we need to handle training mode for LoRA.
            # Safest: always no_grad for frontend (it has no learnable params in frozen mode)
            pass

        # Frontend has no learnable params we train, so always no_grad here
        with torch.no_grad():
            feats, feat_lengths = self.frontend(wav_cpu, len_cpu)

        return feats.to(device), feat_lengths.to(device)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        waveform : (B, T)  raw 16 kHz audio, float32, peak-normalised to [-1, 1]
        lengths  : (B,)    true sample counts (before padding)
        returns  : (B, NUM_CLASSES)  logits
        """
        B      = waveform.shape[0]
        device = waveform.device

        # 1. WavFrontend: raw waveform → LFR+CMVN fbank, shape (B, T'', 560)
        feats, feat_lengths = self._extract_frontend_features(waveform, lengths)
        # feat_lengths is in *frames*, not samples

        # 2. Build dummy `text` tensor for encode().
        #    encode() reads:
        #      text[:, 0] for language token ID (0 = "auto" path)
        #      text[:, 3] for textnorm token ID (25016 → index 14 = "withitn")
        dummy_text       = torch.zeros(B, 4, dtype=torch.long, device=device)
        dummy_text[:, 0] = FIXED_LID_TOKEN
        dummy_text[:, 3] = FIXED_TEXTNORM_TOKEN

        # 3. encode(): prepends 4 prefix token embeddings, runs the encoder.
        #    Input:  feats (B, T'', 560),  feat_lengths (B,)
        #    Output: enc_out (B, 4+T'', encoder_dim)
        enc_out, _ = self.sv_model.encode(feats, feat_lengths, dummy_text)

        # 4. Drop the 4 prefix token positions, mean-pool over speech frames
        speech_enc = enc_out[:, NUM_PREFIX_TOKENS:, :]   # (B, T'', encoder_dim)
        pooled     = speech_enc.mean(dim=1)               # (B, encoder_dim)

        return self.classifier(pooled)


# ─────────────────────────────────────────────
# COLLATE — needed because forward() now takes lengths
# ─────────────────────────────────────────────

def collate_fn(batch):
    waveforms  = torch.stack([b["waveform"]   for b in batch])
    lengths    = torch.tensor([b["length"]    for b in batch], dtype=torch.long)
    emotion_ids = torch.stack([b["emotion_id"] for b in batch])
    return {"waveform": waveforms, "length": lengths, "emotion_id": emotion_ids}


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(preds, labels):
    preds  = np.array(preds)
    labels = np.array(labels)
    acc    = (preds == labels).mean()
    wf1    = f1_score(labels, preds, average="weighted", zero_division=0)
    pcf    = f1_score(labels, preds, average=None,
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    cm     = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(labels, preds, target_names=EMOTIONS,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy":              float(acc),
        "weighted_f1":           float(wf1),
        "per_class_f1":          {EMOTIONS[i]: float(pcf[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def save_training_curves(history, output_dir, strategy):
    if not PLT_AVAILABLE or not history:
        return
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_f1     = [h["val_f1"]     for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{MODEL_SHORT_NAME} – {strategy}", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, train_loss, "b-o", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_loss,   "r-s", label="Val",   linewidth=2)
    axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    best_e = int(np.argmax(val_f1)) + 1
    best_f = max(val_f1)
    axes[0, 1].plot(epochs, val_f1, "g-^", linewidth=2)
    axes[0, 1].axhline(y=best_f, color="gold",   linestyle="--", alpha=0.7)
    axes[0, 1].axvline(x=best_e, color="purple", linestyle="--", alpha=0.7)
    axes[0, 1].set_title(f"Val wF1  (best {best_f:.4f} @ ep{best_e})")
    axes[0, 1].grid(True, alpha=0.3)

    gap = [v - t for v, t in zip(val_loss, train_loss)]
    axes[1, 0].bar(epochs, gap, alpha=0.6, color="orange", edgecolor="black")
    axes[1, 0].axhline(y=0, color="red", linewidth=1.5)
    axes[1, 0].set_title("Overfitting gap"); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, train_loss, "b-", alpha=0.5, label="Train loss")
    axes[1, 1].plot(epochs, val_loss,   "r-", alpha=0.5, label="Val loss")
    ax2 = axes[1, 1].twinx()
    ax2.plot(epochs, val_f1, "g-", linewidth=2, label="Val F1")
    axes[1, 1].set_title("Combined")
    axes[1, 1].legend(loc="upper left"); ax2.legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves → {out}")


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, accelerator, scheduler=None):
    model.train()
    total_loss = 0.0
    local_preds:  list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    t0 = time.time()

    for batch in tqdm(loader, desc="  train", leave=False, dynamic_ncols=True):
        with accelerator.accumulate(model):
            logits = model(batch["waveform"], batch["length"])
            loss   = criterion(logits, batch["emotion_id"])
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        local_preds.append(logits.detach().argmax(dim=-1))
        local_labels.append(batch["emotion_id"])

    all_preds  = accelerator.gather_for_metrics(torch.cat(local_preds))
    all_labels = accelerator.gather_for_metrics(torch.cat(local_labels))
    elapsed    = time.time() - t0

    return (total_loss / max(len(loader), 1),
            compute_metrics(all_preds.cpu().numpy(), all_labels.cpu().numpy()),
            elapsed)


@torch.no_grad()
def evaluate(model, loader, criterion, accelerator):
    model.eval()
    total_loss = 0.0
    local_preds:  list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="  eval ", leave=False, dynamic_ncols=True):
        logits = model(batch["waveform"], batch["length"])
        total_loss += criterion(logits, batch["emotion_id"]).item()
        local_preds.append(logits.argmax(dim=-1))
        local_labels.append(batch["emotion_id"])

    all_preds  = accelerator.gather_for_metrics(torch.cat(local_preds))
    all_labels = accelerator.gather_for_metrics(torch.cat(local_labels))

    return (total_loss / max(len(loader), 1),
            compute_metrics(all_preds.cpu().numpy(), all_labels.cpu().numpy()))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ist       = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist).strftime("%Y%m%d_%H%M%S")

    output_dir  = Path(BASE_RESULTS_DIR) / MODEL_SHORT_NAME / STRATEGY / timestamp
    weights_dir = output_dir / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)

    splits_dir   = Path(SPLITS_DIR).resolve()
    dataset_root = Path(DATASET_ROOT).resolve()
    max_samples  = int(MAX_DURATION * TARGET_SR)

    lr = LEARNING_RATE_LORA if STRATEGY == "lora" else LEARNING_RATE_HEAD

    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS,
                              kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_main_process
    pin     = torch.cuda.is_available()

    config = {
        "model_dir": MODEL_DIR, "strategy": STRATEGY, "timestamp": timestamp,
        "run_id": f"{MODEL_SHORT_NAME}_{STRATEGY}_{timestamp}",
        "output_dir": str(output_dir),
        "hyperparameters": {
            "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS * accelerator.num_processes,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "learning_rate": lr, "weight_decay": WEIGHT_DECAY,
            "focal_gamma": FOCAL_GAMMA, "label_smoothing": LABEL_SMOOTHING,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "max_duration": MAX_DURATION, "num_workers": NUM_WORKERS,
        },
        "emotions": EMOTIONS,
    }

    if is_main:
        eff = BATCH_SIZE * GRAD_ACCUM_STEPS * accelerator.num_processes
        print(f"\n{'='*70}")
        print(f"  SenseVoice-Small Emotion Training")
        print(f"{'='*70}")
        print(f"  Run:        {config['run_id']}")
        print(f"  Strategy:   {STRATEGY}  |  Devices: {accelerator.num_processes}")
        print(f"  Batch/GPU:  {BATCH_SIZE}  |  Accum: {GRAD_ACCUM_STEPS}"
              f"  |  Effective: {eff}")
        print(f"  LR:         {lr}  |  Focal γ: {FOCAL_GAMMA}")
        print(f"  Max audio:  {MAX_DURATION}s = {max_samples:,} samples")
        print(f"{'='*70}\n")
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    tb_writer = (
        SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if is_main and USE_TENSORBOARD and TB_AVAILABLE else None
    )

    if is_main:
        print("Loading datasets ...")

    train_dataset = SenseVoiceEmotionDataset(
        str(splits_dir / "train.json"), str(dataset_root),
        max_samples=max_samples, augment=True)
    val_dataset = SenseVoiceEmotionDataset(
        str(splits_dir / "val.json"), str(dataset_root),
        max_samples=max_samples, augment=False)

    train_sampler = build_weighted_sampler(train_dataset)
    class_weights = compute_class_weights(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
    )

    if is_main:
        print(f"  Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}\n")

    model = SenseVoiceEmotionClassifier(
        MODEL_DIR, strategy=STRATEGY, use_grad_ckpt=USE_GRAD_CKPT)
    criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA,
                          label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )

    total_steps  = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    (model, optimizer,
     train_loader, val_loader, scheduler) = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)
    criterion = criterion.to(accelerator.device)

    best_val_f1      = 0.0
    patience_counter = 0
    history          = []
    epoch_metrics    = []

    for epoch in range(1, MAX_EPOCHS + 1):
        if is_main:
            print(f"\n{'─'*70}")
            print(f"  Epoch {epoch}/{MAX_EPOCHS}")
            print(f"{'─'*70}")

        train_loss, train_m, train_sec = train_one_epoch(
            model, train_loader, optimizer, criterion, accelerator, scheduler)
        val_loss, val_m = evaluate(model, val_loader, criterion, accelerator)

        epoch_metrics.append({
            "epoch":              epoch,
            "train_loss":         train_loss,
            "train_accuracy":     train_m["accuracy"],
            "train_weighted_f1":  train_m["weighted_f1"],
            "train_per_class_f1": train_m["per_class_f1"],
            "val_loss":           val_loss,
            "val_accuracy":       val_m["accuracy"],
            "val_weighted_f1":    val_m["weighted_f1"],
            "val_per_class_f1":   val_m["per_class_f1"],
            "epoch_time_sec":     train_sec,
        })

        if is_main:
            eta_min = train_sec * (MAX_EPOCHS - epoch) / 60
            print(f"  train  loss:{train_loss:.4f}  acc:{train_m['accuracy']:.4f}"
                  f"  wF1:{train_m['weighted_f1']:.4f}  [{train_sec/60:.1f} min]")
            print(f"  val    loss:{val_loss:.4f}  acc:{val_m['accuracy']:.4f}"
                  f"  wF1:{val_m['weighted_f1']:.4f}")
            print(f"  per-class val F1: "
                  + "  ".join(f"{e[:3]}:{v:.2f}"
                               for e, v in val_m["per_class_f1"].items()))
            print(f"  ETA: ~{eta_min:.0f} min")

            if tb_writer:
                tb_writer.add_scalar("Loss/train", train_loss, epoch)
                tb_writer.add_scalar("Loss/val",   val_loss,   epoch)
                tb_writer.add_scalar("F1/train",   train_m["weighted_f1"], epoch)
                tb_writer.add_scalar("F1/val",     val_m["weighted_f1"],   epoch)
                for emo, v in val_m["per_class_f1"].items():
                    tb_writer.add_scalar(f"F1_val/{emo}", v, epoch)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_f1": val_m["weighted_f1"],
        })

        if val_m["weighted_f1"] > best_val_f1:
            best_val_f1      = val_m["weighted_f1"]
            patience_counter = 0
            if is_main:
                print(f"  ✓ New best val wF1: {best_val_f1:.4f} — saving checkpoint")
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), weights_dir / "best_model.pt")
                with open(output_dir / "best_metrics.json", "w") as f:
                    json.dump({"epoch": epoch, "best_val_f1": best_val_f1,
                               "metrics": val_m}, f, indent=2)
        else:
            patience_counter += 1
            if is_main:
                print(f"  patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                if is_main:
                    print(f"\n[Early stopping] after epoch {epoch}")
                break

    if is_main:
        print(f"\n{'='*70}")
        print(f"  Training Complete — best val wF1: {best_val_f1:.4f}")
        print(f"  Results: {output_dir}")
        print(f"{'='*70}\n")
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(epoch_metrics, f, indent=2)
        if SAVE_CURVES:
            save_training_curves(history, output_dir, STRATEGY)
        if tb_writer:
            tb_writer.close()


if __name__ == "__main__":
    main()