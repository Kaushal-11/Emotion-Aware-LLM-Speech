"""
Emotional AI – SenseVoice Small: Test
=======================================
Loads best_model.pt saved by train_sensevoice.py and evaluates on the
held-out test set.

Key requirement: this file's SenseVoiceEmotionClassifier must be
architecturally identical to the one in train_sensevoice.py so that
strict=True state_dict loading works.

- Uses the model's own WavFrontend (LFR lfr_m=7 → 560-dim) instead of
  a hand-rolled MelSpectrogram, matching the train code exactly.
- forward(waveform, lengths) signature matches train code.
- Dataset returns "length" (true sample count) alongside "waveform".
- No accelerator at test time.

Usage:
    python test_sensevoice.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report, accuracy_score)
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_DIR    = "/workspace/audio-em/finetune-results/sensevoice-small/lora/20260521_212218"
SPLITS_DIR   = "/workspace/audio-em/dataset/splits"
DATASET_ROOT = "/workspace/audio-em/dataset"
BATCH_SIZE   = 16
NUM_WORKERS  = 4
OUTPUT_DIR   = None   # None → MODEL_DIR/test_results


# ─────────────────────────────────────────────
# CONSTANTS  (must match train_sensevoice.py exactly)
# ─────────────────────────────────────────────

EMOTIONS         = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]
NUM_CLASSES      = len(EMOTIONS)
TARGET_SR        = 16_000
MAX_DURATION     = 10.0
NUM_PREFIX_TOKENS = 4

FIXED_LID_TOKEN      = 0
FIXED_TEXTNORM_TOKEN = 25016

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ["linear_q_k_v", "linear_out"]   # actual SANM module names


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class SenseVoiceEmotionDataset(Dataset):
    def __init__(self, split_json_path: str, dataset_root: str, max_samples: int):
        with open(split_json_path, encoding="utf-8") as f:
            all_records = json.load(f)

        self.root        = Path(dataset_root).resolve()
        self.max_samples = max_samples
        self.records     = []

        missing = 0
        for r in all_records:
            if r.get("emotion") not in EMOTIONS:
                continue
            ap = self._resolve(r["audio_path"])
            if not ap.exists():
                missing += 1
                continue
            self.records.append({
                "audio_path": str(ap),
                "emotion":    r["emotion"],
                "emotion_id": r["emotion_id"],
            })

        if missing > 0:
            print(f"    WARNING: {missing} audio files not found and skipped")

    def _resolve(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        waveform, length = self._load(r["audio_path"])
        return {
            "waveform":   waveform,                                        # (max_samples,) float32
            "length":     length,                                          # true samples before padding
            "emotion_id": torch.tensor(r["emotion_id"], dtype=torch.long),
        }

    def _load(self, path: str):
        """Returns (waveform padded to max_samples, true_length)."""
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                wav = T.Resample(sr, TARGET_SR)(wav)
            wav = wav.squeeze(0)
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


def collate_fn(batch):
    waveforms   = torch.stack([b["waveform"]   for b in batch])
    lengths     = torch.tensor([b["length"]    for b in batch], dtype=torch.long)
    emotion_ids = torch.stack([b["emotion_id"] for b in batch])
    return {"waveform": waveforms, "length": lengths, "emotion_id": emotion_ids}


# ─────────────────────────────────────────────
# MANUAL LoRA  (must match train_sensevoice.py exactly)
# ─────────────────────────────────────────────

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
        import math
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)
        return base + lora * self.scale


# ─────────────────────────────────────────────
# MODEL  (must match train_sensevoice.py exactly)
# ─────────────────────────────────────────────

class SenseVoiceEmotionClassifier(nn.Module):
    """
    Identical architecture to the one in train_sensevoice.py.

    Uses the model's WavFrontend (LFR lfr_m=7 → 560-dim features) so that
    encode() receives the correct 560-dim input and doesn't crash with:
        "Expected size 560 but got size 80"
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

    # ------------------------------------------------------------------ #

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

        print(f"  [SenseVoice] Model loaded, encoder_dim={encoder_dim}")
        print(f"  [SenseVoice] Frontend: {type(frontend).__name__}, "
              f"output_size={frontend.output_size() if hasattr(frontend, 'output_size') else '?'}")
        return sv_model, frontend, encoder_dim

    def _apply_lora(self):
        for p in self.sv_model.parameters():
            p.requires_grad = False
        injected = 0
        for module in self.sv_model.encoder.modules():
            for attr in LORA_TARGETS:
                original = getattr(module, attr, None)
                if isinstance(original, nn.Linear):
                    setattr(module, attr, LoRALinear(original, LORA_R, LORA_ALPHA, LORA_DROPOUT))
                    injected += 1
        if injected == 0:
            print("  [SenseVoice] WARNING: No LoRA targets found — model stays frozen")

    # ------------------------------------------------------------------ #

    def _extract_frontend_features(
        self,
        waveform: torch.Tensor,   # (B, T)
        lengths:  torch.Tensor,   # (B,) true sample counts
    ):
        """Run WavFrontend on CPU (kaldi.fbank requirement), return on original device."""
        device   = waveform.device
        wav_cpu  = waveform.float().cpu()
        len_cpu  = lengths.cpu()
        with torch.no_grad():
            feats, feat_lengths = self.frontend(wav_cpu, len_cpu)
        return feats.to(device), feat_lengths.to(device)

    # ------------------------------------------------------------------ #

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        waveform : (B, T)  raw 16 kHz, float32, peak-normalised
        lengths  : (B,)    true sample counts before padding
        returns  : (B, NUM_CLASSES) logits
        """
        B      = waveform.shape[0]
        device = waveform.device

        # 1. WavFrontend → (B, T'', 560)
        feats, feat_lengths = self._extract_frontend_features(waveform, lengths)

        # 2. Dummy text tensor for encode()
        dummy_text       = torch.zeros(B, 4, dtype=torch.long, device=device)
        dummy_text[:, 0] = FIXED_LID_TOKEN
        dummy_text[:, 3] = FIXED_TEXTNORM_TOKEN

        # 3. encode() → (B, 4+T'', encoder_dim)
        enc_out, _ = self.sv_model.encode(feats, feat_lengths, dummy_text)

        # 4. Drop 4 prefix tokens, mean-pool speech frames
        speech_enc = enc_out[:, NUM_PREFIX_TOKENS:, :]   # (B, T'', encoder_dim)
        pooled     = speech_enc.mean(dim=1)               # (B, encoder_dim)

        return self.classifier(pooled)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_dir: Path, device: torch.device):
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    strategy   = cfg.get("strategy", "frozen")
    funasr_dir = cfg.get("model_dir", "FunAudioLLM/SenseVoiceSmall")

    print(f"  Strategy  : {strategy}")
    print(f"  FunASR dir: {funasr_dir}")

    ckpt = model_dir / "weights" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"  Checkpoint: {ckpt}")

    model = SenseVoiceEmotionClassifier(funasr_dir, strategy)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable")
    return model, strategy


# ─────────────────────────────────────────────
# METRICS + CONFUSION MATRIX
# ─────────────────────────────────────────────

def compute_metrics(preds, labels):
    acc    = accuracy_score(labels, preds)
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


def save_confusion_matrix(cm, class_names, path, title=""):
    if not PLT_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix → {path}")


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="  evaluating", leave=False):
        wav    = batch["waveform"].to(device)
        lens   = batch["length"].to(device)
        labels = batch["emotion_id"].to(device)

        logits = model(wav, lens)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["predictions"]   = all_preds
    metrics["true_labels"]   = all_labels
    metrics["probabilities"] = [p.tolist() for p in all_probs]
    return metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    model_dir  = Path(MODEL_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else model_dir / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("  SenseVoice-Small Emotion — Test Evaluation")
    print(f"{'='*70}")
    print(f"  Model dir : {model_dir}")
    print(f"  Device    : {device}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*70}\n")

    max_samples = int(MAX_DURATION * TARGET_SR)
    test_dataset = SenseVoiceEmotionDataset(
        str(Path(SPLITS_DIR) / "test.json"),
        DATASET_ROOT,
        max_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    print(f"Test samples: {len(test_dataset):,}\n")

    print("Loading model ...")
    model, strategy = load_model(model_dir, device)
    print("  ✓ Loaded\n")

    print("Evaluating ...")
    metrics = evaluate(model, test_loader, device)

    print(f"\n{'='*70}")
    print("  TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Strategy    : {strategy}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}")
    print(f"\n  Per-class F1:")
    for emo, f1 in metrics["per_class_f1"].items():
        bar = "█" * int(f1 * 20)
        print(f"    {emo:<12}: {f1:.4f}  {bar}")

    results = {
        "model_dir":   str(model_dir),
        "strategy":    strategy,
        "num_samples": len(test_dataset),
        "test_metrics": {
            "accuracy":         metrics["accuracy"],
            "weighted_f1":      metrics["weighted_f1"],
            "per_class_f1":     metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "classification_report": metrics["classification_report"],
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results → {output_dir / 'test_results.json'}")

    save_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        EMOTIONS,
        output_dir / "confusion_matrix.png",
        title=f"{model_dir.name} ({strategy})",
    )

    summary = (
        f"{'='*70}\nSENSEVOICE-SMALL TEST SUMMARY\n{'='*70}\n\n"
        f"Model:       {model_dir}\nStrategy:    {strategy}\n"
        f"Samples:     {len(test_dataset):,}\n\n"
        f"Accuracy:    {metrics['accuracy']:.4f}\n"
        f"Weighted F1: {metrics['weighted_f1']:.4f}\n\n"
        f"Per-class F1:\n"
        + "\n".join(f"  {e:<12}: {v:.4f}"
                    for e, v in metrics["per_class_f1"].items())
        + f"\n\nOutput: {output_dir}\n{'='*70}\n"
    )
    with open(output_dir / "SUMMARY.txt", "w") as f:
        f.write(summary)
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
