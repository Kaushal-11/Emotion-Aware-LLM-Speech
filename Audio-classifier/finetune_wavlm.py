"""
Emotional AI – WavLM Large: Train
===================================
Model:    microsoft/wavlm-large
Strategy: frozen  → freeze encoder, train head only
          lora    → LoRA on attention projections + head


GPU optimizations (same as HuBERT v2):
  - BATCH_SIZE=2, GRAD_ACCUM=8 → effective batch=32 on 2 GPUs
  - Gradient checkpointing to cut VRAM ~40%
  - NUM_WORKERS=2 (safe for worker RAM)
  - persistent_workers to avoid re-fork overhead
  - No speed-perturbation in workers (causes OOM)
  - Plain state_dict checkpoint (no accelerate bundle)

Launch (single GPU):
    python wavlm_train.py

Launch (multi-GPU):
    accelerate launch --num_processes=2 --multi_gpu --mixed_precision=bf16 wavlm_train.py
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

from transformers import WavLMModel
from accelerate import Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, get_peft_model

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

MODEL_NAME       = "microsoft/wavlm-large"
MODEL_SHORT_NAME = "wavlm-large"
FEATURES_DIR     = "/workspace/audio-em/features"
STRATEGY         = "lora"
BASE_RESULTS_DIR = "/workspace/audio-em/finetune-results"

BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 8

MAX_EPOCHS       = 30
PATIENCE         = 7
WARMUP_RATIO     = 0.1
WEIGHT_DECAY     = 1e-4

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_LORA = 3e-5

# WavLM-Large attention modules
# query/value are standard; key and output proj also help
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGET  = ["q_proj", "v_proj", "k_proj", "out_proj"]

FOCAL_GAMMA     = 2.0
LABEL_SMOOTHING = 0.1

NUM_WORKERS     = 2
USE_GRAD_CKPT   = True

USE_TENSORBOARD = True
SAVE_CURVES     = True

EMOTIONS    = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]
NUM_CLASSES = len(EMOTIONS)


# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None,
                 gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce           = F.cross_entropy(logits, targets, weight=self.weight,
                                       label_smoothing=self.ls, reduction="none")
        focal_weight = (1.0 - torch.exp(-ce)) ** self.gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    Loads pre-extracted waveform tensors.
    Augmentation (train only): noise, time-shift, amplitude scaling.
    Speed perturbation intentionally excluded (worker OOM risk).
    """
    def __init__(self, index_path: str, augment: bool = False):
        with open(index_path, encoding="utf-8") as f:
            self.records = json.load(f)
        self.augment   = augment
        self.noise_std = 0.005

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r    = self.records[idx]
        data = torch.load(r["pt_path"], map_location="cpu", weights_only=True)
        w    = data["waveform"]
        if self.augment:
            w = self._augment(w)
        return {
            "waveform":   w,
            "emotion_id": torch.tensor(data["emotion_id"], dtype=torch.long),
        }

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


def build_weighted_sampler(dataset: EmotionDataset) -> WeightedRandomSampler:
    counts = defaultdict(int)
    for r in dataset.records:
        counts[r["emotion_id"]] += 1
    cw = {eid: 1.0 / cnt for eid, cnt in counts.items()}
    weights = [cw[r["emotion_id"]] for r in dataset.records]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def compute_class_weights(dataset: EmotionDataset) -> torch.Tensor:
    counts = torch.zeros(NUM_CLASSES)
    for r in dataset.records:
        counts[r["emotion_id"]] += 1
    w = 1.0 / (counts + 1e-6)
    return w / w.sum() * NUM_CLASSES


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class WavLMEmotionClassifier(nn.Module):
    def __init__(self, strategy: str, use_grad_ckpt: bool = False):
        super().__init__()
        self.strategy = strategy
        self.wavlm    = WavLMModel.from_pretrained(MODEL_NAME)
        hidden        = self.wavlm.config.hidden_size   # 1024 for large

        if use_grad_ckpt:
            self.wavlm.gradient_checkpointing_enable()
            print("  [WavLM] Gradient checkpointing ENABLED")

        # Same classifier head as HuBERT for fair comparison
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
            print("  [WavLM] Encoder FROZEN — training head only.")
        elif strategy == "lora":
            cfg = LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET,
                bias="none",
            )
            self.wavlm = get_peft_model(self.wavlm, cfg)
            self.wavlm.print_trainable_parameters()

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out    = self.wavlm(input_values=input_values)
        hidden = out.last_hidden_state        # (B, T, 1024)
        pooled = hidden.mean(dim=1)           # (B, 1024)
        return self.classifier(pooled)


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
    axes[0, 1].set_title(f"Val weighted-F1  (best {best_f:.4f} @ ep{best_e})")
    axes[0, 1].grid(True, alpha=0.3)

    gap = [v - t for v, t in zip(val_loss, train_loss)]
    axes[1, 0].bar(epochs, gap, alpha=0.6, color="orange", edgecolor="black")
    axes[1, 0].axhline(y=0, color="red", linewidth=1.5)
    axes[1, 0].set_title("Overfitting gap (val − train loss)")
    axes[1, 0].grid(True, alpha=0.3)

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
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, accelerator, scheduler=None):
    model.train()
    total_loss = 0.0
    local_preds:  list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    t0 = time.time()

    for batch in tqdm(loader, desc="  train", leave=False, dynamic_ncols=True):
        with accelerator.accumulate(model):
            logits = model(batch["waveform"])
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
        logits = model(batch["waveform"])
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

    features_dir = Path(FEATURES_DIR).resolve()
    lr = LEARNING_RATE_LORA if STRATEGY == "lora" else LEARNING_RATE_HEAD

    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS,
                              kwargs_handlers=[ddp_kwargs])
    is_main = accelerator.is_main_process
    pin     = torch.cuda.is_available()

    config = {
        "model_name": MODEL_NAME, "strategy": STRATEGY, "timestamp": timestamp,
        "run_id": f"{MODEL_SHORT_NAME}_{STRATEGY}_{timestamp}",
        "output_dir": str(output_dir),
        "hyperparameters": {
            "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS * accelerator.num_processes,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "learning_rate": lr, "weight_decay": WEIGHT_DECAY,
            "focal_gamma": FOCAL_GAMMA, "label_smoothing": LABEL_SMOOTHING,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "grad_ckpt": USE_GRAD_CKPT, "num_workers": NUM_WORKERS,
        },
        "emotions": EMOTIONS,
    }

    if is_main:
        eff = BATCH_SIZE * GRAD_ACCUM_STEPS * accelerator.num_processes
        print(f"\n{'='*70}")
        print(f"  WavLM-Large Emotion Training")
        print(f"{'='*70}")
        print(f"  Run:        {config['run_id']}")
        print(f"  Strategy:   {STRATEGY}  |  Devices: {accelerator.num_processes}")
        print(f"  Batch/GPU:  {BATCH_SIZE}  |  Accum: {GRAD_ACCUM_STEPS}"
              f"  |  Effective: {eff}")
        print(f"  LR:         {lr}  |  Focal γ: {FOCAL_GAMMA}"
              f"  |  Label smooth: {LABEL_SMOOTHING}")
        print(f"  LoRA r/α:   {LORA_R}/{LORA_ALPHA}  |  Grad ckpt: {USE_GRAD_CKPT}")
        print(f"{'='*70}\n")
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    tb_writer = (
        SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if is_main and USE_TENSORBOARD and TB_AVAILABLE else None
    )

    # ── Datasets & loaders ────────────────────
    if is_main:
        print("Loading datasets ...")
    train_dataset = EmotionDataset(str(features_dir / "features_train.json"), augment=True)
    val_dataset   = EmotionDataset(str(features_dir / "features_val.json"),   augment=False)

    train_sampler = build_weighted_sampler(train_dataset)
    class_weights = compute_class_weights(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    if is_main:
        print(f"  Train: {len(train_dataset):,}  |  Val: {len(val_dataset):,}\n")

    # ── Model / loss / optimiser / scheduler ──
    model     = WavLMEmotionClassifier(STRATEGY, use_grad_ckpt=USE_GRAD_CKPT)
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

    # ── Training loop ─────────────────────────
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
            print(f"  ETA (remaining): ~{eta_min:.0f} min")

            if tb_writer:
                tb_writer.add_scalar("Loss/train",     train_loss,             epoch)
                tb_writer.add_scalar("Loss/val",       val_loss,               epoch)
                tb_writer.add_scalar("F1/train",       train_m["weighted_f1"], epoch)
                tb_writer.add_scalar("F1/val",         val_m["weighted_f1"],   epoch)
                tb_writer.add_scalar("Time/epoch_min", train_sec / 60,         epoch)
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