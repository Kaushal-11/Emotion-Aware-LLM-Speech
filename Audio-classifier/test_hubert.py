"""
Emotional AI – Script 4: Test HuBERT-Large
===========================================
Loads the best_model.pt saved by script3_train.py
and evaluates on the held-out test set.

Key fix vs previous version:
  - Loads plain state_dict (best_model.pt) instead of
    accelerate checkpoint bundle → eliminates mode collapse
  - strict=True so any weight mismatch raises an error immediately
  - No accelerator involved at test time (single GPU/CPU)
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report, accuracy_score)
from tqdm import tqdm

from transformers import HubertModel
from peft import LoraConfig, get_peft_model

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False


# ─────────────────────────────────────────────
# CONFIGURATION — set MODEL_DIR to your run folder
# ─────────────────────────────────────────────

MODEL_DIR    = "/workspace/audio-em/finetune-results/hubert-large/lora/20260508_153317"
FEATURES_DIR = "/workspace/audio-em/features"
BATCH_SIZE   = 16
OUTPUT_DIR   = None   # None → MODEL_DIR/test_results


# ─────────────────────────────────────────────
# CONSTANTS  (must match training)
# ─────────────────────────────────────────────

EMOTIONS    = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]
NUM_CLASSES = len(EMOTIONS)

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGET  = ["q_proj", "v_proj", "k_proj", "out_proj"]


# ─────────────────────────────────────────────
# DATASET  (no augmentation at test time)
# ─────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, index_path: str):
        with open(index_path, encoding="utf-8") as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r    = self.records[idx]
        data = torch.load(r["pt_path"], map_location="cpu", weights_only=True)
        return {
            "waveform":   data["waveform"],
            "emotion_id": torch.tensor(data["emotion_id"], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# MODEL  (must exactly match script3_train.py)
# ─────────────────────────────────────────────

class HuBERTEmotionClassifier(nn.Module):
    def __init__(self, strategy: str = "lora"):
        super().__init__()
        self.strategy = strategy
        self.hubert   = HubertModel.from_pretrained(
            "facebook/hubert-large-ls960-ft", use_safetensors=True)
        hidden = self.hubert.config.hidden_size   # 1024

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
            for p in self.hubert.parameters():
                p.requires_grad = False
        elif strategy == "lora":
            cfg = LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET,
                bias="none",
            )
            self.hubert = get_peft_model(self.hubert, cfg)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self.hubert(input_values=input_values).last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_dir: Path, device: torch.device):
    """
    Load strategy from config.json, build model, load state_dict strictly.
    Raises FileNotFoundError / RuntimeError on any mismatch.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)
    strategy = config.get("strategy", "lora")
    print(f"  Strategy from config: {strategy}")

    ckpt_path = model_dir / "weights" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Make sure you trained with script3_train.py (which saves best_model.pt)."
        )

    print(f"  Loading weights from: {ckpt_path}")
    model = HuBERTEmotionClassifier(strategy=strategy)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # strict=True so any key mismatch raises an error — never silently skip weights
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total  |  {trainable:,} trainable")
    return model, strategy


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(preds, labels):
    acc      = accuracy_score(labels, preds)
    wf1      = f1_score(labels, preds, average="weighted", zero_division=0)
    per_cls  = f1_score(labels, preds, average=None,
                        labels=list(range(NUM_CLASSES)), zero_division=0)
    cm       = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    report   = classification_report(labels, preds, target_names=EMOTIONS,
                                     output_dict=True, zero_division=0)
    return {
        "accuracy":              float(acc),
        "weighted_f1":           float(wf1),
        "per_class_f1":          {EMOTIONS[i]: float(per_cls[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, class_names, output_path, title=""):
    if not PLT_AVAILABLE:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix → {output_path}")


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="  evaluating", leave=False):
        waveforms = batch["waveform"].to(device)
        labels    = batch["emotion_id"].to(device)

        logits = model(waveforms)
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
    model_dir    = Path(MODEL_DIR).resolve()
    features_dir = Path(FEATURES_DIR).resolve()
    output_dir   = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else model_dir / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print("  HuBERT Emotion — Test Evaluation")
    print(f"{'='*70}")
    print(f"  Model dir : {model_dir}")
    print(f"  Features  : {features_dir}")
    print(f"  Device    : {device}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*70}\n")

    # ── Dataset ───────────────────────────────
    test_dataset = EmotionDataset(str(features_dir / "features_test.json"))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    # ── Load model ────────────────────────────
    print("\nLoading model ...")
    model, strategy = load_model(model_dir, device)
    print("  ✓ Loaded successfully\n")

    # ── Evaluate ──────────────────────────────
    print("Running evaluation ...")
    metrics = evaluate(model, test_loader, device)

    # ── Print results ─────────────────────────
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

    # ── Save results ──────────────────────────
    results = {
        "model_dir":  str(model_dir),
        "strategy":   strategy,
        "num_samples": len(test_dataset),
        "test_metrics": {
            "accuracy":        metrics["accuracy"],
            "weighted_f1":     metrics["weighted_f1"],
            "per_class_f1":    metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "classification_report": metrics["classification_report"],
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results → {output_dir / 'test_results.json'}")

    # ── Confusion matrix ──────────────────────
    cm = np.array(metrics["confusion_matrix"])
    save_confusion_matrix(cm, EMOTIONS,
                          output_dir / "confusion_matrix.png",
                          title=f"{model_dir.name} ({strategy})")

    # ── Text summary ──────────────────────────
    summary = (
        f"{'='*70}\n"
        f"HUBERT TEST SUMMARY\n"
        f"{'='*70}\n\n"
        f"Model:       {model_dir}\n"
        f"Strategy:    {strategy}\n"
        f"Samples:     {len(test_dataset)}\n\n"
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
    print(f"✓ All results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()