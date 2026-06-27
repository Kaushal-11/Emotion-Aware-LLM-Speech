"""
Emotional AI – WavLM Large: Test
==================================
Loads best_model.pt saved by wavlm_train.py
and evaluates on the held-out test set.

- strict=True so any weight mismatch surfaces immediately
- No accelerator at test time (single GPU/CPU)
- Saves: test_results.json, confusion_matrix.png, SUMMARY.txt
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

from transformers import WavLMModel
from peft import LoraConfig, get_peft_model

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

MODEL_DIR    = "/workspace/audio-em/finetune-results/wavlm-large/lora/20260513_225501"
FEATURES_DIR = "/workspace/audio-em/features"
BATCH_SIZE   = 16
OUTPUT_DIR   = None   # None → MODEL_DIR/test_results


# ─────────────────────────────────────────────
# CONSTANTS  (must match wavlm_train.py exactly)
# ─────────────────────────────────────────────

EMOTIONS    = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]
NUM_CLASSES = len(EMOTIONS)

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGET  = ["q_proj", "v_proj", "k_proj", "out_proj"]


# ─────────────────────────────────────────────
# DATASET
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
# MODEL  (must exactly match wavlm_train.py)
# ─────────────────────────────────────────────

class WavLMEmotionClassifier(nn.Module):
    def __init__(self, strategy: str = "lora"):
        super().__init__()
        self.strategy = strategy
        self.wavlm    = WavLMModel.from_pretrained("microsoft/wavlm-large")
        hidden        = self.wavlm.config.hidden_size   # 1024

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
                r=LORA_R, lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET,
                bias="none",
            )
            self.wavlm = get_peft_model(self.wavlm, cfg)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self.wavlm(input_values=input_values).last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_dir: Path, device: torch.device):
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        config = json.load(f)
    strategy  = config.get("strategy", "lora")
    print(f"  Strategy from config: {strategy}")

    ckpt_path = model_dir / "weights" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Ensure training completed and best_model.pt was saved."
        )
    print(f"  Loading: {ckpt_path}")

    model      = WavLMEmotionClassifier(strategy=strategy)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)   # strict=True — no silent failures
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
    acc     = accuracy_score(labels, preds)
    wf1     = f1_score(labels, preds, average="weighted", zero_division=0)
    pcf     = f1_score(labels, preds, average=None,
                       labels=list(range(NUM_CLASSES)), zero_division=0)
    cm      = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    report  = classification_report(labels, preds, target_names=EMOTIONS,
                                    output_dict=True, zero_division=0)
    return {
        "accuracy":              float(acc),
        "weighted_f1":           float(wf1),
        "per_class_f1":          {EMOTIONS[i]: float(pcf[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix":      cm.tolist(),
        "classification_report": report,
    }


# ─────────────────────────────────────────────
# CONFUSION MATRIX
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
        logits    = model(waveforms)
        probs     = torch.softmax(logits, dim=-1)
        preds     = logits.argmax(dim=-1)

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
    print("  WavLM-Large Emotion — Test Evaluation")
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
    print(f"Test samples: {len(test_dataset):,}")

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

    cm = np.array(metrics["confusion_matrix"])
    save_confusion_matrix(cm, EMOTIONS,
                          output_dir / "confusion_matrix.png",
                          title=f"{model_dir.name} ({strategy})")

    summary = (
        f"{'='*70}\n"
        f"WAVLM-LARGE TEST SUMMARY\n"
        f"{'='*70}\n\n"
        f"Model:       {model_dir}\n"
        f"Strategy:    {strategy}\n"
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
    print(f"✓ All results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()