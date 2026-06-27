"""
Emotional AI – Fixed Script: Fine-tune wav2vec2-Large
=====================================================
Fixes applied vs original:
  1. Class weights explicitly moved to correct device BEFORE criterion
  2. Removed accelerator.gather() — causes issues with DataParallel mode
  3. Weighted sampler reads emotion_id correctly from index records
  4. NaN/Inf guard on loss with skip logic
  5. Waveform clamped to [-1, 1] in dataset
  6. Label dtype enforced as torch.long everywhere
  7. Gradient clipping applied correctly
  8. Model saved as state_dict not accelerate checkpoint
  9. unwrap_model() used during eval to avoid DDP eval issues
  10. drop_last=True on train loader to avoid incomplete batch issues

Run with:
    # Single GPU
    python finetune_wav2vec2.py

    # Multi GPU (recommended)
    accelerate launch --multi_gpu --num_processes=2 finetune_wav2vec2.py
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from transformers import Wav2Vec2Model
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


# ─────────────────────────────────────────────
# CONFIGURATION - CHANGE THESE AS NEEDED
# ─────────────────────────────────────────────

# Paths
FEATURES_DIR = "/workspace/audio-em/features"           # Directory with features_train.json, features_val.json, features_test.json
OUTPUT_DIR = None                                       # Set to None for default: "checkpoints/wav2vec2/<strategy>"

# Training parameters
STRATEGY = "lora"                                       # Options: "frozen" or "lora"
MAX_EPOCHS = 30                                         # Reduced from 50
BATCH_SIZE = 4
PATIENCE = 7                                            # Increased patience

# Learning rates
LEARNING_RATE_HEAD = 5e-5                               # Reduced from 1e-3 (more stable)
LEARNING_RATE_LORA = 1e-5                               # Reduced from 5e-5
WEIGHT_DECAY = 1e-2                                     # Increased for stability

# LoRA parameters (only used if STRATEGY = "lora")
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "out_proj"]

# Other
GRAD_ACCUM_STEPS = 4                                    # Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
WARMUP_RATIO = 0.2                                      # Increased warmup
MAX_GRAD_NORM = 1.0
USE_WANDB = False                                       # Set to True to use Weights & Biases
RUN_NAME = None                                         # Set custom run name, or None for default

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MODEL_NAME  = "facebook/wav2vec2-large-960h"
EMOTIONS    = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]
NUM_CLASSES = len(EMOTIONS)


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
        record  = self.records[idx]
        pt_data = torch.load(record["pt_path"], map_location="cpu",
                             weights_only=True)

        waveform   = pt_data["waveform"].float()
        emotion_id = int(pt_data["emotion_id"])

        # Clamp to [-1, 1] — prevents extreme values causing NaN
        waveform = torch.clamp(waveform, -1.0, 1.0)

        # Replace any residual NaN / Inf
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            waveform = torch.zeros_like(waveform)

        return {
            "waveform":   waveform,
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# CLASS WEIGHTS & SAMPLER
# ─────────────────────────────────────────────

def compute_class_weights(dataset: EmotionDataset) -> torch.Tensor:
    """Returns CPU tensor — caller must move to device."""
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for r in dataset.records:
        counts[int(r["emotion_id"])] += 1

    print(f"  Class counts : {counts.tolist()}")

    for i, c in enumerate(counts):
        if c == 0:
            print(f"  [WARN] {EMOTIONS[i]} has 0 samples — check splits!")

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    print(f"  Class weights: {[round(w, 4) for w in weights.tolist()]}")
    return weights  # CPU — moved to device in main()


def build_weighted_sampler(dataset: EmotionDataset) -> WeightedRandomSampler:
    counts = defaultdict(int)
    for r in dataset.records:
        counts[int(r["emotion_id"])] += 1

    class_w  = {eid: 1.0 / cnt for eid, cnt in counts.items()}
    sample_w = [class_w[int(r["emotion_id"])] for r in dataset.records]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.float32),
        num_samples=len(sample_w),
        replacement=True,
    )


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, strategy: str):
        super().__init__()
        self.strategy = strategy

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            MODEL_NAME,
            ignore_mismatched_sizes=True,  # suppresses lm_head warnings
            use_safetensors=True,            
        )

        hidden_size = self.wav2vec2.config.hidden_size  # 1024 for large

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
        )

        self._init_classifier()
        self.attention_pool = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.attention_pool.weight)
        nn.init.zeros_(self.attention_pool.bias)


        if strategy == "frozen":
            self._freeze_encoder()
        elif strategy == "lora":
            self._apply_lora()

    def _init_classifier(self):
        """Xavier init — prevents vanishing/exploding at start."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _freeze_encoder(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # Unfreeze feature_projection only — bridges CNN → transformer
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [wav2vec2] Encoder FROZEN")
        print(f"  Trainable params: {n_trainable:,}")

    def _apply_lora(self):
        lora_cfg = LoraConfig(
            r=LORA_R, 
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
            init_lora_weights=True,
        )
        self.wav2vec2 = get_peft_model(self.wav2vec2, lora_cfg)
        self.wav2vec2.print_trainable_parameters()
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs       = self.wav2vec2(input_values=input_values)
        hidden_states = outputs.last_hidden_state   # (B, T', 1024)
    
        # Attentive pooling — learns WHICH frames matter for emotion
        # instead of averaging all frames equally
        attn_weights = self.attention_pool(hidden_states)      # (B, T', 1)
        attn_weights = torch.softmax(attn_weights, dim=1)      # (B, T', 1)
        pooled       = (hidden_states * attn_weights).sum(dim=1)  # (B, 1024)
    
        return self.classifier(pooled)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(preds, labels):
    if len(preds) == 0 or len(labels) == 0:
        return {"accuracy": 0.0, "weighted_f1": 0.0,
                "per_class_f1": {e: 0.0 for e in EMOTIONS},
                "confusion_matrix": [[0]*NUM_CLASSES]*NUM_CLASSES}
        
    preds  = np.array(preds)
    labels = np.array(labels)
    acc    = (preds == labels).mean()
    w_f1   = f1_score(labels, preds, average="weighted", zero_division=0)
    per_f1 = f1_score(labels, preds, average=None,
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    cm     = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    return {
        "accuracy":         float(acc),
        "weighted_f1":      float(w_f1),
        "per_class_f1":     {EMOTIONS[i]: float(per_f1[i]) for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    accelerator, scheduler, epoch):
    model.train()
    total_loss  = 0.0
    valid_steps = 0
    nan_batches = 0
    all_preds   = []
    all_labels  = []

    pbar = tqdm(loader, desc=f"  Epoch {epoch} train", leave=False)
    for step, batch in enumerate(pbar):
        waveforms = batch["waveform"]
        labels    = batch["emotion_id"]

        with accelerator.accumulate(model):
            logits = model(waveforms)
            loss   = criterion(logits, labels)

            # ── NaN guard — skip bad batch ──
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                if nan_batches <= 3:
                    print(f"\n  [WARN] NaN/Inf loss step {step}. "
                          f"logits range: [{logits.min():.2f}, {logits.max():.2f}]")
                continue

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss  += loss.item()
        valid_steps += 1

        preds = logits.detach().argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if nan_batches > 0:
        print(f"  [WARN] {nan_batches} batches skipped due to NaN/Inf loss.")

    avg_loss = total_loss / max(valid_steps, 1)
    return avg_loss, compute_metrics(all_preds, all_labels)


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, split_name="val"):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in tqdm(loader, desc=f"  {split_name:>4} eval", leave=False):
        waveforms = batch["waveform"]
        labels    = batch["emotion_id"]
        logits    = model(waveforms)
        loss      = criterion(logits, labels)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / max(len(loader), 1), compute_metrics(all_preds, all_labels)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    features_dir = Path(FEATURES_DIR).resolve()
    
    # Set output directory
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR).resolve()
    else:
        output_dir = Path("checkpoints") / "wav2vec2" / STRATEGY
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set learning rate based on strategy
    lr = LEARNING_RATE_LORA if STRATEGY == "lora" else LEARNING_RATE_HEAD
    
    # Set run name
    run_name = RUN_NAME if RUN_NAME else f"wav2vec2_{STRATEGY}"

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        mixed_precision="fp16",
    )
    device      = accelerator.device
    is_main     = accelerator.is_main_process

    if is_main:
        print(f"\n{'='*60}")
        print(f"  wav2vec2-Large Fine-tuning")
        print(f"  Strategy  : {STRATEGY}")
        print(f"  Device    : {device}")
        print(f"  Processes : {accelerator.num_processes}")
        print(f"  LR        : {lr}")
        print(f"  Features  : {features_dir}")
        print(f"  Output    : {output_dir}")
        print(f"{'='*60}\n")

    tb_writer = None
    if is_main and TB_AVAILABLE:
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    if is_main and USE_WANDB and WANDB_AVAILABLE:
        wandb.init(project="emotional-ai", name=run_name, config={
            "model": MODEL_NAME, "strategy": STRATEGY,
            "lr": lr, "batch_size": BATCH_SIZE,
        })

    # ── Datasets ──
    if is_main:
        print("[Loading datasets ...]")

    train_ds = EmotionDataset(str(features_dir / "features_train.json"))
    val_ds   = EmotionDataset(str(features_dir / "features_val.json"))
    test_ds  = EmotionDataset(str(features_dir / "features_test.json"))

    if is_main:
        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── CRITICAL: compute weights → move to device → create criterion ──
    class_weights = compute_class_weights(train_ds).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    train_sampler = build_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──
    if is_main:
        print("\n[Building model ...]")

    model = Wav2Vec2EmotionClassifier(strategy=STRATEGY)
    # Sanity check — verify model forward pass is clean before training
    if is_main:
        print("\n[Sanity checking model forward pass ...]")
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 16000, dtype=torch.float32, device=next(model.parameters()).device)
            try:
                out = model(dummy)
                if torch.isnan(out).any():
                    print("  [FATAL] Model produces NaN on dummy input — check LoRA init!")
                    return
                else:
                    print(f"  [OK] Forward pass clean. Output shape: {out.shape}")
            except Exception as e:
                print(f"  [FATAL] Forward pass failed: {e}")
                return
        model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr,
                                  weight_decay=WEIGHT_DECAY, eps=1e-8)

    total_steps  = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    if is_main:
        print(f"  Total steps: {total_steps}  Warmup: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Accelerate prepare ──
    model, optimizer, train_loader, val_loader, test_loader, scheduler = \
        accelerator.prepare(model, optimizer, train_loader,
                            val_loader, test_loader, scheduler)

    # ── Training ──
    best_val_f1      = 0.0
    patience_counter = 0
    history          = []

    if is_main:
        print(f"\n[Starting training — {STRATEGY} strategy]\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        if is_main:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"\n{'─'*55}")
            print(f"  Epoch {epoch}/{MAX_EPOCHS}   LR: {cur_lr:.2e}")
            print(f"{'─'*55}")

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, criterion,
            accelerator, scheduler, epoch
        )

        # Eval on unwrapped model — avoids DDP eval issues
        unwrapped = accelerator.unwrap_model(model)
        val_loss, val_m = evaluate(unwrapped, val_loader, criterion, "val")
        val_f1 = val_m["weighted_f1"]

        if is_main:
            print(f"  Train → loss:{train_loss:.4f}  acc:{train_m['accuracy']:.4f}"
                  f"  wF1:{train_m['weighted_f1']:.4f}")
            print(f"  Val   → loss:{val_loss:.4f}  acc:{val_m['accuracy']:.4f}"
                  f"  wF1:{val_f1:.4f}")
            print(f"  Per-class F1:")
            for emo, f1 in val_m["per_class_f1"].items():
                bar = "█" * int(f1 * 20)
                print(f"    {emo:<12} {f1:.4f}  {bar}")

            if tb_writer:
                tb_writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
                tb_writer.add_scalars("F1",   {"train": train_m["weighted_f1"],
                                               "val": val_f1}, epoch)
                tb_writer.add_scalar("LR", cur_lr, epoch)

            if USE_WANDB and WANDB_AVAILABLE:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                           "val_f1": val_f1, "lr": cur_lr, "epoch": epoch,
                           **{f"f1_{k}": v for k, v in val_m["per_class_f1"].items()}})

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            if is_main:
                print(f"\n  ✓ New best val wF1: {best_val_f1:.4f} — saving.")
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    output_dir / "best_model.pt"
                )
        else:
            patience_counter += 1
            if is_main:
                print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                if is_main:
                    print(f"\n[Early stopping] Patience exhausted.")
                break

    # ── Final test evaluation ──
    if is_main:
        print(f"\n{'='*60}")
        print("  FINAL TEST EVALUATION")
        print(f"{'='*60}")

        best_state = torch.load(output_dir / "best_model.pt",
                                map_location=device, weights_only=True)
        accelerator.unwrap_model(model).load_state_dict(best_state)
        unwrapped = accelerator.unwrap_model(model)

        test_loss, test_m = evaluate(unwrapped, test_loader, criterion, "test")

        print(f"\n  Test Loss       : {test_loss:.4f}")
        print(f"  Test Accuracy   : {test_m['accuracy']:.4f}")
        print(f"  Test Weighted F1: {test_m['weighted_f1']:.4f}")
        print(f"\n  Per-class F1:")
        for emo, f1 in test_m["per_class_f1"].items():
            bar = "█" * int(f1 * 30)
            print(f"    {emo:<12} {f1:.4f}  {bar}")

        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        header = "           " + "".join(f"{e[:5]:>7}" for e in EMOTIONS)
        print(f"  {header}")
        for i, row in enumerate(test_m["confusion_matrix"]):
            print(f"  {EMOTIONS[i][:10]:<11}" + "".join(f"{v:>7}" for v in row))

        results = {
            "model": MODEL_NAME, "strategy": STRATEGY,
            "best_val_f1": best_val_f1, "test_metrics": test_m,
            "training_history": history,
            "config": {"lr": lr, "batch_size": BATCH_SIZE,
                       "grad_accum": GRAD_ACCUM_STEPS},
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results → {output_dir / 'results.json'}")

        if tb_writer:
            tb_writer.close()
        if USE_WANDB and WANDB_AVAILABLE:
            wandb.finish()

    print(f"\n[Done] wav2vec2 fine-tuning ({STRATEGY}) complete.")


if __name__ == "__main__":
    main()
