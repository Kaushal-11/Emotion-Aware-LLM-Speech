"""
Emotional AI – SenseVoice: Data Preparation
============================================
SenseVoice does NOT use pre-extracted .pt waveform tensors.
It takes raw audio file paths and computes 80-dim log-Mel fbank
features internally via FunASR at training time.

This script replaces script2_extract.py for SenseVoice.
It reads your existing split JSON files (from script1_split.py)
and converts them to the JSONL format required by FunASR/SenseVoice.

Output structure:
    sensevoice_data/
        train.jsonl
        val.jsonl
        test.jsonl
        train_wav.scp      (key → audio_path)
        train_emo.txt      (key → <|EMOTION|>)
        val_wav.scp
        val_emo.txt
        test_wav.scp
        test_emo.txt

JSONL format (one JSON per line):
    {
      "key": "unique_id",
      "source": "/absolute/path/to/audio.wav",
      "source_len": 160,          ← fbank frames (approx), filled with 0 if unknown
      "text_language": "<|en|>",
      "emo_target": "<|HAPPY|>",
      "event_target": "<|Speech|>",
      "with_or_wo_itn": "<|woitn|>",
      "target": "",               ← empty: we only care about emotion, not ASR
      "target_len": 0
    }

Emotion mapping (your labels → SenseVoice tokens):
    happiness → <|HAPPY|>
    anger     → <|ANGRY|>
    sadness   → <|SAD|>
    disgust   → <|DISGUSTED|>
    fear      → <|FEARFUL|>
    surprise  → <|SURPRISED|>

Requirements:
    pip install funasr torch torchaudio
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ROOT_DIR    = "/workspace/audio-em/dataset"
SPLITS_DIR  = None   # None → ROOT_DIR/splits
OUTPUT_DIR  = None   # None → ROOT_DIR/../sensevoice_data

# Language tag — all your data is English
LANGUAGE_TAG = "<|en|>"

# Event tag — all clips are speech
EVENT_TAG = "<|Speech|>"

# Average fbank frames per second at 10ms shift = 100 frames/sec
# Used to estimate source_len. Actual value computed from duration if available.
FBANK_FRAMES_PER_SEC = 100

# ─────────────────────────────────────────────
# EMOTION MAPPING
# SenseVoice has 7 emotion tokens. We map our 6 emotions.
# Note: SenseVoice also has <|NEUTRAL|> but we don't have that class.
# ─────────────────────────────────────────────

EMOTION_TO_SENSEVOICE = {
    "happiness": "<|HAPPY|>",
    "anger":     "<|ANGRY|>",
    "sadness":   "<|SAD|>",
    "disgust":   "<|DISGUSTED|>",
    "fear":      "<|FEARFUL|>",
    "surprise":  "<|SURPRISED|>",
}

# Reverse map for verification
SENSEVOICE_TO_EMOTION = {v: k for k, v in EMOTION_TO_SENSEVOICE.items()}

EMOTIONS = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def resolve_audio_path(record: dict, root: Path) -> str:
    """Return absolute audio path from a split record."""
    ap = record.get("audio_path", "")
    p  = Path(ap)
    if not p.is_absolute():
        p = root / p
    return str(p)


def estimate_source_len(duration_sec: float) -> int:
    """Estimate fbank frame count from duration."""
    if duration_sec and duration_sec > 0:
        return max(1, int(duration_sec * FBANK_FRAMES_PER_SEC))
    return 160   # fallback: ~1.6s


def build_jsonl_record(record: dict, root: Path, idx: int) -> dict | None:
    """Convert a split record to a SenseVoice JSONL dict."""
    emotion = record.get("emotion", "")
    if emotion not in EMOTION_TO_SENSEVOICE:
        return None   # skip unknown emotions

    audio_path = resolve_audio_path(record, root)
    if not Path(audio_path).exists():
        return None   # skip missing files

    # Unique key: dataset_emotion_idx
    key = f"{record.get('dataset', 'unk')}_{emotion}_{idx:06d}"
    if record.get("speaker_id"):
        key = f"{record['dataset']}_{record['speaker_id']}_{emotion}_{idx:06d}"

    return {
        "key":           key,
        "source":        audio_path,
        "source_len":    estimate_source_len(record.get("duration_sec", 0.0)),
        "text_language": LANGUAGE_TAG,
        "emo_target":    EMOTION_TO_SENSEVOICE[emotion],
        "event_target":  EVENT_TAG,
        "with_or_wo_itn": "<|woitn|>",
        "target":        "",    # no ASR transcription needed
        "target_len":    0,
        # Extra metadata (not used by FunASR but useful for debugging)
        "_emotion":      emotion,
        "_dataset":      record.get("dataset", ""),
        "_speaker_id":   record.get("speaker_id", ""),
    }


# ─────────────────────────────────────────────
# PROCESS ONE SPLIT
# ─────────────────────────────────────────────

def process_split(split_name: str, records: list, root: Path,
                  output_dir: Path) -> dict:
    """
    Convert records to JSONL + scp + emo files for one split.
    Returns summary dict.
    """
    jsonl_records = []
    skipped       = 0
    emotion_counts = defaultdict(int)
    dataset_counts = defaultdict(int)

    for idx, record in enumerate(records):
        jr = build_jsonl_record(record, root, idx)
        if jr is None:
            skipped += 1
            continue
        jsonl_records.append(jr)
        emotion_counts[record["emotion"]] += 1
        dataset_counts[record.get("dataset", "unk")] += 1

    # ── Write JSONL ──────────────────────────
    jsonl_path = output_dir / f"{split_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for jr in jsonl_records:
            # Remove internal metadata keys before writing
            out = {k: v for k, v in jr.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"  [{split_name.upper()}] JSONL → {jsonl_path}  ({len(jsonl_records)} records)")

    # ── Write wav.scp ────────────────────────
    scp_path = output_dir / f"{split_name}_wav.scp"
    with open(scp_path, "w", encoding="utf-8") as f:
        for jr in jsonl_records:
            f.write(f"{jr['key']} {jr['source']}\n")
    print(f"  [{split_name.upper()}] wav.scp → {scp_path}")

    # ── Write emo.txt ────────────────────────
    emo_path = output_dir / f"{split_name}_emo.txt"
    with open(emo_path, "w", encoding="utf-8") as f:
        for jr in jsonl_records:
            f.write(f"{jr['key']} {jr['emo_target']}\n")
    print(f"  [{split_name.upper()}] emo.txt → {emo_path}")

    # ── Write text.txt (empty transcriptions) ─
    text_path = output_dir / f"{split_name}_text.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for jr in jsonl_records:
            f.write(f"{jr['key']} \n")   # empty transcription
    print(f"  [{split_name.upper()}] text.txt → {text_path}")

    return {
        "total":          len(jsonl_records),
        "skipped":        skipped,
        "emotion_counts": dict(emotion_counts),
        "dataset_counts": dict(dataset_counts),
    }


# ─────────────────────────────────────────────
# VERIFY AUDIO FILES
# ─────────────────────────────────────────────

def verify_audio_sample(records: list, root: Path, n: int = 5):
    """Quick sanity check: try loading a few audio files."""
    try:
        import torchaudio
    except ImportError:
        print("  torchaudio not available — skipping audio verification")
        return

    print(f"\n  Verifying {n} audio files ...")
    ok = 0
    for r in records[:n * 3]:   # scan a few more in case of early missing files
        if ok >= n:
            break
        p = resolve_audio_path(r, root)
        if not Path(p).exists():
            print(f"    MISSING: {p}")
            continue
        try:
            wav, sr = torchaudio.load(p)
            print(f"    ✓  {Path(p).name}  shape:{wav.shape}  sr:{sr}")
            ok += 1
        except Exception as e:
            print(f"    ✗  {Path(p).name}  {e}")


# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(summaries: dict):
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  SENSEVOICE DATA PREPARATION SUMMARY")
    print(SEP)

    for split_name, s in summaries.items():
        print(f"\n  [{split_name.upper()}]  {s['total']} records"
              f"  (skipped {s['skipped']})")
        print(f"  {'Emotion':<12}  {'Count':>6}  {'SenseVoice token'}")
        print(f"  {'─'*45}")
        for emo in EMOTIONS:
            count = s["emotion_counts"].get(emo, 0)
            token = EMOTION_TO_SENSEVOICE.get(emo, "—")
            bar   = "█" * (count // 30)
            print(f"  {emo:<12}  {count:>6}  {token:<14}  {bar}")
        print(f"  Datasets: {dict(sorted(s['dataset_counts'].items()))}")

        counts = list(s["emotion_counts"].values())
        counts = [c for c in counts if c > 0]
        if counts:
            ratio = max(counts) / min(counts)
            print(f"  Imbalance ratio: {ratio:.2f}×")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────
# GENERATE FINETUNE CONFIG
# ─────────────────────────────────────────────

def write_finetune_config(output_dir: Path, summaries: dict):
    """
    Write a reference finetune.sh snippet showing how to launch
    FunASR training with the generated data.
    """
    config = f"""#!/bin/bash
# ─────────────────────────────────────────────
# SenseVoice Finetuning Command
# Run from the SenseVoice repo root directory
# ─────────────────────────────────────────────

# Install FunASR first:
#   git clone https://github.com/alibaba/FunASR.git && cd FunASR
#   pip install -e ./

TRAIN_JSONL="{output_dir}/train.jsonl"
VAL_JSONL="{output_dir}/val.jsonl"
MODEL_DIR="FunAudioLLM/SenseVoiceSmall"   # or local path after download
OUTPUT_DIR="/workspace/audio-em/finetune-results/sensevoice"

# Training stats:
# Train samples: {summaries.get('train', {}).get('total', '?')}
# Val samples:   {summaries.get('val',   {}).get('total', '?')}

python -m funasr.bin.train_ds \\
    --config-path conf \\
    --config-name sensevoice.yaml \\
    ++model="{{}}" \\
    ++model_conf.model_dir="$MODEL_DIR" \\
    ++dataset_conf.data_path="['$TRAIN_JSONL']" \\
    ++dataset_conf.data_path_val="['$VAL_JSONL']" \\
    ++train_conf.max_epoch=30 \\
    ++train_conf.save_checkpoint_steps=1000 \\
    ++train_conf.keep_nbest_models=5 \\
    ++train_conf.avg_nbest_model=5 \\
    ++optim_conf.lr=1e-4 \\
    ++output_dir="$OUTPUT_DIR" \\
    ++device="cuda" \\
    ++batch_size=8 \\
    ++accum_grad=4 \\
    ++num_workers=2 \\
    ++dataset_conf.batch_type="example" \\
    ++log_interval=100

# NOTE: SenseVoice training goes through FunASR's train_ds.py.
# The emotion labels in your JSONL (emo_target) are used automatically.
# No custom training loop is needed.
"""
    cfg_path = output_dir / "finetune_command.sh"
    with open(cfg_path, "w") as f:
        f.write(config)
    os.chmod(cfg_path, 0o755)
    print(f"  Finetune command reference → {cfg_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print("  EMOTIONAL AI – SENSEVOICE DATA PREPARATION")
    print(f"{'='*65}")

    root       = Path(ROOT_DIR).resolve()
    splits_dir = Path(SPLITS_DIR).resolve() if SPLITS_DIR else root / "splits"
    output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else root.parent / "sensevoice_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root:       {root}")
    print(f"Splits:     {splits_dir}")
    print(f"Output:     {output_dir}")
    print(f"Lang tag:   {LANGUAGE_TAG}")
    print(f"Event tag:  {EVENT_TAG}")
    print(f"\nEmotion mapping:")
    for emo, token in EMOTION_TO_SENSEVOICE.items():
        print(f"  {emo:<12} → {token}")
    print(f"{'='*65}\n")

    # Quick audio verification on train split
    train_split_path = splits_dir / "train.json"
    if train_split_path.exists():
        with open(train_split_path, encoding="utf-8") as f:
            sample_records = json.load(f)
        verify_audio_sample(sample_records, root, n=3)

    summaries = {}

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.exists():
            print(f"\n[SKIP] {split_path} not found")
            continue

        with open(split_path, encoding="utf-8") as f:
            records = json.load(f)

        print(f"\n[{split_name.upper()}] {len(records)} records")
        summary = process_split(split_name, records, root, output_dir)
        summaries[split_name] = summary

    # Save emotion map reference
    emo_map_path = output_dir / "emotion_mapping.json"
    with open(emo_map_path, "w") as f:
        json.dump({
            "emotion_to_sensevoice": EMOTION_TO_SENSEVOICE,
            "sensevoice_to_emotion": SENSEVOICE_TO_EMOTION,
            "language_tag":          LANGUAGE_TAG,
            "event_tag":             EVENT_TAG,
        }, f, indent=2)
    print(f"\n  Emotion map → {emo_map_path}")

    write_finetune_config(output_dir, summaries)
    print_summary(summaries)

    print(f"{'='*65}")
    print("  DATA PREPARATION COMPLETE!")
    print(f"{'='*65}")
    print(f"\nNext step: run sensevoice_train.py")
    print(f"           (or use finetune_command.sh for FunASR native training)\n")


if __name__ == "__main__":
    main()