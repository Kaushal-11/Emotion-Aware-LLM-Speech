"""
Emotional AI – Script 1: Dataset Splitting
==========================================
Reads metadata.json from each dataset folder.
Applies per-emotion capping to reduce class imbalance.
Splits per dataset using speaker-stratified (or conversation-stratified for MELD)
so no speaker/conversation leaks across train/val/test.

Output:
    splits/train.json
    splits/val.json
    splits/test.json
    splits/split_summary.json
"""

import json
import random
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ROOT_DIR   = "/workspace/audio-em/dataset"
OUTPUT_DIR = None   # None → ROOT_DIR/splits

# Cap per (emotion, dataset) to reduce imbalance.
# Set to None to disable capping entirely.
# Rule of thumb: set to ~2× your smallest per-emotion count.
# Smallest is surprise ~1995, so cap at ~2500 globally.
MAX_PER_EMOTION_TOTAL = 2500   # across all datasets combined, per emotion

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

EMOTIONS      = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}
DATASETS      = ["cremad", "iemocap", "ravdess", "savee", "tess", "meld"]


# ─────────────────────────────────────────────
# LOAD METADATA
# ─────────────────────────────────────────────

def load_dataset_records(dataset_dir: Path, dataset_name: str) -> list:
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        print(f"  [SKIP] No metadata.json in {dataset_dir}")
        return []

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    records = []
    for emotion, emotion_data in meta.get("emotions", {}).items():
        if emotion not in EMOTIONS:
            continue
        for file_info in emotion_data.get("files", []):
            records.append({
                "audio_path":      file_info.get("audio_path"),
                "filename":        file_info.get("filename"),
                "emotion":         emotion,
                "emotion_id":      EMOTION_TO_ID[emotion],
                "dataset":         dataset_name,
                "speaker_id":      file_info.get("speaker_id"),
                "transcription":   file_info.get("transcription", None),
                "duration_sec":    file_info.get("duration_sec", 0.0),
                "conversation_id": file_info.get("conversation_id", None),
                "utterance_id":    file_info.get("utterance_id",
                                       file_info.get("filename", "")),
            })

    print(f"  [{dataset_name.upper()}] Loaded {len(records)} records")
    return records


# ─────────────────────────────────────────────
# BALANCE: cap total per-emotion across all datasets
# We cap AFTER loading all datasets so we can do
# proportional sampling from each dataset source.
# ─────────────────────────────────────────────

def apply_global_emotion_cap(all_records: list, max_per_emotion: int, seed: int) -> list:
    """
    For each emotion, if total count > max_per_emotion, randomly sample
    max_per_emotion records from it, preserving dataset proportionality.
    Speaker integrity is maintained because we sample whole records (not speakers),
    and the downstream split step will then group by speaker.
    """
    rng = random.Random(seed)

    by_emotion = defaultdict(list)
    for r in all_records:
        by_emotion[r["emotion"]].append(r)

    result = []
    print(f"\n  Emotion cap = {max_per_emotion} per class")
    print(f"  {'Emotion':<12}  {'Before':>7}  {'After':>7}  {'Kept%':>6}")
    print(f"  {'─'*40}")

    for emotion in EMOTIONS:
        items = by_emotion[emotion]
        before = len(items)
        if max_per_emotion and before > max_per_emotion:
            # Sample proportionally from each dataset
            by_ds = defaultdict(list)
            for r in items:
                by_ds[r["dataset"]].append(r)

            kept = []
            remaining = max_per_emotion
            ds_names = sorted(by_ds.keys())

            # First pass: proportional allocation
            alloc = {}
            for ds in ds_names:
                proportion = len(by_ds[ds]) / before
                alloc[ds] = max(1, round(proportion * max_per_emotion))

            # Adjust to hit exactly max_per_emotion
            total_alloc = sum(alloc.values())
            if total_alloc != max_per_emotion:
                # Trim/extend from largest dataset
                largest = max(alloc, key=alloc.get)
                alloc[largest] += (max_per_emotion - total_alloc)
                alloc[largest] = max(1, alloc[largest])

            for ds in ds_names:
                ds_items = by_ds[ds]
                rng.shuffle(ds_items)
                n = min(alloc.get(ds, 0), len(ds_items))
                kept.extend(ds_items[:n])

            # Final trim if still over due to rounding
            rng.shuffle(kept)
            items = kept[:max_per_emotion]

        pct = 100 * len(items) / max(before, 1)
        print(f"  {emotion:<12}  {before:>7}  {len(items):>7}  {pct:>5.1f}%")
        result.extend(items)

    return result


# ─────────────────────────────────────────────
# SPEAKER-STRATIFIED SPLIT
# ─────────────────────────────────────────────

def speaker_stratified_split(records: list, dataset_name: str) -> tuple:
    """
    Split by unique speaker so no speaker appears in more than one split.
    Returns (train, val, test) lists of records.
    """
    rng = random.Random(SEED)

    speaker_to_records = defaultdict(list)
    for r in records:
        spk = r.get("speaker_id") or "unknown"
        speaker_to_records[spk].append(r)

    speakers = sorted(speaker_to_records.keys())

    # Edge case: very few speakers (e.g. SAVEE has 4)
    n = len(speakers)
    if n < 3:
        # Fall back to random record-level split
        print(f"    WARNING: only {n} speakers — using record-level split")
        rng.shuffle(records)
        n_train = max(1, int(len(records) * TRAIN_RATIO))
        n_val   = max(1, int(len(records) * VAL_RATIO))
        return records[:n_train], records[n_train:n_train+n_val], records[n_train+n_val:]

    rng.shuffle(speakers)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))

    train_spk = set(speakers[:n_train])
    val_spk   = set(speakers[n_train:n_train + n_val])
    test_spk  = set(speakers[n_train + n_val:])

    train, val, test = [], [], []
    for spk, recs in speaker_to_records.items():
        if spk in train_spk:
            train.extend(recs)
        elif spk in val_spk:
            val.extend(recs)
        else:
            test.extend(recs)

    print(f"    Speakers: {n} total → train:{len(train_spk)} "
          f"val:{len(val_spk)} test:{len(test_spk)}")
    print(f"    Records  → train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test


# ─────────────────────────────────────────────
# CONVERSATION-STRATIFIED SPLIT (MELD)
# ─────────────────────────────────────────────

def conversation_stratified_split(records: list) -> tuple:
    rng = random.Random(SEED)

    conv_to_records = defaultdict(list)
    for r in records:
        conv = r.get("conversation_id") or r.get("speaker_id") or "unknown"
        conv_to_records[conv].append(r)

    convs = sorted(conv_to_records.keys())
    rng.shuffle(convs)

    n       = len(convs)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))

    train_c = set(convs[:n_train])
    val_c   = set(convs[n_train:n_train + n_val])

    train, val, test = [], [], []
    for conv, recs in conv_to_records.items():
        if conv in train_c:
            train.extend(recs)
        elif conv in val_c:
            val.extend(recs)
        else:
            test.extend(recs)

    print(f"    Conversations: {n} total → train:{len(train_c)} "
          f"val:{len(val_c)} test:{n - len(train_c) - len(val_c)}")
    print(f"    Records → train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test


# ─────────────────────────────────────────────
# VERIFY NO SPEAKER LEAKAGE
# ─────────────────────────────────────────────

def verify_no_leakage(train, val, test):
    """
    Check that no (dataset, speaker_id) pair appears in more than one split.
    Prints a warning if leakage is found.
    """
    def get_keys(records):
        return {(r["dataset"], r.get("speaker_id") or "unknown") for r in records}

    train_keys = get_keys(train)
    val_keys   = get_keys(val)
    test_keys  = get_keys(test)

    tv = train_keys & val_keys
    tt = train_keys & test_keys
    vt = val_keys   & test_keys

    print("\n  Leakage verification:")
    if tv or tt or vt:
        print(f"  ⚠ Train∩Val: {len(tv)}  Train∩Test: {len(tt)}  Val∩Test: {len(vt)}")
        for pair in sorted(tv | tt | vt)[:10]:
            print(f"    {pair}")
    else:
        print("  ✓ No speaker leakage detected across splits")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def seconds_to_hms(s: float) -> str:
    s = int(s)
    return f"{s//3600}h {(s%3600)//60}m {s%60}s"


def build_split_summary(train, val, test) -> dict:
    summary = {}
    for split_name, records in [("train", train), ("val", val), ("test", test)]:
        emotion_counts = defaultdict(int)
        emotion_dur    = defaultdict(float)
        dataset_counts = defaultdict(int)

        for r in records:
            emotion_counts[r["emotion"]] += 1
            emotion_dur[r["emotion"]]    += r.get("duration_sec", 0.0)
            dataset_counts[r["dataset"]] += 1

        summary[split_name] = {
            "total_files":    len(records),
            "total_duration": seconds_to_hms(sum(emotion_dur.values())),
            "per_emotion": {
                e: {
                    "count":        emotion_counts.get(e, 0),
                    "duration_hms": seconds_to_hms(emotion_dur.get(e, 0.0)),
                }
                for e in EMOTIONS
            },
            "per_dataset": dict(dataset_counts),
        }
    return summary


def print_summary(summary: dict):
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  FINAL SPLIT SUMMARY")
    print(SEP)
    for split_name in ["train", "val", "test"]:
        s = summary[split_name]
        print(f"\n  [{split_name.upper()}]  {s['total_files']} files  |  {s['total_duration']}")
        print(f"  {'Emotion':<12}  {'Count':>6}  {'Duration'}")
        print(f"  {'─'*40}")
        for emo in EMOTIONS:
            e = s["per_emotion"][emo]
            print(f"  {emo:<12}  {e['count']:>6}  {e['duration_hms']}")
        print(f"  Datasets: {dict(sorted(s['per_dataset'].items()))}")

    # Print imbalance ratio
    for split_name in ["train", "val", "test"]:
        counts = [summary[split_name]["per_emotion"][e]["count"] for e in EMOTIONS]
        counts = [c for c in counts if c > 0]
        if counts:
            ratio = max(counts) / min(counts)
            print(f"\n  [{split_name.upper()}] Imbalance ratio (max/min): {ratio:.2f}×")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print("  EMOTIONAL AI – DATASET SPLITTER")
    print(f"{'='*65}")

    root = Path(ROOT_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else root / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root:    {root}")
    print(f"Output:  {output_dir}")
    print(f"Cap:     {MAX_PER_EMOTION_TOTAL} clips per emotion (total)")
    print(f"{'='*65}")

    # ── Step 1: load all records ──────────────
    all_records = []
    for dataset_name in DATASETS:
        dataset_dir = root / dataset_name
        if not dataset_dir.exists():
            print(f"\n[SKIP] {dataset_name} not found at {dataset_dir}")
            continue
        print(f"\n[Loading {dataset_name.upper()} ...]")
        records = load_dataset_records(dataset_dir, dataset_name)
        all_records.extend(records)

    print(f"\nTotal records loaded: {len(all_records)}")

    # ── Step 2: apply global emotion cap ──────
    if MAX_PER_EMOTION_TOTAL:
        print("\n[Applying emotion cap ...]")
        all_records = apply_global_emotion_cap(all_records, MAX_PER_EMOTION_TOTAL, SEED)
        print(f"Total after cap: {len(all_records)}")

    # ── Step 3: group back by dataset and split
    by_dataset = defaultdict(list)
    for r in all_records:
        by_dataset[r["dataset"]].append(r)

    all_train, all_val, all_test = [], [], []

    for dataset_name in DATASETS:
        records = by_dataset.get(dataset_name, [])
        if not records:
            continue

        print(f"\n[Splitting {dataset_name.upper()} — {len(records)} records ...]")
        if dataset_name == "meld":
            train, val, test = conversation_stratified_split(records)
        else:
            train, val, test = speaker_stratified_split(records, dataset_name)

        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)

    # ── Step 4: shuffle final splits ──────────
    rng = random.Random(SEED)
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    # ── Step 5: verify no leakage ─────────────
    verify_no_leakage(all_train, all_val, all_test)

    # ── Step 6: save ──────────────────────────
    for split_name, records in [("train", all_train), ("val", all_val), ("test", all_test)]:
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved] {split_name}.json → {len(records)} records → {out_path}")

    summary = build_split_summary(all_train, all_val, all_test)
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"{'='*65}")
    print("  SPLITTING COMPLETE!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()