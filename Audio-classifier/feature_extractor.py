"""
Emotional AI – Script 2: Feature Extraction
============================================
Reads split JSON files, loads each audio file,
resamples to 16 kHz mono, normalises, pads/truncates,
and saves each waveform as a .pt tensor.

Augmentation is NOT applied here — it is done on-the-fly
in the Dataset class during training.
This keeps the .pt files clean and reusable.

Output structure:
    features/
        train/   *.pt
        val/     *.pt
        test/    *.pt
        features_train.json
        features_val.json
        features_test.json
        features_config.json
"""

import os
os.environ["TORCHAUDIO_USE_FFMPEG"] = "1"

import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ROOT_DIR     = "/workspace/audio-em/dataset"
SPLITS_DIR   = None   # None → ROOT_DIR/splits
OUTPUT_DIR   = None   # None → ROOT_DIR/../features

MAX_DURATION = 10.0   # seconds — clips longer than this are truncated
NUM_WORKERS  = 4


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TARGET_SR = 16_000
EMOTIONS  = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]


# ─────────────────────────────────────────────
# AUDIO PROCESSING
# ─────────────────────────────────────────────

def load_and_process(audio_path: str, root: Path, max_samples: int):
    """
    Load audio → mono → 16 kHz → normalise → pad/truncate.
    Returns 1-D float32 tensor of shape (max_samples,) or None on error.
    """
    path = Path(audio_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return None

    try:
        waveform, sr = torchaudio.load(str(path))

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != TARGET_SR:
            waveform = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(waveform)

        waveform = waveform.squeeze(0)   # (samples,)

        # Peak-normalise to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # Pad or truncate
        length = waveform.shape[0]
        if length < max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - length))
        else:
            waveform = waveform[:max_samples]

        return waveform.float()

    except Exception as e:
        if not hasattr(load_and_process, "_err_count"):
            load_and_process._err_count = 0
        if load_and_process._err_count < 5:
            print(f"    Load error [{type(e).__name__}]: {str(e)[:120]}")
            load_and_process._err_count += 1
        return None


# ─────────────────────────────────────────────
# WORKER (runs in subprocess)
# ─────────────────────────────────────────────

def process_record(args):
    idx, record, root_str, out_dir_str, max_samples = args

    out_path = Path(out_dir_str) / f"{idx}_{record['emotion']}_{record['dataset']}.pt"

    # Resume: skip if already done
    if out_path.exists():
        return idx, True, str(out_path), None

    waveform = load_and_process(record["audio_path"], Path(root_str), max_samples)
    if waveform is None:
        return idx, False, None, f"Failed: {record['audio_path']}"

    torch.save({
        "waveform":    waveform,
        "emotion":     record["emotion"],
        "emotion_id":  record["emotion_id"],
        "dataset":     record["dataset"],
        "audio_path":  record["audio_path"],
        "duration_sec": record.get("duration_sec", 0.0),
    }, str(out_path))

    return idx, True, str(out_path), None


# ─────────────────────────────────────────────
# PROCESS ONE SPLIT
# ─────────────────────────────────────────────

def process_split(split_name, records, root, output_dir, max_samples, num_workers):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Small splits don't need multiprocessing overhead
    workers = 1 if len(records) < 500 else num_workers

    worker_args = [
        (i, r, str(root), str(split_dir), max_samples)
        for i, r in enumerate(records)
    ]

    index, errors = [], []

    print(f"\n  [{split_name.upper()}] {len(records)} files, {workers} workers ...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_record, a): a[0] for a in worker_args}
        with tqdm(total=len(worker_args), desc=f"  {split_name}", unit="file") as pbar:
            for future in as_completed(futures):
                i, ok, out_path, err = future.result()
                r = records[i]
                if ok:
                    index.append({
                        "pt_path":     out_path,
                        "emotion":     r["emotion"],
                        "emotion_id":  r["emotion_id"],
                        "dataset":     r["dataset"],
                        "speaker_id":  r.get("speaker_id"),
                        "duration_sec": r.get("duration_sec", 0.0),
                        "audio_path":  r["audio_path"],
                    })
                else:
                    errors.append(err)
                pbar.update(1)

    print(f"  [{split_name.upper()}] Done — {len(index)} ok, {len(errors)} failed")
    if errors:
        for e in errors[:5]:
            print(f"    {e}")

    return index


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def print_feature_summary(indices: dict, max_duration: float):
    from collections import defaultdict

    print(f"\n{'='*60}")
    print("  FEATURE EXTRACTION SUMMARY")
    print(f"  Max duration: {max_duration}s")
    print(f"{'='*60}")

    for split_name, index in indices.items():
        if not index:
            print(f"\n  [{split_name.upper()}] No tensors extracted")
            continue
        emotion_counts = defaultdict(int)
        dataset_counts = defaultdict(int)
        for r in index:
            emotion_counts[r["emotion"]] += 1
            dataset_counts[r["dataset"]] += 1

        print(f"\n  [{split_name.upper()}] {len(index)} tensors")
        for emo in EMOTIONS:
            c = emotion_counts[emo]
            bar = "█" * (c // 40)
            print(f"    {emo:<12}: {c:>5}  {bar}")
        print(f"  Datasets: {dict(sorted(dataset_counts.items()))}")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  EMOTIONAL AI – FEATURE EXTRACTION")
    print(f"{'='*60}")

    root       = Path(ROOT_DIR).resolve()
    splits_dir = Path(SPLITS_DIR).resolve() if SPLITS_DIR else root / "splits"
    output_dir = Path(OUTPUT_DIR).resolve() if OUTPUT_DIR else root.parent / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(MAX_DURATION * TARGET_SR)

    print(f"Root:        {root}")
    print(f"Splits:      {splits_dir}")
    print(f"Output:      {output_dir}")
    print(f"Max samples: {max_samples:,}  ({MAX_DURATION}s @ {TARGET_SR} Hz)")
    print(f"Workers:     {NUM_WORKERS}")
    print(f"{'='*60}\n")

    # Quick sanity check on one file
    train_split = splits_dir / "train.json"
    if train_split.exists():
        with open(train_split, encoding="utf-8") as f:
            test_records = json.load(f)
        if test_records:
            tp = root / test_records[0]["audio_path"]
            print(f"Sanity check: {tp.name}  exists={tp.exists()}")
            if tp.exists():
                try:
                    wav, sr = torchaudio.load(str(tp))
                    print(f"  ✓ torchaudio OK — shape {wav.shape}, sr {sr}")
                except Exception as e:
                    print(f"  ✗ torchaudio failed: {e}")
                    print("  Try: apt-get install ffmpeg  or  pip install torchaudio --force-reinstall")
                    return
            print()
    else:
        print(f"WARNING: {train_split} not found — run script1_split.py first.\n")
        return

    all_indices = {}

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.exists():
            print(f"[SKIP] {split_path} not found")
            continue

        with open(split_path, encoding="utf-8") as f:
            records = json.load(f)

        print(f"[{split_name.upper()}] {len(records)} records")

        index = process_split(split_name, records, root, output_dir, max_samples, NUM_WORKERS)

        index_path = output_dir / f"features_{split_name}.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"  Index saved → {index_path}")

        all_indices[split_name] = index

    # Save config
    config = {
        "target_sr":   TARGET_SR,
        "max_duration": MAX_DURATION,
        "max_samples": max_samples,
        "num_classes": len(EMOTIONS),
        "emotions":    EMOTIONS,
        "emotion_to_id": {e: i for i, e in enumerate(EMOTIONS)},
        "features_dir": str(output_dir),
    }
    with open(output_dir / "features_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved → {output_dir / 'features_config.json'}")

    print_feature_summary(all_indices, MAX_DURATION)
    print(f"{'='*60}")
    print("  FEATURE EXTRACTION COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()