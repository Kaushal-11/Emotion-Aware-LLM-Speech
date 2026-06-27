"""
Script 01: Dataset Filtering Pipeline with Train/Test Split
================================================================================
Filters raw audio datasets, creates train/test splits (1000 steering, 300 test)
for 7 emotions (including neutral from ESD), and saves metadata JSON files.
ONLY includes samples that have transcriptions.
"""

import os
import json
import random
import shutil
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
# DIRECT CONFIGURATION VARIABLES
# ==============================================================================

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR = "/workspace/audio-em"

# Dataset paths - each has emotion subdirectories
DATA_DIR_RAVDESS = os.path.join(BASE_DIR, "dataset", "ravdess")
DATA_DIR_IEMOCAP = os.path.join(BASE_DIR, "dataset", "iemocap")
DATA_DIR_ESD = os.path.join(BASE_DIR, "dataset", "esd")
DATA_DIR_CREMA_D = os.path.join(BASE_DIR, "dataset", "cremad")
DATA_DIR_TESS = os.path.join(BASE_DIR, "dataset", "tess")
DATA_DIR_SAVEE = os.path.join(BASE_DIR, "dataset", "savee")
DATA_DIR_MELD = os.path.join(BASE_DIR, "dataset", "meld")

# JSON files with transcriptions and metadata
DATA_ANALYSIS_DIR = os.path.join(BASE_DIR, "dataset", "data_analysis")

# Output directories
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_USED_DIR = os.path.join(OUTPUT_BASE_DIR, "used")
OUTPUT_STEERING_DIR = os.path.join(OUTPUT_USED_DIR, "steering")
OUTPUT_TEST_DIR = os.path.join(OUTPUT_USED_DIR, "test")
OUTPUT_UNUSED_DIR = os.path.join(OUTPUT_BASE_DIR, "unused")

# ─── EMOTIONS ─────────────────────────────────────────────────────────────────
# 7 emotions including neutral (from ESD only)
EMOTIONS = ["anger", "happiness", "sadness", "disgust", "fear", "surprise", "neutral"]
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# Steering and test split sizes
STEERING_SIZE = 1000
TEST_SIZE = 300

# ─── DATASET FILTERING ────────────────────────────────────────────────────────
MIN_DURATION_S = 1.0
MAX_DURATION_S = 20.0
MAX_SILENCE_RATIO = 0.30
MIN_SNR_DB = 10.0

# ─── CORPUS SELECTION ─────────────────────────────────────────────────────────
PROCESS_RAVDESS = True
PROCESS_IEMOCAP = True
PROCESS_ESD = True
PROCESS_CREMA_D = True
PROCESS_TESS = True
PROCESS_SAVEE = True
PROCESS_MELD = True

# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==============================================================================
# LOAD EXISTING METADATA FROM DATA_ANALYSIS JSON FILES
# ==============================================================================

def load_dataset_metadata(dataset_name: str):
    """Load existing metadata from data_analysis JSON files."""
    json_path = os.path.join(DATA_ANALYSIS_DIR, f"{dataset_name}.json")
    
    if not os.path.exists(json_path):
        print(f"  ⚠️  Metadata file not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a lookup dictionary by audio_name
        metadata_lookup = {}
        
        # Handle both list and dict formats
        if isinstance(data, list):
            # If it's a list of files
            for item in data:
                audio_name = item.get("audio_name", "")
                if audio_name:
                    metadata_lookup[audio_name] = item
        elif isinstance(data, dict):
            # If it's a dict with 'files' key
            files = data.get("files", [])
            for item in files:
                audio_name = item.get("audio_name", "")
                if audio_name:
                    metadata_lookup[audio_name] = item
        
        print(f"  ✅ Loaded {len(metadata_lookup)} entries from {dataset_name}.json")
        return metadata_lookup
    
    except Exception as e:
        print(f"  ❌ Error loading {json_path}: {e}")
        return {}

# ==============================================================================
# EMOTION FOLDER NAME MAPPING
# ==============================================================================

EMOTION_FOLDER_MAP = {
    "anger": "anger", "angry": "anger",
    "happiness": "happiness", "happy": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral"
}

# ==============================================================================
# GENERIC PARSER WITH METADATA ENRICHMENT
# ==============================================================================

def extract_speaker_id(wav_path: Path, corpus_name: str) -> str:
    """Extract speaker ID from file path based on dataset naming convention."""
    stem = wav_path.stem
    
    if corpus_name == "cremad":
        parts = stem.split("_")
        return parts[0] if len(parts) >= 1 else "unknown"
    
    elif corpus_name == "ravdess":
        parts = stem.split("-")
        return f"actor_{parts[6]}" if len(parts) >= 7 else "unknown"
    
    elif corpus_name == "iemocap":
        parts = stem.split("_")
        return parts[0] if len(parts) >= 1 else "unknown"
    
    elif corpus_name == "esd":
        if "speaker_" in stem:
            parts = stem.split("_")
            return parts[1] if len(parts) >= 2 else "unknown"
        elif stem[:4].isdigit():
            return stem[:4]
    
    elif corpus_name == "tess":
        parts = stem.split("_")
        return parts[0] if len(parts) >= 1 else "unknown"
    
    elif corpus_name == "savee":
        parts = stem.split("_")
        return parts[0] if len(parts) >= 1 else "unknown"
    
    elif corpus_name == "meld":
        parts = stem.split("_")
        return parts[0] if len(parts) >= 1 else "unknown"
    
    return "unknown"


def parse_emotion_folders(data_dir: str, corpus_name: str, metadata_lookup: dict):
    """
    Parse dataset and enrich with metadata from data_analysis JSON files.
    """
    samples = []
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"  ⚠️  Directory not found: {data_dir}")
        return samples
    
    found_emotions = []
    
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue
        
        folder_name = item.name.lower()
        if folder_name not in EMOTION_FOLDER_MAP:
            continue
            
        emotion = EMOTION_FOLDER_MAP[folder_name]
        if emotion not in EMOTIONS:
            continue
        
        found_emotions.append(emotion)
        
        wav_files = list(item.glob("*.wav")) + list(item.glob("*.WAV"))
        
        for wav_path in wav_files:
            audio_name = wav_path.name
            
            # Get metadata from lookup if available
            meta = metadata_lookup.get(audio_name, {})
            
            # Construct relative audio path (dataset/emotion/filename)
            relative_path = f"{corpus_name}/{emotion}/{audio_name}"
            
            speaker_id = extract_speaker_id(wav_path, corpus_name)
            
            samples.append({
                "audio_path": relative_path,  # Relative path for JSON
                "audio_full_path": str(wav_path),  # Full path for copying
                "audio_name": audio_name,
                "emotion": emotion,
                "emotion_id": EMOTION_TO_ID[emotion],
                "dataset": corpus_name,
                "speaker_id": meta.get("speaker_id", speaker_id),
                "transcription": meta.get("transcription", None),
                "sentence_code": meta.get("sentence_code", None),
                "intensity": meta.get("intensity", None),
                "conversation_id": meta.get("conversation_id", None),
                "utterance_id": meta.get("utterance_id", audio_name.replace(".wav", "")),
                "duration_sec": None  # Will be filled during filtering
            })
        
        print(f"    {emotion:12s}: {len(wav_files):5d} files")
    
    missing_emotions = set(EMOTIONS) - set(found_emotions)
    for emotion in missing_emotions:
        print(f"    {emotion:12s}: 0 files (not found)")
    
    return samples

# ==============================================================================
# DATASET-SPECIFIC PARSERS
# ==============================================================================

def parse_ravdess(data_dir: str, metadata_lookup: dict):
    return parse_emotion_folders(data_dir, "ravdess", metadata_lookup)

def parse_iemocap(data_dir: str, metadata_lookup: dict):
    return parse_emotion_folders(data_dir, "iemocap", metadata_lookup)

def parse_esd(data_dir: str, metadata_lookup: dict):
    return parse_emotion_folders(data_dir, "esd", metadata_lookup)

def parse_cremad(data_dir: str, metadata_lookup: dict):
    return parse_emotion_folders(data_dir, "cremad", metadata_lookup)

def parse_tess(data_dir: str, metadata_lookup: dict):
    return parse_emotion_folders(data_dir, "tess", metadata_lookup)

def parse_savee(data_dir: str, metadata_lookup: dict):
    return parse_emotion_folders(data_dir, "savee", metadata_lookup)

def parse_meld(data_dir: str, metadata_lookup: dict):
    return parse_emotion_folders(data_dir, "meld", metadata_lookup)

# ==============================================================================
# AUDIO QUALITY CHECKS
# ==============================================================================

def load_audio(path: str, target_sr: int = 16000):
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio, sr


def filter_samples_with_stats(samples: list, verbose: bool = True):
    """Apply filtering criteria and update duration."""
    passed = []
    
    emotion_stats = {
        emotion: {
            "total": 0, "too_short": 0, "too_long": 0,
            "too_silent": 0, "low_snr": 0, "passed": 0,
            "total_duration": 0
        } for emotion in EMOTIONS
    }
    
    overall_stats = {
        "total": len(samples),
        "too_short": 0, "too_long": 0,
        "too_silent": 0, "low_snr": 0, "load_error": 0, "passed": 0,
        "total_duration": 0
    }
    
    for sample in tqdm(samples, desc="Filtering", disable=not verbose):
        emotion = sample["emotion"]
        emotion_stats[emotion]["total"] += 1
        
        try:
            audio, sr = load_audio(sample["audio_full_path"])
            duration = len(audio) / sr
            sample["duration_sec"] = round(duration, 3)
        except Exception as e:
            overall_stats["load_error"] += 1
            continue

        if duration < MIN_DURATION_S:
            overall_stats["too_short"] += 1
            emotion_stats[emotion]["too_short"] += 1
            continue
        if duration > MAX_DURATION_S:
            overall_stats["too_long"] += 1
            emotion_stats[emotion]["too_long"] += 1
            continue

        intervals = librosa.effects.split(audio, top_db=30.0)
        if len(intervals) == 0:
            overall_stats["too_silent"] += 1
            emotion_stats[emotion]["too_silent"] += 1
            continue
        
        non_silent_samples = sum(end - start for start, end in intervals)
        silence_ratio = 1.0 - (non_silent_samples / len(audio))
        if silence_ratio > MAX_SILENCE_RATIO:
            overall_stats["too_silent"] += 1
            emotion_stats[emotion]["too_silent"] += 1
            continue

        # SNR check
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        if rms.max() > 1e-10:
            threshold = np.percentile(rms, 30)
            noise_rms = rms[rms <= threshold]
            signal_rms = rms[rms > threshold]
            if len(noise_rms) > 0 and noise_rms.mean() > 1e-10:
                snr = 20 * np.log10(signal_rms.mean() / noise_rms.mean() + 1e-10)
                if snr < MIN_SNR_DB:
                    overall_stats["low_snr"] += 1
                    emotion_stats[emotion]["low_snr"] += 1
                    continue

        passed.append(sample)
        overall_stats["passed"] += 1
        overall_stats["total_duration"] += duration
        emotion_stats[emotion]["passed"] += 1
        emotion_stats[emotion]["total_duration"] += duration

    return passed, overall_stats, emotion_stats


def filter_by_transcription(samples: list, verbose: bool = True):
    """
    Filter samples to only keep those with transcriptions.
    Returns filtered samples and statistics.
    """
    has_transcription = []
    missing_transcription = []
    
    for sample in samples:
        if sample.get("transcription") and sample["transcription"].strip():
            has_transcription.append(sample)
        else:
            missing_transcription.append(sample)
    
    if verbose:
        print(f"\n📝 TRANSCRIPTION ANALYSIS:")
        print(f"   Total samples: {len(samples)}")
        print(f"   With transcription: {len(has_transcription)} ({len(has_transcription)/len(samples)*100:.1f}%)")
        print(f"   Missing transcription: {len(missing_transcription)} ({len(missing_transcription)/len(samples)*100:.1f}%)")
        
        # Show per-dataset stats
        dataset_stats = defaultdict(lambda: {"total": 0, "has_trans": 0})
        for sample in samples:
            dataset_stats[sample["dataset"]]["total"] += 1
            if sample.get("transcription") and sample["transcription"].strip():
                dataset_stats[sample["dataset"]]["has_trans"] += 1
        
        print(f"\n   Per-dataset transcription coverage:")
        for dataset, stats in sorted(dataset_stats.items()):
            coverage = stats["has_trans"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"      {dataset:10s}: {stats['has_trans']:5d}/{stats['total']:5d} ({coverage:5.1f}%)")
    
    return has_transcription, missing_transcription

# ==============================================================================
# TRAIN/TEST SPLIT FUNCTIONS
# ==============================================================================

def create_splits(samples_by_emotion: dict):
    """
    Create steering (1000), test (300), and unused splits for each emotion.
    """
    splits = {
        "steering": {},
        "test": {},
        "unused": {}
    }
    
    split_stats = {}
    
    for emotion, samples in samples_by_emotion.items():
        total = len(samples)
        print(f"\n  {emotion.upper()}: Total with transcriptions = {total}")
        
        # Check if we have enough samples for steering
        if total < STEERING_SIZE:
            print(f"    ⚠️  WARNING: Only {total} samples available for {emotion} (need {STEERING_SIZE} for steering)")
            steering_count = total
            test_count = 0
            unused_count = 0
        else:
            steering_count = STEERING_SIZE
            remaining = total - STEERING_SIZE
            test_count = min(TEST_SIZE, remaining)
            unused_count = remaining - test_count
        
        # Random shuffle
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Split
        steering_samples = shuffled[:steering_count]
        test_samples = shuffled[steering_count:steering_count + test_count]
        unused_samples = shuffled[steering_count + test_count:]
        
        splits["steering"][emotion] = steering_samples
        splits["test"][emotion] = test_samples
        splits["unused"][emotion] = unused_samples
        
        split_stats[emotion] = {
            "total": total,
            "steering": steering_count,
            "test": test_count,
            "unused": unused_count
        }
        
        print(f"    Steering: {steering_count}, Test: {test_count}, Unused: {unused_count}")
    
    return splits, split_stats

# ==============================================================================
# SAVE JSON METADATA FUNCTIONS
# ==============================================================================

def save_split_metadata(split_samples: dict, split_name: str, output_dir: str):
    """
    Save metadata JSON files for each emotion in a split.
    """
    emotion_dir = os.path.join(output_dir, split_name)
    os.makedirs(emotion_dir, exist_ok=True)
    
    all_files = []
    
    for emotion, samples in split_samples.items():
        if not samples:
            continue
        
        # Prepare metadata for this emotion
        emotion_metadata = {
            "split": split_name,
            "emotion": emotion,
            "emotion_id": EMOTION_TO_ID[emotion],
            "total_samples": len(samples),
            "files": []
        }
        
        for sample in samples:
            # Remove full path, keep only relative path for JSON
            file_entry = {
                "audio_path": sample["audio_path"],
                "audio_name": sample["audio_name"],
                "emotion": sample["emotion"],
                "emotion_id": sample["emotion_id"],
                "dataset": sample["dataset"],
                "speaker_id": sample["speaker_id"],
                "transcription": sample.get("transcription"),
                "sentence_code": sample.get("sentence_code"),
                "intensity": sample.get("intensity"),
                "conversation_id": sample.get("conversation_id"),
                "utterance_id": sample.get("utterance_id"),
                "duration_sec": sample["duration_sec"]
            }
            emotion_metadata["files"].append(file_entry)
            all_files.append(file_entry)
        
        # Save per-emotion JSON
        json_path = os.path.join(emotion_dir, f"{emotion}_{split_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(emotion_metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: {json_path}")
    
    # Save combined JSON for this split
    combined_metadata = {
        "split": split_name,
        "total_samples": len(all_files),
        "emotion_counts": {emotion: len(samples) for emotion, samples in split_samples.items() if samples},
        "files": all_files
    }
    
    combined_path = os.path.join(output_dir, f"{split_name}_metadata.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved combined: {combined_path}")
    
    return combined_path


def save_unused_metadata(split_samples: dict, output_dir: str):
    """Save unused samples metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_unused = []
    
    for emotion, samples in split_samples.items():
        if not samples:
            continue
        
        unused_metadata = {
            "split": "unused",
            "emotion": emotion,
            "emotion_id": EMOTION_TO_ID[emotion],
            "total_samples": len(samples),
            "files": []
        }
        
        for sample in samples:
            file_entry = {
                "audio_path": sample["audio_path"],
                "audio_name": sample["audio_name"],
                "emotion": sample["emotion"],
                "emotion_id": sample["emotion_id"],
                "dataset": sample["dataset"],
                "speaker_id": sample["speaker_id"],
                "transcription": sample.get("transcription"),
                "sentence_code": sample.get("sentence_code"),
                "intensity": sample.get("intensity"),
                "duration_sec": sample["duration_sec"]
            }
            unused_metadata["files"].append(file_entry)
            all_unused.append(file_entry)
        
        # Save per-emotion unused JSON
        json_path = os.path.join(output_dir, f"{emotion}_unused.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unused_metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved: {json_path}")
    
    # Save combined unused JSON
    combined_unused = {
        "split": "unused",
        "total_samples": len(all_unused),
        "emotion_counts": {emotion: len(samples) for emotion, samples in split_samples.items() if samples},
        "files": all_unused
    }
    
    combined_path = os.path.join(output_dir, "unused_metadata.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_unused, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Saved combined unused: {combined_path}")


def print_split_summary(split_stats):
    """Print summary of splits."""
    print("\n" + "=" * 80)
    print("📊 FINAL SPLIT SUMMARY (SAMPLES WITH TRANSCRIPTIONS ONLY)")
    print("=" * 80)
    
    total_steering = 0
    total_test = 0
    total_unused = 0
    
    print(f"\n{'Emotion':12s} {'Total':>8s} {'Steering':>10s} {'Test':>8s} {'Unused':>8s}")
    print("-" * 50)
    
    for emotion, stats in split_stats.items():
        total_steering += stats["steering"]
        total_test += stats["test"]
        total_unused += stats["unused"]
        print(f"{emotion:12s} {stats['total']:8d} {stats['steering']:10d} {stats['test']:8d} {stats['unused']:8d}")
    
    print("-" * 50)
    print(f"{'TOTAL':12s} {sum(s['total'] for s in split_stats.values()):8d} {total_steering:10d} {total_test:8d} {total_unused:8d}")
    
    print(f"\n✅ Steering total: {total_steering} samples (all with transcriptions)")
    print(f"✅ Test total: {total_test} samples (all with transcriptions)")
    print(f"✅ Unused total: {total_unused} samples (all with transcriptions)")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("EmoSteer-TTS Dataset Filtering & Split Pipeline (With Transcription Filter)")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Steering size per emotion: {STEERING_SIZE}")
    print(f"   Test size per emotion: {TEST_SIZE}")
    print(f"   Target emotions: {EMOTIONS}")
    print(f"   Duration filter: {MIN_DURATION_S}s - {MAX_DURATION_S}s")
    print(f"   ⚠️  ONLY samples with transcriptions will be included in steering/test sets")
    
    # Load metadata from data_analysis JSON files
    print("\n" + "=" * 80)
    print("📁 LOADING DATASET METADATA")
    print("=" * 80)
    
    datasets = ["cremad", "ravdess", "iemocap", "esd", "tess", "savee", "meld"]
    metadata_lookups = {}
    
    for dataset in datasets:
        print(f"\n📍 Loading {dataset}.json...")
        metadata_lookups[dataset] = load_dataset_metadata(dataset)
    
    # Parse datasets
    print("\n" + "=" * 80)
    print("📁 PARSING DATASETS")
    print("=" * 80)
    
    corpus_parsers = [
        ("ravdess", DATA_DIR_RAVDESS, PROCESS_RAVDESS, parse_ravdess),
        ("iemocap", DATA_DIR_IEMOCAP, PROCESS_IEMOCAP, parse_iemocap),
        ("esd", DATA_DIR_ESD, PROCESS_ESD, parse_esd),
        ("cremad", DATA_DIR_CREMA_D, PROCESS_CREMA_D, parse_cremad),
        ("tess", DATA_DIR_TESS, PROCESS_TESS, parse_tess),
        ("savee", DATA_DIR_SAVEE, PROCESS_SAVEE, parse_savee),
        ("meld", DATA_DIR_MELD, PROCESS_MELD, parse_meld),
    ]
    
    all_samples = []
    corpus_raw_counts = {}
    
    for corpus_name, data_dir, process_flag, parser_func in corpus_parsers:
        if process_flag and data_dir and Path(data_dir).exists():
            print(f"\n📍 {corpus_name.upper()}: {data_dir}")
            raw = parser_func(data_dir, metadata_lookups.get(corpus_name, {}))
            print(f"   ✅ Total: {len(raw)} samples")
            all_samples.extend(raw)
            corpus_raw_counts[corpus_name] = len(raw)
        elif process_flag:
            print(f"\n📍 {corpus_name.upper()}: {data_dir}")
            print(f"   ⚠️  Directory not found - skipping")
    
    print("\n" + "=" * 80)
    print(f"📊 TOTAL RAW SAMPLES: {len(all_samples)}")
    print("=" * 80)
    
    # First filter by audio quality
    print("\n" + "=" * 80)
    print("🔊 STEP 1: AUDIO QUALITY FILTERING")
    print("=" * 80)
    
    quality_passed, overall_stats, emotion_stats = filter_samples_with_stats(all_samples)
    
    print(f"\n   Audio quality passed: {overall_stats['passed']}/{overall_stats['total']} ({overall_stats['passed']/overall_stats['total']*100:.1f}%)")
    
    # Then filter by transcription
    print("\n" + "=" * 80)
    print("📝 STEP 2: TRANSCRIPTION FILTERING")
    print("=" * 80)
    
    samples_with_trans, samples_without_trans = filter_by_transcription(quality_passed, verbose=True)
    
    # Group samples with transcriptions by emotion
    samples_by_emotion = defaultdict(list)
    for sample in samples_with_trans:
        samples_by_emotion[sample["emotion"]].append(sample)
    
    # Print per-emotion counts after transcription filtering
    print("\n" + "=" * 80)
    print("📊 SAMPLES WITH TRANSCRIPTIONS - PER EMOTION")
    print("=" * 80)
    for emotion in EMOTIONS:
        count = len(samples_by_emotion[emotion])
        print(f"   {emotion:12s}: {count:5d} samples")
    
    # Check if neutral exists (only from ESD)
    if "neutral" not in samples_by_emotion or len(samples_by_emotion["neutral"]) == 0:
        print("\n⚠️  WARNING: No neutral samples with transcriptions found!")
        print("   Only ESD dataset provides neutral emotion. Please check ESD metadata.")
    
    # Create splits (only from samples with transcriptions)
    print("\n" + "=" * 80)
    print("📊 STEP 3: CREATING TRAIN/TEST SPLITS")
    print("=" * 80)
    
    splits, split_stats = create_splits(samples_by_emotion)
    
    # Save steering split metadata
    print("\n" + "=" * 80)
    print("💾 SAVING STEERING SPLIT METADATA")
    print("=" * 80)
    save_split_metadata(splits["steering"], "steering", OUTPUT_USED_DIR)
    
    # Save test split metadata
    print("\n" + "=" * 80)
    print("💾 SAVING TEST SPLIT METADATA")
    print("=" * 80)
    save_split_metadata(splits["test"], "test", OUTPUT_USED_DIR)
    
    # Save unused split metadata
    print("\n" + "=" * 80)
    print("💾 SAVING UNUSED SPLIT METADATA")
    print("=" * 80)
    save_unused_metadata(splits["unused"], OUTPUT_UNUSED_DIR)
    
    # Print final summary
    print_split_summary(split_stats)
    
    # Save processing report
    report = {
        "config": {
            "steering_size": STEERING_SIZE,
            "test_size": TEST_SIZE,
            "min_duration_sec": MIN_DURATION_S,
            "max_duration_sec": MAX_DURATION_S,
            "max_silence_ratio": MAX_SILENCE_RATIO,
            "min_snr_db": MIN_SNR_DB,
            "seed": SEED,
            "require_transcription": True
        },
        "filtering_stats": overall_stats,
        "emotion_filtering_stats": emotion_stats,
        "transcription_stats": {
            "total_after_quality": len(quality_passed),
            "with_transcription": len(samples_with_trans),
            "without_transcription": len(samples_without_trans)
        },
        "split_stats": split_stats,
        "corpus_raw_counts": corpus_raw_counts
    }
    
    report_path = os.path.join(OUTPUT_BASE_DIR, "processing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Processing report saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
    print("  ├── used/")
    print("  │   ├── steering/ (samples with transcriptions)")
    print("  │   └── test/ (samples with transcriptions)")
    print("  └── unused/ (remaining samples with transcriptions)")
    print(f"\nMetadata JSON files saved in each directory")
    print("\n⚠️  Note: Only samples with transcriptions are included in steering/test splits")


if __name__ == "__main__":
    main()