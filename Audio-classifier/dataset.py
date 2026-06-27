import os
import re
import csv
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TARGET_EMOTIONS = {"anger", "sadness", "fear", "happiness", "disgust", "surprise"}

EMOTION_TO_ID = {
    "anger":     0,
    "sadness":   1,
    "fear":      2,
    "happiness": 3,
    "disgust":   4,
    "surprise":  5,
}


# ─────────────────────────────────────────────
# AUDIO DURATION
# ─────────────────────────────────────────────

def get_duration_seconds(wav_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return round(float(out.strip()), 3)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# CREMA-D
# ─────────────────────────────────────────────

CREMAD_EMOTION_MAP = {
    "HAP": "happiness",
    "ANG": "anger",
    "SAD": "sadness",
    "DIS": "disgust",
    "FEA": "fear",
}

CREMAD_SENTENCE_MAP = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}


def scan_cremad(raw_dir: Path) -> list:
    records = []
    src = raw_dir / "cremad"
    if not src.exists():
        print(f"  [SKIP] cremad directory not found: {src}")
        return records

    audio_dir = src / "AudioWAV"
    if not audio_dir.exists():
        audio_dir = src
        for subdir in src.iterdir():
            if subdir.is_dir() and list(subdir.glob("*.wav")):
                audio_dir = subdir
                break

    if not list(audio_dir.glob("*.wav")):
        print(f"  [WARN] No .wav files found in {src}")
        return records

    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 4:
            continue

        speaker_id    = parts[0]
        sentence_code = parts[1]
        emotion_code  = parts[2].upper()
        intensity     = parts[3]

        if emotion_code not in CREMAD_EMOTION_MAP:
            continue

        emotion = CREMAD_EMOTION_MAP[emotion_code]
        records.append({
            "audio_path":      str(wav),
            "audio_name":      wav.name,
            "emotion":         emotion,
            "emotion_id":      EMOTION_TO_ID[emotion],
            "dataset":         "cremad",
            "speaker_id":      speaker_id,
            "transcription":   CREMAD_SENTENCE_MAP.get(sentence_code, None),
            "sentence_code":   sentence_code,
            "intensity":       intensity,
            "duration_sec":    get_duration_seconds(wav),
            "conversation_id": None,
            "utterance_id":    wav.stem,
        })

    print(f"  [CREMA-D] Found {len(records)} matching files.")
    return records


# ─────────────────────────────────────────────
# IEMOCAP
# ─────────────────────────────────────────────

IEMOCAP_EMOTION_MAP = {
    "hap": "happiness",
    "exc": "happiness",
    "ang": "anger",
    "sad": "sadness",
    "dis": "disgust",
    "fea": "fear",
    "sur": "surprise",
}


def parse_iemocap_transcriptions(trans_file: Path) -> dict:
    trans = {}
    if not trans_file.exists():
        return trans
    with open(trans_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\S+)\s+\[[\d.\-]+\]:\s*(.+)$", line)
            if match:
                trans[match.group(1).strip()] = match.group(2).strip()
    return trans


def scan_iemocap(raw_dir: Path) -> list:
    records = []
    src = raw_dir / "iemocap"
    if not src.exists():
        print(f"  [SKIP] iemocap directory not found: {src}")
        return records

    for session_dir in sorted(src.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.lower().startswith("session"):
            continue

        emo_eval_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir    = session_dir / "dialog" / "transcriptions"
        wav_base     = session_dir / "sentences" / "wav"

        if not emo_eval_dir.exists():
            print(f"  [WARN] No EmoEvaluation in {session_dir.name}")
            continue

        for label_file in sorted(emo_eval_dir.glob("*.txt")):
            dialog_name    = label_file.stem
            transcriptions = parse_iemocap_transcriptions(trans_dir / f"{dialog_name}.txt")
            wav_dir        = wav_base / dialog_name

            with open(label_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("["):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    utterance_id = parts[3]
                    emotion_code = parts[4].strip().lower()
                    if emotion_code not in IEMOCAP_EMOTION_MAP:
                        continue

                    emotion  = IEMOCAP_EMOTION_MAP[emotion_code]
                    wav_file = wav_dir / f"{utterance_id}.wav"
                    if not wav_file.exists():
                        matches = list(session_dir.rglob(f"{utterance_id}.wav"))
                        if not matches:
                            continue
                        wav_file = matches[0]

                    spk_char   = utterance_id.split("_")[-1][0]
                    speaker_id = f"{session_dir.name}_{spk_char}"

                    records.append({
                        "audio_path":      str(wav_file),
                        "audio_name":      wav_file.name,
                        "emotion":         emotion,
                        "emotion_id":      EMOTION_TO_ID[emotion],
                        "dataset":         "iemocap",
                        "speaker_id":      speaker_id,
                        "transcription":   transcriptions.get(utterance_id, None),
                        "session":         session_dir.name,
                        "dialog":          dialog_name,
                        "duration_sec":    get_duration_seconds(wav_file),
                        "conversation_id": dialog_name,
                        "utterance_id":    utterance_id,
                    })

    print(f"  [IEMOCAP] Found {len(records)} matching files.")
    return records


# ─────────────────────────────────────────────
# RAVDESS
# ─────────────────────────────────────────────

RAVDESS_EMOTION_MAP = {
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

RAVDESS_STATEMENT_MAP = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}

RAVDESS_INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}


def scan_ravdess(raw_dir: Path) -> list:
    records = []
    src = raw_dir / "ravdess"
    if not src.exists():
        print(f"  [SKIP] ravdess directory not found: {src}")
        return records

    for wav in sorted(src.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) < 7:
            continue

        emotion_code   = parts[2]
        intensity_code = parts[3]
        statement_code = parts[4]
        actor_id       = parts[6]

        if emotion_code not in RAVDESS_EMOTION_MAP:
            continue

        emotion = RAVDESS_EMOTION_MAP[emotion_code]
        gender  = "M" if int(actor_id) % 2 != 0 else "F"

        records.append({
            "audio_path":      str(wav),
            "audio_name":      wav.name,
            "emotion":         emotion,
            "emotion_id":      EMOTION_TO_ID[emotion],
            "dataset":         "ravdess",
            "speaker_id":      f"Actor_{actor_id}",
            "gender":          gender,
            "transcription":   RAVDESS_STATEMENT_MAP.get(statement_code, None),
            "intensity":       RAVDESS_INTENSITY_MAP.get(intensity_code, None),
            "duration_sec":    get_duration_seconds(wav),
            "conversation_id": None,
            "utterance_id":    wav.stem,
        })

    print(f"  [RAVDESS] Found {len(records)} matching files.")
    return records


# ─────────────────────────────────────────────
# SAVEE
# ─────────────────────────────────────────────

SAVEE_EMOTION_MAP = {
    "su": "surprise",
    "sa": "sadness",
    "a":  "anger",
    "d":  "disgust",
    "f":  "fear",
    "h":  "happiness",
}


def scan_savee(raw_dir: Path) -> list:
    records = []
    src = raw_dir / "savee"
    if not src.exists():
        print(f"  [SKIP] savee directory not found: {src}")
        return records

    for wav in sorted(src.rglob("*.wav")):
        stem = wav.stem
        if "_" in stem:
            parts      = stem.split("_", 1)
            speaker_id = parts[0]
            emo_part   = parts[1].lower()
        else:
            speaker_id = "unknown"
            emo_part   = stem.lower()

        emotion = None
        for prefix, emo in SAVEE_EMOTION_MAP.items():
            if emo_part.startswith(prefix):
                emotion = emo
                break

        if emotion is None:
            continue

        records.append({
            "audio_path":      str(wav),
            "audio_name":      wav.name,
            "emotion":         emotion,
            "emotion_id":      EMOTION_TO_ID[emotion],
            "dataset":         "savee",
            "speaker_id":      speaker_id,
            "transcription":   None,
            "duration_sec":    get_duration_seconds(wav),
            "conversation_id": None,
            "utterance_id":    wav.stem,
        })

    print(f"  [SAVEE] Found {len(records)} matching files.")
    return records


# ─────────────────────────────────────────────
# TESS
# ─────────────────────────────────────────────

TESS_EMOTION_MAP = {
    "angry":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happiness",
    "sad":      "sadness",
    "surprise": "surprise",
    "ps":       "surprise",
}


def scan_tess(raw_dir: Path) -> list:
    records = []
    src = raw_dir / "tess"
    if not src.exists():
        print(f"  [SKIP] tess directory not found: {src}")
        return records

    for folder in sorted(src.iterdir()):
        if not folder.is_dir() or "_" not in folder.name.lower():
            continue

        spk_prefix, emotion_key = folder.name.lower().split("_", 1)
        speaker_id = "OAF" if "oaf" in spk_prefix else "YAF"

        if emotion_key not in TESS_EMOTION_MAP:
            continue

        emotion = TESS_EMOTION_MAP[emotion_key]

        for wav in sorted(folder.glob("*.wav")):
            stem_parts  = wav.stem.split("_")
            word_spoken = stem_parts[1] if len(stem_parts) >= 3 else None

            records.append({
                "audio_path":      str(wav),
                "audio_name":      wav.name,
                "emotion":         emotion,
                "emotion_id":      EMOTION_TO_ID[emotion],
                "dataset":         "tess",
                "speaker_id":      speaker_id,
                "transcription":   word_spoken,
                "duration_sec":    get_duration_seconds(wav),
                "conversation_id": None,
                "utterance_id":    wav.stem,
            })

    print(f"  [TESS] Found {len(records)} matching files.")
    return records


# ─────────────────────────────────────────────
# MELD
# ─────────────────────────────────────────────

MELD_EMOTION_MAP = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "happiness",
    "sadness":  "sadness",
    "surprise": "surprise",
}


def scan_meld(raw_dir: Path) -> list:
    records = []
    src = raw_dir / "meld"
    if not src.exists():
        print(f"  [SKIP] meld directory not found: {src}")
        return records

    missing_audio = 0

    split_configs = [
        ("train_sent_emo.csv", "train", "train_splits"),
        ("dev_sent_emo.csv",   "dev",   "dev_splits_complete"),
        ("test_sent_emo.csv",  "test",  "output_repeated_splits_test"),
    ]

    for csv_name, split_name, audio_subdir in split_configs:
        csv_path = src / csv_name
        if not csv_path.exists():
            print(f"  [WARN] {csv_name} not found.")
            continue

        audio_dir = src / split_name / audio_subdir
        if not audio_dir.exists():
            audio_dir = src / split_name
            if not audio_dir.exists():
                print(f"  [WARN] Audio directory not found for {split_name}: {audio_dir}")
                continue

        print(f"  Processing {split_name} split from {csv_name}")
        print(f"    Audio path: {audio_dir}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emotion_raw = row.get("Emotion", "").strip().lower()
                if emotion_raw not in MELD_EMOTION_MAP:
                    continue

                emotion = MELD_EMOTION_MAP[emotion_raw]
                dia_id  = row.get("Dialogue_ID",  row.get("DialogueId",  "")).strip()
                utt_id  = row.get("Utterance_ID", row.get("UtteranceId", "")).strip()
                sr_no   = row.get("Sr No", row.get("Sr_No", row.get("Sr", ""))).strip()

                wav_name_candidates = []
                if dia_id and utt_id:
                    wav_name_candidates += [f"dia{dia_id}_utt{utt_id}.wav",
                                            f"dia{dia_id}_utt{utt_id}.mp4"]
                if sr_no:
                    wav_name_candidates += [f"{sr_no}.wav", f"{sr_no}.mp4"]

                wav_file = None
                for candidate in wav_name_candidates:
                    test_path = audio_dir / candidate
                    if test_path.exists():
                        wav_file = test_path
                        break

                if not wav_file and dia_id and utt_id:
                    for ext in ["*.wav", "*.mp4"]:
                        for p in audio_dir.glob(ext):
                            if f"dia{dia_id}" in p.name and f"utt{utt_id}" in p.name:
                                wav_file = p
                                break
                        if wav_file:
                            break

                if not wav_file and sr_no:
                    for p in audio_dir.glob("*.wav"):
                        if sr_no in p.name:
                            wav_file = p
                            break

                if wav_file is None:
                    missing_audio += 1
                    if missing_audio <= 10:
                        print(f"    [DEBUG] No audio for: Dia={dia_id}, Utt={utt_id}, Sr={sr_no}")
                    continue

                speaker    = row.get("Speaker", "").strip() or f"unknown_{split_name}"
                transcript = row.get("Utterance", row.get("UtteranceText", "")).strip()

                records.append({
                    "audio_path":      str(wav_file),
                    "audio_name":      wav_file.name,
                    "emotion":         emotion,
                    "emotion_id":      EMOTION_TO_ID[emotion],
                    "dataset":         "meld",
                    "speaker_id":      speaker,
                    "transcription":   transcript if transcript else None,
                    "split":           split_name,
                    "duration_sec":    get_duration_seconds(wav_file),
                    "conversation_id": f"dia{dia_id}" if dia_id else None,
                    "utterance_id":    f"utt{utt_id}" if utt_id else sr_no,
                })

    if missing_audio > 0:
        print(f"  [WARN] {missing_audio} MELD rows had no matching audio file.")
    print(f"  [MELD] Found {len(records)} matching files.")
    return records


# ─────────────────────────────────────────────
# EMNS  (ylacombe/emns — HuggingFace)
#
# Structure after snapshot_download:
#   emns/repo_snapshot/
#     data/train-*.parquet          ← metadata rows
#     wavs/recorded_audio_*.webm    ← audio files
#
# Parquet columns of interest:
#   id, utterance, emotion, audio_recording (= "wavs/recorded_audio_XXX.webm")
#   audio  (dict with 'path' key pointing to the webm path inside the repo)
#
# Emotion classes:
#   Angry → anger
#   Happy → happiness
#   Sad   → sadness
#   Disgust → disgust
#   Surprised → surprise
#   Excited → skip  (no direct mapping to our 6)
#   Sarcastic → skip
#   Neutral → skip
# ─────────────────────────────────────────────

EMNS_EMOTION_MAP = {
    "angry":     "anger",
    "angry\n":   "anger",       # guard against trailing whitespace
    "happy":     "happiness",
    "sad":       "sadness",
    "disgust":   "disgust",
    "surprised": "surprise",
    # Excited, Sarcastic, Neutral → skip
}


def scan_emns(raw_dir: Path) -> list:
    """
    Scan the EMNS dataset downloaded via HuggingFace snapshot_download.

    Expected layout:
        <raw_dir>/emns/repo_snapshot/
            data/train-*.parquet
            wavs/recorded_audio_*.webm   (or *.wav after conversion)
    """
    records  = []
    src      = raw_dir / "emns"
    snap_dir = src / "repo_snapshot"

    if not snap_dir.exists():
        print(f"  [SKIP] emns repo_snapshot not found: {snap_dir}")
        print(f"         Run the downloader first (source=huggingface, repo=ylacombe/emns)")
        return records

    # ── locate parquet files ──────────────────────────────────────────────────
    parquet_files = list(snap_dir.rglob("*.parquet"))
    if not parquet_files:
        # Fall back: try loading directly with datasets library
        print("  [WARN] No parquet files found; attempting datasets library fallback...")
        return _scan_emns_via_datasets(src)

    try:
        import pandas as pd
    except ImportError:
        print("  [ERROR] pandas is required to read EMNS parquet files.")
        print("          Install with: pip install pandas pyarrow")
        return records

    rows = []
    for pq in sorted(parquet_files):
        df = pd.read_parquet(pq)
        rows.extend(df.to_dict(orient="records"))

    print(f"  Loaded {len(rows)} rows from {len(parquet_files)} parquet file(s).")

    missing = 0
    for row in rows:
        # Normalise emotion string
        emotion_raw = str(row.get("emotion", "")).strip().lower()
        if emotion_raw not in EMNS_EMOTION_MAP:
            continue

        emotion = EMNS_EMOTION_MAP[emotion_raw]

        # Resolve audio file path
        # 'audio_recording' field = "wavs/recorded_audio_XXX.webm"
        audio_rel = str(row.get("audio_recording", "")).strip()
        if not audio_rel:
            # Fallback: try the nested 'audio' dict
            audio_field = row.get("audio", {})
            if isinstance(audio_field, dict):
                audio_rel = audio_field.get("path", "")

        # The webm files may have been decoded to wav by datasets library;
        # check both extensions.
        audio_file = None
        for candidate_rel in [audio_rel, audio_rel.replace(".webm", ".wav")]:
            candidate = snap_dir / candidate_rel
            if candidate.exists():
                audio_file = candidate
                break

        # Broad search fallback using the filename stem
        if audio_file is None:
            stem = Path(audio_rel).stem
            for ext in [".webm", ".wav", ".flac", ".mp3"]:
                matches = list(snap_dir.rglob(f"{stem}{ext}"))
                if matches:
                    audio_file = matches[0]
                    break

        if audio_file is None:
            missing += 1
            if missing <= 5:
                print(f"    [DEBUG] Audio not found: {audio_rel}")
            continue

        utt_id     = str(row.get("id", Path(audio_rel).stem))
        transcript = str(row.get("utterance", "")).strip() or None

        records.append({
            "audio_path":      str(audio_file),
            "audio_name":      audio_file.name,
            "emotion":         emotion,
            "emotion_id":      EMOTION_TO_ID[emotion],
            "dataset":         "emns",
            "speaker_id":      f"user_{row.get('user_id', 'unknown')}",
            "gender":          str(row.get("gender", "")).strip() or None,
            "transcription":   transcript,
            "emotion_level":   int(row.get("level", 0)),
            "duration_sec":    get_duration_seconds(audio_file),
            "conversation_id": None,
            "utterance_id":    utt_id,
        })

    if missing:
        print(f"  [WARN] {missing} EMNS rows had no matching audio file.")
    print(f"  [EMNS] Found {len(records)} matching files.")
    return records


def _scan_emns_via_datasets(src: Path) -> list:
    """Fallback: load EMNS via the HuggingFace datasets library."""
    records = []
    try:
        from datasets import load_dataset
        ds = load_dataset("ylacombe/emns", split="train")
    except Exception as e:
        print(f"  [ERROR] Could not load EMNS via datasets library: {e}")
        return records

    missing = 0
    for row in ds:
        emotion_raw = str(row.get("emotion", "")).strip().lower()
        if emotion_raw not in EMNS_EMOTION_MAP:
            continue

        emotion = EMNS_EMOTION_MAP[emotion_raw]

        audio_field = row.get("audio", {})
        if isinstance(audio_field, dict):
            audio_path_str = audio_field.get("path", "")
        else:
            audio_path_str = str(audio_field)

        audio_file = Path(audio_path_str) if audio_path_str else None
        if audio_file is None or not audio_file.exists():
            missing += 1
            continue

        records.append({
            "audio_path":      str(audio_file),
            "audio_name":      audio_file.name,
            "emotion":         emotion,
            "emotion_id":      EMOTION_TO_ID[emotion],
            "dataset":         "emns",
            "speaker_id":      f"user_{row.get('user_id', 'unknown')}",
            "gender":          str(row.get("gender", "")).strip() or None,
            "transcription":   str(row.get("utterance", "")).strip() or None,
            "emotion_level":   int(row.get("level", 0)),
            "duration_sec":    get_duration_seconds(audio_file),
            "conversation_id": None,
            "utterance_id":    str(row.get("id", audio_file.stem)),
        })

    if missing:
        print(f"  [WARN] {missing} EMNS rows had no audio path.")
    print(f"  [EMNS] Found {len(records)} matching files (via datasets library).")
    return records


# ─────────────────────────────────────────────
# BEAT  (H-Liu1997/BEAT — HuggingFace)
#
# Structure after snapshot_download:
#   beat/repo_snapshot/
#     data/                          ← audio files (.wav, 27-828 s each)
#     metadata.jsonl  (or)
#     refs/convert/parquet/default/  ← auto-converted parquet (label column)
#
# HF viewer shows:
#   audio  (AudioFile)   | label (class label, 22 classes but displayed as "01" etc.)
#
# The label column values on HF correspond to the BEAT emotion CSV per-file
# majority vote; but the dataset card does NOT document the mapping explicitly.
#
# From the official BEAT paper / GitHub:
#   .csv  emotion labels 0-7 → neutral, happiness, anger, sadness,
#                               contempt, surprise, fear, disgust
#
# The HF dataset "label" field stores the sequence ID string, NOT an emotion.
# The actual emotion is encoded in the filename:
#   speaker_<seq_type>_<seq_id>_<start>_<end>.wav
#   seq_id for English speech:
#       0-64   → neutral      (skip)
#       65-72  → happiness
#       73-80  → anger
#       81-86  → sadness
#       87-94  → contempt     (skip)
#       95-102 → surprise
#       103-110→ fear
#       111-118→ disgust
#
# Alternatively, alongside each .wav there is a .csv file with frame-level
# emotion labels (col 0 = time, col 1 = label 0-7).  We take the majority
# label across all frames as the clip emotion.
# ─────────────────────────────────────────────

# Map from 0-7 BEAT emotion index to our 6 targets (None = skip)
BEAT_LABEL_MAP = {
    0: None,          # neutral  → skip
    1: "happiness",
    2: "anger",
    3: "sadness",
    4: None,          # contempt → skip (no mapping in our 6)
    5: "surprise",
    6: "fear",
    7: "disgust",
}

# Sequence-ID range → emotion (for filename-based fallback)
BEAT_SEQID_RANGES = [
    (range(0,   65),  None),          # neutral
    (range(65,  73),  "happiness"),
    (range(73,  81),  "anger"),
    (range(81,  87),  "sadness"),
    (range(87,  95),  None),          # contempt
    (range(95,  103), "surprise"),
    (range(103, 111), "fear"),
    (range(111, 119), "disgust"),
]


def _beat_emotion_from_seqid(seq_id: int):
    """Return emotion string (or None) based on BEAT speech sequence ID."""
    for r, emo in BEAT_SEQID_RANGES:
        if seq_id in r:
            return emo
    return None


def _beat_majority_emotion_from_csv(csv_path: Path):
    """
    Read a BEAT emotion CSV file (frame-level labels) and return the majority
    emotion label as a string, or None if the file can't be parsed.

    CSV format:  time_sec, label_0_to_7  (no header)
    """
    counts = defaultdict(int)
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    label = int(float(row[1]))
                except (IndexError, ValueError):
                    continue
                counts[label] += 1
    except Exception:
        return None

    if not counts:
        return None

    majority_label = max(counts, key=counts.__getitem__)
    return BEAT_LABEL_MAP.get(majority_label, None)

def scan_beat(raw_dir: Path) -> list:
    """
    Scan BEAT dataset - reads emotion labels from CSV files.
    Each WAV has a matching CSV with emotion labels.
    """
    records = []
    src = raw_dir / "beat"
    snap_dir = src / "repo_snapshot"
    
    if not snap_dir.exists():
        print(f"  [SKIP] beat repo_snapshot not found: {snap_dir}")
        return records
    
    # Find all CSV files (each corresponds to a WAV file)
    csv_files = list(snap_dir.rglob("*.csv"))
    
    # Filter out any CSV files that are too large or not emotion-related
    emotion_csvs = []
    for csv_file in csv_files:
        # CSV files are typically small (few KB) compared to WAV files (many MB)
        if csv_file.stat().st_size < 100 * 1024:  # Less than 100KB
            emotion_csvs.append(csv_file)
    
    print(f"  Found {len(csv_files)} total CSV files, {len(emotion_csvs)} potential emotion CSV files")
    
    if not emotion_csvs:
        print("  [WARN] No emotion CSV files found")
        return records
    
    # Map WAV files to their CSV files
    wav_to_csv = {}
    for csv_file in emotion_csvs:
        wav_file = csv_file.with_suffix(".wav")
        if wav_file.exists():
            wav_to_csv[wav_file] = csv_file
    
    print(f"  Found {len(wav_to_csv)} WAV files with matching CSV files")
    
    # Show sample for debugging
    if wav_to_csv:
        sample_wav = list(wav_to_csv.keys())[0]
        print(f"  Sample file: {sample_wav.name}")
        print(f"  Sample CSV: {wav_to_csv[sample_wav].name}")
    
    matched = 0
    skipped_no_emotion = 0
    skipped_no_csv = 0
    
    for wav_file, csv_file in wav_to_csv.items():
        duration = get_duration_seconds(wav_file)
        
        # Extract emotion from CSV file
        emotion = _extract_emotion_from_beat_csv(csv_file)
        
        if emotion is None:
            skipped_no_emotion += 1
            continue
        
        # Extract speaker ID from path or filename
        # Path format: .../1/1_wayne_0_1_1.wav
        parts = wav_file.parts
        speaker_id = "unknown"
        
        # Try to get speaker from filename (e.g., "wayne" in "1_wayne_0_1_1.wav")
        filename_parts = wav_file.stem.split("_")
        if len(filename_parts) >= 2:
            speaker_id = filename_parts[1]  # "wayne"
        
        matched += 1
        records.append({
            "audio_path": str(wav_file),
            "audio_name": wav_file.name,
            "emotion": emotion,
            "emotion_id": EMOTION_TO_ID[emotion],
            "dataset": "beat",
            "speaker_id": speaker_id,
            "transcription": _extract_transcript_from_beat_files(wav_file),
            "duration_sec": duration,
            "conversation_id": None,
            "utterance_id": wav_file.stem,
        })
    
    print(f"  [BEAT] Matched: {matched}, Skipped (no emotion): {skipped_no_emotion}")
    print(f"  [BEAT] Found {len(records)} matching files.")
    return records


def _extract_emotion_from_beat_csv(csv_path: Path) -> str:
    """
    Extract majority emotion from BEAT CSV file.
    
    BEAT CSV format (frame-level labels):
        time, emotion_label
        0.0, 1
        0.1, 1
        ...
    
    Emotion labels (0-7):
        0 = neutral (skip)
        1 = happiness
        2 = anger
        3 = sadness
        4 = contempt (skip)
        5 = surprise
        6 = fear
        7 = disgust
    """
    try:
        import csv
        
        emotion_counts = defaultdict(int)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    # CSV may have 2 columns: time, label
                    # or just one column with label
                    if len(row) >= 2:
                        label = int(float(row[1]))
                    else:
                        label = int(float(row[0]))
                    
                    emotion_counts[label] += 1
                except (ValueError, IndexError):
                    continue
        
        if not emotion_counts:
            return None
        
        # Get majority emotion
        majority_label = max(emotion_counts, key=emotion_counts.get)
        
        # Map BEAT label to target emotion
        BEAT_TO_EMOTION = {
            1: "happiness",
            2: "anger",
            3: "sadness",
            5: "surprise",
            6: "fear",
            7: "disgust",
        }
        
        return BEAT_TO_EMOTION.get(majority_label, None)
        
    except Exception as e:
        return None


def _extract_transcript_from_beat_files(wav_path: Path) -> str:
    """
    Extract transcript from accompanying .txt or .TextGrid file.
    """
    # Try .txt file first
    txt_file = wav_path.with_suffix(".txt")
    if txt_file.exists():
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return content[:500]  # Limit length
        except:
            pass
    
    # Try .TextGrid file (Praat format) - extract text from intervals
    textgrid_file = wav_path.with_suffix(".TextGrid")
    if textgrid_file.exists():
        try:
            with open(textgrid_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple extraction of text between quotes
                import re
                matches = re.findall(r'"([^"]*)"', content)
                if matches:
                    # Join non-empty texts
                    texts = [m for m in matches if m.strip() and not m[0].isdigit()]
                    if texts:
                        return " ".join(texts[:3])[:500]
        except:
            pass
    
    return None


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

def seconds_to_hms(total_seconds: float) -> str:
    total_seconds = int(total_seconds)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h}h {m}m {s}s"


def build_summary(all_records: dict) -> dict:
    emotion_stats  = {e: {"count": 0, "duration_sec": 0.0,
                          "min_duration_sec": None, "max_duration_sec": None}
                      for e in TARGET_EMOTIONS}
    dataset_stats  = {}

    for dataset, records in all_records.items():
        dataset_stats[dataset] = {
            "total_count":        0,
            "total_duration_sec": 0.0,
            "per_emotion": {
                e: {"count": 0, "duration_sec": 0.0,
                    "min_duration_sec": None, "max_duration_sec": None}
                for e in TARGET_EMOTIONS
            },
        }

        for r in records:
            emo = r["emotion"]
            dur = r.get("duration_sec", 0.0) or 0.0

            # Global emotion stats
            emotion_stats[emo]["count"]        += 1
            emotion_stats[emo]["duration_sec"] += dur
            if emotion_stats[emo]["min_duration_sec"] is None or dur < emotion_stats[emo]["min_duration_sec"]:
                emotion_stats[emo]["min_duration_sec"] = dur
            if emotion_stats[emo]["max_duration_sec"] is None or dur > emotion_stats[emo]["max_duration_sec"]:
                emotion_stats[emo]["max_duration_sec"] = dur

            # Per-dataset stats
            ds = dataset_stats[dataset]
            ds["total_count"]        += 1
            ds["total_duration_sec"] += dur
            ds["per_emotion"][emo]["count"]        += 1
            ds["per_emotion"][emo]["duration_sec"] += dur
            pe = ds["per_emotion"][emo]
            if pe["min_duration_sec"] is None or dur < pe["min_duration_sec"]:
                pe["min_duration_sec"] = dur
            if pe["max_duration_sec"] is None or dur > pe["max_duration_sec"]:
                pe["max_duration_sec"] = dur

    # Add human-readable fields
    for emo in emotion_stats:
        s = emotion_stats[emo]
        s["duration_hms"]     = seconds_to_hms(s["duration_sec"])
        s["min_duration_sec"] = round(s["min_duration_sec"] or 0.0, 3)
        s["max_duration_sec"] = round(s["max_duration_sec"] or 0.0, 3)

    for ds_name in dataset_stats:
        ds = dataset_stats[ds_name]
        ds["total_duration_hms"] = seconds_to_hms(ds["total_duration_sec"])
        for emo in ds["per_emotion"]:
            pe = ds["per_emotion"][emo]
            pe["duration_hms"]     = seconds_to_hms(pe["duration_sec"])
            pe["min_duration_sec"] = round(pe["min_duration_sec"] or 0.0, 3)
            pe["max_duration_sec"] = round(pe["max_duration_sec"] or 0.0, 3)

    total_count = sum(s["count"] for s in emotion_stats.values())
    total_dur   = sum(s["duration_sec"] for s in emotion_stats.values())

    return {
        "total_files":        total_count,
        "total_duration_hms": seconds_to_hms(total_dur),
        "total_duration_sec": round(total_dur, 2),
        "per_emotion":        emotion_stats,
        "per_dataset":        dataset_stats,
    }


def print_summary(summary: dict):
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  DATASET SCAN SUMMARY")
    print(SEP)
    print(f"  Total files    : {summary['total_files']}")
    print(f"  Total duration : {summary['total_duration_hms']}")

    print(f"\n  {'─'*68}")
    print(f"  PER EMOTION (global across all datasets)")
    print(f"  {'─'*68}")
    print(f"  {'Emotion':<12}  {'Files':>6}  {'Duration':<14}  {'Min(s)':>8}  {'Max(s)':>8}  Bar")
    print(f"  {'─'*68}")
    for emo in TARGET_EMOTIONS:
        s   = summary["per_emotion"][emo]
        bar = "█" * min(30, s["count"] // 30)
        print(f"  {emo:<12}  {s['count']:>6}  {s['duration_hms']:<14}  "
              f"{s['min_duration_sec']:>8.2f}  {s['max_duration_sec']:>8.2f}  {bar}")

    print(f"\n  {'─'*68}")
    print(f"  PER DATASET")
    print(f"  {'─'*68}")
    for ds, stats in summary["per_dataset"].items():
        print(f"\n  [{ds.upper()}]  {stats['total_count']} files  |  {stats['total_duration_hms']}")
        print(f"    {'Emotion':<12}  {'Files':>5}  {'Duration':<14}  {'Min(s)':>8}  {'Max(s)':>8}")
        for emo in TARGET_EMOTIONS:
            e = stats["per_emotion"][emo]
            if e["count"] > 0:
                print(f"    {emo:<12}  {e['count']:>5}  {e['duration_hms']:<14}  "
                      f"{e['min_duration_sec']:>8.2f}  {e['max_duration_sec']:>8.2f}")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan emotion datasets, build JSON files and print summary."
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory containing dataset subfolders (cremad/, iemocap/, emns/, beat/, ...)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for JSON files. Default: <root>/output/"
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="Comma-separated list or 'all'. Options: cremad,iemocap,ravdess,savee,tess,meld,emns,beat"
    )
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    output_dir = Path(args.output).resolve() if args.output else root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scanners = ["cremad", "iemocap", "ravdess", "savee", "tess", "meld", "emns", "beat"]

    if args.datasets.strip().lower() == "all":
        to_scan = all_scanners
    else:
        to_scan = [d.strip().lower() for d in args.datasets.split(",")]

    scanners = {
        "cremad":  scan_cremad,
        "iemocap": scan_iemocap,
        "ravdess": scan_ravdess,
        "savee":   scan_savee,
        "tess":    scan_tess,
        "meld":    scan_meld,
        "emns":    scan_emns,
        "beat":    scan_beat,
    }

    all_records = {}

    for ds in to_scan:
        if ds not in scanners:
            print(f"[WARN] Unknown dataset '{ds}', skipping.")
            continue

        print(f"\n[Scanning {ds.upper()} ...]")
        records = scanners[ds](root)
        all_records[ds] = records

        if records:
            out_path = output_dir / f"{ds}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  → Saved {len(records)} records to {out_path}")

    summary      = build_summary(all_records)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()

