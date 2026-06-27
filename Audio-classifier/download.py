import os
import sys
import shutil
import zipfile
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm  # You may need to install: pip install tqdm


ROOT_DATA_DIR = "/workspace/audio-em/raw-data/"
DATASETS_TO_DOWNLOAD = ["emns", "beat"]


DATASETS = {
    "cremad": {
        "kaggle_id": "ejlok1/cremad",
        "target":    "cremad",
        "note":      "CREMA-D — 7,442 clips, 91 actors, 6 emotions (no surprise)",
        "source":    "kaggle",
    },
    "ravdess": {
        "kaggle_id": "uwrfkaggler/ravdess-emotional-speech-audio",
        "target":    "ravdess",
        "note":      "RAVDESS — 1,440 clips, 24 actors, 8 emotions",
        "source":    "kaggle",
    },
    "tess": {
        "kaggle_id": "ejlok1/toronto-emotional-speech-set-tess",
        "target":    "tess",
        "note":      "TESS — 2,800 clips, 2 female speakers, 7 emotions",
        "source":    "kaggle",
    },
    "savee": {
        "kaggle_id": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
        "target":    "savee",
        "note":      "SAVEE — 480 clips, 4 male speakers, 7 emotions",
        "source":    "kaggle",
    },
    "meld": {
        "kaggle_id": "zaber666/meld-dataset",
        "target":    "meld",
        "note":      "MELD — 13,000+ clips from Friends TV show, 7 emotions",
        "source":    "kaggle",
        "only_raw":  True,
        "raw_subdir": "MELD-RAW",
    },
    "iemocap": {
        "kaggle_id": "sangayb/iemocap",
        "target":    "iemocap",
        "note":      "IEMOCAP — 10,300 clips, 10 actors, scripted + spontaneous",
        "source":    "kaggle",
    },
    # ── HuggingFace datasets ──────────────────────────────────────────────────
    "emns": {
        "hf_repo":  "ylacombe/emns",
        "target":   "emns",
        "note":     "EMNS — 1,181 clips, 1 female speaker, 8 emotions (angry/happy/sad/disgust/surprised/excited/sarcastic/neutral)",
        "source":   "huggingface",
    },
    "beat": {
        "hf_repo":  "H-Liu1997/BEAT",
        "target":   "beat",
        "note":     "BEAT — 1,100 long audio clips, 30 speakers, 8 emotions (76 h total, frame-level CSV labels)",
        "source":   "huggingface",
    },
}


# ─────────────────────────────────────────────
# HELPERS — Kaggle
# ─────────────────────────────────────────────

def check_kaggle_cli():
    """Make sure the kaggle CLI is available."""
    if shutil.which("kaggle") is None:
        print("[ERROR] kaggle CLI not found.")
        print("        Install it with:  pip install kaggle")
        sys.exit(1)

    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        print("[ERROR] Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print()
        print("  To fix this:")
        print("  1. Go to https://www.kaggle.com → Account → Create New API Token")
        print("  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  3. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    print("[OK] kaggle CLI and credentials found.")


def download_meld_raw_only(dest_dir: Path) -> bool:
    """Download only the MELD-RAW directory from the MELD dataset."""
    import tarfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading MELD-RAW only (~1-2 GB instead of 12 GB)...")

    raw_url    = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
    output_file = dest_dir / "MELD.Raw.tar.gz"

    try:
        print(f"  Downloading from: {raw_url}")
        response = requests.get(raw_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(output_file, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="  Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print("  Extracting...")
        with tarfile.open(output_file, "r:gz") as tar:
            tar.extractall(dest_dir)

        output_file.unlink()
        print(f"  [OK] MELD-RAW downloaded and extracted → {dest_dir}")
        return True

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print("  Trying alternative method with Kaggle API...")
        return download_meld_raw_via_kaggle_api(dest_dir)


def download_meld_raw_via_kaggle_api(dest_dir: Path) -> bool:
    """Alternative: Download only MELD-RAW using Kaggle API with file filtering."""
    print("  Listing dataset files...")
    list_cmd = ["kaggle", "datasets", "files", "zaber666/meld-dataset"]
    result   = subprocess.run(list_cmd, capture_output=True, text=True)

    raw_files = []
    for line in result.stdout.split("\n"):
        if "MELD-RAW" in line or "MELD.Raw" in line.lower():
            parts = line.split()
            if parts:
                raw_files.append(parts[0])

    if not raw_files:
        print("  [WARN] Could not find MELD-RAW files in dataset listing")
        print("  Downloading full dataset as fallback...")
        return download_dataset_fallback("zaber666/meld-dataset", dest_dir)

    success = True
    for raw_file in raw_files:
        print(f"  Downloading {raw_file}...")
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "zaber666/meld-dataset",
            "-f", raw_file,
            "-p", str(dest_dir),
            "--unzip",
        ]
        if subprocess.run(cmd).returncode != 0:
            print(f"  [ERROR] Failed to download {raw_file}")
            success = False

    return success


def download_dataset_fallback(kaggle_id: str, dest_dir: Path) -> bool:
    """Fallback: Download full Kaggle dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading full {kaggle_id} ...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", kaggle_id,
        "-p", str(dest_dir),
        "--unzip",
    ]
    if subprocess.run(cmd).returncode != 0:
        print(f"  [ERROR] Download failed for {kaggle_id}")
        return False
    print(f"  [OK] Downloaded and extracted → {dest_dir}")
    return True


def download_dataset(kaggle_id: str, dest_dir: Path,
                     only_raw: bool = False, raw_subdir: str = None) -> bool:
    """Download and unzip a Kaggle dataset into dest_dir."""
    if only_raw and raw_subdir:
        return download_meld_raw_only(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading {kaggle_id} ...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", kaggle_id,
        "-p", str(dest_dir),
        "--unzip",
    ]
    if subprocess.run(cmd).returncode != 0:
        print(f"  [ERROR] Download failed for {kaggle_id}")
        return False
    print(f"  [OK] Downloaded and extracted → {dest_dir}")
    return True


# ─────────────────────────────────────────────
# HELPERS — HuggingFace
# ─────────────────────────────────────────────

def check_huggingface_cli():
    """Ensure huggingface_hub is importable (pip install huggingface_hub)."""
    try:
        import huggingface_hub  # noqa: F401
        print("[OK] huggingface_hub found.")
        return True
    except ImportError:
        print("[ERROR] huggingface_hub not found.")
        print("        Install it with:  pip install huggingface_hub")
        return False


def download_hf_dataset(hf_repo: str, dest_dir: Path) -> bool:
    """
    Download a HuggingFace dataset using the datasets library.

    For EMNS  (ylacombe/emns):
        Parquet + audio files stored inside the repo.
        We snapshot the whole repo and then export audio from the parquet.

    For BEAT  (H-Liu1997/BEAT):
        Audio folder dataset — snapshot downloads everything including wav files
        and the auto-converted parquet (which carries emotion label per row).

    After download the raw files sit in dest_dir/repo_snapshot/.
    The scanner script reads directly from that path.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = dest_dir / "repo_snapshot"

    print(f"\n  Downloading HuggingFace repo: {hf_repo}")
    print(f"  Target: {snapshot_dir}")
    print("  (This may take a while for large datasets like BEAT ~84 GB)")

    try:
        local_path = snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=str(snapshot_dir),
            local_dir_use_symlinks=False,   # copy files, no symlinks
        )
        print(f"  [OK] Snapshot downloaded → {local_path}")
        return True

    except Exception as e:
        print(f"  [ERROR] HuggingFace download failed: {e}")
        print("  Tip: If the dataset is gated, run `huggingface-cli login` first.")
        return False


# ─────────────────────────────────────────────
# VERIFICATION / UTILS
# ─────────────────────────────────────────────

def print_disk_usage(path: Path):
    try:
        result = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
        size = result.stdout.split()[0] if result.stdout else "unknown"
        print(f"  Disk usage: {size}")
    except Exception:
        pass


def verify_download(target_dir: Path, dataset_key: str) -> bool:
    """Basic sanity check — make sure audio files exist after download."""
    audio_extensions = ["*.wav", "*.mp4", "*.webm", "*.flac", "*.mp3", "*.ogg"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(target_dir.rglob(ext))

    # For HF datasets we also accept parquet files as sign of success
    parquet_files = list(target_dir.rglob("*.parquet"))

    if len(audio_files) == 0 and len(parquet_files) == 0:
        print(f"  [WARN] No audio or parquet files found in {target_dir}")
        print(f"         The download may have failed or the dataset structure is different.")
        return False

    if audio_files:
        print(f"  [VERIFY] Found {len(audio_files)} audio files.")
    if parquet_files:
        print(f"  [VERIFY] Found {len(parquet_files)} parquet files.")
    return True


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 50)
    print("  EMOTIONAL AI - DATASET DOWNLOADER")
    print("=" * 50)

    root    = Path(ROOT_DATA_DIR).resolve()
    raw_dir = root / "raw"

    # Decide which datasets to download
    if DATASETS_TO_DOWNLOAD == "all":
        to_download = list(DATASETS.keys())
    elif isinstance(DATASETS_TO_DOWNLOAD, list):
        to_download = [d.lower() for d in DATASETS_TO_DOWNLOAD]
    else:
        print(f"[ERROR] DATASETS_TO_DOWNLOAD must be 'all' or a list, got {type(DATASETS_TO_DOWNLOAD)}")
        sys.exit(1)

    invalid = [d for d in to_download if d not in DATASETS]
    if invalid:
        print(f"[ERROR] Unknown dataset(s): {', '.join(invalid)}")
        print(f"        Valid options: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    # Determine which backends are needed
    needs_kaggle = any(DATASETS[d]["source"] == "kaggle"       for d in to_download)
    needs_hf     = any(DATASETS[d]["source"] == "huggingface"  for d in to_download)

    if needs_kaggle:
        check_kaggle_cli()
    if needs_hf:
        if not check_huggingface_cli():
            sys.exit(1)

    print(f"\nRoot directory : {root}")
    print(f"Raw directory  : {raw_dir}")
    print(f"Datasets       : {', '.join(to_download)}")
    print()

    results = {}

    for key in to_download:
        info       = DATASETS[key]
        source     = info["source"]
        target_dir = raw_dir / info["target"]

        print(f"{'─' * 50}")
        print(f"[{key.upper()}] {info['note']}")

        # Skip if already downloaded
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"  [SKIP] Already exists: {target_dir}")
            print(f"         Delete the folder to re-download.")
            results[key] = "skipped"
            continue

        # ── Kaggle ────────────────────────────────────────────────────────────
        if source == "kaggle":
            only_raw   = info.get("only_raw",   False)
            raw_subdir = info.get("raw_subdir", None)
            success    = download_dataset(info["kaggle_id"], target_dir, only_raw, raw_subdir)

        # ── HuggingFace ───────────────────────────────────────────────────────
        elif source == "huggingface":
            success = download_hf_dataset(info["hf_repo"], target_dir)

        else:
            print(f"  [ERROR] Unknown source '{source}' for {key}")
            success = False

        if success:
            verify_download(target_dir, key)
            print_disk_usage(target_dir)
            results[key] = "success"
        else:
            results[key] = "failed"

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'=' * 50}")
    for key, status in results.items():
        icon = "✓" if status == "success" else ("↷" if status == "skipped" else "✗")
        print(f"  {icon}  {key:<12}  {status}")

    failed = [k for k, v in results.items() if v == "failed"]
    if failed:
        print(f"\n[WARN] {len(failed)} dataset(s) failed: {', '.join(failed)}")
        print("       Check your credentials and dataset slugs above.")
        print("       You can retry only failed ones by modifying DATASETS_TO_DOWNLOAD.")
    else:
        print(f"\n[DONE] All datasets downloaded to: {raw_dir}")
        print(f"\nNext step — run the extraction script:")
        print(f"  python prepare_emotion_dataset.py --root {root}")

    print(f"{'=' * 50}")


if __name__ == "__main__":
    import tarfile  # noqa: F401 (needed for MELD)
    main()