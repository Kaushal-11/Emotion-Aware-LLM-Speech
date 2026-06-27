import os
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

EMOTIONS = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]

EMOTION_MAP = {
    "happiness": "happiness",
    "anger": "anger",
    "sadness": "sadness",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise"
}

# Minimum samples required to keep an emotion-dataset pair
MIN_SAMPLES_THRESHOLD = 10


# ─────────────────────────────────────────────
# ORGANIZER CLASS
# ─────────────────────────────────────────────

class DatasetOrganizer:
    def __init__(self, root_dir: Path, use_copy: bool = True):
        self.root_dir = Path(root_dir)
        self.use_copy = use_copy  # True = copy, False = symlink
        self.stats = defaultdict(lambda: defaultdict(dict))
        
    def organize_dataset(self, dataset_name: str, scanner_json_path: Path):
        """
        Organize a single dataset into emotion subdirectories.
        """
        print(f"\n{'='*60}")
        print(f"Organizing: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load scanner JSON
        if not scanner_json_path.exists():
            print(f"  [SKIP] Scanner JSON not found: {scanner_json_path}")
            return False
        
        with open(scanner_json_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        # Create dataset directory
        dataset_dir = self.root_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create emotion subdirectories
        for emotion in EMOTIONS:
            emotion_dir = dataset_dir / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
        
        # Organize samples by emotion
        emotion_samples = defaultdict(list)
        
        # Group samples by emotion
        for sample in tqdm(samples, desc=f"  Grouping samples", unit="file"):
            emotion = sample.get('emotion', 'unknown')
            if emotion not in EMOTIONS:
                continue
            
            # Get source audio path
            src_path = Path(sample['audio_path'])
            if not src_path.exists():
                # Try to find relative to root
                alt_path = self.root_dir / sample['audio_path']
                if alt_path.exists():
                    src_path = alt_path
                else:
                    print(f"  [WARN] File not found: {sample['audio_path']}")
                    continue
            
            # Prepare sample metadata
            sample_meta = {
                'audio_path': f"{dataset_name}/{emotion}/{src_path.name}",
                'filename': src_path.name,
                'speaker_id': sample.get('speaker_id', 'unknown'),
                'transcription': sample.get('transcription'),
                'duration_sec': sample.get('duration_sec', 0.0),
            }
            
            # Add dataset-specific fields
            if dataset_name == 'cremad':
                sample_meta['intensity'] = sample.get('intensity')
                sample_meta['sentence_code'] = sample.get('sentence_code')
            elif dataset_name == 'ravdess':
                sample_meta['gender'] = sample.get('gender')
                sample_meta['intensity'] = sample.get('intensity')
            elif dataset_name == 'iemocap':
                sample_meta['session'] = sample.get('session')
                sample_meta['dialog'] = sample.get('dialog')
                sample_meta['conversation_id'] = sample.get('conversation_id')
            elif dataset_name == 'meld':
                sample_meta['split'] = sample.get('split')
                sample_meta['conversation_id'] = sample.get('conversation_id')
                sample_meta['utterance_id'] = sample.get('utterance_id')
            elif dataset_name == 'tess':
                sample_meta['word_spoken'] = sample.get('transcription')
            
            emotion_samples[emotion].append(sample_meta)
            
            # Copy or symlink the file
            dst_path = dataset_dir / emotion / src_path.name
            if not dst_path.exists():
                if self.use_copy:
                    shutil.copy2(src_path, dst_path)
                else:
                    dst_path.symlink_to(src_path)
        
        # Build metadata JSON
        metadata = {
            "dataset_name": dataset_name,
            "total_files": sum(len(v) for v in emotion_samples.values()),
            "total_duration_hours": self._calculate_total_duration(emotion_samples),
            "emotions": {}
        }
        
        # Add emotion data
        for emotion in EMOTIONS:
            samples_list = emotion_samples.get(emotion, [])
            total_duration = sum(s.get('duration_sec', 0.0) for s in samples_list)
            
            metadata["emotions"][emotion] = {
                "count": len(samples_list),
                "duration_seconds": round(total_duration, 2),
                "files": samples_list
            }
        
        # Save metadata.json
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print(f"\n  {'─'*50}")
        print(f"  {dataset_name.upper()} Statistics:")
        print(f"  {'─'*50}")
        for emotion in EMOTIONS:
            count = metadata["emotions"][emotion]["count"]
            if count > 0:
                duration = metadata["emotions"][emotion]["duration_seconds"]
                print(f"    {emotion:<12}: {count:>5} files  ({duration/3600:.2f} hours)")
            else:
                print(f"    {emotion:<12}: {count:>5} files  (no samples)")
        
        print(f"\n  ✓ Metadata saved: {metadata_path}")
        print(f"  ✓ Total files: {metadata['total_files']}")
        
        return True
    
    def _calculate_total_duration(self, emotion_samples):
        """Calculate total duration in hours."""
        total_seconds = 0
        for samples in emotion_samples.values():
            total_seconds += sum(s.get('duration_sec', 0.0) for s in samples)
        return round(total_seconds / 3600, 2)
    
    def create_global_metadata(self):
        """Create a global metadata file combining all datasets."""
        global_metadata = {
            "datasets": {},
            "total_files": 0,
            "total_duration_hours": 0,
            "emotion_summary": {emotion: {"count": 0, "duration_seconds": 0} for emotion in EMOTIONS}
        }
        
        for dataset_name in ['cremad', 'ravdess', 'tess', 'savee', 'meld', 'iemocap']:
            metadata_path = self.root_dir / dataset_name / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                global_metadata["datasets"][dataset_name] = {
                    "total_files": metadata["total_files"],
                    "total_duration_hours": metadata["total_duration_hours"]
                }
                
                global_metadata["total_files"] += metadata["total_files"]
                global_metadata["total_duration_hours"] += metadata["total_duration_hours"]
                
                for emotion in EMOTIONS:
                    stats = metadata["emotions"][emotion]
                    global_metadata["emotion_summary"][emotion]["count"] += stats["count"]
                    global_metadata["emotion_summary"][emotion]["duration_seconds"] += stats["duration_seconds"]
        
        # Save global metadata
        global_metadata_path = self.root_dir / "global_metadata.json"
        with open(global_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(global_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("GLOBAL METADATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {global_metadata['total_files']}")
        print(f"Total duration: {global_metadata['total_duration_hours']:.2f} hours")
        print(f"\nGlobal metadata saved: {global_metadata_path}")
        
        return global_metadata


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Organize emotion datasets into emotion-based directory structure"
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory containing dataset folders (cremad/, ravdess/, etc.)"
    )
    parser.add_argument(
        "--copy", action="store_true", default=True,
        help="Copy files instead of symlinking (default: copy)"
    )
    parser.add_argument(
        "--scanner-output", type=str, default=None,
        help="Directory containing scanner JSON files. Default: <root>/output/"
    )
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    scanner_dir = Path(args.scanner_output).resolve() if args.scanner_output else root_dir / "output"
    
    print(f"\n{'='*60}")
    print("  EMOTIONAL AI - DATASET ORGANIZER")
    print(f"{'='*60}")
    print(f"Root directory:     {root_dir}")
    print(f"Scanner output:     {scanner_dir}")
    print(f"Operation:          {'COPY' if args.copy else 'SYMLINK'}")
    print(f"{'='*60}")
    
    # Check if scanner output exists
    if not scanner_dir.exists():
        print(f"\n[ERROR] Scanner output directory not found: {scanner_dir}")
        print("Please run the scanner first:")
        print(f"  python dataset.py --root {root_dir}")
        return
    
    # Initialize organizer
    organizer = DatasetOrganizer(root_dir, use_copy=args.copy)
    
    # Process each dataset
    datasets = ['cremad', 'ravdess', 'tess', 'savee', 'meld', 'iemocap']
    
    for dataset in datasets:
        scanner_json = scanner_dir / f"{dataset}.json"
        if scanner_json.exists():
            organizer.organize_dataset(dataset, scanner_json)
        else:
            print(f"\n[SKIP] {dataset.upper()} - Scanner JSON not found: {scanner_json}")
    
    # Create global metadata
    organizer.create_global_metadata()
    
    print(f"\n{'='*60}")
    print("  ORGANIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nDirectory structure created at: {root_dir}")
    print("\nEach dataset now has:")
    print("  - Emotion subdirectories (happiness/, anger/, etc.)")
    print("  - metadata.json with complete file information")
    print(f"\nGlobal metadata: {root_dir / 'global_metadata.json'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()