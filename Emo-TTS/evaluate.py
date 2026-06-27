"""
evaluate.py  —  COMPLETE EVALUATION SUITE
=====================================================
Models tested : F5-TTS  |  CosyVoice2
Task          : Emotion-steered TTS (neutral → angry / happy / sad / etc.)
"""

import os
import json
import csv
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  CRITICAL: Set GPU before anything else
# ══════════════════════════════════════════════════════════════════════════════
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 (12GB)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  edit only this section
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "f5tts"        # "f5tts"  or  "cosyvoice2"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR          = "/workspace/audio-em"
INFERENCE_DIR     = os.path.join(BASE_DIR, "emo-tts", "results", "inference-test", MODEL_NAME)
ANCHOR_DIR        = os.path.join(BASE_DIR, "emo-tts", "data", "used", "steering")
NEUTRAL_JSON      = os.path.join(BASE_DIR, "emo-tts", "data", "used", "test", "neutral_test.json")
DATASET_BASE_DIR  = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR       = os.path.join(BASE_DIR, "emo-tts", "results", "evaluation", MODEL_NAME)

EMOTIONS          = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]
ALPHA_VALUES      = [0.5, 1.0, 1.5, 2.0]
SATURATION_ALPHAS = [0, 0.5, 1.0, 1.5, 2.0]

MAX_SAMPLES          = 300   # all samples
SATURATION_N_SAMPLES = 50    # smaller set for the fine-grained saturation curve
ANCHOR_PER_EMO       = 100   # reference recordings per emotion for E-SIM anchor

# ══════════════════════════════════════════════════════════════════════════════
#  STEP BANNER HELPER
# ══════════════════════════════════════════════════════════════════════════════

def banner(step: int, title: str, description: str = ""):
    print("\n" + "═" * 70)
    print(f"  STEP {step}  │  {title}")
    if description:
        print(f"           │  {description}")
    print("═" * 70)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD TRANSCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════════

def step1_load_transcriptions(json_path: str) -> Dict[str, str]:
    banner(1, "Load transcriptions", f"Reading ground-truth text from {json_path}")

    if not os.path.exists(json_path):
        print(f"  [WARN] JSON not found: {json_path} — WER will be skipped")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", data.get("files", []))
    mapping = {}
    for item in samples:
        name = item.get("audio_name", "").replace(".wav", "").strip()
        text = item.get("transcription", "").strip()
        if name and text:
            mapping[name] = text

    print(f"  Loaded {len(mapping)} reference transcriptions")
    return mapping

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — COLLECT AUDIO FILE PAIRS
# ══════════════════════════════════════════════════════════════════════════════

def step2_collect_pairs(inference_dir: str, emotion: str, alpha: float,
                        max_n: int = MAX_SAMPLES) -> List[Tuple[str, str]]:
    gen_dir  = os.path.join(inference_dir, f"alpha-{alpha}", emotion)
    base_dir = os.path.join(inference_dir, "baseline")

    if not os.path.exists(gen_dir):
        return []

    pairs = []
    missing_base = 0
    for wav in sorted(Path(gen_dir).glob("*.wav"))[:max_n]:
        base_wav = os.path.join(base_dir, wav.name)
        if os.path.exists(base_wav):
            pairs.append((str(wav), base_wav))
        else:
            missing_base += 1

    if missing_base:
        print(f"    [WARN] {missing_base} generated files had no baseline match")

    return pairs

# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION LABEL MAPPING (for SenseVoice)
# ══════════════════════════════════════════════════════════════════════════════

_LABEL_MAP = {
    "anger":     ["ang", "angry", "anger"],
    "happiness": ["hap", "happy", "happiness", "joy", "excited"],
    "sadness":   ["sad", "sadness", "depressed"],
    "disgust":   ["dis", "disgust", "disgusted"],
    "fear":      ["fea", "fear", "fearful", "scared", "anxious"],
    "surprise":  ["sur", "surprise", "surprised"],
    "neutral":   ["neu", "neutral", "calm"],
}

def map_to_emotion(label: str) -> str:
    label_lower = label.lower()
    for emotion, aliases in _LABEL_MAP.items():
        if any(alias in label_lower for alias in aliases):
            return emotion
    return "unknown"

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — WER (Whisper)
# ══════════════════════════════════════════════════════════════════════════════

class WERScorer:
    def __init__(self, device=DEVICE):
        self.model = None
        try:
            import whisper
            self.model = whisper.load_model("large-v3", device=device)
            print("  [WER] Loaded Whisper large-v3")
        except Exception as e:
            print(f"  [WER] Whisper not available: {e}  — WER will be -1")

    def _transcribe(self, path: str) -> str:
        if self.model is None:
            return ""
        try:
            return self.model.transcribe(path, language="en")["text"].strip()
        except Exception:
            return ""

    @staticmethod
    def _edit_distance_wer(ref: str, hyp: str) -> float:
        ref_w = ref.lower().split()
        hyp_w = hyp.lower().split()
        if not ref_w:
            return 0.0
        m, n = len(ref_w), len(hyp_w)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                tmp = dp[j]
                if ref_w[i - 1] == hyp_w[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = tmp
        return dp[n] / len(ref_w)

    def score(self, gen_paths: List[str], ref_texts: List[str]) -> Dict:
        valid_pairs = [(p, r) for p, r in zip(gen_paths, ref_texts) if r.strip()]
        if not valid_pairs:
            return {"wer_mean": -1.0, "wer_std": 0.0, "n": 0}

        wers = []
        for path, ref in tqdm(valid_pairs, desc="    WER transcribing"):
            hyp = self._transcribe(path)
            if hyp:
                wers.append(self._edit_distance_wer(ref, hyp))

        return {
            "wer_mean": float(np.mean(wers) * 100) if wers else -1.0,
            "wer_std":  float(np.std(wers)  * 100) if wers else 0.0,
            "n":        len(wers),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — S-SIM (SpeechBrain only - removed Pyannote)
# ══════════════════════════════════════════════════════════════════════════════

class SSIMScorer:
    def __init__(self, device=DEVICE):
        self.model = None
        try:
            from speechbrain.pretrained import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device})
            print("  [S-SIM] Loaded SpeechBrain ECAPA-TDNN")
        except Exception as e:
            print(f"  [S-SIM] No speaker encoder available: {e}  — S-SIM will be -1")

    def _embed(self, path: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            import torchaudio
            sig, sr = torchaudio.load(path)
            if sr != 16000:
                sig = torchaudio.functional.resample(sig, sr, 16000)
            return self.model.encode_batch(sig).squeeze().cpu().numpy()
        except Exception:
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def score(self, gen_paths: List[str], ref_paths: List[str]) -> Dict:
        sims = []
        for g, r in tqdm(zip(gen_paths, ref_paths), desc="    S-SIM embedding",
                         total=len(gen_paths)):
            eg, er = self._embed(g), self._embed(r)
            if eg is not None and er is not None:
                sims.append(self._cosine(eg, er))

        return {
            "ssim_mean": float(np.mean(sims)) if sims else -1.0,
            "ssim_std":  float(np.std(sims))  if sims else 0.0,
            "n":         len(sims),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — UTMOSv2 (Naturalness) - FIXED
# ══════════════════════════════════════════════════════════════════════════════

class UTMOSScorer:
    def __init__(self, device=DEVICE):
        self.model = None
        try:
            import utmosv2
            self.model = utmosv2.create_model(pretrained=True)
            print("  [UTMOS] Loaded UTMOSv2 naturalness scorer")
        except ImportError:
            try:
                import utmos
                self.model = utmos.Score(device="cpu")
                print("  [UTMOS] Loaded UTMOS (legacy) naturalness scorer")
            except Exception as e:
                print(f"  [UTMOS] Not available: {e}  — UTMOS will be -1")
        except Exception as e:
            print(f"  [UTMOS] Not available: {e}  — UTMOS will be -1")

    def score(self, paths: List[str]) -> Dict:
        if self.model is None:
            return {"nmos_mean": -1.0, "nmos_std": 0.0, "n": 0}

        scores = []
        for p in tqdm(paths, desc="    UTMOS scoring"):
            try:
                # Handle both utmos and utmosv2 APIs
                if hasattr(self.model, 'predict'):
                    score = self.model.predict(input_path=p)
                else:
                    score = self.model.score(p)
                scores.append(float(score))
            except Exception:
                pass

        return {
            "nmos_mean": float(np.mean(scores)) if scores else -1.0,
            "nmos_std":  float(np.std(scores))  if scores else 0.0,
            "n":         len(scores),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Emotion2Vec SER (Primary - more reliable than SenseVoice)
# ══════════════════════════════════════════════════════════════════════════════

class Emotion2VecSER:
    """Use emotion2vec for all emotion metrics - provides both embeddings and probabilities"""
    
    def __init__(self, device=DEVICE):
        self.model = None
        self.label_map = {
            "生气/angry": "anger",
            "厌恶/disgusted": "disgust",
            "恐惧/fearful": "fear",
            "开心/happy": "happiness",
            "中立/neutral": "neutral",
            "难过/sad": "sadness",
            "吃惊/surprised": "surprise",
        }
        try:
            from funasr import AutoModel
            self.model = AutoModel(model="iic/emotion2vec_plus_large", device=device)
            print("  [SER] Loaded emotion2vec_plus_large for emotion recognition")
        except Exception as e:
            print(f"  [SER] Could not load emotion2vec: {e} — SER will be -1")

    def predict(self, path: str) -> Tuple[str, Dict[str, float]]:
        if self.model is None:
            return "unknown", {e: 0.0 for e in EMOTIONS}
        try:
            res = self.model.generate(input=path, granularity="utterance")
            if res and len(res) > 0 and "scores" in res[0] and "labels" in res[0]:
                scores = res[0]["scores"]
                labels = res[0]["labels"]
                probs = {emo: 0.0 for emo in EMOTIONS}
                for label, score in zip(labels, scores):
                    for cn_label, en in self.label_map.items():
                        if cn_label in label:
                            probs[en] = max(probs[en], float(score))
                            break
                if any(probs.values()):
                    pred = max(probs, key=probs.get)
                    return pred, probs
        except Exception:
            pass
        return "unknown", {e: 0.0 for e in EMOTIONS}

    def predict_batch(self, paths: List[str], desc: str = "    SER inference") -> List[Tuple[str, Dict[str, float]]]:
        results = []
        for path in tqdm(paths, desc=desc):
            results.append(self.predict(str(path)))
        return results

# Use Emotion2Vec as primary SER (SenseVoice removed due to API issues)
SERScorer = Emotion2VecSER

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — EMOTION METRICS (from SER results)
# ══════════════════════════════════════════════════════════════════════════════

def step7_emotion_metrics(ser_results: List[Tuple[str, Dict]],
                          target_emotion: str) -> Dict:
    acc_count = 0
    confidences = []
    margins = []
    target_probs = []

    for pred, probs in ser_results:
        target_prob = probs.get(target_emotion, 0.0)
        target_probs.append(target_prob)

        if pred == target_emotion:
            acc_count += 1

        confidences.append(target_prob)

        other_probs = [v for k, v in probs.items() if k != target_emotion]
        second_highest = max(other_probs) if other_probs else 0.0
        margins.append(target_prob - second_highest)

    n = len(ser_results)
    return {
        "accuracy":          acc_count / n if n else 0.0,
        "confidence_mean":   float(np.mean(confidences)) if confidences else 0.0,
        "confidence_std":    float(np.std(confidences))  if confidences else 0.0,
        "margin_mean":       float(np.mean(margins))     if margins else 0.0,
        "margin_std":        float(np.std(margins))      if margins else 0.0,
        "target_prob_mean":  float(np.mean(target_probs)) if target_probs else 0.0,
    }

def step7_emotion_gain(baseline_ser_results: List[Tuple[str, Dict]],
                       gen_ser_results: List[Tuple[str, Dict]],
                       target_emotion: str) -> Dict:
    baseline_probs = [probs.get(target_emotion, 0.0) for _, probs in baseline_ser_results]
    gen_probs = [probs.get(target_emotion, 0.0) for _, probs in gen_ser_results]

    gains = [g - b for g, b in zip(gen_probs, baseline_probs)]

    return {
        "gain_mean":     float(np.mean(gains)) if gains else 0.0,
        "gain_std":      float(np.std(gains))  if gains else 0.0,
        "baseline_mean": float(np.mean(baseline_probs)) if baseline_probs else 0.0,
        "steered_mean":  float(np.mean(gen_probs)) if gen_probs else 0.0,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — E-SIM (Emotion Similarity via emotion2vec)
# ══════════════════════════════════════════════════════════════════════════════

class ESIMScorer:
    def __init__(self, device: str = DEVICE):
        self.model = None
        try:
            from funasr import AutoModel
            self.model = AutoModel(model="iic/emotion2vec_plus_large", device=device)
            print("  [E-SIM] Loaded emotion2vec_plus_large")
        except Exception as e:
            print(f"  [E-SIM] Could not load emotion2vec: {e}  — E-SIM will be -1")

    def _embed(self, path: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            res = self.model.generate(input=path, granularity="utterance", extract_embedding=True)
            if res:
                emb = res[0].get("feats", res[0].get("embedding"))
                if emb is not None:
                    return np.array(emb).flatten()
        except Exception:
            pass
        return None

    def build_anchor(self, anchor_dir: str, emotion: str, n: int = ANCHOR_PER_EMO) -> Optional[np.ndarray]:
        wavs = list(Path(anchor_dir).glob(f"**/{emotion}/*.wav"))[:n]
        if not wavs:
            wavs = [w for w in Path(anchor_dir).rglob("*.wav") if w.parent.name.lower() == emotion.lower()][:n]

        if not wavs:
            print(f"    [E-SIM] No anchor wavs found for {emotion}")
            return None

        embs = []
        for w in tqdm(wavs, desc=f"    Anchor {emotion}", leave=False):
            e = self._embed(str(w))
            if e is not None:
                embs.append(e)

        if not embs:
            print(f"    [E-SIM] Could not embed any anchor file for {emotion}")
            return None

        return np.stack(embs).mean(axis=0)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

    def score(self, gen_paths: List[str], anchor_emb: Optional[np.ndarray]) -> Dict:
        if anchor_emb is None:
            return {"esim_mean": -1.0, "esim_std": 0.0, "n": 0}

        sims = []
        for path in tqdm(gen_paths, desc="    E-SIM scoring", leave=False):
            e = self._embed(path)
            if e is not None:
                sims.append(self._cosine(e, anchor_emb))

        return {
            "esim_mean": float(np.mean(sims)) if sims else -1.0,
            "esim_std":  float(np.std(sims))  if sims else 0.0,
            "n":         len(sims),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def step9_confusion_matrix(ser_results: List[Tuple[str, Dict]],
                           target_emotion: str, alpha: float) -> Dict:
    counts = {e: 0 for e in EMOTIONS + ["unknown"]}
    for pred, _ in ser_results:
        counts[pred] = counts.get(pred, 0) + 1

    total = len(ser_results)
    return {
        "target_emotion":   target_emotion,
        "alpha":            alpha,
        "predicted_counts": counts,
        "accuracy":         counts.get(target_emotion, 0) / total if total else 0,
        "total":            total,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10 — ALPHA TREND (Controllability)
# ══════════════════════════════════════════════════════════════════════════════

def step10_alpha_trend(ser_scorer, inference_dir: str, emotion: str,
                       alpha_values: List[float], n_samples: int = MAX_SAMPLES) -> Dict:
    probs_per_alpha = []

    for alpha in alpha_values:
        gen_dir = os.path.join(inference_dir, f"alpha-{alpha}", emotion)
        if not os.path.exists(gen_dir):
            probs_per_alpha.append(0.0)
            continue

        wavs = list(Path(gen_dir).glob("*.wav"))[:n_samples]
        ps = []
        for wav in tqdm(wavs, desc=f"    Trend α={alpha}", leave=False):
            _, probs_dict = ser_scorer.predict(str(wav))
            ps.append(probs_dict.get(emotion, 0.0))
        probs_per_alpha.append(float(np.mean(ps)) if ps else 0.0)

    base_dir = os.path.join(inference_dir, "baseline")
    base_wavs = list(Path(base_dir).glob("*.wav"))[:n_samples] if os.path.exists(base_dir) else []
    base_ps = []
    for wav in base_wavs:
        _, probs_dict = ser_scorer.predict(str(wav))
        base_ps.append(probs_dict.get(emotion, 0.0))
    baseline_prob = float(np.mean(base_ps)) if base_ps else 0.0

    full_alphas = [0.0] + list(alpha_values)
    full_probs = [baseline_prob] + probs_per_alpha

    if len(full_probs) > 1 and float(np.std(full_probs)) > 1e-8:
        corr = float(np.corrcoef(full_alphas, full_probs)[0, 1])
    else:
        corr = 0.0

    monotonic = all(full_probs[i] <= full_probs[i + 1] for i in range(len(full_probs) - 1))

    return {
        "alphas":        full_alphas,
        "probs":         full_probs,
        "correlation":   corr,
        "monotonic":     monotonic,
        "baseline_prob": baseline_prob,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 11 — SATURATION CURVE
# ══════════════════════════════════════════════════════════════════════════════

def step11_saturation_curve(ser_scorer, inference_dir: str, emotion: str,
                            saturation_alphas: List[float], n_samples: int = SATURATION_N_SAMPLES) -> Dict:
    probabilities = []

    for alpha in tqdm(saturation_alphas, desc="    Saturation curve", leave=False):
        if alpha == 0:
            target_dir = os.path.join(inference_dir, "baseline")
        else:
            target_dir = os.path.join(inference_dir, f"alpha-{alpha}", emotion)

        if not os.path.exists(target_dir):
            probabilities.append(0.0)
            continue

        wavs = list(Path(target_dir).glob("*.wav"))[:n_samples]
        ps = []
        for wav in wavs:
            _, probs_dict = ser_scorer.predict(str(wav))
            ps.append(probs_dict.get(emotion, 0.0))
        probabilities.append(float(np.mean(ps)) if ps else 0.0)

    global_max = max(probabilities) if probabilities else 0.0
    saturation_alpha = saturation_alphas[-1]

    for alpha, prob in zip(saturation_alphas, probabilities):
        if prob >= global_max * 0.95:
            saturation_alpha = alpha
            break

    return {
        "alphas":           saturation_alphas,
        "probabilities":    probabilities,
        "saturation_alpha": saturation_alpha,
        "max_probability":  global_max,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 12 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def step12_save_results(all_results: Dict, all_confusion: List[Dict], results_dir: str):
    banner(12, "Save results", f"Writing output files to {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "evaluation_full.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("  Saved evaluation_full.json")

    rows = []
    for emotion in EMOTIONS:
        if emotion not in all_results:
            continue
        for alpha in ALPHA_VALUES:
            key = f"alpha_{alpha}"
            if key not in all_results[emotion]:
                continue
            d = all_results[emotion][key]
            rows.append({
                "emotion": emotion, "alpha": alpha,
                "wer":    d.get("wer_mean", -1),
                "ssim":   d.get("ssim_mean", -1),
                "nmos":   d.get("nmos_mean", -1),
                "acc":    d.get("emotion_accuracy", -1),
                "conf":   d.get("emotion_confidence_mean", -1),
                "margin": d.get("emotion_margin_mean", -1),
                "gain":   d.get("emotion_gain_mean", -1),
                "esim":   d.get("esim_mean", -1),
            })

    if rows:
        with open(os.path.join(results_dir, "evaluation_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("  Saved evaluation_summary.csv")

    with open(os.path.join(results_dir, "confusion_matrices.json"), "w") as f:
        json.dump(all_confusion, f, indent=2)
    print("  Saved confusion_matrices.json")

    trend_rows = []
    for emotion in EMOTIONS:
        trend = all_results.get(emotion, {}).get("alpha_trend", {})
        if not trend:
            continue
        for a, p in zip(trend.get("alphas", []), trend.get("probs", [])):
            trend_rows.append({"emotion": emotion, "alpha": a, "prob": p})

    if trend_rows:
        with open(os.path.join(results_dir, "alpha_trend.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["emotion", "alpha", "prob"])
            writer.writeheader()
            writer.writerows(trend_rows)
        print("  Saved alpha_trend.csv")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 13 — PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def step13_print_table(all_results: Dict):
    banner(13, "Summary table", "All metrics for every (emotion, alpha) cell")

    header = (f"{'Emotion':<12} {'Alpha':<7} {'WER%':<7} {'S-SIM':<7} "
              f"{'NMOS':<7} {'Acc%':<7} {'Conf':<7} {'Margin':<8} "
              f"{'Gain':<7} {'E-SIM':<7}")
    print("\n" + header)
    print("─" * len(header))

    for emotion in EMOTIONS:
        if emotion not in all_results:
            continue
        for alpha in ALPHA_VALUES:
            key = f"alpha_{alpha}"
            if key not in all_results[emotion]:
                continue
            d = all_results[emotion][key]
            print(
                f"{emotion:<12} {alpha:<7} "
                f"{d.get('wer_mean', -1):<7.2f} "
                f"{d.get('ssim_mean', -1):<7.3f} "
                f"{d.get('nmos_mean', -1):<7.2f} "
                f"{d.get('emotion_accuracy', -1)*100:<7.1f} "
                f"{d.get('emotion_confidence_mean', -1):<7.3f} "
                f"{d.get('emotion_margin_mean', -1):<8.3f} "
                f"{d.get('emotion_gain_mean', -1):<7.3f} "
                f"{d.get('esim_mean', -1):<7.3f}"
            )
        print()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 68 + "╗")
    print(f"║  COMPLETE EVALUATION  —  {MODEL_NAME.upper():<42} ║")
    print(f"║  Device: {DEVICE:<59} ║")
    print(f"║  Inference dir: {INFERENCE_DIR[-50:]:<52} ║")
    print("╚" + "═" * 68 + "╝")

    # STEP 1: Load transcriptions
    transcriptions = step1_load_transcriptions(NEUTRAL_JSON)

    # STEP 2-5: Initialize scorers
    banner(3, "Initialise scorers", "Loading Whisper / SpeechBrain / UTMOS / emotion2vec")

    wer_scorer   = WERScorer()
    ssim_scorer  = SSIMScorer()
    utmos_scorer = UTMOSScorer()
    ser_scorer   = SERScorer()
    esim_scorer  = ESIMScorer()

    # STEP 6: Build emotion2vec anchor embeddings
    banner(6, "Build E-SIM anchors", f"Embedding up to {ANCHOR_PER_EMO} reference files per emotion")

    anchors = {}
    for emo in EMOTIONS:
        print(f"\n  Building anchor for: {emo}")
        anchors[emo] = esim_scorer.build_anchor(ANCHOR_DIR, emo)

    # STEP 7: Pre-compute baseline SER once per emotion
    banner(7, "Pre-compute baseline SER", "Running emotion2vec on 300 baseline files per emotion")

    baseline_ser_cache = {}
    base_dir_global = os.path.join(INFERENCE_DIR, "baseline")

    for emotion in EMOTIONS:
        print(f"\n  Baseline SER for: {emotion}")
        base_wavs = sorted(Path(base_dir_global).glob("*.wav"))[:MAX_SAMPLES] if os.path.exists(base_dir_global) else []
        if base_wavs:
            baseline_ser_cache[emotion] = ser_scorer.predict_batch(
                [str(w) for w in base_wavs],
                desc=f"    Baseline SER [{emotion}]")
        else:
            print(f"    [WARN] No baseline wavs found")
            baseline_ser_cache[emotion] = []

    # MAIN EVALUATION LOOP
    all_results = {}
    all_confusion = []

    for emotion in EMOTIONS:
        banner(8, f"Evaluate emotion: {emotion.upper()}", f"Alphas: {ALPHA_VALUES}")
        emo_results = {}

        for alpha in ALPHA_VALUES:
            print(f"\n  ── α = {alpha} ──────────────────────────────────────────")

            pairs = step2_collect_pairs(INFERENCE_DIR, emotion, alpha)
            if not pairs:
                print(f"    [SKIP] No audio files found")
                continue

            gen_paths = [p[0] for p in pairs]
            base_paths = [p[1] for p in pairs]
            print(f"    Collected {len(pairs)} audio pairs")

            ref_texts = [transcriptions.get(Path(p).stem, "") for p in gen_paths]

            metrics = {"n_samples": len(pairs)}

            # QUALITY: WER
            w = wer_scorer.score(gen_paths, ref_texts)
            metrics["wer_mean"] = w["wer_mean"]
            print(f"      WER = {w['wer_mean']:.2f}%")

            # QUALITY: S-SIM
            s = ssim_scorer.score(gen_paths, base_paths)
            metrics["ssim_mean"] = s["ssim_mean"]
            print(f"      S-SIM = {s['ssim_mean']:.3f}")

            # QUALITY: UTMOS
            u = utmos_scorer.score(gen_paths)
            metrics["nmos_mean"] = u["nmos_mean"]
            print(f"      UTMOS = {u['nmos_mean']:.2f}")

            # EMOTION: SER
            gen_ser_results = ser_scorer.predict_batch(gen_paths, desc=f"    SER [{emotion} α={alpha}]")

            emo_m = step7_emotion_metrics(gen_ser_results, emotion)
            metrics["emotion_accuracy"] = emo_m["accuracy"]
            metrics["emotion_confidence_mean"] = emo_m["confidence_mean"]
            metrics["emotion_margin_mean"] = emo_m["margin_mean"]
            print(f"      Accuracy = {emo_m['accuracy']*100:.1f}%")
            print(f"      Confidence = {emo_m['confidence_mean']:.3f}")
            print(f"      Margin = {emo_m['margin_mean']:.3f}")

            # EMOTION: Gain
            gain_m = step7_emotion_gain(baseline_ser_cache.get(emotion, []), gen_ser_results, emotion)
            metrics["emotion_gain_mean"] = gain_m["gain_mean"]
            print(f"      Gain = {gain_m['gain_mean']:+.3f}")

            # EMOTION: E-SIM
            esim_r = esim_scorer.score(gen_paths, anchors.get(emotion))
            metrics["esim_mean"] = esim_r["esim_mean"]
            print(f"      E-SIM = {esim_r['esim_mean']:.3f}")

            # Confusion matrix
            cm = step9_confusion_matrix(gen_ser_results, emotion, alpha)
            if cm:
                all_confusion.append(cm)

            emo_results[f"alpha_{alpha}"] = metrics

        # Alpha trend
        trend = step10_alpha_trend(ser_scorer, INFERENCE_DIR, emotion, ALPHA_VALUES, MAX_SAMPLES)
        emo_results["alpha_trend"] = trend
        print(f"    Alpha correlation = {trend['correlation']:.3f}")
        print(f"    Monotonic = {trend['monotonic']}")

        # Saturation curve
        sat = step11_saturation_curve(ser_scorer, INFERENCE_DIR, emotion, SATURATION_ALPHAS, SATURATION_N_SAMPLES)
        emo_results["saturation_curve"] = sat
        print(f"    Saturation α = {sat['saturation_alpha']}")

        all_results[emotion] = emo_results

    # Save and print results
    step12_save_results(all_results, all_confusion, RESULTS_DIR)
    step13_print_table(all_results)

    print(f"\n✅ Done. Results saved to: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()