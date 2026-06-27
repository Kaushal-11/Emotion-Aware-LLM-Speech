import os
from pathlib import Path

# ============================================================================
# DEVICE
# ============================================================================

import torch
 
_cuda_available = torch.cuda.is_available()
_num_gpus       = torch.cuda.device_count() if _cuda_available else 0
_force_cpu      = os.environ.get("FORCE_CPU", "0") == "1"
 
# Primary device — ASR, SER, Classifier, TTS, Recommender
DEVICE     = "cpu" if (_force_cpu or not _cuda_available) else "cuda:0"
# LLM device — pinned to GPU 1 when available, else falls back to GPU 0 / CPU
LLM_DEVICE = "cpu" if (_force_cpu or not _cuda_available) else (
    "cuda:1" if _num_gpus >= 2 else "cuda:0"
)
 
DTYPE     = torch.float16 if "cuda" in DEVICE     else torch.float32
LLM_DTYPE = torch.float16 if "cuda" in LLM_DEVICE else torch.float32
 
print(f"[Config] DEVICE={DEVICE}  LLM_DEVICE={LLM_DEVICE}  num_gpus={_num_gpus}")
 
# ============================================================================
# TOGGLES — switch components without touching code
# ============================================================================

# SER backend: "sensevoice" (default) or "wavlm"
SER_BACKEND = "sensevoice"

# LLM backend: "mistral" (default) or "qwen"
LLM_BACKEND = "mistral"

# TTS backend: "f5tts" (default) or "cosyvoice2"
TTS_BACKEND = "f5tts"


# ============================================================================
# ASR — Whisper
# ============================================================================

WHISPER_MODEL_SIZE = "large-v3-turbo"   # "large", "large-v3", "large-v3-turbo"
WHISPER_DEVICE     = DEVICE
WHISPER_LANGUAGE   = "en"               


# ============================================================================
# TEXT CLASSIFIER — RoBERTa multi-head (emotion / target / intensity)
# ============================================================================

# Directory containing tokenizer files + best_model.pt
CLASSIFIER_DIR = Path("/workspace/text-em/classification/finetune/checkpoints-2")
CLASSIFIER_MAX_LENGTH = 128

EMOTIONS    = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]
TARGETS_CL  = ["you", "other", "self", "situation"]


# ============================================================================
# SER — Speech Emotion Recognition
# ============================================================================

# 6-class emotion order used by BOTH SER models (must match training)
SER_EMOTIONS = ["happiness", "anger", "sadness", "disgust", "fear", "surprise"]

# ── SenseVoice (default) ───────────────────────────────────────────────────
SENSEVOICE_MODEL_DIR = Path("/workspace/audio-em/finetune-results/sensevoice-small/lora/20260521_212218")
SENSEVOICE_FUNASR_FALLBACK = "FunAudioLLM/SenseVoiceSmall"

SENSEVOICE_LORA_R       = 16
SENSEVOICE_LORA_ALPHA   = 32
SENSEVOICE_LORA_DROPOUT = 0.1
SENSEVOICE_LORA_TARGETS = ["linear_q_k_v", "linear_out"]
SENSEVOICE_NUM_PREFIX_TOKENS = 4
SENSEVOICE_FIXED_LID_TOKEN      = 0
SENSEVOICE_FIXED_TEXTNORM_TOKEN = 25016

# ── WavLM (toggle option) ──────────────────────────────────────────────────
WAVLM_MODEL_DIR = Path("/workspace/audio-em/finetune-results/wavlm-large/lora/20260513_225501")
WAVLM_LORA_R       = 16
WAVLM_LORA_ALPHA   = 32
WAVLM_LORA_DROPOUT = 0.1
WAVLM_LORA_TARGET  = ["q_proj", "v_proj", "k_proj", "out_proj"]

# Common audio settings for SER
SER_TARGET_SR    = 16_000
SER_MAX_DURATION = 20.0  # seconds


# ============================================================================
# LLM — Mistral / Qwen with emotion steering
# ============================================================================

LLM_PATHS = {
    "mistral": "/workspace/text-em/models/Ministral-3B-BF16",
    "qwen":    "Qwen/Qwen3-4B-Instruct-2507",   
}

LLM_MODEL_NAMES = {
    "mistral": "ministral3_3b",
    "qwen":    "qwen3_4b",
}

# Steering parameters
ALPHA  = 8.0          # base steering strength — scaled by vector_intensity each turn
LAYERS = "11-20"      # inclusive layer range for vector injection
LAST_K = 1
SCALE  = "rms"

# Direction vectors — one folder per LLM backend
DIRECTIONS_DIR = {
    "mistral": Path("/workspace/text-em/steering_vector/ministral3_3b/02_emotion_directions"),
    "qwen":    Path("/workspace/text-em/steering_vector/qwen3_4b_instruct_2507/02_emotion_directions"),
}

HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ============================================================================
# TTS — F5-TTS (default) / CosyVoice2 (option) — preset speakers, no voice cloning
# ============================================================================

TTS_EMOTIONS = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]

F5TTS_MODEL_NAME = "F5TTS_v1_Base"   
F5TTS_CKPT_PATH  = Path("/workspace/audio-em/emo-tts/models/f5tts/F5TTS_v1_Base/model_1250000.safetensors")
F5TTS_VOCAB_PATH = Path("/workspace/audio-em/emo-tts/models/f5tts/F5TTS_v1_Base/vocab.txt") 
F5TTS_VECTORS_DIR = Path("/workspace/audio-em/emo-tts/results/activation_vector/f5tts/final")
F5TTS_STEERED_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22]   # DiT transformer block indices
F5TTS_DEFAULT_ALPHA  = 8.0     # multiplied by vector_intensity from decision engine
F5TTS_SAMPLE_RATE    = 24000
F5TTS_NFE_STEP       = 32       # number of denoising steps (lower = faster, less quality)
F5TTS_CFG_STRENGTH   = 2.0
F5TTS_SPEED          = 1.0
 
COSYVOICE_MODEL_DIR    = Path("/workspace/audio-em/emo-tts/models/cosyvoice2")
COSYVOICE_REPO_PATH = "/workspace/audio-em/emo-tts/models/CosyVoice"  
COSYVOICE_VECTORS_DIR = Path("/workspace/audio-em/emo-tts/results/activation_vector/cosyvoice2/final")
TTS_EMOTIONS = ["anger", "happiness", "sadness", "disgust", "fear", "surprise"]
COSYVOICE_STEERED_LAYERS = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51]
COSYVOICE_DEFAULT_ALPHA  = 8.0   # multiplied by vector_intensity from decision engine
COSYVOICE_SAMPLE_RATE    = 22050


# Preset speakers — user picks one in the UI.
# Each entry needs a short reference clip (5-15s, clean) + matching transcript.
PRESET_SPEAKERS = {
    "speaker_1": {
        "label":     "Speaker 1 (Male, Calm)",
        "ref_audio": "/workspace/audio-em/emo-tts/data/calm_male.wav",
        "ref_text":  "So today, I'm going to share a few tips where you can practice in your daily life and then you can, if you practice this, it will be a great help to find peacefulness and calm",
    },
    "speaker_2": {
        "label":     "Speaker 2 (Female, Warm)",
        "ref_audio": "/workspace/audio-em/emo-tts/data/calm_female.wav",
        "ref_text":  "Just sit, even just do deep breathing, you've started being with yourself, that self-discipline, that nothing around me is pulling me. That is my first achievement because people won't even sit for five minutes without picking up something. So 15 minutes",
    },
}
DEFAULT_SPEAKER = "speaker_1"


# ============================================================================
# RECOMMENDER — Sentence-Transformers + curated JSON KB
# ============================================================================

RECOMMENDER_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good enough for short text
RECOMMENDER_KB_PATH    = Path("/workspace/integration/recommendation_kb.json")
RECOMMENDER_TOP_K      = 3


MAX_HISTORY_TURNS = 20   # safety cap on conversation history length kept for prompt