"""
config.py — Restaurant Bot Configuration
"""

import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/workspace/")
DATA_DIR = Path("/workspace/backend/data")

# ── Device Selection (Auto-detect GPU) ────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    LLM_DEVICE = "cuda"
    LLM_DTYPE = torch.bfloat16  # Use FP16 for much faster performance on your Tesla P40
else:
    DEVICE = "cpu"
    LLM_DEVICE = "cpu"
    LLM_DTYPE = torch.float32

# ── Classifier ────────────────────────────────────────────────────────────────
CLASSIFIER_DIR = Path("/workspace/backend/model/text-classifier/checkpoints-2")
CLASSIFIER_MAX_LENGTH = 128

# ── Emotion / Target vocab ────────────────────────────────────────────────────
EMOTIONS = ["anger", "sadness", "fear", "happiness", "disgust", "surprise"]
TARGETS_CL = ["you", "other", "self", "situation"]

# ── Recommender ───────────────────────────────────────────────────────────────
RECOMMENDER_MODEL_NAME = "all-MiniLM-L6-v2"
RECOMMENDER_KB_PATH = DATA_DIR / "recommendation_kb.json"
RECOMMENDER_TOP_K = 3

# ── LLM — Steered Mistral ─────────────────────────────────────────────────────
LLM_BACKEND = "mistral"

LLM_PATHS = {
    "mistral": "mistralai/Ministral-3-3B-Instruct-2512",  
    "qwen": "Qwen/Qwen3-4B-Instruct-2507",          
}

DIRECTIONS_DIR = {
    "mistral": Path("/workspace/backend/model/ministral_3b"),
    "qwen": Path("/workspace/backend/model/qwen3_4b_instruct_2507")
}

# Activation steering params
ALPHA = 20.0
LAYERS = "11-20"
LAST_K = 1
SCALE = "rms"

# HuggingFace token
HF_TOKEN = ""

MAX_HISTORY_TURNS = 10

# ── Restaurant-specific ───────────────────────────────────────────────────────
ESCALATION_ANGER_THRESHOLD = 0.8
ESCALATION_TURNS_REQUIRED = 2
RESTAURANT_NAME = "Spice Garden"