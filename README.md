# 🧠 Sakhya AI — Emotionally Intelligent LLM-Speech

> An end-to-end system that perceives human emotion through voice and text, reasons about it, and responds with a steered, emotionally-aware AI persona — in natural speech.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
  - [1. Audio Classifier (SER)](#1-audio-classifier-ser)
  - [2. Emotional TTS](#2-emotional-tts)
  - [3. Text Emotion Module](#3-text-emotion-module)
  - [4. Sakhya AI Integration](#4-sakhya-ai-integration)
  - [5. Restaurant Bot (Use Case Demo)](#5-restaurant-bot-use-case-demo)

---

## Project Overview

**Sakhya AI** is a research-to-production emotionally intelligent conversational system. It combines:

- **Speech Emotion Recognition (SER)** — detecting how a user *feels* from their voice
- **Text Emotion Classification** — understanding sentiment from *what* they say
- **Multimodal Fusion** — blending acoustic and semantic signals into one emotional representation
- **Activation Steering** — injecting emotion vectors directly into LLM hidden states to control response personality
- **Emotional TTS** — synthesizing voice responses with steered emotional tone
- **Decision Engine + State Memory** — maintaining coherent emotional context across multi-turn conversations
- **Recommender System** — surfacing contextually relevant suggestions (music, activities, quotes, movies)

The full pipeline runs speech-in → speech-out, with a FastAPI backend and a Gradio frontend supporting mic input, text input, and audio file upload.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INPUT (Speech / Text)                      │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
        ┌──────────┐                            ┌──────────────┐
        │  ASR     │  (Whisper)                 │     SER      │  (SenseVoice / HuBERT / WavLM)
        │Transcript│                            │Voice Emotion │
        └────┬─────┘                            └──────┬───────┘
             │                                         │
             ▼                                         │
    ┌──────────────────┐                               │
    │ RoBERTa Classifier│◄──────────────────────────────┘
    │ Text Emotion +   │
    │ Target + Intensity│
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Fusion Layer    │  (Acoustic + Semantic → Unified Emotion State)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │   State Memory + Decision Engine     │
    │  (Decay-based continuity, StyleContract,│
    │   escalation logic, target selection) │
    └────────┬─────────────────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌─────────────────┐
│ Steered  │  │  KB Recommender │  (Cosine similarity via Sentence-Transformers)
│  LLM     │  │  Music/Movies/  │
│(Mistral/ │  │  Activities/    │
│  Qwen)   │  │  Quotes         │
└────┬─────┘  └────────┬────────┘
     │                 │
     └────────┬────────┘
              ▼
    ┌──────────────────┐
    │  Emotional TTS   │  (F5-TTS / CosyVoice2 — Activation Steered)
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  SPEECH RESPONSE │
    └──────────────────┘
```

---

## Repository Structure

```
sakhya-ai/
├── Audio-Classifier/
├── Emo-TTS/
├── Text-Emo/
├── Sakhya-AI-Integration/
└── Restaurant-Bot/
```

---

### 1. Audio Classifier (SER)

Speech Emotion Recognition module. Fine-tunes multiple transformer-based audio models on a multi-corpus emotional speech dataset.

```
Audio-Classifier/
├── dataset/
│   ├── cremad/
│   ├── esd/
│   ├── iemocap/
│   ├── meld/
│   ├── ravdess/
│   ├── savee/
│   └── tess/
├── finetune-results/
│   ├── hubert-large/
│   ├── wavlm-large/
│   └── sensevoice-small/
├── dataset.py                  # Prepares unified dataset for fine-tuning
├── download.py                 # Downloads raw audio corpora
├── feature_extractor.py        # Acoustic feature extraction utilities
├── organize.py                 # Organizes multi-corpus data into unified structure
├── split.py                    # Train/val/test splitting
├── prepare_data_sensevoice.py  # SenseVoice-specific data preparation
├── finetune_hubert.py          # HuBERT fine-tuning script
├── finetune_wavlm.py           # WavLM fine-tuning script
├── finetune_wav2vec.py         # Wav2Vec fine-tuning script
├── finetune_sensevoice.py      # SenseVoice (LoRA) fine-tuning script
├── test_hubert.py
├── test_wavlm.py
├── test_sensevoice.py
└── requirements.txt
```

**Datasets used:** CREMA-D, ESD, IEMOCAP, MELD, RAVDESS, SAVEE, TESS

**Models fine-tuned:** Wav2Vec2, HuBERT-Large, WavLM-Large, SenseVoice-Small (LoRA)


---

### 2. Emotional TTS

Activation steering applied to Text-to-Speech models to produce emotionally expressive voice synthesis — without retraining.

```
Emo-TTS/
├── data/
│   ├── used/                           # Emotional speech samples used for vector extraction
│   └── unused/
├── results/
│   ├── activation-vectors/             # Extracted steering vectors per emotion
│   ├── generated/                      # Steered audio output samples
│   ├── evaluation/                     # Quality evaluation results
│   └── test-with-different-alphas/     # Steering strength experiments
├── cosyvoice2_hooks.py                 # Hooks for CosyVoice2 internal layer access
├── cosyvoice2_layer_analysis.txt       # Layer analysis for optimal injection points
├── extract_cosyvoice2_vectors.py       # Extract steering vectors from CosyVoice2
├── extract_f5tts_vectors.py            # Extract steering vectors from F5-TTS
├── f5tts_hooks.py                      # Hooks for F5-TTS layer access
├── filter_dataset.py                   # Filters high-quality emotional samples
├── inference_steering.py               # Runs steered TTS inference
├── evaluate.py                         # Evaluates output quality
└── token_search.py                     # Searches for optimal token injection positions
```

**Models:** F5-TTS, CosyVoice2

**Approach:** Activation steering (vector injection) — same method used for the LLM, applied to TTS model internals


---

### 3. Text Emotion Module

The core NLP research module. Covers prompt engineering across multiple LLMs, activation steering, AI characterization, and the full multi-turn RoBERTa-based classification + decision pipeline.

```
Text-Emo/
├── prompting/                          # Prompt engineering experiments across LLMs
│   ├── claude.ipynb
│   ├── deepseek.ipynb
│   ├── gemini.ipynb
│   ├── gpt.ipynb
│   ├── mistral.ipynb
│   ├── phi.ipynb
│   └── qwen.ipynb
│
├── steering/                           # Activation steering experiments
│   ├── Mistral.ipynb
│   ├── Mistral_analysis.ipynb
│   ├── Qwen.ipynb
│   └── qwen_analysis.ipynb
│
├── prompt-steer-output/                # Combined results for prompting & steering
│
├── characterization/                   # AI persona / character consistency experiments
│   ├── outputs/
│   ├── Al_Character.ipynb
│   ├── Al_Character_mistral.ipynb
│   ├── Al_Character_qwen.ipynb
│   ├── baseline_mistral.ipynb
│   └── baseline_qwen.ipynb
│
└── text-classifier/                    # RoBERTa multi-head classifier pipeline
    ├── emotion-data/                   # Fine-tuning & synthesized data for classifier
    ├── finetune/                       # Fine-tuned RoBERTa weights & logs
    ├── output/                         # Test results
    ├── testing-scenario/               # 15-turn per emotion evaluation scenarios
    ├── Data.ipynb                      # Dataset preparation
    ├── Finetune.ipynb                  # RoBERTa fine-tuning
    ├── decision_engine.py              # Decides which emotion vector to steer with + intensity
    ├── pure_steering.py                # Vector injection into the LLM hidden states
    ├── state_memory.py                 # Maintains AI emotional state across turns (with decay)
    └── steering_style_contract.py      # Controls response length, tone & structure per emotion-target
```

---

### 4. Sakhya AI Integration

Full end-to-end integration of all modules into a single speech-to-speech pipeline with a FastAPI.

```
Sakhya-AI-Integration/
├── core/                       # Backend modules
│   ├── asr.py                  # Automatic Speech Recognition (Whisper)
│   ├── ser.py                  # Speech Emotion Recognition (fine-tuned SenseVoice)
│   ├── classifier.py           # RoBERTa text emotion + target + intensity
│   ├── fusion.py               # Fuses acoustic (SER) + semantic (text) emotion signals
│   ├── llm.py                  # Steered LLM inference (Mistral / Qwen)
│   ├── tts.py                  # Emotional TTS (steered F5-TTS / CosyVoice2)
│   ├── decision_engine.py      # Selects steering vector + target + response strategy
│   ├── state_memory.py         # Decay-based emotional state across conversation turns
│   └── recommender.py          # KB semantic search & ranking (Sentence-Transformers)
├── frontend/                   # UI assets
├── app.py                      # Gradio UI 
├── pipeline.py                 # Orchestrates full speech-to-speech pipeline
└── config.py                   # Global config (model paths, thresholds, parameters)
```

**Input Modes:**
- 🎙️ Live microphone
- ⌨️ Text input
- 📁 Audio file upload

**Pipeline Flow:**
`Voice Input → ASR (Whisper) → [SER + RoBERTa Classifier] → Fusion → State Memory → Decision Engine → Steered LLM + Recommender → Emotional TTS → Voice Response`

---

### 5. Restaurant Bot (Use Case Demo)

A production-style emotionally intelligent customer support chatbot for a restaurant, built on top of the Sakhya pipeline.

```
Restaurant-Bot/
├── backend/
│   ├── classifier.py           # Text emotion detection
│   ├── decision_engine.py      # Steering vector selection + escalation logic
│   ├── llm.py                  # Steered LLM for empathetic customer responses
│   ├── recommender.py          # Action recommendations (refund, retry, escalate)
│   ├── state_memory.py         # Per-session emotional state tracking
│   ├── config.py
│   ├── main.py                 # FastAPI backend entry point
│   └── requirements.txt
└── frontend/                   # Customer UI + Manager Dashboard
```

**Features:**
- Customer enters name, selects issue type (track order, refund, etc.), and provides order ID — bot fetches real order details
- Every message runs through the 5-step pipeline: **Classify → Memory → Decide → Steer → Respond**
- Automatic escalation to a human manager if anger crosses threshold twice or user requests one
- Manager dashboard shows all conversations grouped by customer, per-session transcripts, and a dedicated escalation queue

---
