#!/bin/bash
# ─────────────────────────────────────────────
# SenseVoice Finetuning Command
# Run from the SenseVoice repo root directory
# ─────────────────────────────────────────────

# Install FunASR first:
#   git clone https://github.com/alibaba/FunASR.git && cd FunASR
#   pip install -e ./

TRAIN_JSONL="/workspace/audio-em/sensevoice_data/train.jsonl"
VAL_JSONL="/workspace/audio-em/sensevoice_data/val.jsonl"
MODEL_DIR="FunAudioLLM/SenseVoiceSmall"   # or local path after download
OUTPUT_DIR="/workspace/audio-em/finetune-results/sensevoice"

# Training stats:
# Train samples: 9679
# Val samples:   2063

python -m funasr.bin.train_ds \
    --config-path conf \
    --config-name sensevoice.yaml \
    ++model="{}" \
    ++model_conf.model_dir="$MODEL_DIR" \
    ++dataset_conf.data_path="['$TRAIN_JSONL']" \
    ++dataset_conf.data_path_val="['$VAL_JSONL']" \
    ++train_conf.max_epoch=30 \
    ++train_conf.save_checkpoint_steps=1000 \
    ++train_conf.keep_nbest_models=5 \
    ++train_conf.avg_nbest_model=5 \
    ++optim_conf.lr=1e-4 \
    ++output_dir="$OUTPUT_DIR" \
    ++device="cuda" \
    ++batch_size=8 \
    ++accum_grad=4 \
    ++num_workers=2 \
    ++dataset_conf.batch_type="example" \
    ++log_interval=100

# NOTE: SenseVoice training goes through FunASR's train_ds.py.
# The emotion labels in your JSONL (emo_target) are used automatically.
# No custom training loop is needed.
