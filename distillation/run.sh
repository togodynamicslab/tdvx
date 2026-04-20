#!/bin/bash
#
# Full distillation pipeline: large-v3 → small (PT+EN only)
#
# Hardware: Runs on RTX 3080 Ti 16GB with gradient checkpointing + fp16
# Time: ~70-90 hours for full training (60K steps)
# Disk: ~50GB for dataset + models
#
# Steps:
#   1. Create student model (copy encoder + 2 decoder layers from teacher)
#   2. Download Common Voice PT+EN and generate pseudo-labels with teacher
#   3. Train student via knowledge distillation
#   4. Convert to CTranslate2/faster-whisper format
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source ../venv/bin/activate

# Set CUDA
export LD_LIBRARY_PATH="../venv/lib/python3.12/site-packages/nvidia/cublas/lib:../venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "=========================================="
echo "  TDVX Whisper Distillation Pipeline"
echo "  Teacher: whisper-large-v3 (1.55B)"
echo "  Student: 2 decoder layers (PT+EN only)"
echo "=========================================="
echo ""

# Step 1: Create student
echo "[1/4] Creating student model..."
python 01_create_student.py
echo ""

# Step 2: Prepare dataset + pseudo-labels
echo "[2/4] Preparing dataset and generating pseudo-labels..."
echo "  This downloads Common Voice PT+EN (~20GB) and runs teacher inference."
echo "  Estimated time: 4-8 hours on RTX 3080 Ti"
python 02_prepare_dataset.py
echo ""

# Step 3: Train
echo "[3/4] Training student model..."
echo "  Estimated time: 70-90 hours on RTX 3080 Ti"
echo "  Monitor with: tail -f distil-whisper-pt-en/trainer_state.json"
python 03_train.py
echo ""

# Step 4: Convert
echo "[4/4] Converting to CTranslate2..."
python 04_convert.py
echo ""

echo "=========================================="
echo "  Done!"
echo "  Model saved to: ../models/distil-whisper-pten"
echo ""
echo "  To use in TDVX, update model_config.py:"
echo "    whisper_model='models/distil-whisper-pten'"
echo "=========================================="
