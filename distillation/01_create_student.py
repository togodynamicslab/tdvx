"""
Step 1: Create a small student model for distillation.

Uses whisper-small architecture (12 encoder layers, 768 hidden dim)
with only 2 decoder layers. This gives ~120M params instead of 756M.

The student's weights are randomly initialized — it learns entirely
from the teacher's pseudo-labels during distillation training.
We copy the tokenizer/processor from the teacher so vocabularies match.

Usage:
    python 01_create_student.py

    # Use 4 decoder layers (~150M params, slightly better quality)
    python 01_create_student.py --decoder-layers 4
"""

import argparse
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEACHER_MODEL = "openai/whisper-large-v3"
STUDENT_OUTPUT = "./student-init"

# Whisper-small architecture
ENCODER_LAYERS = 12
DECODER_LAYERS = 2
D_MODEL = 768
ENCODER_ATTENTION_HEADS = 12
DECODER_ATTENTION_HEADS = 12
ENCODER_FFN_DIM = 3072
DECODER_FFN_DIM = 3072


def create_student(args):
    decoder_layers = args.decoder_layers

    # Load teacher processor (tokenizer + feature extractor)
    # We need the same vocabulary so student output is compatible
    logger.info(f"Loading processor from teacher: {TEACHER_MODEL}")
    processor = WhisperProcessor.from_pretrained(TEACHER_MODEL)

    # Get teacher config for vocab size and special token IDs
    logger.info(f"Loading teacher config...")
    teacher_config = WhisperConfig.from_pretrained(TEACHER_MODEL)

    # Build student config: small encoder + minimal decoder
    student_config = WhisperConfig(
        vocab_size=teacher_config.vocab_size,
        num_mel_bins=teacher_config.num_mel_bins,
        encoder_layers=ENCODER_LAYERS,
        encoder_attention_heads=ENCODER_ATTENTION_HEADS,
        encoder_ffn_dim=ENCODER_FFN_DIM,
        decoder_layers=decoder_layers,
        decoder_attention_heads=DECODER_ATTENTION_HEADS,
        decoder_ffn_dim=DECODER_FFN_DIM,
        d_model=D_MODEL,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_source_positions=teacher_config.max_source_positions,
        max_target_positions=teacher_config.max_target_positions,
        pad_token_id=teacher_config.pad_token_id,
        bos_token_id=teacher_config.bos_token_id,
        eos_token_id=teacher_config.eos_token_id,
        decoder_start_token_id=teacher_config.decoder_start_token_id,
        begin_suppress_tokens=teacher_config.begin_suppress_tokens,
        suppress_tokens=teacher_config.suppress_tokens,
        is_encoder_decoder=True,
        use_cache=True,
        apply_spec_augment=False,
    )

    logger.info(f"Student architecture:")
    logger.info(f"  Encoder: {ENCODER_LAYERS} layers, {D_MODEL} hidden, {ENCODER_ATTENTION_HEADS} heads")
    logger.info(f"  Decoder: {decoder_layers} layers, {D_MODEL} hidden, {DECODER_ATTENTION_HEADS} heads")

    # Initialize student with random weights
    student = WhisperForConditionalGeneration(student_config)

    student_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Student params: {student_params:,}")

    # Reference sizes
    sizes = {
        "whisper-tiny": 39_000_000,
        "whisper-base": 74_000_000,
        "whisper-small": 244_000_000,
        "whisper-medium": 769_000_000,
        "whisper-large-v3": 1_550_000_000,
    }
    for name, size in sizes.items():
        ratio = student_params / size
        logger.info(f"  vs {name}: {ratio:.2f}x ({student_params/1e6:.0f}M vs {size/1e6:.0f}M)")

    # Save student model + teacher's processor
    os.makedirs(STUDENT_OUTPUT, exist_ok=True)
    student.save_pretrained(STUDENT_OUTPUT)
    processor.save_pretrained(STUDENT_OUTPUT)
    logger.info(f"Saved student model to: {STUDENT_OUTPUT}")

    # Print model file size
    model_files = [f for f in os.listdir(STUDENT_OUTPUT) if f.endswith(('.safetensors', '.bin'))]
    for f in model_files:
        size_mb = os.path.getsize(os.path.join(STUDENT_OUTPUT, f)) / 1024 / 1024
        logger.info(f"  {f}: {size_mb:.1f} MB")

    return student


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder-layers", type=int, default=DECODER_LAYERS,
        help=f"Number of decoder layers (default: {DECODER_LAYERS})"
    )
    args = parser.parse_args()
    create_student(args)
