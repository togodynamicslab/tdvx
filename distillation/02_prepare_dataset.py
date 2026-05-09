"""
Step 2: Download and prepare PT+EN training data.

Downloads Mozilla Common Voice for Portuguese and English,
preprocesses audio to 16kHz, and generates pseudo-labels
using the teacher model (whisper-large-v3).

Requires: HF token with access to Common Voice dataset.
Set PYANNOTE_AUTH_TOKEN in .env or HF_TOKEN env var, or run:
    huggingface-cli login

Usage:
    python 02_prepare_dataset.py

    # Resume pseudo-labeling from checkpoint
    python 02_prepare_dataset.py --resume
"""

import argparse
import torch
import os
import logging
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import login
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEACHER_MODEL = "openai/whisper-large-v3"
OUTPUT_DIR = "./dataset-pt-en"
LANGUAGES = ["pt", "en"]
DATASET_NAME = "mozilla-foundation/common_voice_17_0"
BATCH_SIZE = 8  # For pseudo-labeling
MAX_SAMPLES_PER_LANG = None  # Set to e.g. 50000 to limit, None for all


def setup_hf_auth():
    """Set up HuggingFace authentication from .env or env vars."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("PYANNOTE_AUTH_TOKEN")

    if not token:
        # Try reading from .env file
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PYANNOTE_AUTH_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break

    if token:
        login(token=token, add_to_git_credential=False)
        logger.info("Authenticated with HuggingFace")
    else:
        logger.warning("No HF token found. Common Voice requires authentication.")
        logger.warning("Run: huggingface-cli login")


def download_and_prepare():
    """Download Common Voice PT + EN and resample to 16kHz."""
    setup_hf_auth()

    all_datasets = []

    for lang in LANGUAGES:
        logger.info(f"Loading Common Voice {lang}...")
        ds = load_dataset(
            DATASET_NAME,
            lang,
            split="train",
        )

        if MAX_SAMPLES_PER_LANG and len(ds) > MAX_SAMPLES_PER_LANG:
            ds = ds.select(range(MAX_SAMPLES_PER_LANG))
            logger.info(f"  Limited to {MAX_SAMPLES_PER_LANG} samples")

        # Add language column
        ds = ds.map(lambda x: {"language": lang}, num_proc=4)

        # Cast audio to 16kHz
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

        # Keep only needed columns
        keep_cols = ["audio", "sentence", "language"]
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        ds = ds.remove_columns(remove_cols)

        logger.info(f"  {lang}: {len(ds)} samples")
        all_datasets.append(ds)

    # Concatenate
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    logger.info(f"Combined dataset: {len(combined)} samples")

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
    logger.info(f"Saved to {OUTPUT_DIR}/train")

    return combined


def generate_pseudo_labels(dataset, resume=False):
    """
    Generate pseudo-labels using whisper-large-v3 teacher.

    The teacher transcribes each audio sample, and we store the
    teacher's output as the training target for the student.
    """
    output_path = os.path.join(OUTPUT_DIR, "train_with_labels")

    if resume and os.path.exists(output_path):
        from datasets import load_from_disk
        logger.info(f"Resuming from {output_path}")
        return load_from_disk(output_path)

    logger.info(f"Loading teacher model: {TEACHER_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(TEACHER_MODEL)
    teacher = WhisperForConditionalGeneration.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    teacher.eval()

    logger.info("Generating pseudo-labels with teacher...")

    pseudo_labels = []
    errors = 0

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Pseudo-labeling"):
        batch = dataset[i : i + BATCH_SIZE]

        try:
            # Process audio
            audio_arrays = [sample["array"] for sample in batch["audio"]]

            input_features = processor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).input_features.to(device, dtype=torch.float16)

            # Generate with teacher (greedy for speed)
            with torch.no_grad():
                predicted_ids = teacher.generate(
                    input_features,
                    max_new_tokens=448,
                    language=None,  # Auto-detect
                    task="transcribe",
                )

            # Decode
            texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for j, text in enumerate(texts):
                pseudo_labels.append({
                    "teacher_text": text.strip(),
                    "original_text": batch["sentence"][j],
                    "language": batch["language"][j],
                })

        except Exception as e:
            logger.warning(f"Error at batch {i}: {e}")
            errors += 1
            # Fill with original text as fallback
            for j in range(len(batch["audio"])):
                pseudo_labels.append({
                    "teacher_text": batch["sentence"][j],
                    "original_text": batch["sentence"][j],
                    "language": batch["language"][j],
                })

        # Save checkpoint every 1000 batches
        if (i // BATCH_SIZE) % 1000 == 0 and i > 0:
            logger.info(f"Checkpoint at {i}/{len(dataset)} ({errors} errors)")

    logger.info(f"Generated {len(pseudo_labels)} pseudo-labels ({errors} errors)")

    # Add pseudo-labels to dataset
    dataset = dataset.add_column("teacher_text", [pl["teacher_text"] for pl in pseudo_labels])

    # Save
    dataset.save_to_disk(output_path)
    logger.info(f"Saved labeled dataset to {output_path}")

    del teacher
    torch.cuda.empty_cache()

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume pseudo-labeling")
    parser.add_argument("--skip-labels", action="store_true", help="Skip pseudo-labeling (use original text only)")
    args = parser.parse_args()

    dataset = download_and_prepare()

    if not args.skip_labels:
        dataset = generate_pseudo_labels(dataset, resume=args.resume)
    else:
        logger.info("Skipping pseudo-labeling — will use original transcripts")
