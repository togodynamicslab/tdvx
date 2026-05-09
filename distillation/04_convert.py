"""
Step 4: Convert trained model to CTranslate2/faster-whisper format.

Takes the trained HuggingFace model and:
1. Converts to CTranslate2 format with int8_float16 quantization
2. Patches config.json to restrict lang_ids to PT+EN
3. Adds 97 unused language tokens to suppress_ids
4. Validates by running inference

Usage:
    python 04_convert.py

    # Custom input/output paths
    python 04_convert.py --model ./distil-whisper-pt-en/final --output ./models/distil-pten
"""

import argparse
import os
import json
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "./distil-whisper-pt-en/final"
OUTPUT_DIR = "../models/distil-whisper-pten"
QUANTIZATION = "int8_float16"

# Language token IDs in Whisper vocabulary
ALL_LANG_IDS = list(range(50259, 50358))  # 50259 (en) to 50357 (su)
KEEP_LANG_IDS = [50259, 50267]  # en, pt


def convert(args):
    model_dir = args.model
    output_dir = args.output

    logger.info(f"Converting {model_dir} → {output_dir}")
    logger.info(f"Quantization: {QUANTIZATION}")

    # Step 1: Convert with ct2-transformers-converter
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", model_dir,
        "--output_dir", output_dir,
        "--quantization", QUANTIZATION,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--force",
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Try alternative converter command
        logger.warning(f"Python module failed, trying ct2-transformers-converter CLI...")
        cmd_alt = [
            "ct2-transformers-converter",
            "--model", model_dir,
            "--output_dir", output_dir,
            "--quantization", QUANTIZATION,
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--force",
        ]
        result = subprocess.run(cmd_alt, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Conversion failed:\n{result.stderr}")
            sys.exit(1)

    logger.info("CTranslate2 conversion complete")

    # Step 2: Patch config.json — restrict to PT+EN
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        # Restrict lang_ids
        config["lang_ids"] = KEEP_LANG_IDS

        # Add removed languages to suppress_ids
        removed = [lid for lid in ALL_LANG_IDS if lid not in KEEP_LANG_IDS]
        existing_suppress = set(config.get("suppress_ids", []))
        existing_suppress.update(removed)
        config["suppress_ids"] = sorted(list(existing_suppress))

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Patched config.json: lang_ids={KEEP_LANG_IDS}, added {len(removed)} tokens to suppress_ids")
    else:
        logger.warning("No config.json found in output — model may not have language restriction")

    # Step 3: Validate
    logger.info("Validating converted model...")
    try:
        from faster_whisper import WhisperModel
        import numpy as np

        model = WhisperModel(output_dir, device="cpu", compute_type="int8")

        # Test with synthetic audio
        audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        segments, info = model.transcribe(audio, language="pt", beam_size=2)
        for s in segments:
            pass

        logger.info(f"Validation passed! Detected language: {info.language}")
        logger.info(f"Model ready at: {os.path.abspath(output_dir)}")

        # Print model size
        total_size = 0
        for f in os.listdir(output_dir):
            fp = os.path.join(output_dir, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                total_size += size
                logger.info(f"  {f}: {size / 1024 / 1024:.1f} MB")
        logger.info(f"  Total: {total_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.info("The model files were created but could not be validated.")
        logger.info(f"Check: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_DIR, help="Path to trained HuggingFace model")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output path for CTranslate2 model")
    args = parser.parse_args()
    convert(args)
