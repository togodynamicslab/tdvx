#!/usr/bin/env python3
"""
stt_gpu.py — GPU-Optimised Entry Point
=======================================
Quantisation : int8_float16  (pesos em int8, ativações em float16)
Device        : CUDA
VRAM mínima   : ~2.5 GB (medium) | ~5 GB (large-v3)

Uso:
    python stt_gpu.py audio.wav
    python stt_gpu.py audio.wav --model large-v3 --max-speakers 4
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# Carrega .env antes de qualquer importação do pipeline
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from stt_pipeline import AudioTranscriber, PipelineConfig


# ──────────────────────────────────────────────────────
def _validate_gpu() -> None:
    """Aborta se não houver GPU CUDA disponível."""
    if not torch.cuda.is_available():
        print(
            "\n[ERRO] Nenhuma GPU CUDA detectada.\n"
            "  → Verifique os drivers NVIDIA e a instalação do PyTorch com CUDA\n"
            "  → Para rodar sem GPU use: python stt_cpu.py audio.wav\n"
        )
        sys.exit(1)

    # Optimizações globais CUDA
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1_073_741_824  # bytes → GB

    print("\n" + "═" * 52)
    print("  GPU STT Pipeline  —  int8_float16 Quantisation")
    print("═" * 52)
    print(f"  GPU       : {props.name}")
    print(f"  VRAM      : {vram_gb:.1f} GB total  |  "
          f"{torch.cuda.memory_reserved(0) / 1e9:.1f} GB em uso")
    print(f"  CUDA      : {torch.version.cuda}")
    print(f"  cuDNN     : {torch.backends.cudnn.version()}")
    print("═" * 52 + "\n")

    if vram_gb < 2.5:
        print("[AVISO] VRAM < 2.5 GB — use --model small ou tiny para evitar OOM\n")


# ──────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STT Pipeline — GPU (int8_float16)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio",
                   help="Arquivo de áudio (WAV / MP3 / FLAC / OGG …)")
    p.add_argument("--model", default="medium",
                   choices=["tiny", "base", "small", "medium",
                            "large-v2", "large-v3"],
                   help="Tamanho do modelo Whisper")
    p.add_argument("--output", default="transcription_gpu.json",
                   metavar="ARQUIVO",
                   help="Arquivo JSON de saída")
    p.add_argument("--hf-token", default="",
                   metavar="TOKEN",
                   help="Token HuggingFace (fallback: variável HF_TOKEN / .env)")
    p.add_argument("--min-speakers", type=int, default=None,
                   metavar="N",
                   help="Número mínimo de speakers (dica para pyannote)")
    p.add_argument("--max-speakers", type=int, default=None,
                   metavar="N",
                   help="Número máximo de speakers (dica para pyannote)")
    p.add_argument("--similarity-threshold", type=float, default=0.80,
                   metavar="FLOAT",
                   help="Limiar cosine para re-identificação de speaker (0–1)")
    return p


# ──────────────────────────────────────────────────────
def main() -> None:
    _validate_gpu()
    args = _build_parser().parse_args()

    token = args.hf_token or os.getenv("HF_TOKEN", "")

    cfg = PipelineConfig(
        # ── quantização GPU ──────────────────────────
        device="cuda",
        compute_type="int8_float16",   # pesos int8, ativações float16
        # ── modelo ──────────────────────────────────
        whisper_model=args.model,
        num_workers=2,
        # ── diarização ──────────────────────────────
        hf_token=token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        similarity_threshold=args.similarity_threshold,
    )

    transcriber = AudioTranscriber(cfg)
    result = transcriber.process(args.audio)
    AudioTranscriber.save_json(result, args.output)

    # ── Resumo terminal ──────────────────────────────
    meta = result["metadata"]
    print("\n" + "─" * 52)
    print("  Resultado GPU")
    print("─" * 52)
    print(f"  Segmentos  : {meta['total_segments']}")
    print(f"  Speakers   : {meta['unique_speakers']}")
    print(f"  Idioma     : {meta['language_detected']}")
    print(f"  Modelo     : {meta['whisper_model']}  [{meta['compute_type']}]")
    vram_used = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"  VRAM pico  : {vram_used:.2f} GB")
    print(f"  Saída      : {args.output}")
    print("─" * 52 + "\n")


if __name__ == "__main__":
    main()
