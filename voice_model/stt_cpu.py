#!/usr/bin/env python3
"""
stt_cpu.py — CPU-Optimised Entry Point
=======================================
Quantização : int8  (pesos + ativações em int8, ~4× menos RAM vs float32)
Device       : CPU
RAM mínima   : ~1.5 GB (medium) | ~3 GB (large-v3)

Usa todos os núcleos disponíveis automaticamente.

Uso:
    python stt_cpu.py audio.wav
    python stt_cpu.py audio.wav --model small --max-speakers 3
"""
from __future__ import annotations

import argparse
import os

import torch

# Carrega .env antes de qualquer importação do pipeline
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from stt_pipeline import AudioTranscriber, PipelineConfig


# ──────────────────────────────────────────────────────
def _configure_cpu() -> int:
    """Optimiza threads PyTorch para CPU e exibe info do sistema."""
    n_threads = os.cpu_count() or 4

    # Força PyTorch a usar todos os núcleos
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))

    # Informações de memória RAM (Linux/Windows)
    ram_info = ""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1_073_741_824
        ram_avail = psutil.virtual_memory().available / 1_073_741_824
        ram_info = f"  RAM       : {ram_avail:.1f} GB disponível / {ram_gb:.1f} GB total\n"
    except ImportError:
        pass

    print("\n" + "═" * 52)
    print("  CPU STT Pipeline  —  int8 Quantisation")
    print("═" * 52)
    print(f"  CPU       : {n_threads} threads disponíveis")
    if ram_info:
        print(ram_info, end="")
    print(f"  PyTorch   : {torch.__version__}")
    print("═" * 52 + "\n")

    return n_threads


# ──────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STT Pipeline — CPU (int8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio",
                   help="Arquivo de áudio (WAV / MP3 / FLAC / OGG …)")
    p.add_argument("--model", default="medium",
                   choices=["tiny", "base", "small", "medium",
                            "large-v2", "large-v3"],
                   help="Tamanho do modelo Whisper")
    p.add_argument("--output", default="transcription_cpu.json",
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
    n_threads = _configure_cpu()
    args = _build_parser().parse_args()

    token = args.hf_token or os.getenv("HF_TOKEN", "")

    cfg = PipelineConfig(
        # ── quantização CPU ──────────────────────────
        device="cpu",
        compute_type="int8",           # máxima compressão, sem GPU
        cpu_threads=n_threads,
        num_workers=min(4, n_threads), # não exceder núcleos físicos
        # ── modelo ──────────────────────────────────
        whisper_model=args.model,
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
    print("  Resultado CPU")
    print("─" * 52)
    print(f"  Segmentos  : {meta['total_segments']}")
    print(f"  Speakers   : {meta['unique_speakers']}")
    print(f"  Idioma     : {meta['language_detected']}")
    print(f"  Modelo     : {meta['whisper_model']}  [{meta['compute_type']}]")
    print(f"  Threads    : {n_threads}")
    print(f"  Saída      : {args.output}")
    print("─" * 52 + "\n")


if __name__ == "__main__":
    main()
