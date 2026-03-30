#!/usr/bin/env python3
"""
record_mic.py — Grava do microfone e transcreve em tempo real
==============================================================
Dois modos de gravação:
  • Padrão   : grava até você pressionar ENTER
  • --duration N : grava por N segundos e para automaticamente

Uso:
    python record_mic.py                          # grava até Enter
    python record_mic.py --duration 30            # grava 30s
    python record_mic.py --list-devices           # lista microfones
    python record_mic.py --device 2 --model small # microfone específico
"""
from __future__ import annotations

import os
import sys
import argparse
import tempfile
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from stt_pipeline import AudioTranscriber, PipelineConfig

SAMPLE_RATE = 16_000   # Whisper foi treinado em 16 kHz


# ──────────────────────────────────────────────────────
# Dispositivos
# ──────────────────────────────────────────────────────

def list_devices() -> None:
    default_in = sd.default.device[0]
    print("\n── Dispositivos de entrada (microfone) ─────────────────────────")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            marker = "  ◄ padrão" if i == default_in else ""
            print(f"  [{i:2d}]  {d['name']}{marker}")
    print()


# ──────────────────────────────────────────────────────
# Gravação
# ──────────────────────────────────────────────────────

def record_until_enter(device: int | None) -> np.ndarray:
    """Grava continuamente até o usuário pressionar ENTER."""
    print("\n" + "─" * 54)
    print("  GRAVANDO  —  Pressione  ENTER  para parar")
    print("─" * 54 + "\n")

    chunks: list[np.ndarray] = []
    lock = threading.Lock()

    def callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"  [aviso de áudio] {status}", flush=True)
        with lock:
            chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
        blocksize=2048,
    ):
        input()   # bloqueia até Enter

    audio = np.concatenate(chunks, axis=0).flatten()
    return audio


def record_fixed(duration: int, device: int | None) -> np.ndarray:
    """Grava por `duration` segundos e para automaticamente."""
    print(f"\n  Gravando por {duration} segundos... ", end="", flush=True)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("pronto.\n")
    return audio.flatten()


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _audio_stats(audio: np.ndarray) -> None:
    dur = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"  Duração gravada : {dur:.1f}s")
    print(f"  Volume RMS      : {rms:.4f}  ", end="")
    if rms < 0.001:
        print("⚠  muito baixo — verifique o microfone")
    elif rms < 0.01:
        print("(baixo)")
    else:
        print("(ok)")


def _build_config(args: argparse.Namespace, token: str) -> PipelineConfig:
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8_float16" if device == "cuda" else "int8"
    return PipelineConfig(
        device=device,
        compute_type=compute,
        whisper_model=args.model,
        hf_token=token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        vad_filter=True,
        vad_min_silence_duration_ms=400,
    )


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gravação de microfone → STT + Diarização",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--duration", type=int, default=None,
                   metavar="SEG",
                   help="Gravar N segundos (padrão: grava até ENTER)")
    p.add_argument("--device", type=int, default=None,
                   metavar="ID",
                   help="ID do microfone (use --list-devices para ver)")
    p.add_argument("--model", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="Modelo Whisper")
    p.add_argument("--output", default="transcription_mic.json",
                   metavar="ARQUIVO",
                   help="Arquivo JSON de saída")
    p.add_argument("--hf-token", default="",
                   metavar="TOKEN",
                   help="Token HuggingFace (fallback: HF_TOKEN env / .env)")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--list-devices", action="store_true",
                   help="Lista microfones disponíveis e sai")
    p.add_argument("--keep-wav", action="store_true",
                   help="Salvar o WAV gravado (gravacao_mic.wav) além do JSON")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    list_devices()

    # ── Gravar ───────────────────────────────────────────
    if args.duration:
        audio = record_fixed(args.duration, args.device)
    else:
        audio = record_until_enter(args.device)

    _audio_stats(audio)

    if len(audio) / SAMPLE_RATE < 0.5:
        print("\n[ERRO] Áudio muito curto (< 0.5s). Abortando.")
        sys.exit(1)

    # ── Salvar WAV temporário ─────────────────────────────
    if args.keep_wav:
        wav_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gravacao_mic.wav"
        )
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"  WAV salvo      : {wav_path}")
        tmp_path = wav_path
        tmp_file = None
    else:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_file.name, audio, SAMPLE_RATE)
        tmp_path = tmp_file.name

    # ── Transcrever ───────────────────────────────────────
    token = args.hf_token or os.getenv("HF_TOKEN", "")
    cfg   = _build_config(args, token)

    print("\n  Iniciando pipeline STT...\n")
    transcriber = AudioTranscriber(cfg)
    result      = transcriber.process(tmp_path)
    AudioTranscriber.save_json(result, args.output)

    # Limpa temp
    if tmp_file is not None:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass

    # ── Exibir resultado ──────────────────────────────────
    print("\n" + "═" * 60)
    print("  TRANSCRIÇÃO")
    print("═" * 60)
    if not result["segments"]:
        print("  (nenhuma fala detectada — verifique o microfone)")
    for seg in result["segments"]:
        print(
            f"  [{seg['timestamp_start']:6.2f}s → {seg['timestamp_end']:6.2f}s]"
            f"  {seg['user_id']:<14}"
            f"  [{seg['language_detected']}]"
            f"  {seg['text']}"
        )

    meta = result["metadata"]
    print("─" * 60)
    print(f"  Speakers  : {meta['unique_speakers']}")
    print(f"  Idioma    : {meta['language_detected']}")
    print(f"  Segmentos : {meta['total_segments']}")
    print(f"  Modelo    : {meta['whisper_model']}  [{meta['compute_type']}]")
    print(f"  JSON      : {args.output}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
