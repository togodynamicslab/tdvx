"""
benchmark.py — Mede RTF e qualidade do pipeline TDvX v3.

Uso:
    python benchmark.py <audio_file>
    python benchmark.py audio.wav --json results.json
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.services.processor import processor
from app.config import settings


def load_audio(path: str) -> tuple[np.ndarray, float]:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / 16000
    return audio, duration


def run_benchmark(audio: np.ndarray, duration: float) -> dict:
    print(f"\n{'='*60}")
    print(f"  TDvX v3 — Whisper {settings.whisper_model} | {settings.whisper_compute_type or 'auto'}")
    print(f"{'='*60}")
    print(f"  Audio: {duration:.2f}s")

    t0 = time.perf_counter()
    result = processor.process_audio(audio, sample_rate=16000)
    elapsed = time.perf_counter() - t0

    rtf = elapsed / duration if duration > 0 else 0
    speakers = {seg.speaker for seg in result.segments}
    words = sum(len(seg.text.split()) for seg in result.segments)
    avg_conf = (
        sum(seg.confidence for seg in result.segments) / len(result.segments)
        if result.segments else 0.0
    )

    print(f"  Tempo:      {elapsed:.2f}s")
    print(f"  RTF:        {rtf:.3f}x  ({'mais rapido' if rtf < 1 else 'mais lento'} que real-time)")
    print(f"  Idioma:     {result.language}")
    print(f"  Segmentos:  {len(result.segments)}")
    print(f"  Palavras:   {words}")
    print(f"  Speakers:   {len(speakers)} — {sorted(speakers)}")
    print(f"  Confianca:  {avg_conf:.3f}")

    if result.segments:
        print(f"\n  Transcricao:")
        print(f"  {'-'*50}")
        for seg in result.segments:
            print(f"  [{seg.speaker}] {seg.start:.1f}s-{seg.end:.1f}s  {seg.text}")

    return {
        "timestamp": datetime.now().isoformat(),
        "audio_duration": round(duration, 3),
        "processing_time": round(elapsed, 3),
        "rtf": round(rtf, 4),
        "language": result.language,
        "segments": len(result.segments),
        "words": words,
        "speakers": sorted(speakers),
        "avg_confidence": round(avg_conf, 4),
        "whisper_model": settings.whisper_model,
        "compute_type": settings.whisper_compute_type or "auto",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TDvX v3")
    parser.add_argument("audio", help="Arquivo de audio (WAV, MP3, etc.)")
    parser.add_argument("--json", "-o", metavar="FILE", help="Salva resultado em JSON")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"[ERRO] Arquivo nao encontrado: {args.audio}")
        sys.exit(1)

    print(f"Carregando: {args.audio}")
    audio, duration = load_audio(args.audio)

    result = run_benchmark(audio, duration)

    print(f"\n{'='*60}")
    print(f"  RTF: {result['rtf']:.3f}x  |  {result['processing_time']:.1f}s para {result['audio_duration']:.1f}s de audio")
    print(f"{'='*60}")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Salvo em: {args.json}")


if __name__ == "__main__":
    main()
