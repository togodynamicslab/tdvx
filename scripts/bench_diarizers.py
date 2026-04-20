#!/usr/bin/env python3
"""
Pyannote vs NVIDIA Streaming Sortformer — side-by-side diarization benchmark.

Loads BOTH backends (skips whichever can't load, logs why) and runs each on the
same WAV inputs, capturing warm-run wall-clock latency and comparing speaker
timelines (unique speakers, total speech-tagged seconds, segment counts).

Typical use:
    python3 scripts/bench_diarizers.py \
        --audio tests/corpus/pt-BR/some.wav \
        --buffer-seconds 10 \
        --repeats 5

Outputs:
    stdout  — per-file table (backend | p50_ms | p95_ms | speakers | speech_s | segments)
    bench_results.json — all raw timings + per-window outputs for offline analysis

Notes:
  * First run per (backend, window) is discarded as cold-start (model warmup,
    cache hydration, CUDA kernel compilation).
  * --buffer-seconds W slices the input into rolling windows of length W to
    simulate the streaming case. Without it, the whole file is one call.
  * Sortformer requires `nemo_toolkit[asr]` on the target box — not a hard
    dep of this repo. See app/services/sortformer_service.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Allow running from repo root without install: add repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bench")


# ----------------------------- audio I/O ------------------------------------


def load_wav_mono_16k(path: Path) -> Tuple[np.ndarray, int]:
    """Load a WAV as float32 mono at 16kHz (resample if needed)."""
    try:
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError("soundfile not installed") from e

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)

    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
        except ImportError:
            raise RuntimeError(
                f"Audio is {sr}Hz but librosa isn't available to resample to 16kHz. "
                f"Pre-resample or install librosa."
            )
        sr = 16000
    return audio, sr


def window_slices(
    audio: np.ndarray, sample_rate: int, buffer_seconds: Optional[float]
) -> List[Tuple[float, np.ndarray]]:
    """Return [(window_start_seconds, window_audio), ...] for the benchmark.

    When buffer_seconds is None → one window covering the full file.
    Otherwise → non-overlapping slices (the final partial slice is kept).
    """
    if buffer_seconds is None or buffer_seconds <= 0:
        return [(0.0, audio)]

    step = int(round(buffer_seconds * sample_rate))
    if step <= 0:
        return [(0.0, audio)]

    slices: List[Tuple[float, np.ndarray]] = []
    for start in range(0, len(audio), step):
        chunk = audio[start : start + step]
        if chunk.size == 0:
            continue
        slices.append((start / float(sample_rate), chunk))
    return slices


# --------------------------- backend adapters -------------------------------


class PyannoteBackend:
    name = "pyannote"

    def __init__(self) -> None:
        self.svc = None
        self.load_error: Optional[str] = None

    def load(self) -> bool:
        try:
            from app.services.diarization_service import diarization_service
            diarization_service.load_pipeline()
            if diarization_service.pipeline is None:
                self.load_error = "pipeline failed to load (check PYANNOTE_AUTH_TOKEN)"
                return False
            self.svc = diarization_service
            return True
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            return False

    def diarize(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        return self.svc.diarize_audio(audio, sample_rate=sample_rate)


class SortformerBackend:
    name = "sortformer"

    def __init__(self) -> None:
        self.svc = None
        self.load_error: Optional[str] = None

    def load(self) -> bool:
        try:
            from app.services.sortformer_service import sortformer_service
            if not sortformer_service.is_available():
                self.load_error = (
                    sortformer_service._load_failed_reason or "unknown load failure"
                )
                return False
            self.svc = sortformer_service
            return True
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            return False

    def diarize(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        return self.svc.diarize_audio(audio, sample_rate=sample_rate)


# --------------------------- timeline stats ---------------------------------


def timeline_stats(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact descriptors for a diarization output: # speakers, speech seconds,
    segment count. Used to spot-check that the two backends agree on structure
    even when latency differs."""
    if not segments:
        return {"speakers": 0, "speech_s": 0.0, "segments": 0}
    speakers = {s["speaker"] for s in segments}
    speech = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    return {
        "speakers": len(speakers),
        "speech_s": round(speech, 3),
        "segments": len(segments),
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    # statistics.quantiles with method='inclusive' gives nearest-rank style.
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(p * (len(ordered) - 1)))))
    return ordered[k]


# --------------------------- benchmark loop ---------------------------------


def bench_backend_on_windows(
    backend: Any,
    windows: List[Tuple[float, np.ndarray]],
    sample_rate: int,
    repeats: int,
) -> Dict[str, Any]:
    """Run `repeats` trials across all windows. First trial discarded (cold)."""
    all_latencies_ms: List[float] = []
    warm_latencies_ms: List[float] = []
    last_segments: List[List[Dict[str, Any]]] = []

    for trial in range(repeats):
        trial_latencies: List[float] = []
        trial_outputs: List[List[Dict[str, Any]]] = []
        for (_start_s, chunk) in windows:
            t0 = time.perf_counter()
            try:
                segs = backend.diarize(chunk, sample_rate)
            except Exception as e:
                logger.exception("%s diarize failed on trial %d", backend.name, trial)
                segs = []
                trial_latencies.append(float("nan"))
                trial_outputs.append(segs)
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            trial_latencies.append(dt_ms)
            trial_outputs.append(segs)

        all_latencies_ms.extend([x for x in trial_latencies if x == x])  # filter NaN
        if trial > 0:  # discard cold trial
            warm_latencies_ms.extend([x for x in trial_latencies if x == x])
        last_segments = trial_outputs

    # Aggregate segments across windows for whole-file timeline stats.
    merged: List[Dict[str, Any]] = []
    for (start_s, _chunk), segs in zip(windows, last_segments):
        for s in segs:
            merged.append({
                "start": start_s + float(s["start"]),
                "end": start_s + float(s["end"]),
                "speaker": s["speaker"],
            })

    stats = timeline_stats(merged)
    lat_pool = warm_latencies_ms if warm_latencies_ms else all_latencies_ms

    return {
        "backend": backend.name,
        "trials": repeats,
        "windows_per_trial": len(windows),
        "p50_ms": round(percentile(lat_pool, 0.50), 2) if lat_pool else None,
        "p95_ms": round(percentile(lat_pool, 0.95), 2) if lat_pool else None,
        "mean_ms": round(statistics.fmean(lat_pool), 2) if lat_pool else None,
        "warm_latencies_ms": [round(x, 2) for x in warm_latencies_ms],
        "all_latencies_ms": [round(x, 2) for x in all_latencies_ms],
        "timeline": stats,
        "merged_segments": merged,
    }


def format_table(file_rows: List[Dict[str, Any]]) -> str:
    """Render a compact fixed-width table."""
    header = f"| {'backend':<12} | {'p50_ms':>8} | {'p95_ms':>8} | {'speakers':>8} | {'speech_s':>8} | {'segments':>8} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [header, sep]
    for row in file_rows:
        tl = row["timeline"]
        lines.append(
            f"| {row['backend']:<12} | "
            f"{(row['p50_ms'] if row['p50_ms'] is not None else 'n/a'):>8} | "
            f"{(row['p95_ms'] if row['p95_ms'] is not None else 'n/a'):>8} | "
            f"{tl['speakers']:>8} | "
            f"{tl['speech_s']:>8} | "
            f"{tl['segments']:>8} |"
        )
    return "\n".join(lines)


# ------------------------------- main ---------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Diarizer benchmark: Pyannote vs Sortformer")
    ap.add_argument("--audio", nargs="+", required=True, help="One or more WAV paths")
    ap.add_argument("--repeats", type=int, default=3, help="Trials per window (first is cold, discarded)")
    ap.add_argument(
        "--buffer-seconds",
        type=float,
        default=None,
        help="Slice each WAV into rolling windows of this length to simulate streaming. "
             "Default: one call for the whole file.",
    )
    ap.add_argument("--output", type=str, default="bench_results.json", help="Where to write raw results JSON")
    ap.add_argument("--backends", nargs="+", default=["pyannote", "sortformer"],
                    help="Subset of backends to run")
    args = ap.parse_args()

    # Resolve audio paths up front.
    audio_paths = [Path(p).expanduser().resolve() for p in args.audio]
    for p in audio_paths:
        if not p.exists():
            logger.error("Audio not found: %s", p)
            return 2

    # Load backends.
    backends: List[Any] = []
    if "pyannote" in args.backends:
        backends.append(PyannoteBackend())
    if "sortformer" in args.backends:
        backends.append(SortformerBackend())

    loaded: List[Any] = []
    for b in backends:
        logger.info("Loading backend: %s", b.name)
        if b.load():
            logger.info("  %s loaded OK", b.name)
            loaded.append(b)
        else:
            logger.warning("  %s FAILED to load: %s", b.name, b.load_error)

    if not loaded:
        logger.error("No backends loaded; nothing to benchmark.")
        return 3

    results: Dict[str, Any] = {
        "config": {
            "repeats": args.repeats,
            "buffer_seconds": args.buffer_seconds,
            "backends_requested": args.backends,
            "backends_loaded": [b.name for b in loaded],
            "backends_failed": [
                {"name": b.name, "reason": b.load_error} for b in backends if b not in loaded
            ],
        },
        "files": [],
    }

    for audio_path in audio_paths:
        logger.info("=" * 70)
        logger.info("Benchmarking file: %s", audio_path)
        audio, sr = load_wav_mono_16k(audio_path)
        duration = len(audio) / float(sr)
        logger.info("  duration=%.2fs  sample_rate=%d", duration, sr)

        windows = window_slices(audio, sr, args.buffer_seconds)
        logger.info("  windows=%d (buffer_seconds=%s)", len(windows), args.buffer_seconds)

        file_rows: List[Dict[str, Any]] = []
        for backend in loaded:
            logger.info("  running %s (trials=%d)", backend.name, args.repeats)
            row = bench_backend_on_windows(backend, windows, sr, args.repeats)
            file_rows.append(row)
            logger.info(
                "    %s: p50=%s ms p95=%s ms speakers=%d speech=%.2fs segs=%d",
                backend.name, row["p50_ms"], row["p95_ms"],
                row["timeline"]["speakers"], row["timeline"]["speech_s"],
                row["timeline"]["segments"],
            )

        print()
        print(f"### {audio_path.name}  (duration={duration:.2f}s, windows={len(windows)})")
        print(format_table(file_rows))
        print()

        results["files"].append({
            "path": str(audio_path),
            "duration_s": round(duration, 3),
            "windows": len(windows),
            "rows": file_rows,
        })

    out_path = Path(args.output).expanduser().resolve()
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote raw results to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
