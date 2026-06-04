#!/usr/bin/env python3
"""
api_test.py — API de testes para load test com 150 usuários simultâneos
=======================================================================

Endpoints:
    GET  /health                 — status + métricas em tempo real
    POST /transcribe             — transcrição (arquivo de áudio)
    POST /transcribe-diarize     — transcrição + diarização (SortFormer NeMo)
    GET  /metrics                — JSON com latências, throughput, error rate

Uso:
    MODEL_PATH=./models/tdvx-v1.6/tdvx-v1.6-ct2 \
    uvicorn api_test:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Configuração via env
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH     = os.environ.get("MODEL_PATH",     "./models/tdvx-v1.6/tdvx-v1.6-ct2")
DEVICE         = os.environ.get("DEVICE",         "cuda")
COMPUTE_TYPE   = os.environ.get("COMPUTE_TYPE",   "int8_float16")
HF_TOKEN       = os.environ.get("HF_TOKEN",       os.environ.get("PYANNOTE_AUTH_TOKEN", ""))
SORTFORMER_MODEL = os.environ.get("SORTFORMER_MODEL", "nvidia/diar_sortformer_4spk-v1")

# ─────────────────────────────────────────────────────────────────────────────
# Métricas em memória (ring buffer — últimos 1000 requests)
# ─────────────────────────────────────────────────────────────────────────────

_metrics: Dict[str, Any] = {
    "requests_total":  0,
    "requests_ok":     0,
    "requests_error":  0,
    "latencies_ms":    deque(maxlen=1000),
    "start_time":      time.time(),
}
_metrics_lock = asyncio.Lock()


async def _record(latency_ms: float, ok: bool) -> None:
    async with _metrics_lock:
        _metrics["requests_total"] += 1
        if ok:
            _metrics["requests_ok"] += 1
        else:
            _metrics["requests_error"] += 1
        _metrics["latencies_ms"].append(latency_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Modelos — lazy load, thread-safe
# ─────────────────────────────────────────────────────────────────────────────

_whisper = None
_sortformer = None
_whisper_lock = asyncio.Lock()
_sortformer_lock = asyncio.Lock()


async def _get_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    async with _whisper_lock:
        if _whisper is not None:
            return _whisper
        from faster_whisper import WhisperModel
        log.info("Carregando Whisper: %s [%s/%s]", MODEL_PATH, DEVICE, COMPUTE_TYPE)
        _whisper = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
        log.info("Whisper pronto.")
    return _whisper


async def _get_sortformer():
    global _sortformer
    if _sortformer is not None:
        return _sortformer
    async with _sortformer_lock:
        if _sortformer is not None:
            return _sortformer
        try:
            from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
            log.info("Carregando SortFormer: %s", SORTFORMER_MODEL)
            model = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL)
            model.eval()
            _sortformer = model
            log.info("SortFormer pronto.")
        except Exception as exc:
            log.error("SortFormer não disponível: %s", exc)
            raise HTTPException(503, f"SortFormer indisponível: {exc}")
    return _sortformer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_bytes(data: bytes) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(io.BytesIO(data), sr=16_000, mono=True)
    return audio.astype(np.float32)


async def _run_whisper(audio: np.ndarray):
    model = await _get_whisper()
    loop = asyncio.get_event_loop()

    def _infer():
        segments, info = model.transcribe(
            audio,
            language="pt",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        return (
            [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
             for s in segments],
            info.language,
        )

    return await loop.run_in_executor(None, _infer)


async def _run_sortformer(audio: np.ndarray) -> List[Dict]:
    from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
    import ast

    model = await _get_sortformer()
    loop  = asyncio.get_event_loop()

    def _infer():
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16_000)
            tmp = f.name

        cfg = DiarizeConfig(max_num_of_spks=4, verbose=False, num_workers=0)
        results = model.diarize(audio=tmp, batch_size=1, override_config=cfg)
        os.unlink(tmp)

        turns = []
        for entry in (results[0] if results else []):
            parts = entry.strip().split()
            try:
                if len(parts) == 3:
                    label = parts[2].upper().replace("SPEAKER_", "SPK_")
                    turns.append({"start": float(parts[0]), "end": float(parts[1]), "speaker": label})
                else:
                    start, end, spk = ast.literal_eval(entry)
                    turns.append({"start": float(start), "end": float(end), "speaker": f"SPK_{int(spk):02d}"})
            except Exception:
                continue
        turns.sort(key=lambda t: t["start"])
        return turns

    return await loop.run_in_executor(None, _infer)


def _assign_speakers(segments: List[Dict], turns: List[Dict]) -> List[Dict]:
    for seg in segments:
        best, best_ov = "UNK", 0.0
        for t in turns:
            ov = max(0.0, min(seg["end"], t["end"]) - max(seg["start"], t["start"]))
            if ov > best_ov:
                best_ov, best = ov, t["speaker"]
        seg["speaker"] = best
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="TDvX Load Test API", version="1.6")


@app.get("/health")
async def health():
    lats = list(_metrics["latencies_ms"])
    uptime = time.time() - _metrics["start_time"]
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "device": DEVICE,
        "uptime_s": round(uptime, 1),
        "requests_total":  _metrics["requests_total"],
        "requests_ok":     _metrics["requests_ok"],
        "requests_error":  _metrics["requests_error"],
        "error_rate":      round(_metrics["requests_error"] / max(1, _metrics["requests_total"]), 4),
        "throughput_rps":  round(_metrics["requests_total"] / max(1, uptime), 2),
        "latency_p50_ms":  round(float(np.percentile(lats, 50)), 1) if lats else None,
        "latency_p95_ms":  round(float(np.percentile(lats, 95)), 1) if lats else None,
        "latency_p99_ms":  round(float(np.percentile(lats, 99)), 1) if lats else None,
    }


@app.get("/metrics")
async def metrics():
    return await health()


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query("pt"),
):
    t0 = time.perf_counter()
    try:
        audio = _load_audio_bytes(await file.read())
        segments, lang = await _run_whisper(audio)
        latency = (time.perf_counter() - t0) * 1000
        await _record(latency, ok=True)
        return {
            "language": lang,
            "duration_s": round(len(audio) / 16_000, 2),
            "latency_ms": round(latency, 1),
            "segments": segments,
        }
    except HTTPException:
        raise
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        await _record(latency, ok=False)
        raise HTTPException(500, str(exc))


@app.post("/transcribe-diarize")
async def transcribe_diarize(
    file: UploadFile = File(...),
    language: str = Query("pt"),
):
    t0 = time.perf_counter()
    try:
        audio = _load_audio_bytes(await file.read())

        (segments, lang), turns = await asyncio.gather(
            _run_whisper(audio),
            _run_sortformer(audio),
        )

        segments = _assign_speakers(segments, turns)
        latency  = (time.perf_counter() - t0) * 1000
        await _record(latency, ok=True)
        return {
            "language": lang,
            "duration_s": round(len(audio) / 16_000, 2),
            "latency_ms": round(latency, 1),
            "speakers": sorted({t["speaker"] for t in turns}),
            "segments": segments,
        }
    except HTTPException:
        raise
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        await _record(latency, ok=False)
        raise HTTPException(500, str(exc))


if __name__ == "__main__":
    uvicorn.run("api_test:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
