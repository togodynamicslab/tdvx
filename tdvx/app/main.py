"""
main.py — TDvX: Transcrição + Diarização + Finetuning Dataset

Rotas:
  GET  /                       → redireciona para /upload.html
  GET  /upload.html            → GUI de upload de arquivo
  GET  /live                   → GUI de transcrição ao vivo
  GET  /health                 → status do serviço
  GET  /models                 → modelo ativo e quantização
  POST /transcribe             → transcreve arquivo (multipart) + salva dataset
  GET  /transcribe?path=...    → transcreve arquivo local + salva dataset
  GET  /finetuning/stats       → estatísticas do dataset acumulado
  DELETE /finetuning/{id}      → remove entrada do dataset
  WS   /ws/transcribe          → PCM float32 16 kHz mono (stream de bytes)
  WS   /transcribe/live        → WebM/Opus do browser (stream de bytes)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.model_config import get_model_config
from app.models.response import FinetuningInfo, TranscriptionResponse
from app.services.processor import preload_engines, processor
from app.services.audio_buffer import SpeechBuffer
from app.services.saver import FinetuningDatasetSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Finetuning data dir ────────────────────────────────────────────────────────
_TDVX_ROOT = Path(__file__).parent.parent

def _resolve_finetuning_dir() -> Path:
    if settings.finetuning_data_dir:
        return Path(settings.finetuning_data_dir)
    return _TDVX_ROOT / "finetuning_data"

# ── Singleton do saver ─────────────────────────────────────────────────────────
_saver: Optional[FinetuningDatasetSaver] = None

def _get_saver() -> FinetuningDatasetSaver:
    global _saver
    if _saver is None:
        _saver = FinetuningDatasetSaver(base_dir=_resolve_finetuning_dir())
    return _saver

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TDvX Transcription API",
    description="Transcrição com diarização de speaker + salvamento de dataset para finetuning.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = _TDVX_ROOT / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando TDvX v3 …")
    logger.info("Modelo Whisper: %s | compute_type: %s", settings.whisper_model, settings.whisper_compute_type or "auto")
    logger.info("Finetuning: %s | dir: %s", settings.finetuning_enabled, _resolve_finetuning_dir())
    preload_engines()
    if settings.finetuning_enabled:
        _get_saver()  # cria dirs ao iniciar
    logger.info("Startup concluído.")


# ── GUI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root():
    return RedirectResponse(url="/upload.html")


@app.get("/upload.html", response_class=HTMLResponse, include_in_schema=False)
async def upload_page():
    html_path = static_path / "upload.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="upload.html não encontrado em static/")


@app.get("/live", response_class=HTMLResponse, include_in_schema=False)
async def live_page():
    html_path = static_path / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="index.html não encontrado em static/")


# ── Health / Models ────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    import torch
    return {
        "status": "running",
        "model": settings.whisper_model,
        "compute_type": settings.whisper_compute_type or "auto",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "diarization": True,
        "finetuning_enabled": settings.finetuning_enabled,
        "finetuning_data_dir": str(_resolve_finetuning_dir()),
    }


@app.get("/models")
async def list_models():
    cfg = get_model_config()
    return {
        "active_model": "tdv1-fast",
        "name": cfg.name,
        "whisper_model": cfg.whisper_model,
        "description": cfg.description,
        "estimated_speed": cfg.estimated_speed,
        "quantization": settings.whisper_compute_type or "auto (int8_float16 GPU / int8 CPU)",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_audio_file(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio


def _run_and_save(audio: np.ndarray, source_name: str, original_path: str | None = None) -> TranscriptionResponse:
    """Roda o pipeline e opcionalmente salva finetuning data."""
    result = processor.process_audio(audio, sample_rate=16000)

    if settings.finetuning_enabled:
        saved = _get_saver().save(audio=audio, result=result, source_name=source_name, original_path=original_path)
        if saved:
            # Injeta info de finetuning na resposta (campo opcional)
            session_id = saved[0]["session_id"]
            result.finetuning = FinetuningInfo(
                session_id=session_id,
                entries_saved=len(saved),
                entry_ids=[e["id"] for e in saved],
            )

    return result


# ── Transcrição — upload de arquivo ───────────────────────────────────────────
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_upload(file: UploadFile = File(...)):
    """
    Transcreve um arquivo de áudio enviado via multipart upload.

    Formatos aceitos: WAV, MP3, M4A, FLAC, OGG (via librosa/ffmpeg)
    Retorna transcrição completa com diarização de speaker.
    Quando `finetuning_enabled=true`, salva automaticamente os segmentos no dataset.
    """
    t0 = time.time()
    logger.info("Arquivo recebido: %s", file.filename)

    max_bytes = settings.max_audio_file_size_mb * 1024 * 1024
    content = await file.read()

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Máximo: {settings.max_audio_file_size_mb}MB",
        )

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        audio = _load_audio_file(tmp_path)
        os.unlink(tmp_path)
        tmp_path = None

        result = _run_and_save(audio, source_name=file.filename or "upload")

        logger.info(
            "Upload concluído em %.2fs | segmentos=%d | finetuning=%s",
            time.time() - t0,
            len(result.segments),
            f"{result.finetuning.entries_saved} entries" if result.finetuning else "off",
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Erro ao transcrever upload: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Transcrição — arquivo local (path) ────────────────────────────────────────
@app.get("/transcribe", response_model=TranscriptionResponse)
async def transcribe_path(
    path: str = Query(..., description="Caminho absoluto ou relativo do arquivo de áudio no servidor"),
):
    """
    Transcreve um arquivo de áudio já presente no servidor.

    Exemplo:
        GET /transcribe?path=/home/user/audios/reuniao.wav
        GET /transcribe?path=C:/audios/entrevista.mp3
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {path}")
    if not audio_path.is_file():
        raise HTTPException(status_code=400, detail=f"Caminho não é um arquivo: {path}")

    t0 = time.time()
    try:
        audio = _load_audio_file(str(audio_path))
        result = _run_and_save(audio, source_name=audio_path.name, original_path=str(audio_path))

        logger.info(
            "Path transcrito em %.2fs | segmentos=%d",
            time.time() - t0, len(result.segments),
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Erro ao transcrever '%s': %s", path, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Finetuning — estatísticas e gestão ────────────────────────────────────────
@app.get("/finetuning/stats")
async def finetuning_stats():
    """Retorna estatísticas do dataset de finetuning acumulado."""
    if not settings.finetuning_enabled:
        raise HTTPException(status_code=503, detail="Finetuning está desabilitado (FINETUNING_ENABLED=false)")
    return _get_saver().stats()


@app.delete("/finetuning/{entry_id}")
async def delete_finetuning_entry(entry_id: str):
    """Remove uma entrada do dataset (apaga o WAV e remove do manifest)."""
    if not settings.finetuning_enabled:
        raise HTTPException(status_code=503, detail="Finetuning está desabilitado")
    removed = _get_saver().remove_entry(entry_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Entrada não encontrada: {entry_id}")
    return {"removed": entry_id}


# ── WebSocket — PCM float32 16kHz ─────────────────────────────────────────────
async def _process_utterance(
    websocket: WebSocket,
    utterance: np.ndarray,
    chunk_id: int,
    time_offset: float,
) -> None:
    """Roda processor em thread executor e transmite segmentos via WebSocket."""
    loop = asyncio.get_event_loop()
    try:
        result: TranscriptionResponse = await loop.run_in_executor(
            None, processor.process_audio, utterance, 16000
        )
    except Exception as exc:
        logger.error("Erro ao processar utterance chunk_id=%d: %s", chunk_id, exc)
        await websocket.send_text(json.dumps({"error": str(exc), "chunk_id": chunk_id}))
        return

    for seg in result.segments:
        payload = {
            "chunk_id": chunk_id,
            "timestamp": datetime.now().isoformat(),
            "original_language": result.language,
            "segment": {
                "speaker": seg.speaker,
                "start": round(seg.start + time_offset, 3),
                "end": round(seg.end + time_offset, 3),
                "text": seg.text,
                "confidence": seg.confidence,
            },
            "is_final": True,
        }
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket para transcrição em tempo real com PCM float32.

    Protocolo:
      - Enviar chunks de bytes (float32 little-endian, 16 kHz, mono)
      - Enviar ArrayBuffer vazio (b'') para sinalizar fim de stream
    """
    await websocket.accept()
    logger.info("WebSocket /ws/transcribe conectado")

    speech_buf = SpeechBuffer(
        aggressiveness=settings.vad_aggressiveness,
        silence_threshold_s=0.7,
        min_speech_s=0.4,
        max_speech_s=12.0,
    )

    chunk_id = 0
    time_offset = 0.0
    pending: list[asyncio.Task] = []

    try:
        while True:
            data = await websocket.receive_bytes()

            if len(data) == 0:
                logger.info("WebSocket: fim de stream, flushing buffer")
                remaining = speech_buf.get_remaining()
                if remaining is not None and len(remaining) > 0:
                    chunk_id += 1
                    task = asyncio.create_task(
                        _process_utterance(websocket, remaining, chunk_id, time_offset)
                    )
                    pending.append(task)
                break

            audio = np.frombuffer(data, dtype=np.float32)
            time_offset += len(audio) / 16000.0

            utterance = speech_buf.add_chunk(audio)
            if utterance is not None:
                chunk_id += 1
                task = asyncio.create_task(
                    _process_utterance(
                        websocket, utterance, chunk_id,
                        time_offset - len(utterance) / 16000.0,
                    )
                )
                pending.append(task)

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/transcribe desconectado")
        for t in pending:
            t.cancel()
    except Exception as exc:
        logger.error("Erro no WebSocket /ws/transcribe: %s", exc)
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass


# ── WebSocket — WebM/Opus do browser ──────────────────────────────────────────
@app.websocket("/transcribe/live")
async def ws_transcribe_live(websocket: WebSocket):
    """
    WebSocket para áudio de browser (WebM/Opus).
    Decodifica via ffmpeg → float32 16kHz → SpeechBuffer → transcrição.
    Enviar b'' para sinalizar fim de stream.
    """
    await websocket.accept()
    logger.info("WebSocket /transcribe/live conectado")

    import subprocess

    speech_buf = SpeechBuffer(
        aggressiveness=settings.vad_aggressiveness,
        silence_threshold_s=0.7,
        min_speech_s=0.4,
        max_speech_s=12.0,
    )

    chunk_id = 0
    time_offset = 0.0
    pending: list[asyncio.Task] = []

    def _decode_webm(raw_bytes: bytes) -> Optional[np.ndarray]:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
                input=raw_bytes,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode != 0 or len(proc.stdout) == 0:
                return None
            return np.frombuffer(proc.stdout, dtype=np.float32)
        except Exception as exc:
            logger.warning("ffmpeg decode falhou: %s", exc)
            return None

    try:
        while True:
            data = await websocket.receive_bytes()

            if len(data) == 0:
                remaining = speech_buf.get_remaining()
                if remaining is not None and len(remaining) > 0:
                    chunk_id += 1
                    task = asyncio.create_task(
                        _process_utterance(websocket, remaining, chunk_id, time_offset)
                    )
                    pending.append(task)
                break

            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, _decode_webm, data)
            if audio is None or len(audio) == 0:
                continue

            time_offset += len(audio) / 16000.0
            utterance = speech_buf.add_chunk(audio)
            if utterance is not None:
                chunk_id += 1
                task = asyncio.create_task(
                    _process_utterance(
                        websocket, utterance, chunk_id,
                        time_offset - len(utterance) / 16000.0,
                    )
                )
                pending.append(task)

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("WebSocket /transcribe/live desconectado")
        for t in pending:
            t.cancel()
    except Exception as exc:
        logger.error("Erro no WebSocket /transcribe/live: %s", exc)
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
