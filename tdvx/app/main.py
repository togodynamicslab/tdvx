from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import time
from typing import Optional

from app.config import settings
from app.services.processor import processor, preload_engines
from app.services.audio_buffer import SpeechBuffer
from app.models.response import TranscriptionResponse
from app.models.model_config import get_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TDvX Transcription API",
    description="Transcrição com diarização de speaker (TDv1-Fast)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando TDvX …")
    logger.info("Modelo Whisper: %s", settings.whisper_model)
    preload_engines()
    logger.info("Startup concluído.")


@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/upload.html")


@app.get("/upload.html", response_class=HTMLResponse)
async def upload_page():
    html_path = Path(__file__).parent.parent / "static" / "upload.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<html><body><h1>Upload</h1></body></html>"


@app.get("/health")
async def health():
    import torch
    return {
        "status": "running",
        "model": settings.whisper_model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "diarization": True,
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
        "quantization": "int8_float16 (GPU) / int8 (CPU)",
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(file: UploadFile = File(...)):
    """
    Transcreve um arquivo de áudio enviado via upload.
    Aceita: WAV, MP3, M4A, FLAC, etc.
    """
    request_start = time.time()
    logger.info("Arquivo recebido: %s", file.filename)

    max_size = settings.max_audio_file_size_mb * 1024 * 1024
    content = await file.read()

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Máximo: {settings.max_audio_file_size_mb}MB",
        )

    import tempfile
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        import librosa
        audio, _ = librosa.load(tmp_path, sr=16000, mono=True)
        os.unlink(tmp_path)
        tmp_path = None

        result = processor.process_audio(audio, sample_rate=16000)

        logger.info(
            "Requisição concluída em %.2fs | segmentos=%d",
            time.time() - request_start,
            len(result.segments),
        )
        return result

    except Exception as e:
        logger.error("Erro na transcrição: %s", e)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


async def _process_utterance(
    websocket: WebSocket,
    utterance: np.ndarray,
    chunk_id: int,
    time_offset: float,
) -> None:
    """Roda processor.process_audio() em thread executor e transmite os segmentos."""
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

    Protocolo do cliente:
      - Enviar chunks de bytes (float32 little-endian, 16 kHz, mono)
      - Enviar b'' (bytes vazio) para sinalizar fim de stream
    O servidor retorna JSON com cada segmento transcrito.
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
    time_offset = 0.0           # acumula duração real enviada para offset de timestamps
    pending: list[asyncio.Task] = []

    try:
        while True:
            data = await websocket.receive_bytes()

            # Bytes vazio = fim de stream — flush do buffer
            if len(data) == 0:
                logger.info("WebSocket: fim de stream recebido, flushing buffer")
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
                    _process_utterance(websocket, utterance, chunk_id, time_offset - len(utterance) / 16000.0)
                )
                pending.append(task)

        # Aguarda todos os tasks de processamento em paralelo
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


@app.websocket("/transcribe/live")
async def ws_transcribe_live(websocket: WebSocket):
    """
    WebSocket para áudio de browser (WebM/Opus, chunks brutos).

    O cliente envia chunks de WebM/Opus; o servidor decodifica via ffmpeg,
    alimenta o SpeechBuffer e transmite segmentos JSON em tempo real.
    Enviar b'' para sinalizar fim de stream.
    """
    await websocket.accept()
    logger.info("WebSocket /transcribe/live conectado")

    import subprocess
    import io

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
        """Decodifica WebM/Opus para float32 16 kHz usando ffmpeg."""
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", "pipe:0",
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "f32le",
                    "pipe:1",
                ],
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
                    _process_utterance(websocket, utterance, chunk_id, time_offset - len(utterance) / 16000.0)
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
