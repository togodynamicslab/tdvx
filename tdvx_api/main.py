"""
tdvx_api/main.py
================
API FastAPI dedicada a transcrição + diarização com salvamento automático
de dados para finetuning.

Rotas
-----
POST /transcribe  — envia arquivo de áudio, recebe diarização e salva
GET  /transcribe  — passa caminho local de arquivo, recebe diarização e salva
GET  /health      — status do serviço
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Adiciona o tdvx ao path para reutilizar os engines ────────────────────────
_TDVX_ROOT = Path(__file__).parent.parent / "tdvx"
sys.path.insert(0, str(_TDVX_ROOT))

# Carrega o .env do tdvx ANTES de importar app.config, pois Settings() é
# instanciado no nível do módulo e busca o .env no CWD atual (tdvx_api/).
_env_file = _TDVX_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from app.config import settings  # noqa: E402  (tdvx settings)
from app.services.processor import TranscriptionProcessor, preload_engines  # noqa: E402

from saver import FinetuningDatasetSaver  # noqa: E402  (local)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tdvx_api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TDvX Finetuning API",
    description=(
        "Transcrição + diarização com salvamento automático de dados "
        "para finetuning (manifest.jsonl + segmentos WAV por speaker)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────
_processor: TranscriptionProcessor | None = None
_saver: FinetuningDatasetSaver | None = None

DATASET_DIR = Path(__file__).parent / "finetuning_data"
MAX_FILE_MB = 200


def _get_processor() -> TranscriptionProcessor:
    global _processor
    if _processor is None:
        _processor = TranscriptionProcessor()
    return _processor


def _get_saver() -> FinetuningDatasetSaver:
    global _saver
    if _saver is None:
        _saver = FinetuningDatasetSaver(base_dir=DATASET_DIR)
    return _saver


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando TDvX Finetuning API …")
    logger.info("Dataset dir: %s", DATASET_DIR)
    preload_engines()
    _get_processor()
    _get_saver()
    logger.info("Startup concluído.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_audio(path: str) -> tuple[np.ndarray, float]:
    """Carrega áudio do disco, retorna (array_float32_16kHz, duration_s)."""
    audio, _ = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / 16000
    return audio, duration


def _process_and_save(audio: np.ndarray, source_name: str, original_path: str | None = None):
    """Roda o pipeline completo e persiste os dados para finetuning."""
    t0 = time.time()

    result = _get_processor().process_audio(audio, sample_rate=16000)

    saved = _get_saver().save(
        audio=audio,
        result=result,
        source_name=source_name,
        original_path=original_path,
    )

    elapsed = time.time() - t0
    logger.info(
        "Processado '%s' em %.2fs | segmentos=%d | entries salvas=%d",
        source_name,
        elapsed,
        len(result.segments),
        len(saved),
    )

    return result, saved


# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Verifica se o serviço está ativo."""
    import torch

    return {
        "status": "running",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_dir": str(DATASET_DIR),
    }


@app.post("/transcribe")
async def transcribe_upload(file: UploadFile = File(...)):
    """
    Recebe um arquivo de áudio via upload (multipart/form-data).

    - Roda transcrição + diarização
    - Salva segmentos WAV individuais por speaker em `finetuning_data/audio/`
    - Atualiza `finetuning_data/manifest.jsonl`
    - Retorna o JSON de diarização completo

    Formatos aceitos: WAV, MP3, M4A, FLAC, OGG
    """
    max_bytes = MAX_FILE_MB * 1024 * 1024
    content = await file.read()

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Máximo: {MAX_FILE_MB}MB",
        )

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        audio, _ = _load_audio(tmp_path)
        os.unlink(tmp_path)
        tmp_path = None

        result, saved_entries = _process_and_save(
            audio=audio,
            source_name=file.filename or "upload",
        )

        return {
            "transcription": result.model_dump(),
            "finetuning": {
                "entries_saved": len(saved_entries),
                "entry_ids": [e["id"] for e in saved_entries],
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Erro ao processar upload: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/transcribe")
async def transcribe_path(
    path: str = Query(..., description="Caminho absoluto ou relativo do arquivo de áudio no servidor"),
):
    """
    Processa um arquivo de áudio já presente no servidor (passa o caminho como query param).

    Exemplo:
        GET /transcribe?path=/home/user/audios/reuniao.wav
        GET /transcribe?path=C:/audios/entrevista.mp3

    - Roda transcrição + diarização
    - Salva segmentos WAV individuais por speaker em `finetuning_data/audio/`
    - Atualiza `finetuning_data/manifest.jsonl`
    - Retorna o JSON de diarização completo
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {path}")
    if not audio_path.is_file():
        raise HTTPException(status_code=400, detail=f"Caminho não é um arquivo: {path}")

    try:
        audio, _ = _load_audio(str(audio_path))

        result, saved_entries = _process_and_save(
            audio=audio,
            source_name=audio_path.name,
            original_path=str(audio_path),
        )

        return {
            "transcription": result.model_dump(),
            "finetuning": {
                "entries_saved": len(saved_entries),
                "entry_ids": [e["id"] for e in saved_entries],
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Erro ao processar arquivo '%s': %s", path, exc)
        raise HTTPException(status_code=500, detail=str(exc))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
