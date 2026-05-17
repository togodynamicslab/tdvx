#!/usr/bin/env python3
"""
labeler/app.py — Interface web para transcrição + rotulação de áudio/vídeo.

Fluxo:
  1. Usuário faz upload de vídeo ou áudio
  2. Extrai áudio em WAV mono 16 kHz (ffmpeg)
  3. Transcreve com o modelo TDvX CT2 (faster-whisper)
  4. Usuário valida/edita segmentos na interface waveform
  5. Exporta pares WAV + TXT por segmento (prontos para novo treino)

Uso:
  pip install -r requirements-labeler.txt
  python labeler/app.py
  # Abrir http://localhost:7860
"""
import asyncio
import io
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_executor = ThreadPoolExecutor(max_workers=2)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
STATIC_DIR  = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"

UPLOADS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# ── Modelo CT2 ────────────────────────────────────────────────────────────────
# Procura o modelo CT2 do TDvX nos locais esperados
_MODEL_CANDIDATES = [
    Path(__file__).parent.parent / "models" / "tdvx-v4-cv-pt-ct2",
    Path(__file__).parent.parent / "models" / "tdv3-cv-pt-v1-ct2",
    Path(__file__).parent.parent / "models" / "tdv1-cv-pt-ct2",
]
_MODEL_PATH: Optional[Path] = None
for _p in _MODEL_CANDIDATES:
    if _p.exists():
        _MODEL_PATH = _p
        break

_whisper_model = None  # carregado sob demanda


def _load_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    if _MODEL_PATH is None:
        raise RuntimeError(
            "Nenhum modelo CT2 encontrado. Certifique-se de que o modelo está em "
            f"tdvx/models/tdvx-v4-cv-pt-ct2 (ou variantes). Candidatos: {_MODEL_CANDIDATES}"
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper não instalado. Execute: pip install faster-whisper")

    device      = "cuda" if _cuda_available() else "cpu"
    compute     = "int8_float16" if device == "cuda" else "int8"
    print(f"[labeler] Carregando modelo: {_MODEL_PATH} ({device}, {compute})")
    _whisper_model = WhisperModel(str(_MODEL_PATH), device=device, compute_type=compute)
    return _whisper_model


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Áudio helpers ──────────────────────────────────────────────────────────────

def extract_audio(src: Path, dst: Path) -> None:
    """Converte qualquer vídeo/áudio para WAV mono 16 kHz via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", "16000", "-ac", "1",
        "-vn", str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou: {result.stderr.decode()[:500]}")


def transcribe(audio_path: Path) -> List[dict]:
    """Transcreve áudio e retorna lista de segmentos {start, end, text}."""
    model = _load_model()
    segments, _ = model.transcribe(
        str(audio_path),
        language="pt",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return [
        {"id": i, "start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()}
        for i, seg in enumerate(segments)
    ]


def export_segments(audio_path: Path, segments: List[dict], export_dir: Path) -> List[Path]:
    """Corta o WAV em pares (segmento.wav + segmento.txt) e retorna os caminhos."""
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise RuntimeError("soundfile não instalado: pip install soundfile")

    data, sr = sf.read(str(audio_path))
    exported = []

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        start_sample = int(seg["start"] * sr)
        end_sample   = int(seg["end"]   * sr)
        clip         = data[start_sample:end_sample]

        name     = f"seg_{seg['id']:04d}"
        wav_path = export_dir / f"{name}.wav"
        txt_path = export_dir / f"{name}.txt"

        sf.write(str(wav_path), clip, sr)
        txt_path.write_text(text, encoding="utf-8")
        exported.append(wav_path)

    return exported


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="TDvX Labeler", version="1.0")

# Armazena jobs em memória (suficiente para uso local)
_jobs: dict[str, dict] = {}


class ExportRequest(BaseModel):
    segments: List[dict]


def _process_job(job_id: str, orig: Path, wav_path: Path) -> None:
    """Executa extração + transcrição em thread separada."""
    job = _jobs[job_id]
    try:
        job["status"] = "extracting"
        extract_audio(orig, wav_path)
        job["status"] = "transcribing"
        job["segments"] = transcribe(wav_path)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"]  = str(e)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Salva o arquivo, dispara extração+transcrição em background e retorna job_id."""
    job_id  = str(uuid.uuid4())[:8]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix   = Path(file.filename).suffix or ".bin"
    orig     = job_dir / f"original{suffix}"
    orig.write_bytes(await file.read())

    wav_path = job_dir / "audio.wav"
    _jobs[job_id] = {
        "wav":      str(wav_path),
        "filename": file.filename,
        "status":   "pending",
        "segments": [],
        "error":    "",
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _process_job, job_id, orig, wav_path)

    return {"job_id": job_id, "filename": file.filename}


@app.get("/status/{job_id}")
def status(job_id: str):
    """Retorna status do job: pending | extracting | transcribing | done | error."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return {
        "status":   job["status"],
        "segments": job["segments"] if job["status"] == "done" else [],
        "error":    job.get("error", ""),
    }


@app.get("/audio/{job_id}")
def get_audio(job_id: str):
    """Serve o arquivo WAV do job para o player."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return FileResponse(job["wav"], media_type="audio/wav")


@app.post("/export/{job_id}")
def export(job_id: str, body: ExportRequest):
    """Recebe segmentos editados e exporta pares WAV+TXT."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")

    export_dir = EXPORTS_DIR / job_id
    export_dir.mkdir(exist_ok=True)

    try:
        exported = export_segments(Path(job["wav"]), body.segments, export_dir)
    except Exception as e:
        raise HTTPException(500, f"Erro ao exportar: {e}")

    return {
        "exported": len(exported),
        "path": str(export_dir),
        "files": [p.name for p in exported],
    }


@app.get("/jobs")
def list_jobs():
    return [
        {"job_id": jid, "filename": j["filename"], "segments": len(j["segments"])}
        for jid, j in _jobs.items()
    ]


# Serve arquivos estáticos (HTML/JS/CSS) e a raiz como index.html
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"[labeler] Abrindo em http://localhost:{port}")
    if _MODEL_PATH:
        print(f"[labeler] Modelo: {_MODEL_PATH}")
    else:
        print(f"[labeler] AVISO: nenhum modelo CT2 encontrado em {_MODEL_CANDIDATES}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, app_dir=str(BASE_DIR))
