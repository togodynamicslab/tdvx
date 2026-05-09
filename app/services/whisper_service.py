import logging
import os
import sys
from typing import Dict

import numpy as np
import torch
from faster_whisper import WhisperModel

from app.config import settings
from app.models.model_config import ModelType, get_model_config

logger = logging.getLogger(__name__)

# Windows-only ffmpeg PATH shim (skipped on Linux containers)
if sys.platform == "win32":
    ffmpeg_path = r"C:\Users\Matheus\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"
    if ffmpeg_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Added ffmpeg to PATH: {ffmpeg_path}")

logger.info("===== Torch import check =====")
logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Torch file: {torch.__file__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
logger.info("=============================")


class WhisperService:
    """Faster-Whisper transcription service (used by all pipelines)."""

    def __init__(self, model_size: str = "small"):
        self.model = None
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Whisper will use device: {self.device}")

    def load_model(self):
        if self.model is not None:
            return
        logger.info(f"Loading Faster-Whisper model: {self.model_size}")
        compute_type = "int8_float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=compute_type,
        )
        logger.info("Faster-Whisper model loaded successfully")
        logger.info(f"Supported languages: {settings.supported_languages}")
        logger.info(f"Live beam size: {settings.live_beam_size}, File beam size: {settings.file_beam_size}")

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        if self.model is None:
            self.load_model()
        try:
            if sample_rate != 16000:
                logger.warning(f"Audio sample rate is {sample_rate}Hz, Whisper expects 16kHz")
            segments_iter, info = self.model.transcribe(
                audio_data,
                language=None,
                task="transcribe",
                beam_size=settings.live_beam_size,
                vad_filter=False,
            )
            segments, full_text = [], []
            for segment in segments_iter:
                segments.append({"start": segment.start, "end": segment.end, "text": segment.text})
                full_text.append(segment.text)
            return {
                "text": " ".join(full_text).strip(),
                "language": info.language if info.language else "unknown",
                "segments": segments,
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"text": "", "language": "unknown", "segments": []}

    def transcribe_file(self, audio_path: str) -> Dict:
        if self.model is None:
            self.load_model()
        try:
            logger.info(f"Transcribing file at: {audio_path}")
            if os.path.exists(audio_path):
                logger.info(f"File size: {os.path.getsize(audio_path)} bytes")
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=None,
                task="transcribe",
                beam_size=settings.file_beam_size,
                vad_filter=False,
            )
            segments, full_text = [], []
            for segment in segments_iter:
                segments.append({"start": segment.start, "end": segment.end, "text": segment.text})
                full_text.append(segment.text)
            return {
                "text": " ".join(full_text).strip(),
                "language": info.language if info.language else "unknown",
                "segments": segments,
            }
        except Exception as e:
            import traceback
            logger.error(f"File transcription error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"text": "", "language": "unknown", "segments": []}


_SERVICES: dict[str, WhisperService] = {}


def get_or_create_whisper_service(model_type: str | None = None) -> WhisperService:
    """Get or lazily create a cached WhisperService for the given model type."""
    if model_type is None:
        model_type = settings.default_model
    model_type = model_type.lower()
    # Canonicalize aliases (e.g., tdv1-balanced → tdv1-medium)
    cfg = get_model_config(model_type)
    cache_key = cfg.whisper_model  # one service per model size
    if cache_key not in _SERVICES:
        _SERVICES[cache_key] = WhisperService(model_size=cfg.whisper_model)
    return _SERVICES[cache_key]


# Backwards-compat alias: some modules still reference `whisper_service`.
# Points at the default model's singleton (lazily loaded).
whisper_service = get_or_create_whisper_service(settings.default_model)
