import whisper
from faster_whisper import WhisperModel
import torch
import numpy as np
from typing import Dict, List
from abc import ABC, abstractmethod
import logging
import os
from app.config import settings
from app.models.model_config import ModelType, get_model_config

logger = logging.getLogger(__name__)

# Add ffmpeg to PATH for Whisper
ffmpeg_path = r"C:\Users\Matheus\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"
if ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
    logger.info(f"Added ffmpeg to PATH: {ffmpeg_path}")

# Force check at import time
logger.info(f"===== Torch import check =====")
logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Torch file: {torch.__file__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
logger.info(f"=============================")


class BaseWhisperService(ABC):
    """Abstract base class for Whisper transcription services"""

    def __init__(self):
        self.model = None
        cuda_available = torch.cuda.is_available()
        self.device = "cuda" if cuda_available else "cpu"
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"Whisper will use device: {self.device}")

    @abstractmethod
    def load_model(self):
        """Load Whisper model (call once at startup)"""
        pass

    @abstractmethod
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Transcribe audio data.

        Args:
            audio_data: NumPy array of audio samples (float32, normalized to [-1, 1])
            sample_rate: Sample rate of audio (Whisper expects 16kHz)

        Returns:
            Dict with 'text', 'language', and 'segments' keys
        """
        pass

    @abstractmethod
    def transcribe_file(self, audio_path: str) -> Dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with transcription results
        """
        pass


class WhisperServiceOriginal(BaseWhisperService):
    """Original OpenAI Whisper implementation (for TDv1 - high quality)"""

    def __init__(self, model_size: str = "large-v3"):
        super().__init__()
        self.model_size = model_size

    def load_model(self):
        """Load OpenAI Whisper model"""
        if self.model is None:
            logger.info(f"Loading OpenAI Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("OpenAI Whisper model loaded successfully")

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Transcribe audio using OpenAI Whisper"""
        if self.model is None:
            self.load_model()

        try:
            # Whisper expects audio normalized to [-1, 1] at 16kHz
            if sample_rate != 16000:
                logger.warning(f"Audio sample rate is {sample_rate}Hz, Whisper expects 16kHz")

            # Transcribe with language detection
            result = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect language
                task="transcribe",  # Not translate, we'll do that separately
                fp16=(self.device == "cuda"),  # Use FP16 on GPU
                verbose=False
            )

            return {
                'text': result['text'].strip(),
                'language': result['language'],
                'segments': result['segments']
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': []
            }

    def transcribe_file(self, audio_path: str) -> Dict:
        """Transcribe audio file using OpenAI Whisper"""
        if self.model is None:
            self.load_model()

        try:
            logger.info(f"Transcribing file at: {audio_path}")
            logger.info(f"File exists: {os.path.exists(audio_path)}")
            if os.path.exists(audio_path):
                logger.info(f"File size: {os.path.getsize(audio_path)} bytes")

            result = self.model.transcribe(
                audio_path,
                language=None,
                task="transcribe",
                fp16=(self.device == "cuda"),
                verbose=False
            )

            return {
                'text': result['text'].strip(),
                'language': result['language'],
                'segments': result['segments']
            }

        except Exception as e:
            import traceback
            logger.error(f"File transcription error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': []
            }


class WhisperServiceFaster(BaseWhisperService):
    """Faster-Whisper implementation (for TDv1-Fast - real-time)"""

    def __init__(self, model_size: str = "small"):
        super().__init__()
        self.model_size = model_size

    def load_model(self):
        """Load Faster-Whisper model"""
        if self.model is None:
            logger.info(f"Loading Faster-Whisper model: {self.model_size}")
            # Faster-Whisper automatically downloads models to cache
            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type
            )
            logger.info("Faster-Whisper model loaded successfully")

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Transcribe audio using Faster-Whisper"""
        if self.model is None:
            self.load_model()

        try:
            # Whisper expects audio normalized to [-1, 1] at 16kHz
            if sample_rate != 16000:
                logger.warning(f"Audio sample rate is {sample_rate}Hz, Whisper expects 16kHz")

            # Faster-Whisper returns segments iterator
            segments_iter, info = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect language
                task="transcribe",
                beam_size=5,
                vad_filter=False  # We handle VAD separately
            )

            # Convert segments to list and build text
            segments = []
            full_text = []

            for segment in segments_iter:
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                })
                full_text.append(segment.text)

            return {
                'text': ' '.join(full_text).strip(),
                'language': info.language if info.language else 'unknown',
                'segments': segments
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': []
            }

    def transcribe_file(self, audio_path: str) -> Dict:
        """Transcribe audio file using Faster-Whisper"""
        if self.model is None:
            self.load_model()

        try:
            logger.info(f"Transcribing file at: {audio_path}")
            logger.info(f"File exists: {os.path.exists(audio_path)}")
            if os.path.exists(audio_path):
                logger.info(f"File size: {os.path.getsize(audio_path)} bytes")

            # Faster-Whisper can transcribe files directly
            segments_iter, info = self.model.transcribe(
                audio_path,
                language=None,
                task="transcribe",
                beam_size=5,
                vad_filter=False
            )

            # Convert segments to list and build text
            segments = []
            full_text = []

            for segment in segments_iter:
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                })
                full_text.append(segment.text)

            return {
                'text': ' '.join(full_text).strip(),
                'language': info.language if info.language else 'unknown',
                'segments': segments
            }

        except Exception as e:
            import traceback
            logger.error(f"File transcription error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': []
            }


# Factory function to get appropriate whisper service
def get_whisper_service(model_type: str = None) -> BaseWhisperService:
    """
    Get Whisper service instance based on model type.

    Args:
        model_type: "tdv1" or "tdv1-fast". If None, uses default from settings.

    Returns:
        BaseWhisperService instance (either WhisperServiceOriginal or WhisperServiceFaster)
    """
    if model_type is None:
        model_type = settings.default_model

    model_config = get_model_config(model_type)

    if model_config.uses_faster_whisper:
        logger.info(f"Using Faster-Whisper service with model: {model_config.whisper_model}")
        return WhisperServiceFaster(model_size=model_config.whisper_model)
    else:
        logger.info(f"Using OpenAI Whisper service with model: {model_config.whisper_model}")
        return WhisperServiceOriginal(model_size=model_config.whisper_model)


# Singleton instances (lazy loaded)
_whisper_service_tdv1 = None
_whisper_service_tdv1_balanced = None
_whisper_service_tdv1_fast = None


def get_or_create_whisper_service(model_type: str = None) -> BaseWhisperService:
    """
    Get or create cached Whisper service instance.

    Args:
        model_type: "tdv1", "tdv1-balanced", or "tdv1-fast". If None, uses default from settings.

    Returns:
        Cached BaseWhisperService instance
    """
    global _whisper_service_tdv1, _whisper_service_tdv1_balanced, _whisper_service_tdv1_fast

    if model_type is None:
        model_type = settings.default_model

    model_type = model_type.lower()

    if model_type == ModelType.TDV1:
        if _whisper_service_tdv1 is None:
            _whisper_service_tdv1 = get_whisper_service(ModelType.TDV1)
        return _whisper_service_tdv1
    elif model_type == ModelType.TDV1_BALANCED:
        if _whisper_service_tdv1_balanced is None:
            _whisper_service_tdv1_balanced = get_whisper_service(ModelType.TDV1_BALANCED)
        return _whisper_service_tdv1_balanced
    elif model_type == ModelType.TDV1_FAST:
        if _whisper_service_tdv1_fast is None:
            _whisper_service_tdv1_fast = get_whisper_service(ModelType.TDV1_FAST)
        return _whisper_service_tdv1_fast
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Legacy singleton instance (for backward compatibility)
# Uses the original service with the configured whisper_model
whisper_service = WhisperServiceOriginal(model_size=settings.whisper_model)
