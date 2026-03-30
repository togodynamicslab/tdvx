from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class TranscriptionSegment(BaseModel):
    """Segmento de transcrição com speaker e confiança."""
    speaker: str
    start: float
    end: float
    text: str
    confidence: float = 0.0


class TranscriptionResponse(BaseModel):
    """Resposta do endpoint de transcrição de arquivo."""
    timestamp: datetime
    language: str
    duration: Optional[float] = None
    segments: List[TranscriptionSegment]


class ErrorResponse(BaseModel):
    """Resposta de erro."""
    error: str
    detail: Optional[str] = None
