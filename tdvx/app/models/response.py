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


class FinetuningInfo(BaseModel):
    """Metadados do salvamento para finetuning."""
    session_id: str
    entries_saved: int
    entry_ids: List[str]


class TranscriptionResponse(BaseModel):
    """Resposta do endpoint de transcrição."""
    timestamp: datetime
    language: str
    duration: Optional[float] = None
    segments: List[TranscriptionSegment]
    finetuning: Optional[FinetuningInfo] = None  # presente quando save_finetuning=True


class ErrorResponse(BaseModel):
    """Resposta de erro."""
    error: str
    detail: Optional[str] = None
