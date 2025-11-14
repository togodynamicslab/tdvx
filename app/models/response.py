from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class TranscriptionSegment(BaseModel):
    """A single transcription segment with speaker and translation"""
    speaker: str
    start: float
    end: float
    text: str
    translation: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoints"""
    timestamp: datetime
    original_language: str  # 'pt' or 'en'
    target_language: str    # 'en' or 'pt'
    duration: Optional[float] = None
    segments: List[TranscriptionSegment]


class LiveTranscriptionChunk(BaseModel):
    """Real-time transcription chunk for WebSocket streaming"""
    chunk_id: int
    timestamp: datetime
    original_language: str
    target_language: str
    segment: TranscriptionSegment
    is_final: bool = False  # True when audio stream ends


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
