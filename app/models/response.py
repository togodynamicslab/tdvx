from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str
    translation: Optional[str] = None
    is_final: Optional[bool] = False


class TranscriptionResponse(BaseModel):
    timestamp: datetime
    original_language: str
    target_language: str
    duration: Optional[float] = None
    segments: List[TranscriptionSegment] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    message: str
    code: Optional[int] = None
