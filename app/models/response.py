from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class TranscriptionSegment(BaseModel):
    """A single transcription segment with speaker and translation"""
    speaker: str
    start: float
    end: float
    text: str
    translation: Optional[str] = None


class SpeakerResolution(BaseModel):
    """Per-local-speaker resolution detail (one row per local speaker in chunk)."""
    local_label: str           # local label from this chunk (e.g. "SPEAKER_00")
    global_label: str          # resolved global label (e.g. "SPEAKER_03")
    is_new: bool               # True if this minted a new global speaker
    distance: Optional[float]  # cosine distance to matched centroid (None if new w/ empty registry)
    duration_s: float          # seconds of audio attributed to this local speaker in the chunk


class ChunkTelemetry(BaseModel):
    """Per-chunk model internals — what the pipeline actually did."""
    chunk_duration_s: float                       # input audio duration
    whisper_ms: int                               # time spent in Whisper inference
    diarization_ms: int                           # time in Pyannote (0 if skipped)
    embedding_ms: int                             # time extracting per-speaker embeddings
    alignment_ms: int                             # time merging diar + transcript
    total_ms: int                                 # end-to-end request time
    diarize_ran: bool                             # was Pyannote actually invoked?
    locals_detected: int                          # # of local speakers Pyannote found
    resolutions: List[SpeakerResolution] = []     # per-local-speaker registry decision
    registry_size: int = 0                        # global speakers known after this chunk
    model: Optional[str] = None                   # which whisper model
    worker: Optional[str] = None                  # which uvicorn worker handled it (gpu+slot)
    notes: List[str] = []                         # free-form: "skipped: dur<min", "embedding model unavailable", etc.
    # Rolling-buffer diarization telemetry (optional; 0/None when buffer mode not in use)
    buffer_seconds: float = 0.0                   # total audio in the rolling buffer at processing time
    speakers_in_buffer: int = 0                   # distinct speakers Pyannote saw across the WHOLE buffer
    chunk_offset_seconds: float = 0.0             # where this chunk starts within the buffer timeline
    flush_reason: Optional[str] = None            # client-set: "vad" | "cap" | "stop" (None server-side)


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoints"""
    timestamp: datetime
    original_language: str  # 'pt' or 'en'
    target_language: str    # 'en' or 'pt'
    duration: Optional[float] = None
    segments: List[TranscriptionSegment]
    telemetry: Optional[ChunkTelemetry] = None


class LiveTranscriptionChunk(BaseModel):
    """Real-time transcription chunk for WebSocket streaming"""
    chunk_id: int
    timestamp: datetime
    original_language: str
    target_language: str
    segment: TranscriptionSegment
    is_final: bool = False  # True when audio stream ends
    telemetry: Optional[ChunkTelemetry] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
