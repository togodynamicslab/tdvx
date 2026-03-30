"""
audio_buffer.py — VAD-based utterance segmentation buffer.

SpeechBuffer accumulates audio frames and fires a complete utterance when:
  1. Silence is detected for >= silence_threshold_s after speech started, OR
  2. The buffer exceeds max_speech_s (hard cap for continuous speech).

This replaces the old fixed-time AudioBuffer approach and solves:
  - Text fragmentation: sentences no longer split at arbitrary time cuts
  - Speaker detection: pyannote receives the full utterance (better embeddings)
"""
import numpy as np
import webrtcvad
import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_FRAME_MS = 30          # webrtcvad supports 10 / 20 / 30 ms
_SAMPLE_RATE = 16000    # Whisper expects 16 kHz


class SpeechBuffer:
    """
    Frame-level VAD buffer that emits complete utterances.

    Args:
        aggressiveness:       webrtcvad aggressiveness 0-3 (3 = strictest)
        silence_threshold_s:  seconds of silence that ends an utterance
        min_speech_s:         minimum speech duration to emit (noise guard)
        max_speech_s:         hard cap — emit even if speech is continuous
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_threshold_s: float = 0.7,
        min_speech_s: float = 0.4,
        max_speech_s: float = 12.0,
    ) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_samples = int(_SAMPLE_RATE * _FRAME_MS / 1000)  # 480 samples

        self._silence_frames_needed = int(silence_threshold_s * 1000 / _FRAME_MS)
        self._min_speech_frames = int(min_speech_s * 1000 / _FRAME_MS)
        self._max_frames = int(max_speech_s * 1000 / _FRAME_MS)

        # State
        self._frames: list[np.ndarray] = []   # accumulated float32 frames
        self._raw: np.ndarray = np.array([], dtype=np.float32)  # sub-frame remainder
        self._in_speech: bool = False
        self._speech_frame_count: int = 0
        self._silence_frame_count: int = 0

        logger.info(
            "SpeechBuffer: aggressiveness=%d  silence=%.1fs  min=%.1fs  max=%.1fs",
            aggressiveness, silence_threshold_s, min_speech_s, max_speech_s,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def add_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Push raw audio (float32, 16 kHz) into the buffer.

        Returns a complete utterance (float32 numpy array) when a silence
        boundary is detected, otherwise returns None.

        Only one utterance is returned per call — call repeatedly if you need
        to drain multiple queued utterances (unlikely in normal use).
        """
        # Normalize int16-range input
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        self._raw = np.concatenate([self._raw, audio])

        while len(self._raw) >= self._frame_samples:
            frame = self._raw[: self._frame_samples]
            self._raw = self._raw[self._frame_samples :]
            utterance = self._process_frame(frame)
            if utterance is not None:
                return utterance

        return None

    def get_remaining(self) -> Optional[np.ndarray]:
        """Flush whatever is in the buffer (called on WebSocket disconnect)."""
        if self._frames and self._speech_frame_count >= self._min_speech_frames:
            return self._flush()
        self._reset_state()
        return None

    def clear(self) -> None:
        self._reset_state()
        self._raw = np.array([], dtype=np.float32)

    def get_buffer_duration(self) -> float:
        return (len(self._frames) * self._frame_samples + len(self._raw)) / _SAMPLE_RATE

    # ──────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        frame_int16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            is_speech = self._vad.is_speech(frame_int16.tobytes(), _SAMPLE_RATE)
        except Exception:
            is_speech = True  # assume speech on error

        self._frames.append(frame)

        if is_speech:
            self._in_speech = True
            self._speech_frame_count += 1
            self._silence_frame_count = 0
        elif self._in_speech:
            self._silence_frame_count += 1

        # Conditions to emit
        silence_boundary = (
            self._in_speech
            and self._silence_frame_count >= self._silence_frames_needed
            and self._speech_frame_count >= self._min_speech_frames
        )
        max_hit = len(self._frames) >= self._max_frames

        if silence_boundary or max_hit:
            if self._speech_frame_count >= self._min_speech_frames:
                return self._flush()
            else:
                self._reset_state()

        return None

    def _flush(self) -> np.ndarray:
        utterance = np.concatenate(self._frames)
        logger.info(
            "SpeechBuffer: utterance emitted %.2fs (speech_frames=%d, silence_frames=%d)",
            len(utterance) / _SAMPLE_RATE,
            self._speech_frame_count,
            self._silence_frame_count,
        )
        self._reset_state()
        return utterance

    def _reset_state(self) -> None:
        self._frames = []
        self._in_speech = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0


# Keep old class available for backward compatibility
class AudioBuffer:
    """Legacy fixed-duration buffer — use SpeechBuffer for live transcription."""

    def __init__(self, sample_rate: int = 16000, chunk_duration: Optional[float] = None):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration or settings.chunk_duration_seconds
        self.buffer = np.array([], dtype=np.float32)
        self.min_samples = int(self.sample_rate * self.chunk_duration)

    def add_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / 32768.0
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        if len(self.buffer) >= self.min_samples:
            chunk = self.buffer[: self.min_samples]
            self.buffer = self.buffer[self.min_samples :]
            return chunk
        return None

    def get_remaining(self) -> Optional[np.ndarray]:
        if len(self.buffer) > 0:
            chunk = self.buffer.copy()
            self.buffer = np.array([], dtype=np.float32)
            return chunk
        return None

    def clear(self):
        self.buffer = np.array([], dtype=np.float32)

    def get_buffer_duration(self) -> float:
        return len(self.buffer) / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
