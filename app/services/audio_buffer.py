import numpy as np
from typing import Optional
import logging
from app.config import settings
from app.services.vad_service import vad_service

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Manages audio chunk buffering for WebSocket live transcription.

    Uses VAD-aware flushing: flushes early when silence is detected after speech,
    instead of always waiting for the full chunk duration. This reduces latency
    for short utterances while still capping at max_duration for continuous speech.
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: Optional[float] = None):
        self.sample_rate = sample_rate
        self.max_duration = chunk_duration or settings.chunk_duration_seconds
        self.buffer = np.array([], dtype=np.float32)
        self.max_samples = int(self.sample_rate * self.max_duration)

        # VAD-aware flush state
        self.min_speech_samples = int(self.sample_rate * 0.5)  # need at least 0.5s before considering flush
        self.had_speech = False  # tracks if we've seen speech in current buffer
        self.silence_frames = 0  # consecutive silent incoming chunks
        self.silence_flush_threshold = 3  # flush after this many silent chunks following speech

        logger.info(f"AudioBuffer initialized: max {self.max_duration}s chunks at {self.sample_rate}Hz, VAD-aware flush enabled")

    def add_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Add audio chunk to buffer. Returns processable audio when:
        1. Buffer reaches max duration (cap for continuous speech), OR
        2. Silence detected after speech (VAD-aware early flush)
        """
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / 32768.0

        self.buffer = np.concatenate([self.buffer, audio_chunk])

        # Hard cap: always flush if buffer is full
        if len(self.buffer) >= self.max_samples:
            chunk_to_process = self.buffer[:self.max_samples]
            self.buffer = self.buffer[self.max_samples:]
            self._reset_vad_state()
            return chunk_to_process

        # VAD-aware early flush: check incoming chunk for speech
        if settings.enable_vad and len(self.buffer) >= self.min_speech_samples:
            chunk_has_speech = vad_service.is_speech(audio_chunk, self.sample_rate) if len(audio_chunk) >= 480 else True

            if chunk_has_speech:
                self.had_speech = True
                self.silence_frames = 0
            elif self.had_speech:
                self.silence_frames += 1

                if self.silence_frames >= self.silence_flush_threshold:
                    # Speech ended — flush what we have
                    chunk_to_process = self.buffer.copy()
                    self.buffer = np.array([], dtype=np.float32)
                    self._reset_vad_state()
                    logger.info(f"VAD flush: {len(chunk_to_process)/self.sample_rate:.2f}s (silence after speech)")
                    return chunk_to_process

        return None

    def _reset_vad_state(self):
        self.had_speech = False
        self.silence_frames = 0

    def get_remaining(self) -> Optional[np.ndarray]:
        if len(self.buffer) > 0:
            chunk = self.buffer.copy()
            self.buffer = np.array([], dtype=np.float32)
            self._reset_vad_state()
            return chunk
        return None

    def clear(self):
        self.buffer = np.array([], dtype=np.float32)
        self._reset_vad_state()

    def get_buffer_duration(self) -> float:
        return len(self.buffer) / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
