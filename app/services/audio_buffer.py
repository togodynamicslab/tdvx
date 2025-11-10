import numpy as np
from typing import Optional
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Manages audio chunk buffering for WebSocket live transcription.
    Accumulates audio chunks until sufficient duration is reached.
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: Optional[float] = None):
        """
        Initialize audio buffer.

        Args:
            sample_rate: Audio sample rate (default: 16kHz for Whisper)
            chunk_duration: Duration in seconds before processing (default: from settings)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration or settings.chunk_duration_seconds
        self.buffer = np.array([], dtype=np.float32)
        self.min_samples = int(self.sample_rate * self.chunk_duration)

        logger.info(f"AudioBuffer initialized: {self.chunk_duration}s chunks at {self.sample_rate}Hz")

    def add_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Add audio chunk to buffer.

        Args:
            audio_chunk: NumPy array of audio samples (float32)

        Returns:
            Audio data ready for processing if buffer is full, None otherwise
        """
        # Ensure audio is float32 and normalized
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize to [-1, 1] if needed
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / 32768.0  # Assuming int16 input

        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk])

        # Check if we have enough samples
        if len(self.buffer) >= self.min_samples:
            # Extract chunk for processing
            chunk_to_process = self.buffer[:self.min_samples]

            # Keep remaining data in buffer
            self.buffer = self.buffer[self.min_samples:]

            return chunk_to_process

        return None

    def get_remaining(self) -> Optional[np.ndarray]:
        """
        Get any remaining audio in buffer (for final processing).

        Returns:
            Remaining audio data, or None if buffer is empty
        """
        if len(self.buffer) > 0:
            chunk = self.buffer.copy()
            self.buffer = np.array([], dtype=np.float32)
            return chunk
        return None

    def clear(self):
        """Clear the buffer"""
        self.buffer = np.array([], dtype=np.float32)

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds"""
        return len(self.buffer) / self.sample_rate

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
