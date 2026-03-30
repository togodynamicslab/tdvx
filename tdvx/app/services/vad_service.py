import webrtcvad
import numpy as np
import logging
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)


class VADService:
    """Voice Activity Detection using WebRTC VAD"""

    def __init__(self, aggressiveness: int = 3):
        """
        Initialize VAD service.

        Args:
            aggressiveness: VAD aggressiveness mode (0-3)
                0 = least aggressive (more permissive)
                3 = most aggressive (only clear speech)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000  # WebRTC VAD only supports 8000, 16000, 32000, 48000 Hz
        logger.info(f"VAD initialized with aggressiveness: {aggressiveness}")

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect if audio chunk contains speech.

        Args:
            audio_chunk: Audio data as numpy array (float32)
            sample_rate: Sample rate (must be 8000, 16000, 32000, or 48000)

        Returns:
            True if speech is detected, False otherwise
        """
        try:
            # WebRTC VAD requires specific sample rates
            if sample_rate not in [8000, 16000, 32000, 48000]:
                logger.warning(f"Unsupported sample rate {sample_rate}, defaulting to 16000")
                sample_rate = 16000

            # WebRTC VAD requires frame lengths of 10, 20, or 30 ms
            # For 16kHz: 160 samples (10ms), 320 samples (20ms), 480 samples (30ms)
            frame_duration_ms = 30  # Use 30ms frames
            frame_length = int(sample_rate * frame_duration_ms / 1000)

            # Convert float32 to int16 (WebRTC VAD expects int16)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            # Process in frames and check if any frame contains speech
            speech_frames = 0
            total_frames = 0

            for i in range(0, len(audio_int16) - frame_length, frame_length):
                frame = audio_int16[i:i + frame_length]

                # VAD requires exact frame length
                if len(frame) != frame_length:
                    continue

                total_frames += 1

                # Convert to bytes
                frame_bytes = frame.tobytes()

                # Check if frame contains speech
                if self.vad.is_speech(frame_bytes, sample_rate):
                    speech_frames += 1

            # Return True if at least 30% of frames contain speech
            if total_frames == 0:
                return False

            speech_ratio = speech_frames / total_frames
            return speech_ratio >= 0.3

        except Exception as e:
            logger.error(f"VAD error: {e}")
            # On error, assume it contains speech to avoid dropping audio
            return True

    def get_speech_ratio(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Get the ratio of speech frames in the audio chunk.

        Args:
            audio_chunk: Audio data as numpy array (float32)
            sample_rate: Sample rate

        Returns:
            Ratio of frames containing speech (0.0 to 1.0)
        """
        try:
            if sample_rate not in [8000, 16000, 32000, 48000]:
                sample_rate = 16000

            frame_duration_ms = 30
            frame_length = int(sample_rate * frame_duration_ms / 1000)
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            speech_frames = 0
            total_frames = 0

            for i in range(0, len(audio_int16) - frame_length, frame_length):
                frame = audio_int16[i:i + frame_length]

                if len(frame) != frame_length:
                    continue

                total_frames += 1
                frame_bytes = frame.tobytes()

                if self.vad.is_speech(frame_bytes, sample_rate):
                    speech_frames += 1

            if total_frames == 0:
                return 0.0

            return speech_frames / total_frames

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return 1.0  # Assume speech on error


# Singleton instance
vad_service = VADService(aggressiveness=settings.vad_aggressiveness)
