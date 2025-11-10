import torch
import numpy as np
from typing import List, Dict
import logging
import tempfile
import soundfile as sf
import os
from app.config import settings

# Set HF token before importing Pyannote
os.environ["HF_TOKEN"] = settings.pyannote_auth_token

from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)


class DiarizationService:
    """Handles speaker diarization using Pyannote"""

    def __init__(self):
        self.pipeline = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Diarization will use device: {self.device}")

    def load_pipeline(self):
        """Load Pyannote pipeline (call once at startup)"""
        if self.pipeline is None and settings.enable_diarization:
            try:
                logger.info("Loading Pyannote diarization pipeline...")
                logger.info(f"Pyannote settings: min_speakers={settings.pyannote_min_speakers}, "
                           f"max_speakers={settings.pyannote_max_speakers}, "
                           f"clustering_threshold={settings.pyannote_clustering_threshold}, "
                           f"segmentation_onset={settings.pyannote_segmentation_onset}")
                # HF token is set in environment variable above
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
                self.pipeline.to(self.device)

                # Instantiate pipeline with custom clustering threshold
                # Based on pyannote.audio documentation and examples
                # Note: segmentation-3.0 only supports min_duration_off (not threshold)
                try:
                    self.pipeline.instantiate({
                        'segmentation': {
                            'min_duration_off': 0.0
                        },
                        'clustering': {
                            'method': 'centroid',
                            'min_cluster_size': 2,
                            'threshold': settings.pyannote_clustering_threshold
                        }
                    })
                    logger.info(f"Successfully set clustering threshold to: {settings.pyannote_clustering_threshold}")
                except Exception as e:
                    logger.warning(f"Could not instantiate pipeline with custom threshold: {e}")
                    logger.warning("Using default clustering threshold")

                logger.info("Pyannote pipeline loaded successfully")
                logger.info(f"Will use clustering threshold: {settings.pyannote_clustering_threshold}")
            except Exception as e:
                logger.error(f"Failed to load Pyannote pipeline: {e}")
                logger.warning("Diarization will be disabled")
                self.pipeline = None

    def diarize_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, clustering_threshold: float = None) -> List[Dict]:
        """
        Perform speaker diarization on audio data.

        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of audio
            clustering_threshold: Optional clustering threshold override (if None, uses default from settings)

        Returns:
            List of diarization segments: [{'start': float, 'end': float, 'speaker': str}, ...]
        """
        if not settings.enable_diarization or self.pipeline is None:
            # Return single default speaker if diarization is disabled
            duration = len(audio_data) / sample_rate
            return [{'start': 0.0, 'end': duration, 'speaker': 'SPEAKER_00'}]

        try:
            # Use provided threshold or fall back to default
            threshold = clustering_threshold if clustering_threshold is not None else settings.pyannote_clustering_threshold

            # Re-instantiate pipeline with the specified threshold
            try:
                self.pipeline.instantiate({
                    'segmentation': {
                        'min_duration_off': 0.0
                    },
                    'clustering': {
                        'method': 'centroid',
                        'min_cluster_size': 2,
                        'threshold': threshold
                    }
                })
            except Exception as e:
                logger.warning(f"Could not re-instantiate pipeline with threshold {threshold}: {e}")

            # Pyannote requires audio file, so create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                tmp_path = tmp_file.name

            # Run diarization with parameters
            # num_speakers: None = auto-detect, or set to expected number
            # min_speakers: minimum number of speakers to detect (1 = auto-detect)
            # max_speakers: maximum number of speakers to detect
            # Clustering threshold is set above via instantiate()
            diarization = self.pipeline(
                tmp_path,
                min_speakers=settings.pyannote_min_speakers,
                max_speakers=settings.pyannote_max_speakers
            )

            # Convert to list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

            # Clean up temp file
            import os
            os.unlink(tmp_path)

            return segments if segments else [{'start': 0.0, 'end': len(audio_data) / sample_rate, 'speaker': 'SPEAKER_00'}]

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            # Return default speaker on error
            duration = len(audio_data) / sample_rate
            return [{'start': 0.0, 'end': duration, 'speaker': 'SPEAKER_00'}]

    def diarize_file(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of diarization segments
        """
        if not settings.enable_diarization or self.pipeline is None:
            return [{'start': 0.0, 'end': 0.0, 'speaker': 'SPEAKER_00'}]

        try:
            diarization = self.pipeline(
                audio_path,
                min_speakers=settings.pyannote_min_speakers,
                max_speakers=settings.pyannote_max_speakers
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

            return segments if segments else [{'start': 0.0, 'end': 0.0, 'speaker': 'SPEAKER_00'}]

        except Exception as e:
            logger.error(f"File diarization error: {e}")
            return [{'start': 0.0, 'end': 0.0, 'speaker': 'SPEAKER_00'}]


# Singleton instance
diarization_service = DiarizationService()
