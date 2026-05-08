import torch
import torch.serialization
import numpy as np
from typing import List, Dict
import logging
import tempfile
import soundfile as sf
import os
from app.config import settings

# PyTorch 2.6+ defaults torch.load to weights_only=True, which blocks pyannote
# AND speechbrain model loading. They also pass weights_only=True explicitly in
# some code paths, so setdefault isn't enough — we force False unconditionally.
# Trust assumption: model files come from the pinned pyannote/speechbrain HF
# repos we already authenticated against. Same trust we'd extend to any pip dep.
import functools
_orig_torch_load = torch.load
@functools.wraps(_orig_torch_load)
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

# Set HF token before importing Pyannote
os.environ["HF_TOKEN"] = settings.pyannote_auth_token

from pyannote.audio import Pipeline, Inference, Model

logger = logging.getLogger(__name__)


class DiarizationService:
    """Handles speaker diarization using Pyannote"""

    def __init__(self):
        self.pipeline = None
        self.embedding_inference = None  # pyannote/embedding for cross-chunk speaker ID
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

            # Load embedding model separately for cross-chunk speaker ID. Used by
            # the session registry in /transcribe-batch to keep speaker labels
            # consistent across HTTP-stateless chunks. Optional — diarization
            # still works without it (just no cross-chunk identity).
            try:
                logger.info("Loading Pyannote embedding model (pyannote/embedding)...")
                emb_model = Model.from_pretrained("pyannote/embedding")
                self.embedding_inference = Inference(emb_model, window="whole", device=self.device)
                logger.info("Embedding model loaded — cross-chunk speaker ID enabled")
            except Exception as e:
                logger.warning(f"Could not load pyannote/embedding: {e}")
                logger.warning("Cross-chunk speaker ID will be disabled (per-chunk labels only)")
                self.embedding_inference = None

    def diarize_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, clustering_threshold: float = None) -> List[Dict]:
        """
        Perform speaker diarization on audio data.

        Uses in-memory waveform dict to avoid temp file disk I/O.
        """
        if not settings.enable_diarization or self.pipeline is None:
            duration = len(audio_data) / sample_rate
            return [{'start': 0.0, 'end': duration, 'speaker': 'SPEAKER_00'}]

        try:
            threshold = clustering_threshold if clustering_threshold is not None else settings.pyannote_clustering_threshold

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

            # Pass audio in-memory as a waveform dict — no temp file needed
            # Pyannote expects shape (channels, samples) as a torch Tensor
            waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}

            diarization = self.pipeline(
                audio_input,
                min_speakers=settings.pyannote_min_speakers,
                max_speakers=settings.pyannote_max_speakers
            )

            # pyannote.audio 4.0 wraps the Annotation in DiarizeOutput.
            # 3.x returned the Annotation directly. Handle both.
            if hasattr(diarization, "speaker_diarization"):
                diarization = diarization.speaker_diarization

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

            return segments if segments else [{'start': 0.0, 'end': len(audio_data) / sample_rate, 'speaker': 'SPEAKER_00'}]

        except Exception as e:
            logger.error(f"Diarization error: {e}")
            duration = len(audio_data) / sample_rate
            return [{'start': 0.0, 'end': duration, 'speaker': 'SPEAKER_00'}]

    def diarize_file(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization on audio file.
        """
        if not settings.enable_diarization or self.pipeline is None:
            return [{'start': 0.0, 'end': 0.0, 'speaker': 'SPEAKER_00'}]

        try:
            diarization = self.pipeline(
                audio_path,
                min_speakers=settings.pyannote_min_speakers,
                max_speakers=settings.pyannote_max_speakers
            )

            # pyannote.audio 4.0 wraps Annotation in DiarizeOutput.
            if hasattr(diarization, "speaker_diarization"):
                diarization = diarization.speaker_diarization

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

    def diarize_file_with_embeddings(
        self, audio_path: str, min_segment_seconds: float = 0.5
    ) -> tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Diarize a file AND extract one speaker embedding per local speaker.

        Used by /transcribe-batch to feed the per-session speaker registry.
        Returns:
            segments: same shape as diarize_file()
            embeddings: {local_speaker_label: np.ndarray(512,)} — only for
              speakers whose total active duration meets min_segment_seconds.
              Speakers with too little audio for a reliable embedding are omitted.
        """
        if not settings.enable_diarization or self.pipeline is None:
            return [{'start': 0.0, 'end': 0.0, 'speaker': 'SPEAKER_00'}], {}

        try:
            diarization = self.pipeline(
                audio_path,
                min_speakers=settings.pyannote_min_speakers,
                max_speakers=settings.pyannote_max_speakers,
            )
            # pyannote.audio 4.0 wraps Annotation in DiarizeOutput.
            if hasattr(diarization, "speaker_diarization"):
                diarization = diarization.speaker_diarization
        except Exception as e:
            logger.error(f"File diarization error: {e}")
            return [{'start': 0.0, 'end': 0.0, 'speaker': 'SPEAKER_00'}], {}

        segments: List[Dict] = []
        per_speaker_duration: Dict[str, float] = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
            per_speaker_duration[speaker] = per_speaker_duration.get(speaker, 0.0) + (turn.end - turn.start)

        if not segments:
            return [{'start': 0.0, 'end': 0.0, 'speaker': 'SPEAKER_00'}], {}

        embeddings: Dict[str, np.ndarray] = {}
        if self.embedding_inference is None:
            return segments, embeddings

        # Extract one embedding per local speaker by running pyannote/embedding
        # on the union of that speaker's active regions. We use the diarization
        # Annotation's per-label timeline to crop precisely.
        try:
            from pyannote.core import Segment
            for label in diarization.labels():
                if per_speaker_duration.get(label, 0.0) < min_segment_seconds:
                    continue
                # Pick the longest contiguous turn for this speaker — fast, and
                # avoids the cost of cropping + concatenating multiple slices.
                best_turn: Segment | None = None
                best_len = 0.0
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    if spk != label:
                        continue
                    if (turn.end - turn.start) > best_len:
                        best_len = turn.end - turn.start
                        best_turn = turn
                if best_turn is None or best_len < min_segment_seconds:
                    continue
                # Pyannote's segmentation can emit turns that extend a few ms past
                # the actual file end (frame rounding). Clamp the crop region to
                # the real file duration so Inference.crop doesn't refuse to read.
                try:
                    import soundfile as _sf
                    info = _sf.info(audio_path)
                    file_dur = info.frames / float(info.samplerate) if info.samplerate else 0.0
                except Exception:
                    file_dur = 0.0
                clamp_end = min(best_turn.end, file_dur) if file_dur > 0 else best_turn.end
                clamp_start = min(best_turn.start, max(0.0, clamp_end - 0.1))
                if clamp_end - clamp_start < min_segment_seconds:
                    continue
                clamped = Segment(clamp_start, clamp_end)
                try:
                    emb = self.embedding_inference.crop(audio_path, clamped)
                    # Inference(window="whole") returns a 1D numpy array of shape (512,).
                    embeddings[label] = np.asarray(emb).reshape(-1)
                except Exception as e:
                    logger.warning(
                        f"Embedding extraction failed for {label} turn={clamped} "
                        f"(file_dur={file_dur:.3f}s): {type(e).__name__}: {e}"
                    )
        except Exception as e:
            logger.warning(f"Embedding extraction loop failed: {type(e).__name__}: {e}", exc_info=True)

        return segments, embeddings


# Singleton instance
diarization_service = DiarizationService()
