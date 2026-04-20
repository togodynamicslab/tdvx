"""
NVIDIA Streaming Sortformer diarization — prototype wrapper.

Why this exists: Pyannote 3.x is our current diarizer but it's the live-path
latency bottleneck (200-300ms per call, no batching, windowed buffer hack for
cross-chunk identity). NVIDIA's streaming-native Sortformer is designed exactly
for the real-time case — end-to-end, with built-in Arrival-Order Speaker Cache
(AOSC) for stable identity across chunks — and is capped at 4 speakers in
v2.1, which is fine for an Omi-style wearable.

This module is a PROTOTYPE behind `settings.diarizer_backend`. It is NOT wired
into the live endpoints yet — the benchmark script (scripts/bench_diarizers.py)
exercises it directly so we can decide go/no-go before swapping anything.

API reference (verified 2026-04-19):
  https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1

NeMo is imported lazily inside the load method so a missing dep doesn't break
the rest of the service — consistent with our "prototype, not yet a hard dep"
stance. Deployer note: install `nemo_toolkit[asr]` on the target box; we do NOT
add it to requirements.txt automatically.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class SortformerService:
    """Thin wrapper around NeMo's SortformerEncLabelModel.

    Lazy-loads on first use. Exposes diarize_audio() with the same shape as
    DiarizationService.diarize_audio() so callers (and the benchmark) can swap
    backends transparently.
    """

    def __init__(self) -> None:
        self.model = None
        self._load_attempted = False
        self._load_failed_reason: Optional[str] = None
        self._device = "cuda"
        # Configure logger so "nothing is happening" states are diagnosable.
        logger.info(
            "SortformerService initialized (lazy load). backend=%s model=%s",
            settings.diarizer_backend,
            settings.sortformer_model_name,
        )

    def is_available(self) -> bool:
        """Try to load; return True if usable. Safe to call repeatedly."""
        self.load()
        return self.model is not None

    def load(self) -> None:
        """Lazy-load the NeMo model. Idempotent. Swallows ImportError so the
        rest of the service keeps working when NeMo isn't installed."""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            # Local import on purpose: NeMo is a ~2GB dep tree (torch, hydra,
            # pytorch-lightning, etc.) and we don't want module-import-time cost
            # on services that never touch Sortformer.
            from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore
        except Exception as e:
            self._load_failed_reason = f"NeMo import failed: {type(e).__name__}: {e}"
            logger.warning(
                "Sortformer unavailable — %s. Install with: pip install 'nemo_toolkit[asr]'",
                self._load_failed_reason,
            )
            return

        try:
            import torch  # type: ignore
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._device = "cpu"

        t0 = time.perf_counter()
        try:
            logger.info(
                "Loading Sortformer model '%s' on device=%s ...",
                settings.sortformer_model_name,
                self._device,
            )
            model = SortformerEncLabelModel.from_pretrained(settings.sortformer_model_name)
            model.eval()

            # Streaming config — these live on .sortformer_modules per the HF
            # model card. Units are 80ms frames. Guard each assignment so an
            # API drift in a future NeMo release gives a useful error instead
            # of a cryptic AttributeError buried in diarize().
            try:
                mods = model.sortformer_modules
                mods.chunk_len = settings.sortformer_chunk_len
                mods.chunk_right_context = settings.sortformer_chunk_right_context
                mods.fifo_len = settings.sortformer_fifo_len
                mods.spkcache_update_period = settings.sortformer_spkcache_update_period
                mods.spkcache_len = settings.sortformer_spkcache_len
                # Validator method documented on the model card.
                if hasattr(mods, "_check_streaming_parameters"):
                    mods._check_streaming_parameters()
                logger.info(
                    "Sortformer streaming params: chunk_len=%d right_ctx=%d fifo=%d "
                    "spkcache_update=%d spkcache_len=%d",
                    settings.sortformer_chunk_len,
                    settings.sortformer_chunk_right_context,
                    settings.sortformer_fifo_len,
                    settings.sortformer_spkcache_update_period,
                    settings.sortformer_spkcache_len,
                )
            except AttributeError as e:
                logger.warning(
                    "Sortformer streaming-params API differs from expected (%s). "
                    "Continuing with model defaults.",
                    e,
                )

            self.model = model
            load_s = time.perf_counter() - t0
            logger.info("Sortformer loaded in %.2fs", load_s)
        except Exception as e:
            self._load_failed_reason = f"Sortformer load failed: {type(e).__name__}: {e}"
            logger.exception("Sortformer load failed")
            self.model = None

    def diarize_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[Dict[str, Any]]:
        """Run Sortformer on an in-memory waveform.

        Returns the same shape as DiarizationService.diarize_audio:
            [{'start': float, 'end': float, 'speaker': 'SPEAKER_NN'}, ...]

        Speaker labels within a single call are stable (AOSC). Across calls
        with separate audio arrays, identity is NOT preserved unless the model
        state persists — this prototype is per-call, which is sufficient for
        the benchmark but NOT for live sessions. A future session-aware
        variant would need to hold a single model instance per session and
        feed sequential chunks; that's a follow-up once benchmarks say go.
        """
        if not self.is_available():
            reason = self._load_failed_reason or "not loaded"
            raise RuntimeError(f"SortformerService unavailable: {reason}")

        if sample_rate != 16000:
            raise ValueError(
                f"Sortformer v2.1 requires 16kHz mono audio; got sample_rate={sample_rate}"
            )

        audio = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []

        t0 = time.perf_counter()
        try:
            # Per the HF model card: diarize() accepts a numpy array but needs
            # sample_rate to be passed explicitly in that case.
            try:
                predicted = self.model.diarize(
                    audio=audio, batch_size=1, sample_rate=sample_rate
                )
            except TypeError:
                # Defensive fallback: older/newer NeMo builds may drop the
                # sample_rate kwarg. In that case numpy path expects 16kHz.
                predicted = self.model.diarize(audio=audio, batch_size=1)
        except Exception as e:
            logger.exception("Sortformer diarize() failed")
            raise RuntimeError(f"Sortformer inference failed: {type(e).__name__}: {e}") from e

        infer_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Sortformer inference: %.1f ms for %.2fs of audio",
                    infer_ms, audio.size / sample_rate)

        # predicted is list-per-input; batch of 1 so predicted[0] is the segment list.
        # Each segment is "begin end speaker_index" — can be a tuple, list, or a
        # space-separated string depending on NeMo release. Handle all three.
        raw_segments = predicted[0] if predicted and len(predicted) > 0 else []

        segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            begin, end, spk = self._parse_segment(seg)
            if begin is None:
                continue
            segments.append({
                "start": float(begin),
                "end": float(end),
                "speaker": f"SPEAKER_{int(spk):02d}",
            })

        if not segments:
            # Mirror DiarizationService behavior: never return empty — caller
            # code assumes at least one segment covering the whole clip.
            duration = audio.size / float(sample_rate)
            return [{"start": 0.0, "end": duration, "speaker": "SPEAKER_00"}]
        return segments

    @staticmethod
    def _parse_segment(seg: Any):
        """Normalize a single Sortformer segment into (begin, end, speaker_idx).

        NeMo has shipped this as tuple, list, and "b e s" strings at various
        points. Return (None, None, None) if we can't parse it — caller skips.
        """
        try:
            if isinstance(seg, str):
                parts = seg.strip().split()
                if len(parts) < 3:
                    return None, None, None
                return float(parts[0]), float(parts[1]), int(float(parts[2]))
            if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                return float(seg[0]), float(seg[1]), int(float(seg[2]))
            # Some NeMo versions return dicts.
            if isinstance(seg, dict):
                begin = seg.get("begin", seg.get("start"))
                end = seg.get("end")
                spk = seg.get("speaker", seg.get("speaker_index"))
                if begin is None or end is None or spk is None:
                    return None, None, None
                return float(begin), float(end), int(float(spk))
        except (TypeError, ValueError):
            return None, None, None
        return None, None, None


# Singleton — mirrors diarization_service pattern. Lazy load on first use so
# importing this module is free even when NeMo isn't installed.
sortformer_service = SortformerService()
