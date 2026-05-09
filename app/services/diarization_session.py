"""
Per-session speaker registry for stateless batched transcription.

Why this exists: /transcribe-batch is stateless HTTP, so each chunk's Pyannote
pass produces local speaker labels that have no relation to prior chunks. This
module keeps a server-side registry keyed on a client-supplied session_id so
that the same voice gets the same global SPEAKER_NN across chunks.

Algorithm (the diart pattern, simplified for HTTP):
  1. Run Pyannote on the chunk → local speaker segments + per-segment audio.
  2. Extract a speaker embedding (pyannote/embedding, 512-dim x-vector) for
     each local speaker by averaging its frames.
  3. For each local embedding, look up nearest centroid in the session
     registry (cosine distance).
       - distance < delta_match  → assign existing global speaker, EMA-update centroid.
       - distance >= delta_match → mint new global speaker.
  4. Cannot-link: two local speakers from the same chunk must map to different
     globals (forces the loser to a new global if collision would occur).
  5. Rewrite the chunk's diarization segments with global speaker IDs.

State lives in process memory (LRU + TTL eviction). Sticky sessions required
if you scale workers horizontally — clients should hash session_id to a worker.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tunables (start with diart's DIHARD-tuned values, expose via env later if needed).
DELTA_MATCH = 1.0       # cosine distance threshold for "same speaker"
EMA_ALPHA = 0.2         # centroid update weight for new evidence
MIN_SEGMENT_SECONDS = 0.5  # skip embedding extraction for very short local segments
SESSION_TTL_SECONDS = 30 * 60  # evict sessions idle > 30 min
MAX_SESSIONS = 1024
MAX_BUFFER_SECONDS = 10.0  # rolling audio buffer length for windowed diarization (halved from 20s for latency; 10s still gives stable embeddings per diart literature)


@dataclass
class _Speaker:
    centroid: np.ndarray         # (512,) float32, L2-normalized
    embedding_count: int = 1
    total_duration: float = 0.0  # seconds of audio attributed to this speaker
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class _Session:
    speakers: Dict[int, _Speaker] = field(default_factory=dict)
    next_speaker_id: int = 0
    last_activity: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    buffer_sample_rate: int = 16000


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Both inputs assumed L2-normalized.
    return float(1.0 - np.dot(a, b))


class SessionRegistry:
    """Process-wide store of per-session speaker registries."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _Session] = {}
        self._global_lock = threading.Lock()

    def get_or_create(self, session_id: str) -> _Session:
        with self._global_lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                self._maybe_evict_locked()
                sess = _Session()
                self._sessions[session_id] = sess
                logger.info(f"[diar-session] created session={session_id[:8]} (total={len(self._sessions)})")
            sess.last_activity = time.time()
            return sess

    def drop(self, session_id: str) -> None:
        with self._global_lock:
            if self._sessions.pop(session_id, None) is not None:
                logger.info(f"[diar-session] dropped session={session_id[:8]}")

    def _maybe_evict_locked(self) -> None:
        now = time.time()
        stale = [sid for sid, s in self._sessions.items() if now - s.last_activity > SESSION_TTL_SECONDS]
        for sid in stale:
            self._sessions.pop(sid, None)
        if len(self._sessions) >= MAX_SESSIONS:
            # Drop the LRU session to make room.
            oldest = min(self._sessions.items(), key=lambda kv: kv[1].last_activity)
            self._sessions.pop(oldest[0], None)
            logger.warning(f"[diar-session] LRU evict session={oldest[0][:8]}")
        if stale:
            logger.info(f"[diar-session] TTL evicted {len(stale)} sessions")


registry = SessionRegistry()


@dataclass
class ResolutionRow:
    """One per local speaker in a chunk — what the registry decided and why."""
    local_label: str
    global_label: str
    is_new: bool
    distance: Optional[float]   # cosine distance to matched centroid; None when registry was empty
    duration_s: float


@dataclass
class AssignmentResult:
    segments: List[dict]
    resolutions: List[ResolutionRow]
    registry_size: int


@dataclass
class WindowedAssignmentResult(AssignmentResult):
    buffer_seconds: float = 0.0
    speakers_in_buffer: int = 0
    chunk_offset_seconds: float = 0.0


def _assign_global_speakers_locked(
    sess: _Session,
    local_segments: List[dict],
    local_embeddings: Dict[str, np.ndarray],
) -> AssignmentResult:
    """Core assignment logic. Caller MUST hold sess.lock.

    Split out of assign_global_speakers() so the windowed path can call it
    without re-acquiring the lock (the windowed path grabs sess.lock once,
    mutates the audio buffer, runs diarization, and then assigns — all under
    the same critical section).
    """
    if not local_segments:
        return AssignmentResult(segments=local_segments, resolutions=[], registry_size=len(sess.speakers))

    norm_embeds: Dict[str, np.ndarray] = {
        k: _l2_normalize(np.asarray(v, dtype=np.float32))
        for k, v in local_embeddings.items()
    }

    local_to_global: Dict[str, int] = {}
    resolutions: List[ResolutionRow] = []

    used_globals: set[int] = set()

    local_durations: Dict[str, float] = {}
    for seg in local_segments:
        local_durations.setdefault(seg["speaker"], 0.0)
        local_durations[seg["speaker"]] += max(0.0, seg["end"] - seg["start"])
    ordered_locals = sorted(norm_embeds.keys(), key=lambda k: -local_durations.get(k, 0.0))

    for local_label in ordered_locals:
        emb = norm_embeds[local_label]
        global_id, dist, is_new = _resolve_or_create(sess, emb, blocked=used_globals)
        local_to_global[local_label] = global_id
        used_globals.add(global_id)

        spk = sess.speakers[global_id]
        if not is_new:
            spk.centroid = _l2_normalize((1.0 - EMA_ALPHA) * spk.centroid + EMA_ALPHA * emb)
            spk.embedding_count += 1
        spk.total_duration += local_durations.get(local_label, 0.0)
        spk.last_seen = time.time()

        resolutions.append(ResolutionRow(
            local_label=local_label,
            global_label=f"SPEAKER_{global_id:02d}",
            is_new=is_new,
            distance=dist,
            duration_s=local_durations.get(local_label, 0.0),
        ))

    registry_size_after = len(sess.speakers)

    out: List[dict] = []
    for seg in local_segments:
        local = seg["speaker"]
        if local in local_to_global:
            global_id = local_to_global[local]
            new_seg = dict(seg)
            new_seg["speaker"] = f"SPEAKER_{global_id:02d}"
            out.append(new_seg)
        else:
            out.append(dict(seg))

    return AssignmentResult(segments=out, resolutions=resolutions, registry_size=registry_size_after)


def assign_global_speakers(
    session_id: str,
    local_segments: List[dict],
    local_embeddings: Dict[str, np.ndarray],
) -> AssignmentResult:
    """Map per-chunk local speaker IDs to persistent global IDs for a session.

    Returns the rewritten segments plus per-local resolution telemetry so the
    caller can surface "we matched local A→global 3 at distance 0.42, minted
    global 4 from local B because nearest was 1.05 > delta_match".
    """
    sess = registry.get_or_create(session_id)
    with sess.lock:
        return _assign_global_speakers_locked(sess, local_segments, local_embeddings)


def assign_global_speakers_windowed_sortformer(
    session_id: str,
    new_audio: np.ndarray,
    sample_rate: int,
    diarization_service,  # for embedding extraction across chunks
    sortformer_service,   # for per-buffer segmentation
) -> WindowedAssignmentResult:
    """Sortformer variant of assign_global_speakers_windowed.

    Sortformer gives us per-buffer segments + per-buffer speaker labels
    (SPEAKER_00..SPEAKER_03 in label space, capped at 4). To keep speaker
    identity stable across buffers (and therefore across WS chunks), we
    extract a pyannote/embedding x-vector for the longest turn of each
    Sortformer-detected speaker in the buffer, then run the same registry
    matching that the pyannote path uses. Sortformer-only output would
    relabel speakers per buffer.

    Trade-off vs the pyannote path: gain ~24× faster segmentation (measured
    99ms vs 2348ms p50 on 30s windows for vO9Z5zXSnhk), pay one extra
    embedding call per detected speaker per buffer (small, ~30-50ms total).
    """
    sess = registry.get_or_create(session_id)
    new_audio = np.asarray(new_audio, dtype=np.float32).reshape(-1)
    new_chunk_duration = len(new_audio) / float(sample_rate) if sample_rate > 0 else 0.0

    with sess.lock:
        if sess.buffer.size > 0 and sess.buffer_sample_rate != sample_rate:
            logger.warning(
                f"[diar-session-sf] sample rate changed {sess.buffer_sample_rate}->{sample_rate}; resetting buffer"
            )
            sess.buffer = np.zeros(0, dtype=np.float32)
        sess.buffer_sample_rate = sample_rate

        combined = np.concatenate([sess.buffer, new_audio]) if sess.buffer.size else new_audio
        max_samples = int(MAX_BUFFER_SECONDS * sample_rate)
        if combined.shape[0] > max_samples:
            combined = combined[-max_samples:]
        sess.buffer = combined

        buffer_total_seconds = sess.buffer.shape[0] / float(sample_rate) if sample_rate > 0 else 0.0
        chunk_offset = max(0.0, buffer_total_seconds - new_chunk_duration)

        # Sortformer segmentation on the rolling buffer.
        try:
            full_segments = sortformer_service.diarize_audio(sess.buffer, sample_rate=sample_rate)
        except Exception as e:
            logger.error(f"[diar-session-sf] sortformer failed, falling back to pyannote: {e}")
            full_segments, embeddings = diarization_service.diarize_audio_with_embeddings(
                sess.buffer, sample_rate=sample_rate,
            )
            assignment = _assign_global_speakers_locked(sess, full_segments, embeddings)
            speakers_in_buffer = len({s["speaker"] for s in assignment.segments}) if assignment.segments else 0
            cropped = _crop_segments_to_chunk(assignment.segments, chunk_offset)
            return WindowedAssignmentResult(
                segments=cropped, resolutions=assignment.resolutions,
                registry_size=assignment.registry_size, buffer_seconds=buffer_total_seconds,
                speakers_in_buffer=speakers_in_buffer, chunk_offset_seconds=chunk_offset,
            )

        # Sortformer's per-buffer SPEAKER_NN labels do not match across calls.
        # To resolve them to global IDs we extract one embedding per detected
        # speaker (longest contiguous turn) and feed the registry, same as the
        # pyannote path. Reuses diarization_service.embedding_inference.
        embeddings = _extract_embeddings_for_segments(
            full_segments, sess.buffer, sample_rate, diarization_service,
        )
        assignment = _assign_global_speakers_locked(sess, full_segments, embeddings)
        speakers_in_buffer = len({s["speaker"] for s in assignment.segments}) if assignment.segments else 0
        cropped = _crop_segments_to_chunk(assignment.segments, chunk_offset)
        return WindowedAssignmentResult(
            segments=cropped, resolutions=assignment.resolutions,
            registry_size=assignment.registry_size, buffer_seconds=buffer_total_seconds,
            speakers_in_buffer=speakers_in_buffer, chunk_offset_seconds=chunk_offset,
        )


def _extract_embeddings_for_segments(
    segments: List[dict], audio_buffer: np.ndarray, sample_rate: int, diarization_service,
) -> Dict[str, np.ndarray]:
    """For each unique speaker label in `segments`, find their longest turn
    and extract one pyannote x-vector embedding from that turn. Used by the
    Sortformer path so the registry has something to match on. Skips
    speakers with too little audio for a reliable embedding."""
    if not segments or diarization_service.embedding_inference is None:
        return {}
    import torch
    from pyannote.core import Segment

    longest_per_speaker: Dict[str, Tuple[float, float]] = {}
    for s in segments:
        spk = s["speaker"]
        dur = float(s["end"]) - float(s["start"])
        prev = longest_per_speaker.get(spk)
        if prev is None or dur > (prev[1] - prev[0]):
            longest_per_speaker[spk] = (float(s["start"]), float(s["end"]))

    waveform = torch.from_numpy(audio_buffer.astype(np.float32)).unsqueeze(0)
    audio_input = {"waveform": waveform, "sample_rate": int(sample_rate)}
    file_dur = audio_buffer.size / float(sample_rate) if sample_rate > 0 else 0.0

    out: Dict[str, np.ndarray] = {}
    for spk, (start, end) in longest_per_speaker.items():
        if (end - start) < MIN_SEGMENT_SECONDS:
            continue
        # Same eps trick as diarize_audio_with_embeddings: dodge the strict
        # "end > duration" check on exact-match boundaries.
        eps = 1.0 / float(sample_rate) if sample_rate > 0 else 1e-3
        clamp_end = min(end, max(0.0, file_dur - eps))
        clamp_start = min(start, max(0.0, clamp_end - 0.1))
        if clamp_end - clamp_start < MIN_SEGMENT_SECONDS:
            continue
        try:
            emb = diarization_service.embedding_inference.crop(audio_input, Segment(clamp_start, clamp_end))
            out[spk] = np.asarray(emb).reshape(-1)
        except Exception as e:
            logger.warning(f"[diar-session-sf] embedding failed for {spk}: {type(e).__name__}: {e}")
    return out


def _crop_segments_to_chunk(segments: List[dict], chunk_offset: float) -> List[dict]:
    """Crop buffer-relative segments to the new chunk window and re-base."""
    cropped: List[dict] = []
    for seg in segments:
        if seg["end"] <= chunk_offset:
            continue
        new_seg = dict(seg)
        new_seg["start"] = max(0.0, seg["start"] - chunk_offset)
        new_seg["end"] = max(new_seg["start"], seg["end"] - chunk_offset)
        cropped.append(new_seg)
    return cropped


def assign_global_speakers_windowed(
    session_id: str,
    new_audio: np.ndarray,
    sample_rate: int,
    diarization_service,
) -> WindowedAssignmentResult:
    """Buffered-diarization variant for live streaming chunks.

    Keeps a rolling per-session audio buffer (≤ MAX_BUFFER_SECONDS). Each call:
      1. Appends new_audio to the session buffer, trimming oldest samples.
      2. Runs Pyannote on the entire buffer so speaker turns span across
         chunk boundaries — embeddings are extracted from 5-15s of audio per
         speaker instead of the 1-4s a single chunk would give.
      3. Resolves local→global speaker IDs via the existing registry logic.
      4. Crops returned segments to the new chunk's time window and re-bases
         offsets so (0, new_chunk_duration) aligns with the caller's
         transcript segments for this chunk.

    Thread safety: the session lock is held for the full duration (buffer
    mutation + diarization + assignment). This serializes overlapping WS
    chunks for the same session, which is desirable: the buffer would be
    corrupted by concurrent mutations, and Pyannote on 20s of audio is
    ~150-300ms, well within live budget.
    """
    sess = registry.get_or_create(session_id)

    new_audio = np.asarray(new_audio, dtype=np.float32).reshape(-1)
    new_chunk_duration = len(new_audio) / float(sample_rate) if sample_rate > 0 else 0.0

    with sess.lock:
        # Reset buffer if sample rate changes mid-session (shouldn't happen,
        # but guard against a stale buffer polluting fresh audio).
        if sess.buffer.size > 0 and sess.buffer_sample_rate != sample_rate:
            logger.warning(
                f"[diar-session] sample rate changed {sess.buffer_sample_rate}->{sample_rate}; resetting buffer"
            )
            sess.buffer = np.zeros(0, dtype=np.float32)
        sess.buffer_sample_rate = sample_rate

        # Append new audio then trim to the rolling window.
        combined = np.concatenate([sess.buffer, new_audio]) if sess.buffer.size else new_audio
        max_samples = int(MAX_BUFFER_SECONDS * sample_rate)
        if combined.shape[0] > max_samples:
            combined = combined[-max_samples:]
        sess.buffer = combined

        buffer_total_seconds = sess.buffer.shape[0] / float(sample_rate) if sample_rate > 0 else 0.0
        # Where does the new chunk start within the buffer timeline?
        # If buffer hasn't filled yet, the new chunk starts at (buffer_total - new_chunk_duration).
        chunk_offset = max(0.0, buffer_total_seconds - new_chunk_duration)

        # Run diarization on the in-memory rolling buffer. Avoids both the
        # tmp-WAV roundtrip cost AND the pyannote 4.0 / torchcodec issue on
        # py3.12+cu130 (torchcodec wants the full CUDA 13 system toolkit).
        full_segments, embeddings = diarization_service.diarize_audio_with_embeddings(
            sess.buffer, sample_rate=sample_rate,
        )

        # Resolve against the session registry (already holding the lock).
        assignment = _assign_global_speakers_locked(sess, full_segments, embeddings)

        speakers_in_buffer = len({s["speaker"] for s in assignment.segments}) if assignment.segments else 0

        # Crop to the new chunk's time window and re-base to chunk-relative offsets.
        cropped: List[dict] = []
        for seg in assignment.segments:
            # Keep any segment that overlaps the new chunk window.
            if seg["end"] <= chunk_offset:
                continue
            new_seg = dict(seg)
            new_seg["start"] = max(0.0, seg["start"] - chunk_offset)
            new_seg["end"] = max(new_seg["start"], seg["end"] - chunk_offset)
            cropped.append(new_seg)

        return WindowedAssignmentResult(
            segments=cropped,
            resolutions=assignment.resolutions,
            registry_size=assignment.registry_size,
            buffer_seconds=buffer_total_seconds,
            speakers_in_buffer=speakers_in_buffer,
            chunk_offset_seconds=chunk_offset,
        )


def _resolve_or_create(
    sess: _Session, emb: np.ndarray, blocked: set[int]
) -> Tuple[int, Optional[float], bool]:
    """Find nearest unblocked centroid; create new speaker if too far.

    Returns (global_id, distance_to_matched_or_nearest, is_new).
    distance is None only when the registry was completely empty (first ever speaker).
    """
    best_id: Optional[int] = None
    best_dist = float("inf")
    for spk_id, spk in sess.speakers.items():
        if spk_id in blocked:
            continue
        d = _cosine_distance(emb, spk.centroid)
        if d < best_dist:
            best_dist = d
            best_id = spk_id

    if best_id is not None and best_dist < DELTA_MATCH:
        return best_id, best_dist, False

    new_id = sess.next_speaker_id
    sess.next_speaker_id += 1
    sess.speakers[new_id] = _Speaker(centroid=emb.copy())
    # Surface the nearest-but-rejected distance so telemetry shows why we minted.
    return new_id, (best_dist if best_id is not None else None), True


def new_session_id() -> str:
    return uuid.uuid4().hex
