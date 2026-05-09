"""
RTTM helpers for the PT-BR diarization eval set.

Glue between hand-labeled ground truth (RTTM files) and the formats the
bench / pyannote.metrics consume:

  RTTM file ──parse_rttm──▶ list[segment_dict] ──segments_to_annotation──▶ pyannote.core.Annotation

Also provides Audacity label-track import for people labeling in Audacity
(easiest visual labeling tool) and a writer for generating ground truth
programmatically (synthetic concatenation, etc.).

Why a separate file: the bench script (scripts/bench_diarizers.py) needs
these helpers but tests/ is the natural home for the corpus. Bench imports
from here via a relative path. Keeping helpers next to the corpus makes them
discoverable for anyone labeling new clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# RTTM is a NIST format. Each line is whitespace-separated:
#   Type FileID Channel Start Duration <NA> <NA> SpeakerID <NA> <NA>
# We only emit / parse SPEAKER lines — other types (SPKR-INFO, etc.) are
# tolerated on read but not written.
_RTTM_TYPE = "SPEAKER"


@dataclass(frozen=True)
class DiarSegment:
    """One speaker turn. Times are seconds from clip start."""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, object]:
        return {"start": self.start, "end": self.end, "speaker": self.speaker}


def parse_rttm(rttm_path: Path | str) -> List[DiarSegment]:
    """Read an RTTM file. Skip non-SPEAKER lines. Tolerate `<NA>` columns.

    Returns segments in file order. Caller can sort by start if needed.
    """
    rttm_path = Path(rttm_path)
    segments: List[DiarSegment] = []
    with rttm_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts or parts[0] != _RTTM_TYPE:
                continue
            # Standard layout: 10 tokens. Be lenient if a tool drops trailing <NA>s.
            if len(parts) < 8:
                raise ValueError(
                    f"{rttm_path}:{lineno}: malformed SPEAKER line, "
                    f"expected >= 8 tokens, got {len(parts)}: {line!r}"
                )
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError as e:
                raise ValueError(
                    f"{rttm_path}:{lineno}: non-numeric start/duration: {line!r}"
                ) from e
            speaker = parts[7]
            segments.append(DiarSegment(start=start, end=start + duration, speaker=speaker))
    return segments


def write_rttm(
    rttm_path: Path | str,
    segments: Iterable[DiarSegment | Dict[str, object]],
    file_id: Optional[str] = None,
) -> None:
    """Write segments to an RTTM file.

    `file_id` defaults to the rttm filename without extension. This must
    match the audio file's basename for pyannote.metrics to align them.
    """
    rttm_path = Path(rttm_path)
    if file_id is None:
        file_id = rttm_path.stem
    with rttm_path.open("w", encoding="utf-8") as fh:
        for s in segments:
            if isinstance(s, dict):
                start = float(s["start"])
                end = float(s["end"])
                speaker = str(s["speaker"])
            else:
                start = s.start
                end = s.end
                speaker = s.speaker
            duration = max(0.0, end - start)
            # Standard 10-token RTTM SPEAKER line.
            fh.write(
                f"{_RTTM_TYPE} {file_id} 1 "
                f"{start:.3f} {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )


def segments_to_annotation(
    segments: Iterable[DiarSegment | Dict[str, object]],
    uri: Optional[str] = None,
):
    """Convert segments to pyannote.core.Annotation for pyannote.metrics.

    Imported lazily so this module stays usable in environments without
    pyannote.audio (e.g. labeling tools, CI lint).
    """
    from pyannote.core import Annotation, Segment  # type: ignore

    ann = Annotation(uri=uri)
    for s in segments:
        if isinstance(s, dict):
            start = float(s["start"])
            end = float(s["end"])
            speaker = str(s["speaker"])
        else:
            start = s.start
            end = s.end
            speaker = s.speaker
        if end <= start:
            continue
        ann[Segment(start, end)] = speaker
    return ann


# ----- Audacity label track import ------------------------------------------
#
# Audacity exports labels as TSV: start \t end \t label
# Convention for diarization labeling: label = SPEAKER_NN (matching RTTM
# speaker IDs). Anything that doesn't match the SPEAKER_NN pattern is treated
# as a non-speaker label (e.g., "noise", "silence") and dropped.

import re

_SPEAKER_LABEL_RE = re.compile(r"^SPEAKER_\d{2,}$")


def parse_audacity_labels(
    labels_path: Path | str,
    speaker_label_pattern: re.Pattern = _SPEAKER_LABEL_RE,
) -> List[DiarSegment]:
    """Read an Audacity labels.txt file. Drop labels that don't match the
    SPEAKER_NN pattern (so labelers can mark non-speech regions without
    polluting ground truth)."""
    labels_path = Path(labels_path)
    segments: List[DiarSegment] = []
    with labels_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                # Audacity sometimes uses spaces; tolerate that.
                parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{labels_path}:{lineno}: expected 'start\\tend\\tlabel', got {line!r}"
                )
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError as e:
                raise ValueError(
                    f"{labels_path}:{lineno}: non-numeric times: {line!r}"
                ) from e
            label = parts[2].strip()
            if not speaker_label_pattern.match(label):
                continue
            segments.append(DiarSegment(start=start, end=end, speaker=label))
    return segments


def audacity_labels_to_rttm(
    labels_path: Path | str,
    rttm_path: Path | str,
    file_id: Optional[str] = None,
) -> int:
    """One-shot: read Audacity labels, write RTTM. Returns segment count.

    Use from the labeling workflow after exporting labels from Audacity.
    """
    segments = parse_audacity_labels(labels_path)
    write_rttm(rttm_path, segments, file_id=file_id)
    return len(segments)


# ----- Sanity checks --------------------------------------------------------


def validate_segments(segments: List[DiarSegment]) -> List[str]:
    """Return a list of warnings about a segment list. Empty list = clean.

    Catches common labeling mistakes before they reach pyannote.metrics:
      - Negative or zero duration
      - Segments outside [0, very_large) range
      - Speakers with fewer than 1s total speech (likely a labeling typo)

    Does NOT enforce non-overlap — overlapping speech is real and pyannote.metrics
    handles it correctly.
    """
    warnings: List[str] = []
    if not segments:
        warnings.append("no segments parsed")
        return warnings

    speaker_total: Dict[str, float] = {}
    for i, s in enumerate(segments):
        if s.duration <= 0:
            warnings.append(f"segment {i}: non-positive duration ({s.duration:.3f}s)")
        if s.start < 0:
            warnings.append(f"segment {i}: negative start ({s.start:.3f}s)")
        speaker_total[s.speaker] = speaker_total.get(s.speaker, 0.0) + s.duration

    for spk, total in speaker_total.items():
        if total < 1.0:
            warnings.append(
                f"speaker {spk}: only {total:.3f}s total speech (likely typo)"
            )

    return warnings


__all__ = [
    "DiarSegment",
    "parse_rttm",
    "write_rttm",
    "segments_to_annotation",
    "parse_audacity_labels",
    "audacity_labels_to_rttm",
    "validate_segments",
]
