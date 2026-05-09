#!/usr/bin/env python3
"""
Build synthetic multi-speaker PT-BR diarization eval clips.

Why this exists:
  The hand-labeled eval set is the gold standard, but it takes days to record
  and label. While that's in flight, we need clips today so the rest of the
  pipeline (DER computation, bench script, Sortformer wiring) can be developed
  and tested. Synthetic clips give us mechanically-perfect ground truth for
  free, which is also useful as a sanity floor: if a backend can't get high
  DER on stitched clean voices with no overlap, it's broken.

What it does:
  Concatenates short clips from the existing single-speaker tests/corpus/pt-BR/
  voice corpus into multi-speaker conversations. Outputs:
    - {clip_id}.wav  ← the stitched audio (16 kHz mono float32 → PCM_16)
    - {clip_id}.rttm ← ground truth from the splice points
    - {clip_id}.json ← metadata flagging this as synthetic

Caveats:
  Synthetic clips are NOT a real diarization benchmark. They have:
    - No speaker overlap (real conversation has 5-15% overlap)
    - No backchannels ("uh-huh", "right")
    - Studio-quiet recording quality
    - Unrealistically clean turn-taking
  The hand-labeled set is required to make the real go/no-go decision. This
  exists for plumbing work only.

Usage:
  python scripts/build_synthetic_diar_eval.py --n-clips 4 --speakers-per-clip 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Helpers live with the corpus — import via path.
sys.path.insert(0, str(_REPO_ROOT / "tests" / "corpus" / "pt-BR" / "diar-eval"))
from rttm_helpers import DiarSegment, write_rttm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("synth-diar")


def _voice_id(path: Path) -> str:
    """Pull a stable speaker key out of a voice clip filename.

    The corpus filenames look like 'Maria_-_Calm_and_Resonating.wav'. Use
    everything before the first ' - ' or '__' as the speaker identity so all
    clips of 'Maria - Calm' map to the same speaker.
    """
    stem = path.stem
    for sep in (" - ", "__", "_-_"):
        if sep in stem:
            stem = stem.split(sep, 1)[0]
            break
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def _load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load any WAV/MP3 as float32 mono 16 kHz."""
    import soundfile as sf
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        # Fallback to librosa (handles MP3 etc.).
        import librosa
        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def _gather_voices(corpus_dir: Path, min_per_voice: int = 1) -> dict[str, List[Path]]:
    """Group corpus clips by inferred speaker. Drop speakers with too few clips."""
    by_speaker: dict[str, List[Path]] = {}
    for p in corpus_dir.glob("*.wav"):
        spk = _voice_id(p)
        by_speaker.setdefault(spk, []).append(p)
    for p in corpus_dir.glob("*.mp3"):
        spk = _voice_id(p)
        by_speaker.setdefault(spk, []).append(p)
    return {k: sorted(v) for k, v in by_speaker.items() if len(v) >= min_per_voice}


def _build_one_clip(
    clip_id: str,
    out_dir: Path,
    speakers: List[str],
    voices_by_speaker: dict[str, List[Path]],
    target_duration_s: float,
    sample_rate: int,
    rng: random.Random,
) -> dict:
    """Build a single synthetic clip. Returns its manifest entry."""
    chunks: List[np.ndarray] = []
    segments: List[DiarSegment] = []
    cursor_s = 0.0

    while cursor_s < target_duration_s:
        spk_idx = rng.randrange(len(speakers))
        speaker_label = f"SPEAKER_{spk_idx:02d}"
        source_path = rng.choice(voices_by_speaker[speakers[spk_idx]])
        try:
            audio = _load_audio(source_path, sample_rate)
        except Exception as e:
            logger.warning("skip %s: %s", source_path, e)
            continue
        if audio.size < int(0.3 * sample_rate):  # skip <300ms snippets
            continue
        # Optional: trim very long clips so we don't end up with a 60s monologue.
        max_samples = int(rng.uniform(2.5, 6.0) * sample_rate)
        if audio.size > max_samples:
            start = rng.randrange(0, audio.size - max_samples)
            audio = audio[start : start + max_samples]
        # Tiny silence between turns (50-250ms) — realistic enough for a sanity floor.
        gap_s = rng.uniform(0.05, 0.25)
        gap = np.zeros(int(gap_s * sample_rate), dtype=np.float32)
        chunks.append(audio)
        chunks.append(gap)
        seg_start = cursor_s
        seg_end = cursor_s + audio.size / sample_rate
        segments.append(DiarSegment(start=seg_start, end=seg_end, speaker=speaker_label))
        cursor_s = seg_end + gap_s

    full_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    duration_s = full_audio.size / sample_rate

    # Write outputs.
    import soundfile as sf
    wav_path = out_dir / f"{clip_id}.wav"
    sf.write(str(wav_path), full_audio, sample_rate, subtype="PCM_16")

    rttm_path = out_dir / f"{clip_id}.rttm"
    write_rttm(rttm_path, segments, file_id=clip_id)

    meta = {
        "clip_id": clip_id,
        "duration_s": round(duration_s, 3),
        "n_speakers": len(speakers),
        "speakers": [
            {
                "id": f"SPEAKER_{i:02d}",
                "role": "synthetic",
                "source_voice": speakers[i],
                "gender": "unknown",
            }
            for i in range(len(speakers))
        ],
        "recording": {
            "source": "synthetic",
            "device": "concatenation",
            "noise_level": "clean",
            "overlap": "none",
        },
        "notes": (
            "Synthetic stitched from tests/corpus/pt-BR/ single-speaker clips. "
            "No overlap, no backchannels, studio-clean. Use as sanity floor only."
        ),
    }
    json_path = out_dir / f"{clip_id}.json"
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info("wrote %s (%.1fs, %d turns, %d speakers)",
                clip_id, duration_s, len(segments), len(speakers))
    return {
        "clip_id": clip_id,
        "audio": f"clips/{clip_id}.wav",
        "rttm": f"clips/{clip_id}.rttm",
        "metadata": f"clips/{clip_id}.json",
        "synthetic": True,
        "duration_s": round(duration_s, 3),
        "n_speakers": len(speakers),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path,
                    default=_REPO_ROOT / "tests" / "corpus" / "pt-BR",
                    help="Source single-speaker voice corpus")
    ap.add_argument("--out-dir", type=Path,
                    default=_REPO_ROOT / "tests" / "corpus" / "pt-BR" / "diar-eval" / "clips",
                    help="Where stitched clips land")
    ap.add_argument("--manifest", type=Path,
                    default=_REPO_ROOT / "tests" / "corpus" / "pt-BR" / "diar-eval" / "manifest.json",
                    help="Manifest JSON to update")
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--speakers-per-clip", type=int, default=3)
    ap.add_argument("--seconds-per-clip", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--prefix", type=str, default="synth")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    voices = _gather_voices(args.corpus_dir, min_per_voice=1)
    if len(voices) < args.speakers_per_clip:
        logger.error(
            "Need >= %d distinct voices, found %d in %s",
            args.speakers_per_clip, len(voices), args.corpus_dir,
        )
        return 2
    logger.info("found %d distinct voices in %s", len(voices), args.corpus_dir)

    rng = random.Random(args.seed)
    voice_keys = sorted(voices.keys())

    manifest_entries = []
    for i in range(args.n_clips):
        clip_id = f"{args.prefix}-{i:02d}"
        speakers = rng.sample(voice_keys, args.speakers_per_clip)
        entry = _build_one_clip(
            clip_id=clip_id,
            out_dir=args.out_dir,
            speakers=speakers,
            voices_by_speaker=voices,
            target_duration_s=args.seconds_per_clip,
            sample_rate=16000,
            rng=rng,
        )
        manifest_entries.append(entry)

    # Update manifest: replace any prior entries with the same prefix.
    existing = {}
    if args.manifest.exists():
        existing = json.loads(args.manifest.read_text())
    keep = [c for c in existing.get("clips", []) if not c.get("clip_id", "").startswith(args.prefix)]
    existing["schema_version"] = existing.get("schema_version", 1)
    existing["clips"] = keep + manifest_entries
    args.manifest.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    logger.info("updated manifest: %d total clips (%d new)",
                len(existing["clips"]), len(manifest_entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
