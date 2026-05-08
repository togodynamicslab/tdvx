"""
YouTube ingest helper: download a YouTube URL's audio via yt-dlp, transcode
to 16 kHz mono float32 PCM via ffmpeg, and cache by video ID.

Why server-side: the browser cannot fetch YouTube media directly due to CORS,
and the YouTube IFrame Player API exposes timestamps but not audio bytes.
For transcription we need the raw audio, so the backend extracts it.

Cache layout:
  <cache_root>/<video_id>/
    audio.wav         16 kHz mono float32 PCM (the canonical artifact)
    metadata.json     {video_id, title, duration_s, channel, fetched_at}

Cache hits are by video ID (not full URL) so different youtube.com /
youtu.be / share-link variants of the same video share one cache entry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default cache lives under <repo>/.cache/youtube/. Persists across runs so
# you don't re-download the same video for every iteration. Bypass by setting
# settings.youtube_cache_dir if you want a different location.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "youtube"

# Regex covers youtube.com/watch?v=, youtu.be/, /shorts/, /embed/, /v/.
# Pulled from yt-dlp's own extractor patterns; intentionally permissive on
# trailing query params.
_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|v/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)


@dataclass
class YouTubeAudio:
    """Resolved YouTube audio asset on disk."""
    video_id: str
    title: str
    duration_s: float
    channel: str
    audio_path: Path
    sample_rate: int = 16000

    def metadata_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "duration_s": self.duration_s,
            "channel": self.channel,
            "sample_rate": self.sample_rate,
        }


def extract_video_id(url: str) -> Optional[str]:
    """Pull a YouTube video ID out of any common URL format. Returns None
    if the URL doesn't look like YouTube. Strict 11-char ID match — no false
    positives on random strings."""
    m = _VIDEO_ID_RE.search(url.strip())
    if m:
        return m.group(1)
    # Bare 11-char ID? Allow it.
    bare = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", bare):
        return bare
    return None


def _cache_dir_for(video_id: str, cache_root: Path) -> Path:
    return cache_root / video_id


def _load_cached(video_id: str, cache_root: Path) -> Optional[YouTubeAudio]:
    cdir = _cache_dir_for(video_id, cache_root)
    audio = cdir / "audio.wav"
    meta = cdir / "metadata.json"
    if not (audio.exists() and meta.exists()):
        return None
    try:
        m = json.loads(meta.read_text())
        return YouTubeAudio(
            video_id=video_id,
            title=m.get("title", ""),
            duration_s=float(m.get("duration_s", 0.0)),
            channel=m.get("channel", ""),
            audio_path=audio,
            sample_rate=int(m.get("sample_rate", 16000)),
        )
    except Exception as e:
        logger.warning("Cached metadata for %s unreadable, re-fetching: %s", video_id, e)
        return None


def fetch_youtube_audio(
    url: str,
    cache_root: Path = DEFAULT_CACHE_DIR,
    sample_rate: int = 16000,
) -> YouTubeAudio:
    """Resolve a YouTube URL to a 16 kHz mono PCM WAV on disk.

    Uses the cache if a previous fetch of the same video ID exists. On a
    cache miss, runs yt-dlp + ffmpeg synchronously (yt-dlp downloads the
    best audio stream, ffmpeg transcodes to 16 kHz mono).

    Raises ValueError on a non-YouTube URL or extraction failure.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Not a recognizable YouTube URL: {url!r}")

    cache_root = Path(cache_root)
    cached = _load_cached(video_id, cache_root)
    if cached is not None:
        logger.info("youtube cache HIT for %s (%.1fs)", video_id, cached.duration_s)
        return cached

    logger.info("youtube cache MISS for %s, fetching", video_id)
    cdir = _cache_dir_for(video_id, cache_root)
    cdir.mkdir(parents=True, exist_ok=True)

    # Step 1 — extract metadata + best audio stream URL via yt-dlp.
    # We use the yt_dlp Python API rather than shelling out for clearer errors.
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed. pip install yt-dlp") from e

    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(cdir / "raw.%(ext)s"),
        # Don't write the per-video subtitles / thumbnails — we only need audio.
        "writesubtitles": False,
        "writeinfojson": False,
        "writethumbnail": False,
    }

    # Run yt-dlp with retry-aware error surfacing.
    info: dict
    raw_path: Path
    t0 = time.perf_counter()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)  # type: ignore[arg-type]
            # extract_info returns the merged dict; the actual filename comes back
            # via prepare_filename (post-merge, with the real extension).
            raw_path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:  # type: ignore[attr-defined]
        # Clean up the half-written cache dir so a retry doesn't pick up garbage.
        shutil.rmtree(cdir, ignore_errors=True)
        raise ValueError(f"yt-dlp download failed for {url!r}: {e}") from e
    except Exception as e:
        shutil.rmtree(cdir, ignore_errors=True)
        raise ValueError(f"yt-dlp failed for {url!r}: {type(e).__name__}: {e}") from e

    download_s = time.perf_counter() - t0
    logger.info("yt-dlp downloaded %s in %.1fs to %s", video_id, download_s, raw_path)

    # Step 2 — transcode to 16 kHz mono PCM_16 WAV via ffmpeg.
    out_path = cdir / "audio.wav"
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(raw_path),
                "-ar", str(sample_rate),
                "-ac", "1",
                "-f", "wav",
                "-acodec", "pcm_s16le",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        shutil.rmtree(cdir, ignore_errors=True)
        stderr = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else ""
        raise ValueError(f"ffmpeg transcode failed for {video_id}: {stderr}") from e
    except FileNotFoundError as e:
        shutil.rmtree(cdir, ignore_errors=True)
        raise RuntimeError("ffmpeg not on PATH") from e
    finally:
        # Drop the original codec-specific download — we keep the canonical WAV only.
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass

    transcode_s = time.perf_counter() - t0

    # Step 3 — write metadata.
    meta = {
        "video_id": video_id,
        "title": info.get("title", ""),
        "duration_s": float(info.get("duration", 0.0) or 0.0),
        "channel": info.get("uploader", "") or info.get("channel", ""),
        "sample_rate": sample_rate,
        "fetched_at": time.time(),
        "download_s": round(download_s, 2),
        "transcode_s": round(transcode_s, 2),
    }
    (cdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return YouTubeAudio(
        video_id=video_id,
        title=meta["title"],
        duration_s=meta["duration_s"],
        channel=meta["channel"],
        audio_path=out_path,
        sample_rate=sample_rate,
    )


def load_audio_pcm_f32(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a cached WAV as float32 mono. Used by the streaming endpoint."""
    import soundfile as sf
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, int(sr)


__all__ = [
    "YouTubeAudio",
    "extract_video_id",
    "fetch_youtube_audio",
    "load_audio_pcm_f32",
]
