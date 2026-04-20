"""
Import downloaded ElevenLabs voice samples as the load-test corpus.

Converts all MP3s in ~/Downloads (under 500 KB — the voice preview size) to
16 kHz mono PCM16 WAV under tests/corpus/pt-BR/. The stochastic test runner
builds variable-duration clips by concatenating these short samples, so we
get multi-speaker realism without spending ElevenLabs credits.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path.home() / "Downloads"
DEST_DIR = REPO_ROOT / "tests" / "corpus" / "pt-BR"
MAX_SIZE = 500 * 1024
MANIFEST = REPO_ROOT / "tests" / "corpus" / "manifest.json"


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(out.strip())


def convert(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            str(dst),
        ],
        check=True,
    )


def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR} not found", file=sys.stderr)
        return 1

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    candidates = sorted(p for p in SRC_DIR.glob("*.mp3") if p.stat().st_size < MAX_SIZE)
    if not candidates:
        print(f"ERROR: no MP3 samples (<500KB) in {SRC_DIR}", file=sys.stderr)
        return 1

    print(f"Converting {len(candidates)} samples → {DEST_DIR}")
    manifest = []
    for src in candidates:
        dst = DEST_DIR / (src.stem.replace(" ", "_") + ".wav")
        if not dst.exists():
            convert(src, dst)
        dur = probe_duration(dst)
        manifest.append({"name": dst.stem, "path": str(dst.relative_to(REPO_ROOT)), "duration_s": round(dur, 3)})

    manifest.sort(key=lambda x: x["duration_s"])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"samples": manifest}, indent=2))

    total = sum(m["duration_s"] for m in manifest)
    durs = sorted(m["duration_s"] for m in manifest)
    p50 = durs[len(durs) // 2]
    p95 = durs[int(len(durs) * 0.95)]
    print(f"Imported {len(manifest)} clips | total {total:.1f}s | p50={p50:.2f}s p95={p95:.2f}s max={max(durs):.2f}s")
    print(f"Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
