"""
Quick WER / latency comparison between tdv1-fast (small), tdv1-medium (medium),
and tdv1 (large-v3). Picks 10 corpus clips spanning short/medium/long durations,
sends each one to /transcribe-fast with each model, measures latency and prints
the text output for side-by-side inspection.

Run from repo root:
    python3 scripts/wer_compare.py [--endpoint URL] [--n 10]

No ground-truth transcripts available, so we print outputs for visual diff and
compute a word-level similarity metric between model pairs (as a proxy for how
much the smaller model diverges from the large one).
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "corpus" / "manifest.json"
DEFAULT_ENDPOINT = "http://96.38.133.243:22961"
MODELS = ["tdv1-fast", "tdv1-medium", "tdv1"]


def post_file(endpoint: str, model: str, wav_path: Path, timeout: float = 60.0):
    """POST the WAV to /transcribe-fast and return (latency_s, json_body)."""
    # urllib requires us to assemble multipart by hand — cheaper than adding deps.
    boundary = f"----wer{random.randint(0, 10**9):x}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{wav_path.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(wav_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    url = f"{endpoint.rstrip('/')}/transcribe-fast?model={model}"
    req = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return time.monotonic() - t0, {"error": f"HTTP {e.code}", "detail": e.read()[:200].decode("utf-8", "replace")}
    return time.monotonic() - t0, data


def extract_text(resp: dict) -> str:
    segs = resp.get("segments", [])
    return " ".join(s.get("text", "").strip() for s in segs).strip()


def word_similarity(a: str, b: str) -> float:
    """Token-level similarity ratio using difflib on word sequences."""
    wa, wb = a.split(), b.split()
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return difflib.SequenceMatcher(None, wa, wb).ratio()


def pick_clips(n: int) -> list[Path]:
    manifest = json.loads(MANIFEST.read_text())
    samples = manifest["samples"]
    # Sort by duration, pick evenly-spaced clips to cover short/medium/long
    samples.sort(key=lambda s: s["duration_s"])
    idxs = [int(i * (len(samples) - 1) / max(1, n - 1)) for i in range(n)]
    return [REPO_ROOT / samples[i]["path"] for i in idxs]


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


def warm(endpoint: str, wav: Path) -> None:
    """Send one request per model to warm the appropriate workers."""
    for m in MODELS:
        post_file(endpoint, m, wav, timeout=60.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--n", type=int, default=8, help="number of clips")
    args = parser.parse_args()

    clips = pick_clips(args.n)
    print(f"Selected {len(clips)} clips spanning short→long:")
    for c in clips:
        print(f"  {c.name}  ({probe_duration(c):.2f}s)")
    print()

    print("Warming workers with one request per model...")
    warm(args.endpoint, clips[0])
    print()

    results: list[dict] = []
    for clip in clips:
        dur = probe_duration(clip)
        entry = {"clip": clip.name, "duration_s": dur, "outputs": {}}
        print(f"=== {clip.name}  ({dur:.2f}s) ===")
        for m in MODELS:
            lat, resp = post_file(args.endpoint, m, clip)
            if "error" in resp:
                text = f"[ERR {resp['error']}]"
                rtf = None
            else:
                text = extract_text(resp)
                rtf = lat / dur if dur > 0 else None
            entry["outputs"][m] = {"latency_s": round(lat, 3), "rtf": round(rtf, 3) if rtf else None, "text": text}
            print(f"  {m:12s}  lat={lat:5.2f}s  rtf={rtf:.3f}  {text[:100]}{'…' if len(text) > 100 else ''}")
        # Similarity: tdv1-fast vs tdv1-medium, and tdv1 as baseline
        fast_text = entry["outputs"]["tdv1-fast"]["text"]
        med_text = entry["outputs"]["tdv1-medium"]["text"]
        large_text = entry["outputs"]["tdv1"]["text"]
        entry["sim_fast_vs_large"] = round(word_similarity(fast_text, large_text), 3)
        entry["sim_med_vs_large"] = round(word_similarity(med_text, large_text), 3)
        entry["sim_fast_vs_med"] = round(word_similarity(fast_text, med_text), 3)
        print(f"  similarity vs large-v3: fast={entry['sim_fast_vs_large']}  medium={entry['sim_med_vs_large']}")
        print(f"  similarity fast↔medium: {entry['sim_fast_vs_med']}")
        print()
        results.append(entry)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for m in MODELS:
        lats = [r["outputs"][m]["latency_s"] for r in results if r["outputs"][m].get("latency_s") is not None]
        rtfs = [r["outputs"][m]["rtf"] for r in results if r["outputs"][m].get("rtf") is not None]
        if lats:
            print(f"{m:14s}  mean_lat={sum(lats)/len(lats):.2f}s  "
                  f"mean_rtf={sum(rtfs)/len(rtfs):.3f}  "
                  f"max_lat={max(lats):.2f}s")

    sims_fast = [r["sim_fast_vs_large"] for r in results]
    sims_med = [r["sim_med_vs_large"] for r in results]
    print()
    print(f"tdv1-fast   vs large-v3: mean word-similarity = {sum(sims_fast)/len(sims_fast):.3f}  "
          f"min = {min(sims_fast):.3f}")
    print(f"tdv1-medium vs large-v3: mean word-similarity = {sum(sims_med)/len(sims_med):.3f}  "
          f"min = {min(sims_med):.3f}")

    # Save raw
    out_path = REPO_ROOT / "results" / "wer_compare.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRaw results: {out_path}")


if __name__ == "__main__":
    main()
