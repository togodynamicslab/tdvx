"""
Generate Deepgram reference transcripts for the test corpus.

For each WAV in tests/corpus/<lang>/, POST to Deepgram's /v1/listen with
diarization enabled. Save the full JSON response under tests/corpus/<lang>/
.reference/<name>.json. These references become the ground-truth (best
available proxy) that future eval runs score our pipeline against.

Usage:
  python3 scripts/build_deepgram_reference.py                 # pt-BR, all files
  python3 scripts/build_deepgram_reference.py --lang en-US    # English corpus
  python3 scripts/build_deepgram_reference.py --only Anita    # one file (substring match)
  python3 scripts/build_deepgram_reference.py --force         # re-fetch even if reference exists

Idempotent by default: skips files whose .reference/<name>.json already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DG_URL = "https://api.deepgram.com/v1/listen"


def load_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPGRAM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    print("DEEPGRAM_API_KEY not set in env or .env", file=sys.stderr)
    sys.exit(2)


def deepgram_lang_code(corpus_lang: str) -> str:
    # Deepgram uses BCP-47-style codes; our corpus dirs are already close.
    return {"pt-BR": "pt-BR", "pt": "pt-BR", "en-US": "en-US", "en": "en-US"}.get(corpus_lang, corpus_lang)


def transcribe(api_key: str, wav_path: Path, lang: str, model: str = "nova-3") -> dict:
    params = {
        "model": model,
        "language": lang,
        "diarize": "true",
        "punctuate": "true",
        "smart_format": "true",
        "utterances": "true",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    with wav_path.open("rb") as f:
        body = f.read()
    resp = requests.post(DG_URL, params=params, headers=headers, data=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="pt-BR", help="Corpus subdir under tests/corpus/")
    parser.add_argument("--model", default="nova-3", help="Deepgram model")
    parser.add_argument("--only", default=None, help="Substring filter on file name")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if reference exists")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N files (for cost control)")
    args = parser.parse_args()

    corpus_dir = REPO_ROOT / "tests" / "corpus" / args.lang
    if not corpus_dir.is_dir():
        print(f"corpus dir not found: {corpus_dir}", file=sys.stderr)
        sys.exit(2)

    ref_dir = corpus_dir / ".reference"
    ref_dir.mkdir(exist_ok=True)

    api_key = load_api_key()
    dg_lang = deepgram_lang_code(args.lang)

    wavs = sorted(corpus_dir.glob("*.wav"))
    if args.only:
        wavs = [w for w in wavs if args.only.lower() in w.name.lower()]
    if args.limit:
        wavs = wavs[: args.limit]

    print(f"[deepgram-ref] corpus={corpus_dir}  files={len(wavs)}  model={args.model}  lang={dg_lang}")

    done = skipped = failed = 0
    t0 = time.time()
    for i, wav in enumerate(wavs, 1):
        ref_path = ref_dir / (wav.stem + ".json")
        if ref_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            t = time.time()
            result = transcribe(api_key, wav, dg_lang, args.model)
            dt = time.time() - t
            ref_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            done += 1
            # Surface a one-line preview so you see what was transcribed.
            try:
                alt = result["results"]["channels"][0]["alternatives"][0]
                preview = (alt.get("transcript") or "").strip()[:80]
            except Exception:
                preview = "<no transcript>"
            print(f"[{i:>2}/{len(wavs)}] {dt:5.2f}s  {wav.name}  | {preview}")
        except requests.HTTPError as e:
            failed += 1
            body = e.response.text[:200] if e.response is not None else ""
            print(f"[{i:>2}/{len(wavs)}] FAIL  {wav.name}  HTTP {e.response.status_code if e.response else '?'}  {body}", file=sys.stderr)
        except Exception as e:
            failed += 1
            print(f"[{i:>2}/{len(wavs)}] FAIL  {wav.name}  {type(e).__name__}: {e}", file=sys.stderr)

    print(f"[deepgram-ref] done in {time.time() - t0:.1f}s  done={done}  skipped={skipped}  failed={failed}  → {ref_dir}")


if __name__ == "__main__":
    main()
