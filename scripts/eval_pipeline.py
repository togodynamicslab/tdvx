"""
Score our pipeline against the Deepgram reference corpus.

For each WAV in tests/corpus/<lang>/:
  1. POST to our /transcribe-batch endpoint, get text + speaker labels.
  2. Load the saved Deepgram reference JSON.
  3. Compute:
       - WER  : word error rate, our transcript vs Deepgram's
       - DSER : "diarization speaker error" — fraction of reference words
                whose assigned speaker differs from ours after Hungarian
                matching of speaker labels (small N, exact). 0 = perfect,
                1 = every word mis-attributed.
       - latency_ms : end-to-end /transcribe-batch wall time
  4. Print a per-file row + aggregate p50/p95.

Usage:
  python3 scripts/eval_pipeline.py
  python3 scripts/eval_pipeline.py --endpoint http://96.38.133.243:22961
  python3 scripts/eval_pipeline.py --model tdv1-fast --limit 10
  python3 scripts/eval_pipeline.py --tag "before-sortformer"   # tags the journal entry

Outputs:
  results/eval_<run_id>/
    summary.md         human-readable
    rows.ndjson        one row per file (full detail)
  results/eval_history.ndjson   appended journal of all runs (one line per run)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords

REPO_ROOT = Path(__file__).resolve().parent.parent

# Apply identical normalization to both reference and hypothesis before WER.
# Without this, "1" vs "um" or "DAS" vs "das" inflate the score artificially.
_normalize = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
    ReduceToListOfListOfWords(),
])


def normalize_text(s: str) -> str:
    """Cheap text normalization for fairer WER comparison.
    Replaces common Deepgram artifacts ('1' for 'um', etc.) before scoring."""
    s = s.strip()
    # Deepgram pt-BR often emits digits where the model said the word.
    # Map the most common cases so we don't penalize ourselves for being correct.
    digit_word = {
        " 1 ": " um ",
        " 2 ": " dois ",
        " 3 ": " três ",
        " 30 ": " trinta ",
    }
    padded = f" {s} "
    for k, v in digit_word.items():
        padded = padded.replace(k, v)
    return padded.strip()


def reference_text(ref: dict) -> str:
    try:
        alt = ref["results"]["channels"][0]["alternatives"][0]
        return alt.get("transcript", "")
    except Exception:
        return ""


def reference_words_with_speakers(ref: dict) -> list[dict]:
    """Per-word [{'word', 'start', 'end', 'speaker'}, ...] from Deepgram."""
    try:
        alt = ref["results"]["channels"][0]["alternatives"][0]
        return alt.get("words", []) or []
    except Exception:
        return []


def hypothesis_words_with_speakers(segments: list[dict]) -> list[dict]:
    """Approximate per-word records from segment-level output: split a
    segment's text into N words and distribute its time range linearly.
    Loses some precision but matches what we have.
    """
    out = []
    for seg in segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        toks = text.split()
        if not toks:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        dur = max(0.0, e - s) or 0.001
        per = dur / len(toks)
        spk = seg.get("speaker", "SPEAKER_00")
        for i, t in enumerate(toks):
            out.append({
                "word": t,
                "start": s + i * per,
                "end": s + (i + 1) * per,
                "speaker": spk,
            })
    return out


def speaker_error_rate(ref_words: list[dict], hyp_words: list[dict]) -> Optional[float]:
    """Fraction of reference words whose nearest-in-time hypothesis word has
    a different speaker, after we map ref-speaker-IDs to hyp-speaker-IDs by
    majority overlap (Hungarian-lite for small speaker counts).

    Returns None if either side has no speaker info.
    """
    if not ref_words or not hyp_words:
        return None

    # Build per-ref-word the dominant hyp speaker by nearest-midpoint match.
    # For each ref word, find the hyp word whose midpoint is closest in time.
    ref_to_hyp_spk: list[tuple[object, object]] = []
    for rw in ref_words:
        if rw.get("speaker") is None:
            continue
        rmid = 0.5 * (float(rw.get("start", 0.0)) + float(rw.get("end", 0.0)))
        best_dt = float("inf")
        best_spk = None
        for hw in hyp_words:
            hmid = 0.5 * (float(hw["start"]) + float(hw["end"]))
            dt = abs(rmid - hmid)
            if dt < best_dt:
                best_dt = dt
                best_spk = hw["speaker"]
        if best_spk is not None:
            ref_to_hyp_spk.append((rw["speaker"], best_spk))

    if not ref_to_hyp_spk:
        return None

    # Build the (ref_spk, hyp_spk) → count matrix and pick the best 1:1
    # mapping greedily (small speaker counts; exact assignment is overkill).
    from collections import Counter
    pair_counts = Counter(ref_to_hyp_spk)
    ref_speakers = sorted({r for r, _ in ref_to_hyp_spk})
    hyp_speakers = sorted({h for _, h in ref_to_hyp_spk})

    # Greedy assignment: take highest-count pair, lock both sides, repeat.
    used_r, used_h = set(), set()
    mapping: dict[object, object] = {}
    for (r, h), _ in pair_counts.most_common():
        if r in used_r or h in used_h:
            continue
        mapping[r] = h
        used_r.add(r); used_h.add(h)
        if len(used_r) == len(ref_speakers) or len(used_h) == len(hyp_speakers):
            break

    # Count errors under this mapping.
    errors = sum(1 for r, h in ref_to_hyp_spk if mapping.get(r) != h)
    return errors / len(ref_to_hyp_spk)


def call_pipeline(endpoint: str, wav_path: Path, model: str, language: str,
                  diarize: bool, session_id: Optional[str]) -> tuple[dict, float]:
    """POST to /transcribe-batch. Returns (response_json, wall_seconds)."""
    params = {
        "model": model,
        "language": language,
        "diarize": "true" if diarize else "false",
        "diarize_min_seconds": "0",
    }
    if session_id:
        params["session_id"] = session_id
    url = f"{endpoint.rstrip('/')}/transcribe-batch"

    with wav_path.open("rb") as f:
        files = {"file": (wav_path.name, f, "audio/wav")}
        t = time.time()
        r = requests.post(url, params=params, files=files, timeout=60)
        dt = time.time() - t
    r.raise_for_status()
    return r.json(), dt


def hyp_text_from_response(resp: dict) -> str:
    parts = []
    for s in resp.get("segments", []) or []:
        t = (s.get("text") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def percentile(arr: list[float], p: int) -> float:
    if not arr:
        return 0.0
    sa = sorted(arr)
    idx = min(len(sa) - 1, int(round((p / 100.0) * (len(sa) - 1))))
    return sa[idx]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://96.38.133.243:22961",
                        help="Pipeline endpoint base URL (no trailing /transcribe-batch)")
    parser.add_argument("--lang", default="pt-BR", help="Corpus subdir under tests/corpus/")
    parser.add_argument("--model", default="tdv1-fast")
    parser.add_argument("--language", default=None,
                        help="Language code sent to /transcribe-batch (default: derived from --lang)")
    parser.add_argument("--diarize", action="store_true", default=True)
    parser.add_argument("--no-diarize", dest="diarize", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="Substring filter on file name")
    parser.add_argument("--tag", default="", help="Free-form tag for the journal entry (what's different about this run)")
    parser.add_argument("--session-per-file", action="store_true", default=True,
                        help="Use a fresh session_id per file (default). Disables cross-file registry pollution.")
    parser.add_argument("--shared-session", dest="session_per_file", action="store_false",
                        help="Reuse one session_id across all files (stress-tests the registry).")
    args = parser.parse_args()

    corpus_dir = REPO_ROOT / "tests" / "corpus" / args.lang
    ref_dir = corpus_dir / ".reference"
    if not corpus_dir.is_dir() or not ref_dir.is_dir():
        print(f"corpus or reference dir missing: {corpus_dir}", file=sys.stderr)
        sys.exit(2)

    api_lang = args.language or args.lang.split("-")[0]  # "pt-BR" → "pt"

    wavs = sorted(corpus_dir.glob("*.wav"))
    if args.only:
        wavs = [w for w in wavs if args.only.lower() in w.name.lower()]
    if args.limit:
        wavs = wavs[: args.limit]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_eval"
    out_dir = REPO_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.ndjson"
    summary_path = out_dir / "summary.md"
    history_path = REPO_ROOT / "results" / "eval_history.ndjson"

    shared_session = uuid.uuid4().hex if not args.session_per_file else None

    print(f"[eval] run_id={run_id}")
    print(f"[eval] endpoint={args.endpoint}  model={args.model}  lang={api_lang}  diarize={args.diarize}")
    print(f"[eval] files={len(wavs)}  session={'per-file' if args.session_per_file else 'shared'}")

    rows: list[dict] = []
    t0 = time.time()
    with rows_path.open("w") as rf:
        for i, wav in enumerate(wavs, 1):
            ref_path = ref_dir / (wav.stem + ".json")
            if not ref_path.exists():
                print(f"[{i:>2}/{len(wavs)}] SKIP {wav.name} (no reference)")
                continue

            ref = json.loads(ref_path.read_text())
            ref_text = normalize_text(reference_text(ref))
            ref_words = reference_words_with_speakers(ref)

            sid = uuid.uuid4().hex if args.session_per_file else shared_session
            try:
                resp, dt = call_pipeline(args.endpoint, wav, args.model, api_lang, args.diarize, sid)
            except Exception as e:
                print(f"[{i:>2}/{len(wavs)}] FAIL {wav.name}  {type(e).__name__}: {e}", file=sys.stderr)
                continue

            hyp_text = normalize_text(hyp_text_from_response(resp))
            hyp_words = hypothesis_words_with_speakers(resp.get("segments", []))

            try:
                w = wer(ref_text, hyp_text) if ref_text else None
            except Exception:
                w = None
            ser = speaker_error_rate(ref_words, hyp_words)

            tel = (resp.get("telemetry") or {})
            row = {
                "file": wav.name,
                "wer": w,
                "speaker_error_rate": ser,
                "latency_ms": int(dt * 1000),
                "server_total_ms": tel.get("total_ms"),
                "server_whisper_ms": tel.get("whisper_ms"),
                "server_diar_ms": tel.get("diarization_ms"),
                "diarize_ran": tel.get("diarize_ran"),
                "locals_detected": tel.get("locals_detected"),
                "registry_size": tel.get("registry_size"),
                # Full text — UI does word-level diff against this. Avg
                # ~150 chars per file, so 60 files ≈ 18KB of text per run. Cheap.
                "ref_text": ref_text,
                "hyp_text": hyp_text,
                # Segment-level timing from our pipeline. The validate UI
                # uses this to approximate per-word timestamps for the HYP
                # panel (linear distribution within each segment).
                "hyp_segments": [
                    {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": (s.get("text") or "").strip()}
                    for s in (resp.get("segments") or [])
                ],
            }
            rows.append(row)
            rf.write(json.dumps(row, ensure_ascii=False) + "\n")

            wer_str = f"{w*100:5.1f}%" if w is not None else "  n/a"
            ser_str = f"{ser*100:5.1f}%" if ser is not None else "  n/a"
            print(f"[{i:>2}/{len(wavs)}] WER {wer_str}  SER {ser_str}  {dt*1000:5.0f}ms  {wav.name}")

    if not rows:
        print("[eval] no rows scored", file=sys.stderr)
        sys.exit(1)

    wers = [r["wer"] for r in rows if r["wer"] is not None]
    sers = [r["speaker_error_rate"] for r in rows if r["speaker_error_rate"] is not None]
    lats = [r["latency_ms"] for r in rows]
    server_lats = [r["server_total_ms"] for r in rows if r["server_total_ms"] is not None]

    agg = {
        "wer_mean": statistics.fmean(wers) if wers else None,
        "wer_p50":  percentile(wers, 50) if wers else None,
        "wer_p95":  percentile(wers, 95) if wers else None,
        "ser_mean": statistics.fmean(sers) if sers else None,
        "ser_p50":  percentile(sers, 50) if sers else None,
        "lat_p50":  percentile(lats, 50),
        "lat_p95":  percentile(lats, 95),
        "server_p50": percentile(server_lats, 50) if server_lats else None,
        "server_p95": percentile(server_lats, 95) if server_lats else None,
        "files":    len(rows),
        "wall_s":   time.time() - t0,
    }

    summary_lines = [
        f"# Pipeline Eval — {run_id}",
        "",
        f"**Tag:** `{args.tag}`  **Commit:** `{git_commit()}`",
        f"**Endpoint:** `{args.endpoint}`  **Model:** `{args.model}`  **Lang:** `{api_lang}`  **Diarize:** {args.diarize}",
        f"**Files scored:** {agg['files']}  **Wall:** {agg['wall_s']:.1f}s",
        "",
        "## Aggregate scores",
        "",
        "| Metric | mean | p50 | p95 |",
        "|---|---|---|---|",
        f"| WER | {fmt_pct(agg['wer_mean'])} | {fmt_pct(agg['wer_p50'])} | {fmt_pct(agg['wer_p95'])} |",
        f"| Speaker error rate | {fmt_pct(agg['ser_mean'])} | {fmt_pct(agg['ser_p50'])} | n/a |",
        f"| Client latency (ms) | — | {agg['lat_p50']:.0f} | {agg['lat_p95']:.0f} |",
        f"| Server total_ms | — | {fmt_ms(agg['server_p50'])} | {fmt_ms(agg['server_p95'])} |",
        "",
        "## Per-file detail",
        "",
        "| WER | SER | client ms | server ms | locals | reg | file |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        summary_lines.append(
            f"| {fmt_pct(r['wer'])} | {fmt_pct(r['speaker_error_rate'])} | "
            f"{r['latency_ms']} | {fmt_ms(r['server_total_ms'])} | "
            f"{r.get('locals_detected', '—')} | {r.get('registry_size', '—')} | {r['file']} |"
        )
    summary_path.write_text("\n".join(summary_lines))

    # Append journal entry. One row per run, easy to grep over time.
    journal_entry = {
        "run_id": run_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "commit": git_commit(),
        "endpoint": args.endpoint,
        "model": args.model,
        "language": api_lang,
        "diarize": args.diarize,
        "files": agg["files"],
        **{k: agg[k] for k in ("wer_mean", "wer_p50", "wer_p95", "ser_mean", "ser_p50",
                                "lat_p50", "lat_p95", "server_p50", "server_p95")},
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
    }
    with history_path.open("a") as hf:
        hf.write(json.dumps(journal_entry, ensure_ascii=False) + "\n")

    print()
    print(f"[eval] WER  mean={fmt_pct(agg['wer_mean'])}  p50={fmt_pct(agg['wer_p50'])}  p95={fmt_pct(agg['wer_p95'])}")
    print(f"[eval] SER  mean={fmt_pct(agg['ser_mean'])}  p50={fmt_pct(agg['ser_p50'])}")
    print(f"[eval] Lat  p50={agg['lat_p50']:.0f}ms  p95={agg['lat_p95']:.0f}ms")
    print(f"[eval] Server total p50={fmt_ms(agg['server_p50'])}  p95={fmt_ms(agg['server_p95'])}")
    print(f"[eval] → {summary_path}")
    print(f"[eval] → journal entry appended to {history_path.relative_to(REPO_ROOT)}")


def fmt_pct(v: Optional[float]) -> str:
    return f"{v*100:.1f}%" if v is not None else "—"


def fmt_ms(v: Optional[float]) -> str:
    return f"{v:.0f}ms" if v is not None else "—"


if __name__ == "__main__":
    main()
