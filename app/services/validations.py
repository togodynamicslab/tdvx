"""
Human validation store for the eval corpus.

Goal: turn the noisy "Deepgram-as-reference" eval into a real ground-truth
eval over time. A human listens to each file, sees ref vs hyp, and picks
which (if any) is correct. Saved verdicts accumulate in a per-language JSON
file under tests/corpus/<lang>/.validations.json.

File shape:
{
  "Anita.wav": {
    "verdict": "ref" | "hyp" | "neither" | "equivalent",
    "corrected_text": null | str,           # optional, for "neither"
    "validated_at": "2026-04-19T18:30:00",
    "validator": "matheus",                 # free-form, defaults to "human"
    "notes": ""                             # free-form
  },
  ...
}

Why JSON-per-lang and not SQLite: the corpus is small (60 files), commits
are easy to inspect in diff, and we never need queries beyond "give me all".
If we ever scale past a few thousand validated samples, swap the storage
without touching the API surface.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_ROOT = REPO_ROOT / "tests" / "corpus"

VALID_VERDICTS = {"ref", "hyp", "neither", "equivalent"}

# One lock per language file. Verdicts come in one at a time from a single
# operator, so contention is near zero — but the lock is cheap insurance.
_locks: dict[str, threading.Lock] = {}
_locks_root_lock = threading.Lock()


def _lock_for(lang: str) -> threading.Lock:
    with _locks_root_lock:
        lock = _locks.get(lang)
        if lock is None:
            lock = threading.Lock()
            _locks[lang] = lock
        return lock


def _validate_lang(lang: str) -> Path:
    if "/" in lang or ".." in lang or not lang:
        raise ValueError(f"invalid lang: {lang!r}")
    d = CORPUS_ROOT / lang
    if not d.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {d}")
    return d


def _path_for(lang: str) -> Path:
    return _validate_lang(lang) / ".validations.json"


def _validate_filename(lang: str, filename: str) -> Path:
    if "/" in filename or ".." in filename or not filename.endswith(".wav"):
        raise ValueError(f"invalid filename: {filename!r}")
    target = (CORPUS_ROOT / lang / filename).resolve()
    if (CORPUS_ROOT / lang).resolve() not in target.parents:
        raise ValueError("path escape")
    if not target.is_file():
        raise FileNotFoundError(f"audio file not found: {filename}")
    return target


def list_validations(lang: str) -> dict[str, Any]:
    """Return all stored validations for a language (may be empty dict)."""
    _validate_lang(lang)
    path = _path_for(lang)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"[validations] failed reading {path}: {e}")
        return {}


def save_validation(
    lang: str,
    filename: str,
    verdict: str,
    corrected_text: Optional[str] = None,
    validator: str = "human",
    notes: str = "",
) -> dict[str, Any]:
    """Persist a single verdict. Returns the stored entry."""
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict must be one of {VALID_VERDICTS}, got {verdict!r}")
    _validate_filename(lang, filename)

    entry = {
        "verdict": verdict,
        "corrected_text": corrected_text,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "validator": validator or "human",
        "notes": notes or "",
    }

    path = _path_for(lang)
    lock = _lock_for(lang)
    with lock:
        current = list_validations(lang)
        current[filename] = entry
        # Atomic write: stage a sibling tmp file, then rename. Avoids partial
        # writes if the process is killed mid-save.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(current, indent=2, ensure_ascii=False))
        tmp.replace(path)
    return entry


def delete_validation(lang: str, filename: str) -> bool:
    """Drop a previously-stored verdict so the file shows as un-validated again."""
    _validate_lang(lang)
    path = _path_for(lang)
    if not path.is_file():
        return False
    lock = _lock_for(lang)
    with lock:
        current = list_validations(lang)
        if filename not in current:
            return False
        del current[filename]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(current, indent=2, ensure_ascii=False))
        tmp.replace(path)
    return True


def _normalize_for_equivalence(s: str) -> str:
    """Aggressive normalization to detect 'equivalent' transcripts.

    Strips punctuation, lowercases, collapses whitespace, and folds common
    Portuguese contractions / spelling variants we'd never want to penalize.
    """
    import re
    s = s.lower()
    s = re.sub(r"[.,!?;:\"'`´()\[\]{}—–\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Common pt-BR equivalences. Keep the list short and obviously-safe.
    aliases = [
        (r"\bpara\b", "pra"),
        (r"\bde isso\b", "disso"),
        (r"\bem isso\b", "nisso"),
        (r"\bde aqui\b", "daqui"),
        (r"\bvocê\b", "voce"),
        (r"\bestá\b", "esta"),
        (r"\bnão\b", "nao"),
        (r"\bé\b", "e"),
    ]
    for pat, repl in aliases:
        s = re.sub(pat, repl, s)
    return s


def auto_prefill_equivalent(
    lang: str,
    pairs: list[dict[str, Any]],
    validator: str = "auto-equiv",
) -> dict[str, int]:
    """Bulk-mark files as equivalent when ref/hyp normalize to the same text.

    Args:
      pairs: [{"file": "...", "ref_text": "...", "hyp_text": "..."}, ...]
      validator: stamp on each entry so a human can later filter these out.

    Skips files that already have a verdict (won't overwrite human work).
    Returns counts: {"matched": N, "skipped_existing": N, "skipped_different": N}.
    """
    _validate_lang(lang)
    existing = list_validations(lang)
    matched = 0
    skipped_existing = 0
    skipped_different = 0
    lock = _lock_for(lang)
    with lock:
        # Re-read inside the lock so concurrent saves don't get clobbered.
        current = list_validations(lang)
        for p in pairs:
            fname = p.get("file") or ""
            if not fname.endswith(".wav"):
                continue
            if fname in current and fname in existing:
                skipped_existing += 1
                continue
            ref = p.get("ref_text") or ""
            hyp = p.get("hyp_text") or ""
            if _normalize_for_equivalence(ref) == _normalize_for_equivalence(hyp):
                current[fname] = {
                    "verdict": "equivalent",
                    "corrected_text": None,
                    "validated_at": datetime.now().isoformat(timespec="seconds"),
                    "validator": validator,
                    "notes": "auto-prefilled (normalized texts match)",
                }
                matched += 1
            else:
                skipped_different += 1

        if matched > 0:
            path = _path_for(lang)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(current, indent=2, ensure_ascii=False))
            tmp.replace(path)
    return {
        "matched": matched,
        "skipped_existing": skipped_existing,
        "skipped_different": skipped_different,
    }


def stats(lang: str) -> dict[str, Any]:
    """Aggregate counts for a language — used by the UI progress bar."""
    _validate_lang(lang)
    items = list_validations(lang)
    counts = {"ref": 0, "hyp": 0, "neither": 0, "equivalent": 0}
    for v in items.values():
        verdict = v.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    total_files = sum(1 for _ in (CORPUS_ROOT / lang).glob("*.wav"))
    return {
        "total_files": total_files,
        "validated": len(items),
        "remaining": max(0, total_files - len(items)),
        "verdict_counts": counts,
    }
