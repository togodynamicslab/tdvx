"""
Read-only access to pipeline eval runs created by scripts/eval_pipeline.py.

Mirror of stress_results.py but for eval data — separate file so the two
result types (stress vs eval) stay independent and can evolve apart.

Source of truth on disk:
  results/eval_history.ndjson           — one JSON line per run (the journal)
  results/<run_id>/summary.md           — markdown report for a single run
  results/<run_id>/rows.ndjson          — one JSON line per file scored

We treat eval_history.ndjson as the index (cheap to scan) and lazy-load
per-run rows.ndjson only on detail lookup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
HISTORY_FILE = RESULTS_ROOT / "eval_history.ndjson"


def _safe_load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError as e:
                logger.warning(f"[eval_results] bad ndjson line in {path}: {e}")
    return out


def list_runs() -> list[dict[str, Any]]:
    """Return all eval-history entries, newest first.

    Each entry is the full journal row (already small — ~15 keys).
    """
    rows = _safe_load_json_lines(HISTORY_FILE)
    rows.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return rows


def get_run(run_id: str) -> Optional[dict[str, Any]]:
    """Return one run's full detail: journal entry + per-file rows + raw summary.md."""
    rows = _safe_load_json_lines(HISTORY_FILE)
    journal = next((r for r in rows if r.get("run_id") == run_id), None)
    if journal is None:
        return None

    run_dir = RESULTS_ROOT / run_id
    per_file = _safe_load_json_lines(run_dir / "rows.ndjson")

    summary_path = run_dir / "summary.md"
    summary_md: Optional[str] = None
    if summary_path.is_file():
        try:
            summary_md = summary_path.read_text()
        except Exception as e:
            logger.warning(f"[eval_results] failed reading {summary_path}: {e}")

    return {
        **journal,
        "rows": per_file,
        "summary_md": summary_md,
    }


def run_dir_for(run_id: str) -> Path:
    """Used by route handlers that stream individual artifact files."""
    return RESULTS_ROOT / run_id
