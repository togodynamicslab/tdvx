"""
Read stress-test results from disk and compute summary rollups.

Used by both the /api/runs FastAPI endpoints and the offline
scripts/build_results_index.py (which writes static JSON for old-style bundling).
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"

logger = logging.getLogger(__name__)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round(p * (len(vs) - 1)))))
    return vs[k]


def parse_run(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "config.json"
    ndjson_path = run_dir / "requests.ndjson"
    if not cfg_path.exists() or not ndjson_path.exists():
        return None

    config = json.loads(cfg_path.read_text())
    records: list[dict[str, Any]] = []
    for line in ndjson_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not records:
        return None

    ok = [r for r in records if r.get("status") == "ok"]
    ttfr = [r["ttfr_ms"] for r in ok if r.get("ttfr_ms") is not None]
    rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
    durations = [r["audio_duration_s"] for r in records if r.get("audio_duration_s") is not None]

    success_rate = 100.0 * len(ok) / len(records) if records else 0.0

    # Table A — per sim-hour
    by_hour: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        h = r.get("sim_hour")
        if h is not None:
            by_hour[int(h)].append(r)
    table_a = []
    for h in range(24):
        bucket = by_hour.get(h, [])
        if not bucket:
            continue
        b_ok = [r for r in bucket if r.get("status") == "ok"]
        b_ttfr = [r["ttfr_ms"] for r in b_ok if r.get("ttfr_ms") is not None]
        b_rtf = [r["rtf"] for r in b_ok if r.get("rtf") is not None]
        table_a.append({
            "hour": h,
            "requests": len(bucket),
            "ttfr_p50_s": round(_pct(b_ttfr, 0.5) / 1000, 3),
            "ttfr_p95_s": round(_pct(b_ttfr, 0.95) / 1000, 3),
            "ttfr_p99_s": round(_pct(b_ttfr, 0.99) / 1000, 3),
            "rtf_p95": round(_pct(b_rtf, 0.95), 3),
            "success_pct": round(100.0 * len(b_ok) / len(bucket), 2),
        })

    # Table D — failures. Records can be missing 'status' entirely (new
    # ws-stoch format emits partial records on connect errors), so default.
    fail_types = sorted({r.get("status", "unknown") for r in records if r.get("status") != "ok"})
    table_d_rows = []
    for h in range(24):
        bucket = [r for r in by_hour.get(h, []) if r.get("status") != "ok"]
        if not bucket:
            continue
        row: dict[str, Any] = {"hour": h, "total": len(bucket)}
        for t in fail_types:
            row[t] = sum(1 for r in bucket if r.get("status") == t)
        table_d_rows.append(row)

    # Table B — stochastic validation
    try:
        times = sorted(
            datetime.fromisoformat(r["wall_ts"].replace("Z", "+00:00")).timestamp()
            for r in records if r.get("wall_ts")
        )
        deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)] if len(times) > 1 else []
        cv = (
            statistics.stdev(deltas) / statistics.mean(deltas)
            if len(deltas) > 1 and statistics.mean(deltas) > 0
            else 0.0
        )
    except Exception:
        cv = 0.0
        deltas = []
    burst_count = sum(1 for r in records if r.get("is_burst"))
    burst_pct = 100.0 * burst_count / len(records) if records else 0.0
    expected = config["users"] * config["audios_per_user_day"]
    table_b = [
        {
            "metric": "Total requests",
            "expected": f"~{expected}",
            "measured": len(records),
            "status": "PASS" if abs(len(records) - expected) < expected * 0.3 else "WARN",
        },
        {
            "metric": "CV of inter-arrivals",
            "expected": "~1.0",
            "measured": round(cv, 3),
            "status": "PASS" if 0.7 < cv < 1.4 else "WARN",
        },
        {
            "metric": "Burst fraction",
            "expected": "~5%",
            "measured": f"{burst_pct:.2f}%",
            "status": "PASS" if 3 < burst_pct < 8 else "WARN",
        },
    ]

    # Executive verdict
    peak = [r for r in ok if 14 <= r.get("sim_hour", -1) < 18]
    peak_ttfr = [r["ttfr_ms"] for r in peak if r.get("ttfr_ms") is not None]
    peak_rtf = [r["rtf"] for r in peak if r.get("rtf") is not None]

    p95_s = _pct(ttfr, 0.95) / 1000 if ttfr else 0.0
    if success_rate < 99.0 or p95_s > 30:
        verdict = "FAIL"
    elif success_rate < 99.9 or p95_s > 15:
        verdict = "DEGRADED"
    else:
        verdict = "PASS"

    # Duration histogram
    edges = [0, 5, 10, 15, 30, 60, 120, 300, 600, 9999]
    hist = [0] * (len(edges) - 1)
    for d in durations:
        for i in range(len(edges) - 1):
            if edges[i] <= d < edges[i + 1]:
                hist[i] += 1
                break
    dur_histogram = [
        {
            "bucket": f"{edges[i]}-{edges[i+1]}s" if edges[i+1] < 9999 else f"{edges[i]}s+",
            "count": hist[i],
        }
        for i in range(len(hist))
    ]

    if len(deltas) > 2000:
        step = len(deltas) // 2000
        deltas_sample = deltas[::step]
    else:
        deltas_sample = deltas

    status_counts = dict(Counter(r.get("status", "unknown") for r in records))

    summary = {
        "run_id": config["run_id"],
        "preset": config.get("preset"),
        "scenario": config.get("scenario"),
        "verdict": verdict,
        "started_at": config["run_id"][:15],
        "real_duration_min": round(config["real_duration_s"] / 60, 1),
        "simulated_hours": config["simulated_hours"],
        "compression": config["compression"],
        "model": config.get("model"),
        "endpoint": config.get("endpoint"),
        "users": config.get("users"),
        "audios_per_user_day": config.get("audios_per_user_day"),
        "total_requests": len(records),
        "success_rate_pct": round(success_rate, 2),
        "ttfr_p50_s": round(_pct(ttfr, 0.5) / 1000, 3) if ttfr else None,
        "ttfr_p95_s": round(_pct(ttfr, 0.95) / 1000, 3) if ttfr else None,
        "ttfr_p99_s": round(_pct(ttfr, 0.99) / 1000, 3) if ttfr else None,
        "rtf_p50": round(_pct(rtfs, 0.5), 3) if rtfs else None,
        "rtf_p95": round(_pct(rtfs, 0.95), 3) if rtfs else None,
        "peak_ttfr_p95_s": round(_pct(peak_ttfr, 0.95) / 1000, 3) if peak_ttfr else None,
        "peak_rtf_p95": round(_pct(peak_rtf, 0.95), 3) if peak_rtf else None,
        "status_counts": status_counts,
    }

    detail = {
        **summary,
        "config": config,
        "table_a": table_a,
        "table_b": table_b,
        "table_d": {"fail_types": fail_types, "rows": table_d_rows},
        "duration_histogram": dur_histogram,
        "inter_arrival_sample": [round(d, 3) for d in deltas_sample[:2000]],
    }
    return detail


def list_runs() -> list[dict[str, Any]]:
    """Return summaries for all complete runs, newest first.

    Skips runs whose config shape doesn't match this parser (e.g. the newer
    ws_stress/ws_stoch formats). They're viewable via /api/evals or by
    reading their summary.md directly.
    """
    if not RESULTS_ROOT.exists():
        return []
    out = []
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        try:
            detail = parse_run(run_dir)
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"[stress_results] skipping {run_dir.name}: {type(e).__name__}: {e}")
            continue
        if detail is None:
            continue
        summary_keys = {
            "run_id", "preset", "scenario", "verdict", "started_at",
            "real_duration_min", "simulated_hours", "compression",
            "model", "endpoint", "users", "audios_per_user_day",
            "total_requests", "success_rate_pct",
            "ttfr_p50_s", "ttfr_p95_s", "ttfr_p99_s",
            "rtf_p50", "rtf_p95",
            "peak_ttfr_p95_s", "peak_rtf_p95", "status_counts",
        }
        out.append({k: v for k, v in detail.items() if k in summary_keys})
    out.sort(key=lambda r: r["started_at"], reverse=True)
    return out


def get_run(run_id: str) -> dict[str, Any] | None:
    run_dir = RESULTS_ROOT / run_id
    if not run_dir.is_dir():
        return None
    return parse_run(run_dir)
