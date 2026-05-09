"""
Stochastic day simulation over the /ws/transcribe path.

Why this exists: tests/ws_stress_test.py runs a FLAT load (N users streaming
constantly for 24 min). The existing tests/stochastic_test.py generates a
REALISTIC day (log-normal sessions, diurnal peak hours, burst probability)
but drives /transcribe-batch. This script combines the two: same stochastic
day shape as stochastic_test.py, but each scheduled session opens a
WebSocket, streams its chunks, and closes.

Mirrors the same NDJSON + summary.md output format so results land in
/runs alongside the HTTP stress runs.

Usage:
  python3 tests/ws_stoch_test.py --endpoint http://127.0.0.1:8000 \
      --users 500 --real-duration-s 1440 --simulated-hours 24 \
      --model tdv1-fast --language pt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import struct
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import soundfile as sf
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

REPO_ROOT = Path(__file__).resolve().parent.parent
# Import the sibling module directly — importing as `tests.stochastic_test`
# only works when run from the repo root, and the server invokes us from
# various cwds.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stochastic_test import (  # type: ignore
    DIURNAL_POINTS,
    PRESETS,
    SCENARIOS,
    RunConfig,
    SimClock,
    diurnal_multiplier,
    load_corpus,
    _build_omi_schedule,
)


# ---------- Record / output shape ----------

@dataclass
class ChunkRecord:
    wall_ts: str
    sim_ts: str
    sim_hour: int
    user_id: str
    session_id: str
    chunk_idx: int
    audio_duration_s: float
    ttfr_ms: Optional[float] = None    # send → first server response for this chunk
    whisper_ms: Optional[int] = None
    diarization_ms: Optional[int] = None
    server_total_ms: Optional[int] = None
    worker: Optional[str] = None
    speaker: Optional[str] = None
    registry_size: Optional[int] = None
    status: str = "ok"
    error: Optional[str] = None
    is_burst: bool = False


# ---------- Audio helpers ----------

def _load_corpus_clips() -> list[Path]:
    """Prefer the manifest-based corpus (so classic + omi match), else scan."""
    try:
        corpus = load_corpus()
        return [c.path for c in corpus]
    except SystemExit:
        # Manifest missing — fall back to scanning tests/corpus/pt-BR/*.wav
        pass
    scan_dir = REPO_ROOT / "tests" / "corpus" / "pt-BR"
    return sorted(scan_dir.glob("*.wav"))


def _read_clip_as_float32(path: Path, target_sr: int = 16000) -> bytes:
    """Read a WAV file and return float32 PCM bytes at target_sr, mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # Cheap resample via stride — corpus is already 16kHz so this rarely fires.
        ratio = target_sr / sr
        n = int(len(audio) * ratio)
        idx = (len(audio) * (range(n))) if False else None  # unused
        import numpy as np  # local import, only needed in the rare resample path
        new_idx = (np.arange(n) / ratio).astype(int)
        new_idx = new_idx[new_idx < len(audio)]
        audio = audio[new_idx]
    return audio.astype("<f4").tobytes()  # little-endian float32


# ---------- Schedule adapter ----------

@dataclass
class _Session:
    user_id: str
    session_id: str
    start_sim_s: float
    chunks: list[tuple[float, bool]]   # (sim_s, is_burst) per chunk, sorted


def _schedule_to_sessions(omi_schedule: list[tuple[float, str, bool]]) -> list[_Session]:
    """Group the flat (sim_s, user, is_burst) schedule into WS-friendly sessions.

    A session is a contiguous run of chunks for the same user whose inter-arrival
    stays below the in-session interval window (~60s). When gaps exceed that,
    we split into a new session — matches real Omi: device records a bit,
    disconnects, reconnects later with a new session_id.
    """
    if not omi_schedule:
        return []
    SPLIT_GAP_S = 120.0  # >2 min gap in SIM seconds → new session
    by_user: dict[str, list[tuple[float, bool]]] = {}
    for t, uid, is_burst in omi_schedule:
        by_user.setdefault(uid, []).append((t, is_burst))

    out: list[_Session] = []
    for uid, pairs in by_user.items():
        pairs.sort(key=lambda p: p[0])
        current: list[tuple[float, bool]] = []
        session_start = 0.0
        for t, is_burst in pairs:
            if not current or (t - current[-1][0]) > SPLIT_GAP_S:
                if current:
                    out.append(_Session(
                        user_id=uid,
                        session_id=uuid.uuid4().hex,
                        start_sim_s=session_start,
                        chunks=current,
                    ))
                current = [(t, is_burst)]
                session_start = t
            else:
                current.append((t, is_burst))
        if current:
            out.append(_Session(
                user_id=uid,
                session_id=uuid.uuid4().hex,
                start_sim_s=session_start,
                chunks=current,
            ))
    out.sort(key=lambda s: s.start_sim_s)
    return out


# ---------- WS client (one session) ----------

def _ws_url(http_endpoint: str, model: str, language: str, session_id: str, diarize: bool) -> str:
    base = http_endpoint.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    q = {
        "model": model,
        "language": language,
        "session_id": session_id,
        "diarize": "true" if diarize else "false",
    }
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    return f"{base}/ws/transcribe?{qs}"


async def _run_session(
    session: _Session,
    cfg: dict[str, Any],
    clock: SimClock,
    corpus: list[Path],
    rng: random.Random,
    log_f: Any,
    records: list[ChunkRecord],
    in_flight: set[asyncio.Task],
) -> None:
    """Open one WS, stream this session's chunks at their scheduled sim times, close.

    Each chunk picks a random corpus clip, reads it as float32 PCM, and sends
    over the WS in one binary frame. The receive loop runs concurrently and
    pops per-chunk telemetry as it arrives; latency = recv_time - send_time
    of the matching chunk (FIFO).
    """
    url = _ws_url(cfg["endpoint"], cfg["model"], cfg["language"], session.session_id, cfg["diarize"])
    wall_start = clock.wall_start
    deadline = wall_start + cfg["real_duration_s"]

    pending_sends: asyncio.Queue[tuple[int, float, float, bool]] = asyncio.Queue()

    async def receiver(ws: Any) -> None:
        """Consume telemetry envelopes, match to pending sends by FIFO order."""
        queued: list[tuple[int, float, float, bool]] = []
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    if pending_sends.empty() and not queued:
                        # No sends pending and nothing queued — we're idle; loop.
                        continue
                    continue
                except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError):
                    break
                recv_t = time.monotonic()
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                tel = data.get("telemetry")
                # Drain any newly-sent items from the queue so we can match FIFO.
                while True:
                    try:
                        queued.append(pending_sends.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if tel is None:
                    # Non-telemetry message (partial segment w/o telemetry). Skip.
                    continue
                if not queued:
                    # Stray telemetry — shouldn't happen but ignore gracefully.
                    continue
                chunk_idx, send_t, dur_s, is_burst = queued.pop(0)
                seg = data.get("segment") or {}
                rec = ChunkRecord(
                    wall_ts=datetime.now(timezone.utc).isoformat(),
                    sim_ts=datetime.fromtimestamp(clock.sim_seconds(recv_t), tz=timezone.utc).isoformat(),
                    sim_hour=int((clock.sim_seconds(recv_t) / 3600.0) % 24.0),
                    user_id=session.user_id,
                    session_id=session.session_id,
                    chunk_idx=chunk_idx,
                    audio_duration_s=round(dur_s, 3),
                    ttfr_ms=round((recv_t - send_t) * 1000.0, 1),
                    whisper_ms=tel.get("whisper_ms"),
                    diarization_ms=tel.get("diarization_ms"),
                    server_total_ms=tel.get("total_ms"),
                    worker=tel.get("worker"),
                    speaker=seg.get("speaker"),
                    registry_size=tel.get("registry_size"),
                    is_burst=is_burst,
                )
                log_f.write(json.dumps(asdict(rec)) + "\n")
                log_f.flush()
                records.append(rec)
        except Exception as e:
            # Don't crash the session on receive-side errors — log a stub record
            # so we can see the failure in the NDJSON.
            rec = ChunkRecord(
                wall_ts=datetime.now(timezone.utc).isoformat(),
                sim_ts=datetime.now(timezone.utc).isoformat(),
                sim_hour=0,
                user_id=session.user_id,
                session_id=session.session_id,
                chunk_idx=-1,
                audio_duration_s=0.0,
                status="recv_error",
                error=f"{type(e).__name__}: {e}"[:200],
            )
            log_f.write(json.dumps(asdict(rec)) + "\n")
            log_f.flush()
            records.append(rec)

    try:
        async with websockets.connect(url, max_size=None, open_timeout=10.0, ping_interval=30.0, ping_timeout=10.0) as ws:
            recv_task = asyncio.create_task(receiver(ws))
            try:
                for chunk_idx, (sim_t, is_burst) in enumerate(session.chunks):
                    # Wait until this chunk's scheduled wall time.
                    target_wall = wall_start + sim_t / cfg["compression"]
                    now = time.monotonic()
                    sleep_for = target_wall - now
                    if sleep_for > 0:
                        try:
                            await asyncio.sleep(min(sleep_for, deadline - now))
                        except asyncio.CancelledError:
                            break
                    if time.monotonic() >= deadline:
                        break

                    # Pick a corpus clip, encode as float32 PCM bytes.
                    clip_path = corpus[rng.randrange(len(corpus))]
                    try:
                        pcm = _read_clip_as_float32(clip_path)
                    except Exception as e:
                        rec = ChunkRecord(
                            wall_ts=datetime.now(timezone.utc).isoformat(),
                            sim_ts=datetime.fromtimestamp(clock.sim_seconds(), tz=timezone.utc).isoformat(),
                            sim_hour=int(clock.sim_hour()),
                            user_id=session.user_id,
                            session_id=session.session_id,
                            chunk_idx=chunk_idx,
                            audio_duration_s=0.0,
                            status="clip_read_err",
                            error=f"{type(e).__name__}: {e}"[:200],
                            is_burst=is_burst,
                        )
                        log_f.write(json.dumps(asdict(rec)) + "\n")
                        log_f.flush()
                        records.append(rec)
                        continue

                    dur_s = len(pcm) / 4.0 / 16000.0  # float32 stereo-mono count → seconds
                    send_t = time.monotonic()
                    await pending_sends.put((chunk_idx, send_t, dur_s, is_burst))
                    try:
                        await ws.send(pcm)
                    except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError) as e:
                        rec = ChunkRecord(
                            wall_ts=datetime.now(timezone.utc).isoformat(),
                            sim_ts=datetime.fromtimestamp(clock.sim_seconds(send_t), tz=timezone.utc).isoformat(),
                            sim_hour=int(clock.sim_hour(send_t)),
                            user_id=session.user_id,
                            session_id=session.session_id,
                            chunk_idx=chunk_idx,
                            audio_duration_s=round(dur_s, 3),
                            status="conn_closed",
                            error=f"{type(e).__name__}: {e}"[:200],
                            is_burst=is_burst,
                        )
                        log_f.write(json.dumps(asdict(rec)) + "\n")
                        log_f.flush()
                        records.append(rec)
                        break

                # Wait briefly for in-flight receipts after last send.
                await asyncio.sleep(2.0)
            finally:
                # Signal end-of-stream then close.
                try:
                    await ws.send("end")
                except Exception:
                    pass
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
    except Exception as e:
        rec = ChunkRecord(
            wall_ts=datetime.now(timezone.utc).isoformat(),
            sim_ts=datetime.now(timezone.utc).isoformat(),
            sim_hour=0,
            user_id=session.user_id,
            session_id=session.session_id,
            chunk_idx=-1,
            audio_duration_s=0.0,
            status="connect_err",
            error=f"{type(e).__name__}: {e}"[:200],
        )
        log_f.write(json.dumps(asdict(rec)) + "\n")
        log_f.flush()
        records.append(rec)


# ---------- Runner ----------

async def run_test(cfg: dict[str, Any], rng: random.Random, out_dir: Path) -> list[ChunkRecord]:
    corpus = _load_corpus_clips()
    if not corpus:
        print("ERROR: no corpus clips found", file=sys.stderr)
        sys.exit(2)
    print(f"Corpus: {len(corpus)} clips")

    # Build a RunConfig shim just so we can reuse _build_omi_schedule().
    rc = RunConfig(
        preset=cfg["preset"],
        scenario=cfg["scenario"],
        workload="omi",
        real_duration_s=cfg["real_duration_s"],
        simulated_hours=cfg["simulated_hours"],
        compression=cfg["compression"],
        users=cfg["users"],
        audios_per_user_day=cfg["audios_per_user_day"],
        endpoint=cfg["endpoint"],
        model=cfg["model"],
        endpoint_path="/ws/transcribe",
        max_concurrency=cfg["max_concurrency"],
        request_timeout_s=cfg["request_timeout_s"],
        seed=cfg["seed"],
        run_id=cfg["run_id"],
        corpus_path=str(REPO_ROOT / "tests" / "corpus" / "pt-BR"),
        extra_query="",
        sessions_per_user_day_mean=cfg["sessions_per_user_day_mean"],
        in_session_interval_mean_s=cfg["in_session_interval_mean_s"],
        chunks_per_session_min=cfg["chunks_per_session_min"],
        chunks_per_session_max=cfg["chunks_per_session_max"],
    )
    flat_schedule = _build_omi_schedule(rc, rng)
    sessions = _schedule_to_sessions(flat_schedule)
    print(f"Omi schedule: {len(flat_schedule)} chunks → {len(sessions)} sessions "
          f"across {cfg['users']} users (~{len(sessions)/cfg['users']:.1f} sessions/user)")
    if sessions:
        first5 = [f"{s.start_sim_s:.1f}s" for s in sessions[:5]]
        print(f"  first 5 session starts (sim_s): {first5}  compression={cfg['compression']:.1f}x")

    wall_start = time.monotonic()
    wall_deadline = wall_start + cfg["real_duration_s"]
    clock = SimClock(cfg["compression"], wall_start)

    ndjson_path = out_dir / "requests.ndjson"
    records: list[ChunkRecord] = []
    in_flight: set[asyncio.Task] = set()
    last_progress = wall_start

    semaphore = asyncio.Semaphore(cfg["max_concurrency"])

    launch_count = 0
    async def launch(s: _Session) -> None:
        nonlocal launch_count
        launch_count += 1
        try:
            async with semaphore:
                await _run_session(s, cfg, clock, corpus, rng, log_f, records, in_flight)
        except Exception as e:
            # Don't let a session crash kill the runner.
            print(f"[launch] session {s.session_id[:8]} failed: {type(e).__name__}: {e}", file=sys.stderr)

    log_f = open(ndjson_path, "w")
    try:
        s_idx = 0
        while s_idx < len(sessions):
            now = time.monotonic()
            if now >= wall_deadline:
                break
            s = sessions[s_idx]
            target_wall = wall_start + s.start_sim_s / cfg["compression"]
            sleep_for = min(target_wall, wall_deadline) - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            now = time.monotonic()
            if now >= wall_deadline:
                break
            task = asyncio.create_task(launch(s))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)
            s_idx += 1

            if now - last_progress >= 30.0:
                elapsed = now - wall_start
                pct = 100.0 * elapsed / cfg["real_duration_s"]
                sim_h = clock.sim_hour(now)
                print(f"  [{elapsed:6.1f}s / {cfg['real_duration_s']:.0f}s  {pct:5.1f}%]  "
                      f"sim_hour={sim_h:5.2f}  scheduled={s_idx}/{len(sessions)}  "
                      f"inflight={len(in_flight):4d}  done={len(records):5d}  "
                      f"queued={len(sessions) - s_idx}")
                last_progress = now

        print(f"\nArrival loop finished. Scheduled {s_idx} sessions (launched {launch_count}). "
              f"Waiting for {len(in_flight)} in-flight sessions...")
        if in_flight:
            await asyncio.wait(in_flight, timeout=cfg["request_timeout_s"] + 60)
    finally:
        log_f.close()

    return records


# ---------- Summary / report ----------

def percentile(arr: list[float], p: float) -> float:
    if not arr:
        return 0.0
    s = sorted(arr)
    return s[min(len(s) - 1, int((p / 100.0) * (len(s) - 1)))]


def write_summary(cfg: dict[str, Any], records: list[ChunkRecord], out_dir: Path) -> None:
    by_hour: dict[int, list[ChunkRecord]] = {}
    status_counts: dict[str, int] = {}
    worker_counts: dict[str, int] = {}
    ttfr_ms: list[float] = []
    server_total: list[float] = []
    whisper: list[float] = []
    diar: list[float] = []

    for r in records:
        by_hour.setdefault(r.sim_hour, []).append(r)
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        if r.worker:
            worker_counts[r.worker] = worker_counts.get(r.worker, 0) + 1
        if r.status == "ok":
            if r.ttfr_ms is not None:
                ttfr_ms.append(r.ttfr_ms)
            if r.server_total_ms is not None:
                server_total.append(r.server_total_ms)
            if r.whisper_ms is not None:
                whisper.append(r.whisper_ms)
            if r.diarization_ms is not None:
                diar.append(r.diarization_ms)

    total = len(records)
    ok = status_counts.get("ok", 0)
    success_pct = (ok / total * 100.0) if total else 0.0

    ttfr_p95 = percentile(ttfr_ms, 95)
    verdict = "PASS"
    if ttfr_p95 > 3000 or success_pct < 95:
        verdict = "FAIL"

    lines = [
        f"# WS Stochastic Day — {cfg['run_id']}",
        "",
        f"**Preset:** `{cfg['preset']}` · **Scenario:** `{cfg['scenario']}`",
        f"**Endpoint:** `{cfg['endpoint']}/ws/transcribe`",
        f"**Users:** {cfg['users']}  **Sim hours:** {cfg['simulated_hours']}  "
        f"**Real:** {cfg['real_duration_s']:.0f}s  **Compression:** {cfg['compression']:.0f}x",
        f"**Model:** `{cfg['model']}`  **Language:** `{cfg['language']}`  **Diarize:** {cfg['diarize']}",
        "",
        f"**Total chunks:** {total}  **OK:** {ok}  **Success:** {success_pct:.2f}%",
        f"**TTFR (client RTT) p50 / p95 / p99:** "
        f"{percentile(ttfr_ms,50):.0f} / {percentile(ttfr_ms,95):.0f} / {percentile(ttfr_ms,99):.0f} ms",
        f"**Server total_ms p50 / p95:** "
        f"{percentile(server_total,50):.0f} / {percentile(server_total,95):.0f} ms",
        f"**Whisper p50 / p95:** "
        f"{percentile(whisper,50):.0f} / {percentile(whisper,95):.0f} ms",
        "",
        f"**Verdict:** {verdict}  (p95 TTFR = {ttfr_p95:.0f} ms, success = {success_pct:.2f}%)",
        "",
        "## Status breakdown",
        "",
        "| status | count | % |",
        "|---|---|---|",
    ]
    for st, cnt in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {st} | {cnt} | {cnt/total*100:.2f}% |")
    lines.append("")

    lines += [
        "## Per-worker fan-out",
        "",
        "| Worker | Chunks | % |",
        "|---|---|---|",
    ]
    for w, cnt in sorted(worker_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{w}` | {cnt} | {cnt/ok*100:.1f}% |")
    lines.append("")

    lines += [
        "## By simulated hour",
        "",
        "| Sim Hour | Chunks | OK | TTFR p50 | TTFR p95 | TTFR p99 | Server p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for hr in sorted(by_hour.keys()):
        rows = by_hour[hr]
        ok_rows = [r for r in rows if r.status == "ok"]
        ttfrs = [r.ttfr_ms for r in ok_rows if r.ttfr_ms is not None]
        srvs = [r.server_total_ms for r in ok_rows if r.server_total_ms is not None]
        lines.append(
            f"| {hr:02d} | {len(rows)} | {len(ok_rows)} "
            f"| {percentile(ttfrs,50):.0f} | {percentile(ttfrs,95):.0f} | {percentile(ttfrs,99):.0f} "
            f"| {percentile(srvs,95):.0f} |"
        )

    (out_dir / "summary.md").write_text("\n".join(lines))


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:8000",
                   help="Base endpoint; ws:// will be derived from this")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="smoke-omi")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="normal")
    p.add_argument("--users", type=int, default=500)
    p.add_argument("--audios-per-user-day", type=int, default=8,
                   help="Omi: chunks/user/day. Ignored for classic workloads.")
    p.add_argument("--real-duration-s", type=float, default=None,
                   help="Wall-clock test duration. Defaults to preset real_min * 60.")
    p.add_argument("--simulated-hours", type=float, default=None,
                   help="Simulated hours to span. Defaults to preset.")
    p.add_argument("--model", default="tdv1-fast")
    p.add_argument("--language", default="pt")
    p.add_argument("--diarize", action="store_true", default=True)
    p.add_argument("--no-diarize", dest="diarize", action="store_false")
    p.add_argument("--max-concurrency", type=int, default=3000,
                   help="Max concurrent WS sessions in flight at once")
    p.add_argument("--request-timeout-s", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--sessions-per-user-day-mean", type=float, default=8.0)
    p.add_argument("--in-session-interval-mean-s", type=float, default=10.0)
    p.add_argument("--chunks-per-session-min", type=int, default=5)
    p.add_argument("--chunks-per-session-max", type=int, default=30)
    args = p.parse_args()

    real_min, sim_hours, _workload = PRESETS[args.preset]
    real_duration_s = args.real_duration_s if args.real_duration_s is not None else real_min * 60.0
    simulated_hours = args.simulated_hours if args.simulated_hours is not None else float(sim_hours)
    compression = (simulated_hours * 3600.0) / real_duration_s

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_ws-stoch_{args.scenario}"
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "results" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "preset": args.preset,
        "scenario": args.scenario,
        "real_duration_s": real_duration_s,
        "simulated_hours": simulated_hours,
        "compression": compression,
        "users": args.users,
        "audios_per_user_day": args.audios_per_user_day,
        "endpoint": args.endpoint,
        "model": args.model,
        "language": args.language,
        "diarize": args.diarize,
        "max_concurrency": args.max_concurrency,
        "request_timeout_s": args.request_timeout_s,
        "seed": args.seed,
        "run_id": run_id,
        "sessions_per_user_day_mean": args.sessions_per_user_day_mean,
        "in_session_interval_mean_s": args.in_session_interval_mean_s,
        "chunks_per_session_min": args.chunks_per_session_min,
        "chunks_per_session_max": args.chunks_per_session_max,
    }

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[ws-stoch] run_id={run_id}")
    print(f"[ws-stoch] endpoint={args.endpoint}/ws/transcribe")
    print(f"[ws-stoch] users={args.users}  preset={args.preset}/{args.scenario}  "
          f"compression={compression:.1f}x  real={real_duration_s:.0f}s  sim={simulated_hours}h")
    print(f"[ws-stoch] output → {out_dir}")

    rng = random.Random(args.seed)
    t0 = time.monotonic()
    try:
        records = asyncio.run(run_test(cfg, rng, out_dir))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(1)
    elapsed = time.monotonic() - t0

    write_summary(cfg, records, out_dir)
    print(f"[ws-stoch] done in {elapsed:.1f}s → {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
