"""
Stochastic (Poisson) load test for TD Nooto transcription service.

Simulates a full day of usage compressed into a shorter wall-clock window.
Inter-arrival times are compressed by the preset factor; audio durations are
NOT compressed (a 30s clip stays 30s of real audio).

Default: smoke preset → 24 min real / 24 h simulated / 60× compression.

Outputs (under results/<run_id>/):
  - requests.ndjson     one line per request with wall_ts, sim_ts, latency, etc.
  - summary.md          tables A/B/D/E/H per spec
  - config.json         echo of the run parameters
  - charts/*.png        (if matplotlib available)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Install aiohttp: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "tests" / "corpus" / "pt-BR"
MANIFEST = REPO_ROOT / "tests" / "corpus" / "manifest.json"
RESULTS_ROOT = REPO_ROOT / "results"


# ---------- Presets ----------

PRESETS = {
    # name        : (real_min, sim_hours, workload)
    "smoke":        (24,       24,        "classic"),
    "smoke-omi":    (24,       24,        "omi"),
    "half-day":     (24,       12,        "classic"),
    "realistic":    (12 * 60,  24,        "classic"),
    "realtime":     (12 * 60,  12,        "classic"),
    "production":   (24 * 60,  24,        "classic"),
}


# Diurnal multipliers (hour-of-day → multiplier on λ_base).
# Bucket edges per spec, interpolated linearly at sample time.
DIURNAL_POINTS = [
    (0,  0.1),
    (6,  1.5),
    (9,  2.0),
    (12, 1.2),
    (14, 2.5),   # peak
    (18, 1.0),
    (22, 0.3),
    (24, 0.1),   # wrap
]


SCENARIOS = ("normal", "peak", "burst", "chaos", "soak")


# ---------- Configuration ----------

@dataclass
class RunConfig:
    preset: str
    scenario: str
    workload: str                   # "classic" | "omi"
    real_duration_s: float
    simulated_hours: float
    compression: float
    users: int
    audios_per_user_day: int        # classic: audios/user/day; omi: chunks/user/day
    endpoint: str
    model: str
    endpoint_path: str              # e.g. "/transcribe" or "/transcribe-fast"
    max_concurrency: int
    request_timeout_s: float
    seed: int
    run_id: str
    corpus_path: str
    extra_query: str = ""           # appended to URL (without leading &), e.g. "diarize=true"
    burst_chance: float = 0.05
    duration_clamp_min: float = 3.0
    duration_clamp_max: float = 600.0
    duration_mu: float = math.log(30.0)
    duration_sigma: float = 0.8
    # Omi workload only
    sessions_per_user_day_mean: float = 8.0   # ~6-12 sessions/user/day
    in_session_interval_mean_s: float = 10.0  # poisson within-session interval
    chunks_per_session_min: int = 5
    chunks_per_session_max: int = 30
    # Telemetry
    gpu_telemetry: bool = False


# ---------- Simulated clock ----------

class SimClock:
    """Maps wall-clock time to simulated time via compression factor.

    sim_seconds = (wall_now - wall_start) * compression
    """

    def __init__(self, compression: float, wall_start: float):
        self.compression = compression
        self.wall_start = wall_start

    def sim_seconds(self, wall_now: Optional[float] = None) -> float:
        if wall_now is None:
            wall_now = time.monotonic()
        return (wall_now - self.wall_start) * self.compression

    def sim_hour(self, wall_now: Optional[float] = None) -> float:
        return (self.sim_seconds(wall_now) / 3600.0) % 24.0


# ---------- Diurnal λ ----------

def diurnal_multiplier(hour: float) -> float:
    """Piecewise-linear interpolation between DIURNAL_POINTS at hour-of-day."""
    h = hour % 24.0
    for (h0, m0), (h1, m1) in zip(DIURNAL_POINTS, DIURNAL_POINTS[1:]):
        if h0 <= h <= h1:
            if h1 == h0:
                return m0
            t = (h - h0) / (h1 - h0)
            return m0 + t * (m1 - m0)
    return DIURNAL_POINTS[-1][1]


def compute_lambda_base(users: int, audios_per_user_day: int) -> float:
    """Solve for λ_base such that ∫₀²⁴ λ_base · diurnal(h) dh = users · audios."""
    total_audios = users * audios_per_user_day
    integral = 0.0
    steps = 240
    for i in range(steps):
        h = 24.0 * i / steps
        integral += diurnal_multiplier(h)
    integral *= 24.0 / steps
    return total_audios / integral


# ---------- Corpus ----------

@dataclass
class CorpusClip:
    path: Path
    duration_s: float


def load_corpus() -> list[CorpusClip]:
    if not MANIFEST.exists():
        print(f"ERROR: corpus manifest missing ({MANIFEST}). Run scripts/import_corpus.py first.", file=sys.stderr)
        sys.exit(2)
    data = json.loads(MANIFEST.read_text())
    clips = []
    for s in data["samples"]:
        p = REPO_ROOT / s["path"]
        if p.exists():
            clips.append(CorpusClip(path=p, duration_s=s["duration_s"]))
    if not clips:
        print("ERROR: no corpus clips found on disk", file=sys.stderr)
        sys.exit(2)
    return clips


async def build_clip_async(target_duration: float, corpus: list[CorpusClip], rng: random.Random) -> Path:
    """Async version: concatenate random corpus clips, trim to target duration.

    Uses asyncio.create_subprocess_exec so the event loop stays unblocked while
    ffmpeg runs. Returns path to a temp WAV file (caller should delete).
    """
    chosen: list[CorpusClip] = []
    total = 0.0
    while total < target_duration:
        c = rng.choice(corpus)
        chosen.append(c)
        total += c.duration_s

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    if len(chosen) == 1 and chosen[0].duration_s >= target_duration:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(chosen[0].path),
            "-t", f"{target_duration:.3f}",
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            str(tmp_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, "ffmpeg")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
            for c in chosen:
                lst.write(f"file '{c.path}'\n")
            lst_path = lst.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", lst_path,
                "-t", f"{target_duration:.3f}",
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                str(tmp_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, "ffmpeg")
        finally:
            Path(lst_path).unlink(missing_ok=True)

    return tmp_path


# ---------- Request ----------

@dataclass
class RequestRecord:
    wall_ts: str
    sim_ts: str
    sim_hour: int
    user_id: str
    audio_duration_s: float
    audio_language: str
    ttfr_ms: Optional[float] = None
    ttlt_ms: Optional[float] = None
    rtf: Optional[float] = None
    upload_ms: Optional[float] = None
    queue_wait_ms: Optional[float] = None
    processing_ms: Optional[float] = None
    gpu_routed: Optional[int] = None
    batch_size: Optional[int] = None
    status: str = "ok"
    error: Optional[str] = None
    is_burst: bool = False


async def send_request(
    session: aiohttp.ClientSession,
    cfg: RunConfig,
    clip_path: Path,
    record: RequestRecord,
    timeout: float,
) -> None:
    """Send clip to the configured transcription endpoint and populate record fields."""
    url = cfg.endpoint.rstrip("/") + cfg.endpoint_path + f"?model={cfg.model}"
    if cfg.extra_query:
        url = url + "&" + cfg.extra_query.lstrip("&")
    t_start = time.monotonic()
    try:
        with open(clip_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=clip_path.name, content_type="audio/wav")
            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                body = await resp.read()
                t_end = time.monotonic()
                total_ms = (t_end - t_start) * 1000.0
                record.ttlt_ms = total_ms
                record.ttfr_ms = total_ms  # non-streaming endpoint
                record.rtf = total_ms / 1000.0 / record.audio_duration_s if record.audio_duration_s > 0 else None
                if resp.status != 200:
                    record.status = f"http_{resp.status}"
                    record.error = body[:200].decode("utf-8", "replace")
    except asyncio.TimeoutError:
        record.status = "timeout"
        record.error = f"timeout after {timeout}s"
        record.ttlt_ms = timeout * 1000.0
    except aiohttp.ClientConnectorError as e:
        record.status = "conn_err"
        record.error = str(e)[:200]
    except aiohttp.ServerDisconnectedError as e:
        record.status = "conn_reset"
        record.error = str(e)[:200]
    except Exception as e:
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"[:200]


# ---------- GPU Telemetry ----------

GPU_TELEMETRY_SSH_ARGS = [
    "ssh",
    "-p", "15072",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=5",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=1",
    "root@ssh8.vast.ai",
    "nvidia-smi",
    "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu",
    "--format=csv,noheader,nounits",
]


async def gpu_telemetry_loop(
    cfg: RunConfig,
    clock: "SimClock",
    out_dir: Path,
    stop_event: asyncio.Event,
) -> None:
    """Sample remote nvidia-smi every 2s and append per-GPU rows to gpu_telemetry.csv.

    Best-effort: failed ssh samples are logged and skipped, loop continues until
    stop_event is set. Writes a CSV header only if the file does not exist.
    """
    csv_path = out_dir / "gpu_telemetry.csv"
    write_header = not csv_path.exists()
    try:
        csv_f = open(csv_path, "a", buffering=1)
    except Exception as e:
        print(f"[gpu-telemetry] could not open {csv_path}: {e}", file=sys.stderr)
        return
    if write_header:
        csv_f.write("wall_ts,sim_hour,gpu_index,util_pct,mem_used_mib,power_w,temp_c\n")

    interval = 2.0
    samples = 0
    failures = 0
    try:
        while not stop_event.is_set():
            tick_start = time.monotonic()
            wall_ts = datetime.now(timezone.utc).isoformat()
            sim_hour = int(clock.sim_hour() )

            try:
                proc = await asyncio.create_subprocess_exec(
                    *GPU_TELEMETRY_SSH_ARGS,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=8.0)
                except asyncio.TimeoutError:
                    proc.kill()
                    try:
                        await proc.wait()
                    except Exception:
                        pass
                    failures += 1
                    print(f"[gpu-telemetry] ssh sample timed out (#{failures})", file=sys.stderr)
                else:
                    if proc.returncode != 0:
                        failures += 1
                        err = stderr_b.decode("utf-8", "replace").strip()[:200]
                        print(f"[gpu-telemetry] ssh rc={proc.returncode} err={err}", file=sys.stderr)
                    else:
                        text = stdout_b.decode("utf-8", "replace")
                        rows_written = 0
                        for line in text.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) < 5:
                                continue
                            gpu_index, util_pct, mem_used_mib, power_w, temp_c = parts[:5]
                            csv_f.write(
                                f"{wall_ts},{sim_hour},{gpu_index},{util_pct},"
                                f"{mem_used_mib},{power_w},{temp_c}\n"
                            )
                            rows_written += 1
                        if rows_written > 0:
                            samples += 1
            except Exception as e:
                failures += 1
                print(f"[gpu-telemetry] sample error: {type(e).__name__}: {e}", file=sys.stderr)

            # Sleep the remainder of the 2s cadence, but wake up if stop is set.
            elapsed = time.monotonic() - tick_start
            remaining = max(0.0, interval - elapsed)
            if remaining > 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
    finally:
        try:
            csv_f.close()
        except Exception:
            pass
        print(f"[gpu-telemetry] stopped: {samples} samples ok, {failures} failures -> {csv_path}")


# ---------- Runner ----------

def _build_omi_schedule(cfg: RunConfig, rng: random.Random) -> list[tuple[float, str, bool]]:
    """Precompute Omi-shaped arrivals in SIMULATED seconds.

    For each user: pick N sessions distributed across the sim day with the
    diurnal curve as weighting, each session emits M chunks at Poisson(λ=1/interval).

    Returns a list of (sim_seconds_from_start, user_id, is_burst) sorted by time.
    """
    sim_total_s = cfg.simulated_hours * 3600.0

    # Precompute CDF of diurnal curve so we can sample session-start hours
    # proportionally to activity level. 240 buckets over 24h = 6-minute resolution.
    steps = 240
    weights = []
    for i in range(steps):
        h = 24.0 * i / steps
        weights.append(diurnal_multiplier(h))
    total_w = sum(weights)
    cdf = []
    running = 0.0
    for w in weights:
        running += w / total_w
        cdf.append(running)

    def sample_session_start_sim_s() -> float:
        u = rng.random()
        # binary search CDF
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        bucket_frac = lo / steps
        jitter = rng.random() / steps
        sim_h = 24.0 * (bucket_frac + jitter)
        return sim_h * 3600.0

    schedule: list[tuple[float, str, bool]] = []
    scenario_mult = 1.5 if cfg.scenario == "peak" else 1.0

    for u_idx in range(cfg.users):
        user_id = f"u_{u_idx:05d}"
        n_sessions = max(1, int(round(rng.gauss(cfg.sessions_per_user_day_mean, 2.0) * scenario_mult)))
        for _ in range(n_sessions):
            start_sim_s = sample_session_start_sim_s()
            n_chunks = rng.randint(cfg.chunks_per_session_min, cfg.chunks_per_session_max)
            rate_per_s = 1.0 / cfg.in_session_interval_mean_s
            t = start_sim_s
            for chunk_idx in range(n_chunks):
                # is_burst flag = "this chunk is part of a multi-chunk session"
                # First chunk isn't a burst; rest are.
                schedule.append((t, user_id, chunk_idx > 0))
                t += rng.expovariate(rate_per_s)
                if t >= sim_total_s:
                    break
    schedule.sort(key=lambda x: x[0])
    return schedule


async def run_test(cfg: RunConfig, rng: random.Random, out_dir: Path) -> list[RequestRecord]:
    corpus = load_corpus()
    print(f"Corpus: {len(corpus)} clips, total {sum(c.duration_s for c in corpus):.1f}s")

    wall_start = time.monotonic()
    wall_deadline = wall_start + cfg.real_duration_s
    clock = SimClock(cfg.compression, wall_start)

    ndjson_path = out_dir / "requests.ndjson"
    records: list[RequestRecord] = []
    pending: set[asyncio.Task] = set()
    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    last_progress = wall_start

    # Precompute Omi schedule up front (deterministic given seed).
    omi_schedule: list[tuple[float, str, bool]] = []
    omi_idx = 0
    if cfg.workload == "omi":
        omi_schedule = _build_omi_schedule(cfg, rng)
        print(f"Omi schedule: {len(omi_schedule)} chunks across {cfg.users} users "
              f"(~{len(omi_schedule)/cfg.users:.1f} chunks/user)")
    else:
        lambda_base = compute_lambda_base(cfg.users, cfg.audios_per_user_day)
        if cfg.scenario == "peak":
            lambda_base *= 1.5
        print(f"λ_base = {lambda_base:.3f} req/sim-hour "
              f"(target total: ~{cfg.users * cfg.audios_per_user_day} requests)")

    # Optional GPU telemetry: samples remote nvidia-smi via ssh every 2s.
    stop_event: Optional[asyncio.Event] = None
    telemetry_task: Optional[asyncio.Task] = None
    if cfg.gpu_telemetry:
        stop_event = asyncio.Event()
        telemetry_task = asyncio.create_task(
            gpu_telemetry_loop(cfg, clock, out_dir, stop_event)
        )
        print(f"GPU telemetry: enabled -> {out_dir / 'gpu_telemetry.csv'}")

    connector = aiohttp.TCPConnector(limit=cfg.max_concurrency, ssl=False)
    log_f = open(ndjson_path, "w")
    async with aiohttp.ClientSession(connector=connector) as session:

        # Cap total in-flight tasks (built clip + in semaphore + completed-but-unlogged)
        # to 2× max_concurrency. Beyond that, we shed load with status="runner_overflow"
        # — real-world backpressure, avoids OOM from 4k pending coroutines.
        overflow_cap = cfg.max_concurrency * 2

        async def send_one(user_id: str, is_burst: bool):
            dur = rng.lognormvariate(cfg.duration_mu, cfg.duration_sigma)
            dur = max(cfg.duration_clamp_min, min(cfg.duration_clamp_max, dur))

            wall_now = time.monotonic()
            sim_s = clock.sim_seconds(wall_now)
            sim_hour = int((sim_s / 3600.0) % 24.0)

            rec = RequestRecord(
                wall_ts=datetime.now(timezone.utc).isoformat(),
                sim_ts=datetime.fromtimestamp(sim_s, tz=timezone.utc).isoformat(),
                sim_hour=sim_hour,
                user_id=user_id,
                audio_duration_s=round(dur, 3),
                audio_language="pt-BR",
                is_burst=is_burst,
            )

            # Shed load if we're already past 2× capacity.
            if len(pending) > overflow_cap:
                rec.status = "runner_overflow"
                rec.error = f"pending={len(pending)} > cap={overflow_cap}"
                log_f.write(json.dumps(asdict(rec)) + "\n")
                log_f.flush()
                records.append(rec)
                return

            # Build the clip *inside* the semaphore. Outside the semaphore,
            # holding temp files and ffmpeg processes for 4k queued tasks
            # wastes disk + FDs for requests that may time out anyway.
            async with semaphore:
                try:
                    clip_path = await build_clip_async(dur, corpus, rng)
                except Exception as e:
                    rec.status = "clip_build_err"
                    rec.error = f"{type(e).__name__}: {e}"[:200]
                    log_f.write(json.dumps(asdict(rec)) + "\n")
                    log_f.flush()
                    records.append(rec)
                    return

                await send_request(session, cfg, clip_path, rec, cfg.request_timeout_s)

                try:
                    clip_path.unlink(missing_ok=True)
                except Exception:
                    pass

            log_f.write(json.dumps(asdict(rec)) + "\n")
            log_f.flush()
            records.append(rec)

        # -------- Classic Poisson loop --------
        if cfg.workload == "classic":
            while True:
                now = time.monotonic()
                if now >= wall_deadline:
                    break
                sim_h = clock.sim_hour(now)
                lam_sim_per_hour = lambda_base * diurnal_multiplier(sim_h)
                lam_real_per_s = (lam_sim_per_hour * cfg.compression) / 3600.0
                lam_real_per_s = max(lam_real_per_s, 1e-6)

                dt = rng.expovariate(lam_real_per_s)
                sleep_for = min(now + dt, wall_deadline) - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                now = time.monotonic()
                if now >= wall_deadline:
                    break

                is_burst = rng.random() < cfg.burst_chance
                user_id = f"u_{rng.randint(0, cfg.users - 1):05d}"
                task = asyncio.create_task(send_one(user_id, is_burst=False))
                pending.add(task)
                task.add_done_callback(pending.discard)

                if is_burst:
                    for _ in range(rng.randint(1, 3)):
                        btask = asyncio.create_task(send_one(user_id, is_burst=True))
                        pending.add(btask)
                        btask.add_done_callback(pending.discard)

                if now - last_progress >= 30.0:
                    elapsed = now - wall_start
                    pct = 100.0 * elapsed / cfg.real_duration_s
                    print(f"  [{elapsed:6.1f}s / {cfg.real_duration_s:.0f}s  {pct:5.1f}%]  "
                          f"sim_hour={sim_h:5.2f}  λ={lam_real_per_s:.3f}/s  "
                          f"inflight={len(pending):4d}  done={len(records):5d}")
                    last_progress = now

        # -------- Omi session-based loop --------
        else:
            while omi_idx < len(omi_schedule):
                now = time.monotonic()
                if now >= wall_deadline:
                    break

                # When in WALL time should the next scheduled SIM event fire?
                # wall_fire = wall_start + (sim_s_from_start / compression)
                target_sim_s, user_id, is_burst = omi_schedule[omi_idx]
                target_wall = wall_start + target_sim_s / cfg.compression
                sleep_for = min(target_wall, wall_deadline) - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

                now = time.monotonic()
                if now >= wall_deadline:
                    break

                task = asyncio.create_task(send_one(user_id, is_burst=is_burst))
                pending.add(task)
                task.add_done_callback(pending.discard)
                omi_idx += 1

                if now - last_progress >= 30.0:
                    elapsed = now - wall_start
                    pct = 100.0 * elapsed / cfg.real_duration_s
                    sim_h = clock.sim_hour(now)
                    remaining = len(omi_schedule) - omi_idx
                    print(f"  [{elapsed:6.1f}s / {cfg.real_duration_s:.0f}s  {pct:5.1f}%]  "
                          f"sim_hour={sim_h:5.2f}  scheduled={omi_idx}/{len(omi_schedule)}  "
                          f"inflight={len(pending):4d}  done={len(records):5d}  "
                          f"queued={remaining}")
                    last_progress = now

        print(f"\nArrival loop finished. Waiting for {len(pending)} in-flight requests...")
        if pending:
            await asyncio.wait(pending, timeout=cfg.request_timeout_s + 60)

    log_f.close()

    # Stop GPU telemetry loop (if running) after drain.
    if telemetry_task is not None and stop_event is not None:
        stop_event.set()
        try:
            await asyncio.wait_for(telemetry_task, timeout=15.0)
        except asyncio.TimeoutError:
            telemetry_task.cancel()
            try:
                await telemetry_task
            except (asyncio.CancelledError, Exception):
                pass
        except Exception as e:
            print(f"[gpu-telemetry] task exit error: {type(e).__name__}: {e}", file=sys.stderr)

    return records


# ---------- Reporting ----------

def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round(p * (len(vs) - 1)))))
    return vs[k]


def write_summary(cfg: RunConfig, records: list[RequestRecord], out_dir: Path) -> None:
    md = []
    md.append(f"# Stochastic Load Test — {cfg.preset} / {cfg.scenario}")
    md.append(f"\n**Run ID:** `{cfg.run_id}`")
    md.append(f"**Endpoint:** `{cfg.endpoint}`  **Model:** `{cfg.model}`")
    md.append(f"**Real duration:** {cfg.real_duration_s/60:.1f} min   **Simulated:** {cfg.simulated_hours:.1f} h   **Compression:** {cfg.compression:.1f}×")
    md.append(f"**Total requests:** {len(records)}")

    ok = [r for r in records if r.status == "ok"]
    success_rate = 100.0 * len(ok) / len(records) if records else 0.0
    md.append(f"**Success rate:** {success_rate:.2f}%  ({len(ok)}/{len(records)})")

    # Global latency
    ttfr = [r.ttfr_ms for r in ok if r.ttfr_ms is not None]
    rtfs = [r.rtf for r in ok if r.rtf is not None]
    if ttfr:
        md.append(f"**TTFR global:** p50={pct(ttfr,0.5)/1000:.2f}s  p95={pct(ttfr,0.95)/1000:.2f}s  p99={pct(ttfr,0.99)/1000:.2f}s")
    if rtfs:
        md.append(f"**RTF global:** p50={pct(rtfs,0.5):.3f}  p95={pct(rtfs,0.95):.3f}  max={max(rtfs):.3f}")

    # Table A: latency by simulated hour
    md.append("\n## A) Latency by simulated hour\n")
    md.append("| Sim Hour | Req | TTFR p50 (s) | TTFR p95 (s) | TTFR p99 (s) | RTF p95 | Success % |")
    md.append("|----------|-----|--------------|--------------|--------------|---------|-----------|")
    for h in range(24):
        bucket = [r for r in records if r.sim_hour == h]
        b_ok = [r for r in bucket if r.status == "ok"]
        b_ttfr = [r.ttfr_ms for r in b_ok if r.ttfr_ms is not None]
        b_rtf = [r.rtf for r in b_ok if r.rtf is not None]
        if not bucket:
            continue
        md.append(f"| {h:02d} | {len(bucket)} | "
                  f"{pct(b_ttfr,0.5)/1000:.2f} | {pct(b_ttfr,0.95)/1000:.2f} | {pct(b_ttfr,0.99)/1000:.2f} | "
                  f"{pct(b_rtf,0.95):.3f} | "
                  f"{100*len(b_ok)/len(bucket):.1f}% |")

    # Table B: stochastic validation
    md.append("\n## B) Stochastic validation\n")
    expected = cfg.users * cfg.audios_per_user_day
    # Compute coefficient of variation of inter-arrival times
    # Reconstruct from wall_ts order
    times_sorted = sorted(datetime.fromisoformat(r.wall_ts.replace("Z","+00:00")).timestamp() for r in records)
    if len(times_sorted) > 2:
        deltas = [times_sorted[i+1] - times_sorted[i] for i in range(len(times_sorted)-1)]
        mean = statistics.mean(deltas)
        stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        cv = stdev / mean if mean > 0 else 0.0
    else:
        cv = 0.0
    burst_count = sum(1 for r in records if r.is_burst)
    burst_pct = 100.0 * burst_count / len(records) if records else 0.0
    md.append("| Metric | Expected | Measured | Status |")
    md.append("|---|---|---|---|")
    md.append(f"| Total requests | ~{expected} | {len(records)} | {'PASS' if abs(len(records)-expected) < expected*0.3 else 'WARN'} |")
    md.append(f"| CV of inter-arrivals | ~1.0 | {cv:.3f} | {'PASS' if 0.7 < cv < 1.4 else 'WARN'} |")
    md.append(f"| Burst fraction | ~5% | {burst_pct:.2f}% | {'PASS' if 3 < burst_pct < 8 else 'WARN'} |")

    # Table D: failures by type
    md.append("\n## D) Failures by type × sim hour\n")
    fail_types = sorted({r.status for r in records if r.status != "ok"})
    if fail_types:
        md.append("| Sim Hour | " + " | ".join(fail_types) + " | Total |")
        md.append("|----------|" + "|".join(["---"] * (len(fail_types) + 1)) + "|")
        for h in range(24):
            bucket = [r for r in records if r.sim_hour == h and r.status != "ok"]
            if not bucket:
                continue
            cols = []
            for t in fail_types:
                cols.append(str(sum(1 for r in bucket if r.status == t)))
            md.append(f"| {h:02d} | " + " | ".join(cols) + f" | {len(bucket)} |")
    else:
        md.append("_No failures._")

    # Executive summary (H)
    md.append("\n## H) Executive summary\n")
    peak_hours = [r for r in ok if 14 <= r.sim_hour < 18]
    peak_ttfr = [r.ttfr_ms for r in peak_hours if r.ttfr_ms is not None]
    peak_rtf = [r.rtf for r in peak_hours if r.rtf is not None]

    verdict_parts = []
    if ttfr:
        p95_s = pct(ttfr, 0.95) / 1000
        if p95_s < 15:
            verdict_parts.append(f"TTFR p95 {p95_s:.1f}s < 15s SLO PASS")
        elif p95_s < 30:
            verdict_parts.append(f"TTFR p95 {p95_s:.1f}s DEGRADED")
        else:
            verdict_parts.append(f"TTFR p95 {p95_s:.1f}s > 30s FAIL")
    if success_rate >= 99.9:
        verdict_parts.append(f"success {success_rate:.2f}% PASS")
    elif success_rate >= 99.0:
        verdict_parts.append(f"success {success_rate:.2f}% WARN")
    else:
        verdict_parts.append(f"success {success_rate:.2f}% FAIL")

    overall = "PASS" if all("PASS" in v for v in verdict_parts) else ("DEGRADED" if any("FAIL" in v for v in verdict_parts) is False else "FAIL")
    md.append(f"**Verdict:** {overall}  \n" + "  \n".join(verdict_parts))
    if peak_ttfr:
        md.append(f"\n**Peak (14-18h):** TTFR p95 = {pct(peak_ttfr,0.95)/1000:.2f}s, RTF p95 = {pct(peak_rtf,0.95):.3f}")

    (out_dir / "summary.md").write_text("\n".join(md))


def write_charts(records: list[RequestRecord], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping charts (pip install matplotlib)")
        return

    charts = out_dir / "charts"
    charts.mkdir(exist_ok=True)

    # Latency vs sim hour
    ok = [r for r in records if r.status == "ok" and r.ttfr_ms is not None]
    if ok:
        by_hour = {h: [] for h in range(24)}
        for r in ok:
            by_hour[r.sim_hour].append(r.ttfr_ms / 1000)
        hours = [h for h in range(24) if by_hour[h]]
        p50s = [pct(by_hour[h], 0.5) for h in hours]
        p95s = [pct(by_hour[h], 0.95) for h in hours]
        p99s = [pct(by_hour[h], 0.99) for h in hours]

        plt.figure(figsize=(10, 5))
        plt.plot(hours, p50s, label="p50", marker="o")
        plt.plot(hours, p95s, label="p95", marker="s")
        plt.plot(hours, p99s, label="p99", marker="^")
        plt.axhline(15, linestyle="--", color="red", label="SLO 15s")
        plt.xlabel("Simulated hour")
        plt.ylabel("TTFR (s)")
        plt.title("Latency vs simulated hour")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(charts / "latency_vs_sim_hour.png", dpi=120)
        plt.close()

    # Inter-arrival histogram
    times_sorted = sorted(datetime.fromisoformat(r.wall_ts.replace("Z","+00:00")).timestamp() for r in records)
    if len(times_sorted) > 10:
        deltas = [times_sorted[i+1] - times_sorted[i] for i in range(len(times_sorted)-1)]
        plt.figure(figsize=(10, 5))
        plt.hist(deltas, bins=50, density=True, alpha=0.7, label="observed")
        if deltas:
            mean_d = statistics.mean(deltas)
            xs = [i * mean_d * 4 / 100 for i in range(100)]
            ys = [(1/mean_d) * math.exp(-x/mean_d) for x in xs]
            plt.plot(xs, ys, "r--", label=f"exp(λ=1/{mean_d:.3f})")
        plt.xlabel("Inter-arrival (s, wall-clock)")
        plt.ylabel("Density")
        plt.title("Inter-arrival distribution")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(charts / "inter_arrival_histogram.png", dpi=120)
        plt.close()

    # Audio duration histogram
    durs = [r.audio_duration_s for r in records]
    if durs:
        plt.figure(figsize=(10, 5))
        plt.hist(durs, bins=50, density=True, alpha=0.7, label="observed")
        plt.xlabel("Audio duration (s)")
        plt.ylabel("Density")
        plt.title("Sampled audio durations (log-normal target)")
        plt.xscale("log")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(charts / "audio_duration_histogram.png", dpi=120)
        plt.close()


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stochastic load test for TD Nooto")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="smoke")
    p.add_argument("--scenario", choices=SCENARIOS, default="normal")
    p.add_argument("--real-duration-min", type=float, default=None,
                   help="Override preset real duration (minutes)")
    p.add_argument("--simulated-hours", type=float, default=None,
                   help="Override preset simulated hours")
    p.add_argument("--users", type=int, default=500)
    p.add_argument("--audios-per-user-dia", dest="audios_per_user_day", type=int, default=8,
                   help="classic preset: audios/user/day. omi preset: not used (chunks derived from sessions).")
    p.add_argument("--endpoint", default="http://96.38.133.243:22961")
    p.add_argument("--endpoint-path", default=None,
                   help="Transcription path. Default: /transcribe-fast for omi workload, /transcribe otherwise.")
    p.add_argument("--extra-query", default="",
                   help="Extra query params appended to each request (no leading &). e.g. 'diarize=true&diarize_min_seconds=10'")
    p.add_argument("--model", default="tdv1-fast")
    p.add_argument("--max-concurrency", type=int, default=500)
    p.add_argument("--request-timeout-s", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=42)

    # Omi-specific (ignored for classic workloads)
    p.add_argument("--omi-sessions-per-user-day", dest="omi_sessions_per_user_day", type=float, default=8.0,
                   help="[omi] Mean sessions per user per simulated day (default 8)")
    p.add_argument("--omi-interval-s", dest="omi_interval_s", type=float, default=10.0,
                   help="[omi] Mean inter-chunk interval within a session (sim seconds, default 10)")
    p.add_argument("--omi-chunks-min", dest="omi_chunks_min", type=int, default=5,
                   help="[omi] Min chunks per session (default 5)")
    p.add_argument("--omi-chunks-max", dest="omi_chunks_max", type=int, default=30,
                   help="[omi] Max chunks per session (default 30)")
    p.add_argument("--gpu-telemetry", dest="gpu_telemetry", action="store_true", default=False,
                   help="Sample remote nvidia-smi every 2s via ssh and write gpu_telemetry.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    real_min, sim_hours, workload = PRESETS[args.preset]
    if args.real_duration_min is not None:
        real_min = args.real_duration_min
    if args.simulated_hours is not None:
        sim_hours = args.simulated_hours

    real_duration_s = real_min * 60.0
    compression = (sim_hours * 60.0) / real_min

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.preset + "_" + args.scenario
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Workload-specific defaults for duration distribution
    if workload == "omi":
        duration_mu = math.log(7.0)
        duration_sigma = 0.6
        duration_clamp_min = 2.0
        duration_clamp_max = 60.0
    else:
        duration_mu = math.log(30.0)
        duration_sigma = 0.8
        duration_clamp_min = 3.0
        duration_clamp_max = 600.0

    # Default endpoint path: /transcribe-fast for Omi, /transcribe for classic.
    endpoint_path = args.endpoint_path
    if endpoint_path is None:
        endpoint_path = "/transcribe-fast" if workload == "omi" else "/transcribe"

    cfg = RunConfig(
        preset=args.preset,
        scenario=args.scenario,
        workload=workload,
        real_duration_s=real_duration_s,
        simulated_hours=sim_hours,
        compression=compression,
        users=args.users,
        audios_per_user_day=args.audios_per_user_day,
        endpoint=args.endpoint,
        endpoint_path=endpoint_path,
        extra_query=args.extra_query,
        model=args.model,
        max_concurrency=args.max_concurrency,
        request_timeout_s=args.request_timeout_s,
        seed=args.seed,
        run_id=run_id,
        corpus_path=str(CORPUS_DIR),
        duration_mu=duration_mu,
        duration_sigma=duration_sigma,
        duration_clamp_min=duration_clamp_min,
        duration_clamp_max=duration_clamp_max,
        sessions_per_user_day_mean=args.omi_sessions_per_user_day,
        in_session_interval_mean_s=args.omi_interval_s,
        chunks_per_session_min=args.omi_chunks_min,
        chunks_per_session_max=args.omi_chunks_max,
        gpu_telemetry=args.gpu_telemetry,
    )

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print(f"=== {cfg.preset} / {cfg.scenario}  (workload={workload}) ===")
    print(f"Real: {real_min} min   Simulated: {sim_hours} h   Compression: {compression:.1f}×")
    if workload == "omi":
        est_chunks = cfg.users * cfg.sessions_per_user_day_mean * (cfg.chunks_per_session_min + cfg.chunks_per_session_max) / 2
        print(f"Users: {cfg.users} × {cfg.sessions_per_user_day_mean:.1f} sessions × "
              f"{cfg.chunks_per_session_min}-{cfg.chunks_per_session_max} chunks "
              f"= ~{int(est_chunks)} total requests")
        print(f"Duration: log-normal(μ=ln({math.exp(duration_mu):.1f}), σ={duration_sigma}) "
              f"clamp [{duration_clamp_min:.0f}s, {duration_clamp_max:.0f}s]")
    else:
        print(f"Users: {cfg.users} × {cfg.audios_per_user_day} audios/day = ~{cfg.users * cfg.audios_per_user_day} total requests")
    print(f"Endpoint: {cfg.endpoint}{cfg.endpoint_path}   Model: {cfg.model}")
    print(f"Results: {out_dir}\n")

    rng = random.Random(args.seed)
    try:
        records = asyncio.run(run_test(cfg, rng, out_dir))
    except KeyboardInterrupt:
        print("\nInterrupted — writing partial results")
        records = []
        # Try to parse whatever we already logged
        ndjson = out_dir / "requests.ndjson"
        if ndjson.exists():
            for line in ndjson.read_text().splitlines():
                try:
                    d = json.loads(line)
                    records.append(RequestRecord(**d))
                except Exception:
                    pass

    print(f"\nCompleted: {len(records)} requests logged")
    write_summary(cfg, records, out_dir)
    write_charts(records, out_dir)
    print(f"\nSummary: {out_dir}/summary.md")
    print(f"Raw log: {out_dir}/requests.ndjson")
    return 0


if __name__ == "__main__":
    sys.exit(main())
