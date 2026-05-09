"""
WebSocket stress test for TDVX /ws/transcribe.

Mirrors the stochastic/Omi HTTP stress test (tests/stochastic_test.py,
tests/stress_test.py) but targets the LIVE-streaming WebSocket path so we
can measure how the live pipeline (per-session diarization registry,
VAD-bounded bursts, whisper-s2t batcher) behaves under N concurrent
long-running sessions.

Each simulated user:
  - opens a WebSocket to /ws/transcribe?model=...&language=...&session_id=...&diarize=...
  - streams a random corpus WAV at REAL-TIME pace, looping when it runs out
  - feeds the stream through a Python port of web/src/lib/vadChunker.ts so
    the server sees VAD-bounded bursts that look like a browser client
  - records per-chunk telemetry (whisper_ms / diarization_ms / total_ms,
    registry_size, speakers_in_buffer, worker tag, client-observed RTT)

Outputs under results/<RUN_ID>_ws-stress/:
  - config.json
  - requests.ndjson  (one line per telemetry event or error)
  - summary.md       (throughput, latency percentiles, label stability,
                      per-worker fan-out, PASS/FAIL verdict)
  - charts/*.png     (optional, only if matplotlib is available)

Example:
  python tests/ws_stress_test.py \
      --endpoint ws://96.38.133.243:22961 \
      --users 50 --duration-s 120 --model tdv1-fast --language pt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import soundfile as sf
except ImportError:
    print("soundfile missing — pip install soundfile", file=sys.stderr)
    sys.exit(1)

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed,
        ConnectionClosedError,
        ConnectionClosedOK,
        InvalidStatusCode,
    )
except ImportError:
    print("websockets missing — pip install 'websockets>=11'", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Python port of web/src/lib/vadChunker.ts
# ---------------------------------------------------------------------------
#
# Faithfulness notes (vs the TS implementation):
#   - Same 20 ms analysis frames, same RMS-in-dBFS feature.
#   - Same adaptive noise-floor EMA (only updates during silent frames).
#   - Same hangover logic (silent frames while voice active stay in buffer,
#     silence-run counts against silenceFramesRequired).
#   - Same min/max chunk gates and "cap" trigger regardless of voice state.
#   - Carry-over behaviour (tail < one frame) preserved.
# Differences:
#   - No JS-style per-frame copy into arrays — we append numpy views and
#     concat only once at emit-time. Semantics match; allocation profile differs.
#   - We use float32 numpy for the RMS calculation (fastest in pure Python).


class VadChunker:
    def __init__(
        self,
        sample_rate: int = 16000,
        min_chunk_ms: int = 1500,
        max_chunk_ms: int = 6000,
        silence_ms: int = 400,
        frame_ms: int = 20,
        energy_threshold_db: float = 8.0,
        initial_noise_floor_db: float = -55.0,
        noise_floor_alpha: float = 0.02,
    ) -> None:
        import numpy as np  # local to keep import surface small

        self.np = np
        self.frame_samples = max(1, round(frame_ms * sample_rate / 1000))
        self.min_chunk_samples = round(min_chunk_ms * sample_rate / 1000)
        self.max_chunk_samples = round(max_chunk_ms * sample_rate / 1000)
        self.silence_frames_required = max(1, round(silence_ms / frame_ms))
        self.energy_threshold_db = energy_threshold_db
        self.noise_floor_alpha = noise_floor_alpha
        self.noise_floor_db = initial_noise_floor_db

        self._carry: "np.ndarray" = np.zeros(0, dtype=np.float32)
        self._buffered: list = []
        self._buffered_samples = 0
        self._voice_active = False
        self._silent_run = 0

    # --- public API --------------------------------------------------------

    def feed(self, samples) -> list:
        """Push new float32 samples; return list of (buffer, reason) tuples
        for any chunks that fired while consuming this input."""
        np = self.np
        emitted: list = []
        if len(samples) == 0:
            return emitted
        if self._carry.size == 0:
            work = samples
        else:
            work = np.concatenate([self._carry, samples])

        total = work.shape[0]
        i = 0
        fs = self.frame_samples
        while i + fs <= total:
            frame = work[i : i + fs]
            out = self._process_frame(frame)
            if out is not None:
                emitted.append(out)
            i += fs
        self._carry = work[i:].copy() if i < total else np.zeros(0, dtype=np.float32)
        return emitted

    def flush_now(self, reason: str = "stop"):
        np = self.np
        if self._carry.size > 0:
            self._append(self._carry)
            self._carry = np.zeros(0, dtype=np.float32)
        if self._buffered_samples == 0:
            return None
        return self._emit(reason)

    # --- internals ---------------------------------------------------------

    def _process_frame(self, frame):
        db = _rms_db(frame)
        threshold = self.noise_floor_db + self.energy_threshold_db
        is_voice = db > threshold

        emitted = None
        if is_voice:
            self._voice_active = True
            self._silent_run = 0
            self._append(frame)
        else:
            if math.isfinite(db):
                self.noise_floor_db = (
                    (1 - self.noise_floor_alpha) * self.noise_floor_db
                    + self.noise_floor_alpha * db
                )
                if self.noise_floor_db < -90:
                    self.noise_floor_db = -90
                elif self.noise_floor_db > -20:
                    self.noise_floor_db = -20
            if self._voice_active:
                self._silent_run += 1
                self._append(frame)
                if self._silent_run >= self.silence_frames_required:
                    if self._buffered_samples >= self.min_chunk_samples:
                        emitted = self._emit("vad")
                    self._voice_active = False
                    self._silent_run = 0

        # Cap trigger regardless of voice state.
        if self._buffered_samples >= self.max_chunk_samples:
            cap = self._emit("cap")
            self._silent_run = 0
            if emitted is None:
                emitted = cap
            else:
                # Pathological: both fired this frame. Very rare; return the
                # later one (cap) and drop the earlier — matches TS behavior
                # where the cap check runs after vad and overrides emit state.
                emitted = cap
        return emitted

    def _append(self, frame) -> None:
        # frame is a view; copy to detach from the sliding `work` buffer.
        self._buffered.append(frame.copy())
        self._buffered_samples += frame.shape[0]

    def _emit(self, reason: str):
        np = self.np
        if self._buffered_samples == 0:
            return None
        merged = np.concatenate(self._buffered) if len(self._buffered) > 1 else self._buffered[0].copy()
        self._buffered = []
        self._buffered_samples = 0
        return (merged, reason)


def _rms_db(frame) -> float:
    # frame: np.ndarray float32
    import numpy as np

    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


@dataclass
class Record:
    wall_ts: str
    user_id: str
    session_id: str
    kind: str  # "chunk" | "error" | "connect" | "close"
    speaker: Optional[str] = None
    latency_ms: Optional[float] = None
    whisper_ms: Optional[int] = None
    diarization_ms: Optional[int] = None
    total_ms: Optional[int] = None
    locals_detected: Optional[int] = None
    speakers_in_buffer: Optional[int] = None
    registry_size: Optional[int] = None
    buffer_seconds: Optional[float] = None
    worker: Optional[str] = None
    flush_reason: Optional[str] = None
    burst_samples: Optional[int] = None
    error_type: Optional[str] = None
    error: Optional[str] = None


@dataclass
class UserStats:
    chunks: int = 0
    errors: int = 0
    speakers_seen: set = field(default_factory=set)
    registry_sizes: list = field(default_factory=list)
    latencies_ms: list = field(default_factory=list)


def _wall_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_ws_endpoint(endpoint: str) -> str:
    ep = endpoint.strip().rstrip("/")
    if ep.startswith("http://"):
        ep = "ws://" + ep[len("http://") :]
    elif ep.startswith("https://"):
        ep = "wss://" + ep[len("https://") :]
    elif not (ep.startswith("ws://") or ep.startswith("wss://")):
        ep = "ws://" + ep
    return ep


def _pct(values, p):
    if not values:
        return None
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round(p * (len(vs) - 1)))))
    return float(vs[k])


def _load_corpus(corpus_path: Path) -> list[Path]:
    if not corpus_path.exists():
        return []
    wavs = sorted(p for p in corpus_path.rglob("*.wav") if p.is_file())
    return wavs


def _load_wav_mono_16k(path: Path):
    import numpy as np

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")
    if sr != 16000:
        # Cheap linear resample — avoids pulling librosa into the hot path.
        target_len = int(round(len(audio) * 16000 / sr))
        xp = np.linspace(0.0, 1.0, num=len(audio), dtype=np.float64)
        x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
        audio = np.interp(x, xp, audio).astype("float32")
    return audio


class NdjsonWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._fh = open(path, "w", buffering=1)

    async def write(self, rec: Record) -> None:
        line = json.dumps({k: _json_default(v) for k, v in asdict(rec).items()})
        async with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _json_default(v):
    if isinstance(v, set):
        return sorted(v)
    return v


async def run_user(
    user_idx: int,
    ws_url_base: str,
    query_params: dict,
    audio: "np.ndarray",  # noqa: F821  (forward ref; numpy imported lazily)
    vad_cfg: dict,
    writer: NdjsonWriter,
    stop_event: asyncio.Event,
    stats: UserStats,
    reconnect: bool,
) -> None:
    import numpy as np

    user_id = f"u_{user_idx:04d}"
    # 4096 samples @16 kHz → 256 ms, matches the browser ScriptProcessor frame.
    frame_size = 4096
    frame_period_s = frame_size / 16000.0

    while not stop_event.is_set():
        session_id = uuid.uuid4().hex
        params = {**query_params, "session_id": session_id}
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{ws_url_base}/ws/transcribe?{qs}"

        chunker = VadChunker(
            sample_rate=16000,
            min_chunk_ms=vad_cfg["min_ms"],
            max_chunk_ms=vad_cfg["max_ms"],
            silence_ms=vad_cfg["silence_ms"],
        )

        # FIFO of (send_time, reason, n_samples) for each burst we've sent but
        # haven't seen telemetry for yet. Each telemetry envelope consumes one.
        pending: list[tuple[float, str, int]] = []

        try:
            connect_t = time.perf_counter()
            async with websockets.connect(
                url,
                open_timeout=15,
                close_timeout=5,
                max_size=8 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                await writer.write(
                    Record(
                        wall_ts=_wall_ts(),
                        user_id=user_id,
                        session_id=session_id,
                        kind="connect",
                        total_ms=int((time.perf_counter() - connect_t) * 1000),
                    )
                )

                async def receiver():
                    try:
                        while not stop_event.is_set():
                            raw = await ws.recv()
                            recv_t = time.perf_counter()
                            if isinstance(raw, (bytes, bytearray)):
                                continue
                            try:
                                data = json.loads(raw)
                            except Exception as e:
                                await writer.write(Record(
                                    wall_ts=_wall_ts(),
                                    user_id=user_id,
                                    session_id=session_id,
                                    kind="error",
                                    error_type="parse",
                                    error=str(e)[:200],
                                ))
                                stats.errors += 1
                                continue

                            if isinstance(data, dict) and data.get("error"):
                                await writer.write(Record(
                                    wall_ts=_wall_ts(),
                                    user_id=user_id,
                                    session_id=session_id,
                                    kind="error",
                                    error_type="server_error",
                                    error=str(data.get("error"))[:300],
                                ))
                                stats.errors += 1
                                continue

                            tel = data.get("telemetry") if isinstance(data, dict) else None
                            seg = data.get("segment") if isinstance(data, dict) else None
                            speaker = None
                            if isinstance(seg, dict):
                                speaker = seg.get("speaker")

                            # Match against the oldest pending burst if we have
                            # telemetry; otherwise this is a non-first segment
                            # from the same burst (telemetry attaches to first).
                            lat_ms = None
                            flush_reason = None
                            burst_samples = None
                            if tel is not None and pending:
                                send_t, flush_reason, burst_samples = pending.pop(0)
                                lat_ms = (recv_t - send_t) * 1000.0

                            if tel is not None:
                                stats.chunks += 1
                                if speaker:
                                    stats.speakers_seen.add(speaker)
                                rs = tel.get("registry_size")
                                if isinstance(rs, int):
                                    stats.registry_sizes.append(rs)
                                if lat_ms is not None:
                                    stats.latencies_ms.append(lat_ms)
                                await writer.write(Record(
                                    wall_ts=_wall_ts(),
                                    user_id=user_id,
                                    session_id=session_id,
                                    kind="chunk",
                                    speaker=speaker,
                                    latency_ms=lat_ms,
                                    whisper_ms=tel.get("whisper_ms"),
                                    diarization_ms=tel.get("diarization_ms"),
                                    total_ms=tel.get("total_ms"),
                                    locals_detected=tel.get("locals_detected"),
                                    speakers_in_buffer=tel.get("speakers_in_buffer"),
                                    registry_size=tel.get("registry_size"),
                                    buffer_seconds=tel.get("buffer_seconds"),
                                    worker=tel.get("worker"),
                                    flush_reason=flush_reason,
                                    burst_samples=burst_samples,
                                ))
                            else:
                                # Non-first segment of a multi-segment burst.
                                if speaker:
                                    stats.speakers_seen.add(speaker)
                    except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
                        return
                    except Exception as e:
                        await writer.write(Record(
                            wall_ts=_wall_ts(),
                            user_id=user_id,
                            session_id=session_id,
                            kind="error",
                            error_type="recv",
                            error=str(e)[:200],
                        ))
                        stats.errors += 1

                recv_task = asyncio.create_task(receiver())

                # Stream at real-time pace, looping the WAV forever.
                cursor = 0
                n = audio.shape[0]
                next_tick = time.perf_counter()
                try:
                    while not stop_event.is_set():
                        if cursor >= n:
                            cursor = 0  # loop
                        end = min(cursor + frame_size, n)
                        frame = audio[cursor:end]
                        cursor = end

                        bursts = chunker.feed(frame)
                        for buf, reason in bursts:
                            send_t = time.perf_counter()
                            pending.append((send_t, reason, int(buf.shape[0])))
                            try:
                                await ws.send(buf.astype("float32").tobytes())
                            except Exception as e:
                                await writer.write(Record(
                                    wall_ts=_wall_ts(),
                                    user_id=user_id,
                                    session_id=session_id,
                                    kind="error",
                                    error_type="send",
                                    error=str(e)[:200],
                                ))
                                stats.errors += 1
                                raise

                        # Sleep until the next real-time frame boundary.
                        next_tick += frame_period_s
                        delay = next_tick - time.perf_counter()
                        if delay > 0:
                            try:
                                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                                break  # stop fired
                            except asyncio.TimeoutError:
                                pass
                        else:
                            # Falling behind — don't let next_tick drift.
                            if delay < -1.0:
                                next_tick = time.perf_counter()

                    # Normal stop: drain chunker, send end marker.
                    tail = chunker.flush_now("stop")
                    if tail is not None:
                        buf, reason = tail
                        send_t = time.perf_counter()
                        pending.append((send_t, reason, int(buf.shape[0])))
                        try:
                            await ws.send(buf.astype("float32").tobytes())
                        except Exception:
                            pass
                    try:
                        await ws.send("end")
                    except Exception:
                        pass
                finally:
                    # Give the receiver a moment to drain any final telemetry.
                    try:
                        await asyncio.wait_for(recv_task, timeout=10.0)
                    except asyncio.TimeoutError:
                        recv_task.cancel()
                    except Exception:
                        pass

                await writer.write(Record(
                    wall_ts=_wall_ts(),
                    user_id=user_id,
                    session_id=session_id,
                    kind="close",
                ))
                return  # normal exit — don't reconnect

        except (InvalidStatusCode, OSError, asyncio.TimeoutError, ConnectionClosedError) as e:
            await writer.write(Record(
                wall_ts=_wall_ts(),
                user_id=user_id,
                session_id=session_id,
                kind="error",
                error_type="connect",
                error=str(e)[:200],
            ))
            stats.errors += 1
        except Exception as e:
            await writer.write(Record(
                wall_ts=_wall_ts(),
                user_id=user_id,
                session_id=session_id,
                kind="error",
                error_type="session",
                error=str(e)[:200],
            ))
            stats.errors += 1

        if not reconnect or stop_event.is_set():
            return
        # Back off a little before reconnecting so we don't hammer a dead server.
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
            return
        except asyncio.TimeoutError:
            pass


def _find_default_corpus(repo_root: Path) -> Path:
    pt = repo_root / "tests" / "corpus" / "pt-BR"
    if pt.exists():
        return pt
    return repo_root / "tests" / "corpus"


def _write_config(out_dir: Path, args, ws_url_base: str, corpus_files: list[Path]) -> None:
    cfg = {
        "run_id": out_dir.name,
        "endpoint": ws_url_base,
        "users": args.users,
        "duration_s": args.duration_s,
        "model": args.model,
        "language": args.language,
        "corpus_path": str(args.corpus_path),
        "corpus_files": [str(p) for p in corpus_files[:50]],
        "corpus_total": len(corpus_files),
        "diarize": bool(args.diarize),
        "translate": bool(args.translate),
        "seed": args.seed,
        "ramp_s": args.ramp_s,
        "vad_min_ms": args.vad_min_ms,
        "vad_max_ms": args.vad_max_ms,
        "vad_silence_ms": args.vad_silence_ms,
        "reconnect": bool(args.reconnect),
        "speaker_count_expected": args.speaker_count_expected,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _analyze_and_write_summary(
    out_dir: Path,
    args,
    wall_time_s: float,
    ws_url_base: str,
) -> None:
    """Second pass: re-read NDJSON and produce summary.md + charts."""

    ndjson_path = out_dir / "requests.ndjson"
    chunks: list[dict] = []
    errors: list[dict] = []
    per_session_speakers: dict[str, set] = defaultdict(set)
    per_session_registry_max: dict[str, int] = defaultdict(int)
    per_session_registry_end: dict[str, int] = {}
    per_session_registry_monotonic: dict[str, bool] = {}
    per_session_registry_prev: dict[str, int] = {}
    worker_counts: Counter = Counter()
    error_counts: Counter = Counter()

    with open(ndjson_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            kind = r.get("kind")
            if kind == "chunk":
                chunks.append(r)
                sid = r.get("session_id") or "-"
                sp = r.get("speaker")
                if sp:
                    per_session_speakers[sid].add(sp)
                rs = r.get("registry_size")
                if isinstance(rs, int):
                    per_session_registry_max[sid] = max(per_session_registry_max[sid], rs)
                    per_session_registry_end[sid] = rs
                    prev = per_session_registry_prev.get(sid)
                    if prev is not None and rs < prev:
                        per_session_registry_monotonic[sid] = False
                    else:
                        per_session_registry_monotonic.setdefault(sid, True)
                    per_session_registry_prev[sid] = rs
                w = r.get("worker")
                if w:
                    worker_counts[w] += 1
                else:
                    worker_counts["(unset)"] += 1
            elif kind == "error":
                errors.append(r)
                error_counts[r.get("error_type") or "unknown"] += 1

    n_chunks = len(chunks)
    latencies = [c["latency_ms"] for c in chunks if isinstance(c.get("latency_ms"), (int, float))]
    whisper = [c["whisper_ms"] for c in chunks if isinstance(c.get("whisper_ms"), (int, float))]
    diar = [c["diarization_ms"] for c in chunks if isinstance(c.get("diarization_ms"), (int, float))]
    total = [c["total_ms"] for c in chunks if isinstance(c.get("total_ms"), (int, float))]

    throughput = n_chunks / wall_time_s if wall_time_s > 0 else 0.0

    p95_lat = _pct(latencies, 0.95) or 0.0
    err_rate = (len(errors) / max(1, n_chunks + len(errors))) if (n_chunks + len(errors)) else 0.0
    verdict = "PASS" if (p95_lat < 3000 and err_rate < 0.05) else "FAIL"

    # Label-stability scan: flag sessions where registry_size grows past
    # speaker_count_expected and keeps climbing.
    expected = args.speaker_count_expected
    unstable_sessions = []
    for sid, mx in per_session_registry_max.items():
        if mx > expected and per_session_registry_monotonic.get(sid, True):
            unstable_sessions.append((sid, mx, per_session_registry_end.get(sid, mx)))
    unstable_sessions.sort(key=lambda x: -x[1])

    md: list[str] = []
    md.append(f"# WebSocket Stress Test — {out_dir.name}")
    md.append("")
    md.append(f"**Endpoint:** `{ws_url_base}/ws/transcribe`")
    md.append(f"**Users:** {args.users}  **Duration:** {args.duration_s}s  "
              f"**Model:** `{args.model}`  **Language:** `{args.language}`  "
              f"**Diarize:** {args.diarize}")
    md.append(f"**Ramp:** {args.ramp_s}s  "
              f"**VAD:** min={args.vad_min_ms}ms max={args.vad_max_ms}ms silence={args.vad_silence_ms}ms")
    md.append(f"**Wall time:** {wall_time_s:.1f}s  "
              f"**Chunks:** {n_chunks}  **Errors:** {len(errors)}")
    md.append("")
    md.append(f"**Verdict:** {verdict}  "
              f"(p95 client latency = {p95_lat:.0f}ms, error rate = {100*err_rate:.2f}%)")
    md.append("")

    md.append("## Throughput")
    md.append("")
    md.append(f"- Telemetry events / sec: **{throughput:.2f}**")
    md.append(f"- Per-user chunks / sec: **{throughput / max(1, args.users):.3f}**")
    md.append(f"- Total chunks: {n_chunks}")
    md.append(f"- Total errors: {len(errors)}")
    md.append("")

    def _row(name, values, unit="ms"):
        if not values:
            return f"| {name} | - | - | - | - | - |"
        return (
            f"| {name} | {min(values):.0f} | {_pct(values,0.5):.0f} | "
            f"{_pct(values,0.95):.0f} | {_pct(values,0.99):.0f} | {max(values):.0f} |"
        )

    md.append("## Latency percentiles (ms)")
    md.append("")
    md.append("| Metric | min | p50 | p95 | p99 | max |")
    md.append("|---|---|---|---|---|---|")
    md.append(_row("Client RTT (send→telemetry)", latencies))
    md.append(_row("Server total_ms", total))
    md.append(_row("Whisper ms", whisper))
    md.append(_row("Diarization ms", diar))
    md.append("")

    md.append("## Per-worker fan-out")
    md.append("")
    if not worker_counts:
        md.append("_No worker tags reported — WORKER_TAG env var likely unset on server._")
    else:
        md.append("| Worker | Chunks | % |")
        md.append("|---|---|---|")
        total_w = sum(worker_counts.values()) or 1
        for w, c in worker_counts.most_common():
            md.append(f"| `{w}` | {c} | {100*c/total_w:.1f}% |")
    md.append("")

    md.append("## Errors by type")
    md.append("")
    if not error_counts:
        md.append("_No errors._")
    else:
        md.append("| Type | Count |")
        md.append("|---|---|")
        for t, c in error_counts.most_common():
            md.append(f"| `{t}` | {c} |")
    md.append("")

    md.append("## Label stability")
    md.append("")
    md.append(f"- Sessions with telemetry: {len(per_session_registry_max)}")
    md.append(f"- Expected speaker count per session: {expected}")
    if per_session_registry_max:
        mx_values = list(per_session_registry_max.values())
        md.append(
            f"- Registry size (final): p50={_pct(list(per_session_registry_end.values()),0.5):.0f}  "
            f"p95={_pct(list(per_session_registry_end.values()),0.95):.0f}  "
            f"max={max(per_session_registry_end.values()) if per_session_registry_end else 0}"
        )
        md.append(
            f"- Registry size (peak):  p50={_pct(mx_values,0.5):.0f}  "
            f"p95={_pct(mx_values,0.95):.0f}  max={max(mx_values)}"
        )
    speaker_cardinalities = [len(v) for v in per_session_speakers.values()]
    if speaker_cardinalities:
        md.append(
            f"- Distinct speaker labels per session: "
            f"p50={_pct(speaker_cardinalities,0.5):.0f}  "
            f"p95={_pct(speaker_cardinalities,0.95):.0f}  "
            f"max={max(speaker_cardinalities)}"
        )
    if unstable_sessions:
        md.append("")
        md.append(f"**{len(unstable_sessions)} session(s) show monotonically growing registry past expected "
                  f"speaker count — possible identity-drift bug:**")
        md.append("")
        md.append("| Session | Peak registry | Final registry |")
        md.append("|---|---|---|")
        for sid, mx, end in unstable_sessions[:15]:
            md.append(f"| `{sid[:12]}…` | {mx} | {end} |")
        if len(unstable_sessions) > 15:
            md.append(f"| … {len(unstable_sessions) - 15} more | | |")
    else:
        md.append("- No monotonic registry growth past the expected speaker count detected.")
    md.append("")

    (out_dir / "summary.md").write_text("\n".join(md))

    # Optional charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    charts = out_dir / "charts"
    charts.mkdir(exist_ok=True)

    if latencies:
        plt.figure(figsize=(8, 4))
        plt.hist(latencies, bins=50, color="#4c78a8")
        plt.axvline(_pct(latencies, 0.95), color="red", linestyle="--", label=f"p95 = {_pct(latencies,0.95):.0f}ms")
        plt.xlabel("Client-observed RTT (ms)")
        plt.ylabel("chunks")
        plt.title("WS client RTT distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts / "client_rtt.png", dpi=120)
        plt.close()

    if total:
        plt.figure(figsize=(8, 4))
        plt.hist(total, bins=50, color="#54a24b")
        plt.axvline(_pct(total, 0.95), color="red", linestyle="--", label=f"p95 = {_pct(total,0.95):.0f}ms")
        plt.xlabel("Server total_ms")
        plt.ylabel("chunks")
        plt.title("Server total_ms distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(charts / "server_total_ms.png", dpi=120)
        plt.close()

    if worker_counts:
        plt.figure(figsize=(8, 4))
        labels, counts = zip(*worker_counts.most_common())
        plt.bar(labels, counts, color="#e45756")
        plt.xlabel("Worker tag")
        plt.ylabel("Chunks")
        plt.title("Per-worker chunk fan-out")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(charts / "worker_fanout.png", dpi=120)
        plt.close()


async def _main_async(args) -> None:
    import numpy as np  # noqa: F401

    ws_url_base = _normalize_ws_endpoint(args.endpoint)
    corpus_path = Path(args.corpus_path).resolve()
    corpus_files = _load_corpus(corpus_path)
    if not corpus_files:
        print(f"ERROR: no .wav files under {corpus_path}", file=sys.stderr)
        sys.exit(2)

    random.seed(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ws-stress"
    out_dir = Path(args.out_dir).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_config(out_dir, args, ws_url_base, corpus_files)

    writer = NdjsonWriter(out_dir / "requests.ndjson")
    stop_event = asyncio.Event()

    # Pre-load audio files (cache by path). Small corpus → fits in RAM easily.
    audio_cache: dict[str, "np.ndarray"] = {}
    for p in corpus_files:
        try:
            audio_cache[str(p)] = _load_wav_mono_16k(p)
        except Exception as e:
            print(f"skipping {p}: {e}", file=sys.stderr)

    if not audio_cache:
        print("ERROR: no usable WAVs in corpus", file=sys.stderr)
        sys.exit(2)

    print(f"[ws-stress] run_id={run_id}")
    print(f"[ws-stress] endpoint={ws_url_base}/ws/transcribe")
    print(f"[ws-stress] users={args.users}  duration={args.duration_s}s  "
          f"model={args.model}  lang={args.language}  diarize={args.diarize}")
    print(f"[ws-stress] corpus={len(corpus_files)} files  ramp={args.ramp_s}s")
    print(f"[ws-stress] output → {out_dir}")

    query_params = {
        "model": args.model,
        "language": args.language,
        "diarize": "true" if args.diarize else "false",
        "translate": "true" if args.translate else "false",
    }

    vad_cfg = {
        "min_ms": args.vad_min_ms,
        "max_ms": args.vad_max_ms,
        "silence_ms": args.vad_silence_ms,
    }

    per_user_stats: list[UserStats] = [UserStats() for _ in range(args.users)]

    tasks: list[asyncio.Task] = []
    ramp_delay = (args.ramp_s / args.users) if args.users > 0 else 0.0

    start_wall = time.perf_counter()

    async def _spawn_all():
        for i in range(args.users):
            if stop_event.is_set():
                break
            wav_path = random.choice(list(audio_cache.keys()))
            audio = audio_cache[wav_path]
            task = asyncio.create_task(
                run_user(
                    user_idx=i,
                    ws_url_base=ws_url_base,
                    query_params=query_params,
                    audio=audio,
                    vad_cfg=vad_cfg,
                    writer=writer,
                    stop_event=stop_event,
                    stats=per_user_stats[i],
                    reconnect=args.reconnect,
                )
            )
            tasks.append(task)
            if ramp_delay > 0 and i < args.users - 1:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=ramp_delay)
                    break
                except asyncio.TimeoutError:
                    pass

    spawn_task = asyncio.create_task(_spawn_all())

    # Main clock: wait duration_s then signal stop.
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=args.duration_s)
    except asyncio.TimeoutError:
        stop_event.set()
    except KeyboardInterrupt:
        print("\n[ws-stress] interrupt — signalling stop")
        stop_event.set()

    # Wait for everyone to wrap up (bounded).
    try:
        await asyncio.wait_for(spawn_task, timeout=5.0)
    except asyncio.TimeoutError:
        pass
    if tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30.0)
        except asyncio.TimeoutError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    wall_time = time.perf_counter() - start_wall
    writer.close()

    _analyze_and_write_summary(out_dir, args, wall_time, ws_url_base)

    print(f"[ws-stress] done in {wall_time:.1f}s → {out_dir}/summary.md")


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    default_corpus = _find_default_corpus(repo_root)

    p = argparse.ArgumentParser(
        description="Concurrent-session stress test for TDVX /ws/transcribe",
    )
    p.add_argument("--endpoint", default="ws://localhost:8000",
                   help="WS base URL (http:// / https:// are auto-rewritten). Default: ws://localhost:8000")
    p.add_argument("--users", type=int, default=50,
                   help="Concurrent live WS sessions")
    p.add_argument("--duration-s", type=float, default=300.0,
                   help="Total test duration in wall-clock seconds")
    p.add_argument("--model", default="tdv1-fast",
                   help="Model name (tdv1 | tdv1-balanced | tdv1-fast)")
    p.add_argument("--language", default="pt", choices=["pt", "en"],
                   help="Source language")
    p.add_argument("--corpus-path", default=str(default_corpus),
                   help="Directory of WAVs to stream (recursively)")
    p.add_argument("--diarize", type=_str2bool, default=True,
                   help="Request per-chunk diarization")
    p.add_argument("--translate", type=_str2bool, default=False,
                   help="Request per-segment translation")
    p.add_argument("--out-dir", default=str(repo_root / "results"),
                   help="Results root directory")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for WAV picks")
    p.add_argument("--ramp-s", type=float, default=30.0,
                   help="Spread session openings linearly across this many seconds")
    p.add_argument("--vad-min-ms", type=int, default=1500,
                   help="VAD minimum chunk size (ms)")
    p.add_argument("--vad-max-ms", type=int, default=6000,
                   help="VAD hard-cap chunk size (ms)")
    p.add_argument("--vad-silence-ms", type=int, default=400,
                   help="VAD trailing-silence threshold (ms)")
    p.add_argument("--reconnect", type=_str2bool, default=True,
                   help="Reconnect a user's session after an error")
    p.add_argument("--speaker-count-expected", type=int, default=2,
                   help="Expected distinct speakers per session (for label-stability flag)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
