"""
WebSocket concurrency stress test for TDVX live transcription.

Simulates N concurrent clients sending audio over WebSocket and measures:
- Per-client latency (time from buffer-flush to response)
- Throughput (total chunks processed / wall time)
- Error rate
- GPU memory under load

Usage:
    # 1 client baseline
    python tests/stress_test.py

    # 5 concurrent clients
    python tests/stress_test.py --clients 5

    # 10 clients, 10s of audio each, with a real audio file
    python tests/stress_test.py --clients 10 --duration 10 --audio tests/sample.wav

    # Ramp-up: add 1 client every 2s up to 8 clients
    python tests/stress_test.py --clients 8 --ramp-delay 2
"""

import asyncio
import argparse
import csv
import json
import os
import statistics
import struct
import sys
import time
from datetime import datetime

import numpy as np

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


def generate_sine_audio(duration_s: float, sample_rate: int = 16000, freq: float = 440.0) -> np.ndarray:
    """Generate a sine-wave tone as float32 PCM (simulates speech-like audio)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    # Mix a few frequencies to be less trivial for VAD/Whisper
    audio = (
        0.3 * np.sin(2 * np.pi * freq * t) +
        0.2 * np.sin(2 * np.pi * (freq * 1.5) * t) +
        0.1 * np.sin(2 * np.pi * (freq * 2) * t)
    )
    # Add slight noise
    audio += np.random.randn(len(audio)).astype(np.float32) * 0.02
    return audio.astype(np.float32)


def load_audio_file(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file and resample to 16kHz mono float32."""
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        print("librosa not installed, falling back to synthetic audio")
        return None


def _response_is_nonempty(data: dict) -> bool:
    """Check whether a JSON response actually carries transcription text."""
    if not isinstance(data, dict):
        return False
    seg = data.get("segment")
    if isinstance(seg, dict) and seg.get("text"):
        return True
    if data.get("text"):
        return True
    return False


async def run_client(
    client_id: int,
    ws_url: str,
    audio_data: np.ndarray,
    chunk_size: int,
    send_delay: float,
    results: dict,
    expected_compute_ms: float = 3000.0,
):
    """
    Single WebSocket client that streams audio and collects timing metrics.
    """
    latencies = []
    segment_latencies_ms = []
    responses = []
    errors = []
    chunks_sent = 0
    ttfr_ms = None

    t_connect_start = time.perf_counter()

    try:
        async with websockets.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
            t_connected = time.perf_counter()
            connect_time = t_connected - t_connect_start

            # Stream audio in chunks
            total_samples = len(audio_data)
            t_first_send = None
            t_last_send = None
            t_last_response = None

            for offset in range(0, total_samples, chunk_size):
                chunk = audio_data[offset : offset + chunk_size]
                payload = chunk.tobytes()

                t_send = time.perf_counter()
                if t_first_send is None:
                    t_first_send = t_send

                await ws.send(payload)
                t_last_send = t_send
                chunks_sent += 1

                # Non-blocking check for responses
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.001)
                        t_recv = time.perf_counter()
                        t_last_response = t_recv
                        data = json.loads(raw)
                        responses.append(data)
                        # Latency = time from this recv back to the most recent send
                        latencies.append(t_recv - t_send)
                        # TTFR: time from ws connect return to first non-empty response
                        if ttfr_ms is None and _response_is_nonempty(data):
                            ttfr_ms = (t_recv - t_connected) * 1000
                        # Segment latency: time from last send preceding recv to recv
                        if _response_is_nonempty(data) and t_last_send is not None:
                            segment_latencies_ms.append((t_recv - t_last_send) * 1000)
                except asyncio.TimeoutError:
                    pass

                if send_delay > 0:
                    await asyncio.sleep(send_delay)

            # Send end signal and drain remaining responses
            await ws.send("end")
            drain_start = time.perf_counter()
            # After "end", the last meaningful send timestamp remains t_last_send
            # (we do NOT advance it to drain_start because "end" is a control msg, not audio).

            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    t_recv = time.perf_counter()
                    t_last_response = t_recv
                    data = json.loads(raw)
                    responses.append(data)
                    latencies.append(t_recv - drain_start)
                    if ttfr_ms is None and _response_is_nonempty(data):
                        ttfr_ms = (t_recv - t_connected) * 1000
                    if _response_is_nonempty(data) and t_last_send is not None:
                        segment_latencies_ms.append((t_recv - t_last_send) * 1000)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

            t_done = time.perf_counter()

            queue_wait_ms = None
            if ttfr_ms is not None:
                queue_wait_ms = max(0.0, ttfr_ms - expected_compute_ms)

            results[client_id] = {
                "status": "ok",
                "connect_time_ms": connect_time * 1000,
                "total_time_s": t_done - t_connect_start,
                "chunks_sent": chunks_sent,
                "responses": len(responses),
                "errors": errors,
                "latencies_ms": [l * 1000 for l in latencies],
                "segment_latencies_ms": segment_latencies_ms,
                "ttfr_ms": ttfr_ms,
                "queue_wait_ms": queue_wait_ms,
                "transcription_segments": [
                    r.get("segment", {}).get("text", r.get("text", ""))
                    for r in responses
                    if "segment" in r or "text" in r
                ],
            }

    except Exception as e:
        results[client_id] = {
            "status": "error",
            "error": str(e),
            "connect_time_ms": None,
            "total_time_s": time.perf_counter() - t_connect_start,
            "chunks_sent": chunks_sent,
            "responses": len(responses),
            "errors": [str(e)],
            "latencies_ms": [l * 1000 for l in latencies],
            "segment_latencies_ms": segment_latencies_ms,
            "ttfr_ms": ttfr_ms,
            "queue_wait_ms": None,
            "transcription_segments": [],
        }


def get_gpu_stats():
    """Get GPU memory usage via nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        parts = out.split(", ")
        return {
            "gpu_memory_used_mib": int(parts[0]),
            "gpu_memory_total_mib": int(parts[1]),
            "gpu_utilization_pct": int(parts[2]),
        }
    except Exception:
        return None


def _stats_block(label: str, values: list, unit: str = "ms"):
    """Print min/median/mean/p95/max for a list of numeric values."""
    print(f"\n  --- {label} ---")
    if not values:
        print(f"  (no samples)")
        return
    print(f"  Min:                {min(values):.1f} {unit}")
    print(f"  Median:             {statistics.median(values):.1f} {unit}")
    print(f"  Mean:               {statistics.mean(values):.1f} {unit}")
    print(f"  P95:                {np.percentile(values, 95):.1f} {unit}")
    print(f"  Max:                {max(values):.1f} {unit}")


def print_report(
    results: dict,
    wall_time: float,
    gpu_before: dict,
    gpu_after: dict,
    audio_seconds_processed: float = 0.0,
    expected_compute_ms: float = 3000.0,
):
    """Print a human-readable stress test report."""
    n = len(results)
    ok = [r for r in results.values() if r["status"] == "ok"]
    failed = [r for r in results.values() if r["status"] != "ok"]

    print("\n" + "=" * 70)
    print("  TDVX CONCURRENCY STRESS TEST REPORT")
    print("=" * 70)

    print(f"\n  Clients:          {n}")
    print(f"  Succeeded:        {len(ok)}")
    print(f"  Failed:           {len(failed)}")
    print(f"  Wall time:        {wall_time:.2f}s")

    if failed:
        print("\n  FAILURES:")
        for cid, r in results.items():
            if r["status"] != "ok":
                print(f"    Client {cid}: {r.get('error', 'unknown')}")

    if ok:
        all_latencies = []
        all_segment_latencies = []
        ttfr_values = []
        queue_wait_values = []
        total_responses = 0
        total_chunks = 0

        for r in ok:
            all_latencies.extend(r["latencies_ms"])
            all_segment_latencies.extend(r.get("segment_latencies_ms", []))
            if r.get("ttfr_ms") is not None:
                ttfr_values.append(r["ttfr_ms"])
            if r.get("queue_wait_ms") is not None:
                queue_wait_values.append(r["queue_wait_ms"])
            total_responses += r["responses"]
            total_chunks += r["chunks_sent"]

        connect_times = [r["connect_time_ms"] for r in ok]

        print(f"\n  --- Connection ---")
        print(f"  Avg connect time:   {statistics.mean(connect_times):.1f} ms")
        print(f"  Max connect time:   {max(connect_times):.1f} ms")

        print(f"\n  --- Throughput ---")
        print(f"  Total chunks sent:  {total_chunks}")
        print(f"  Total responses:    {total_responses}")
        print(f"  Responses/sec:      {total_responses / wall_time:.1f}")
        print(f"  Audio processed:    {audio_seconds_processed:.2f} s")
        print(f"  Wall time:          {wall_time:.2f} s")
        throughput_rtf = (audio_seconds_processed / wall_time) if wall_time > 0 else 0.0
        print(f"  Throughput RTF:     {throughput_rtf:.2f}x (>1.0 = keeps up with realtime)")

        if all_latencies:
            print(f"\n  --- Response Latency ---")
            print(f"  Min:                {min(all_latencies):.1f} ms")
            print(f"  Median:             {statistics.median(all_latencies):.1f} ms")
            print(f"  Mean:               {statistics.mean(all_latencies):.1f} ms")
            print(f"  P95:                {np.percentile(all_latencies, 95):.1f} ms")
            print(f"  P99:                {np.percentile(all_latencies, 99):.1f} ms")
            print(f"  Max:                {max(all_latencies):.1f} ms")

        _stats_block("Time-to-First-Response (TTFR)", ttfr_values, unit="ms")
        _stats_block(
            f"Queue Wait (TTFR - expected_compute={expected_compute_ms:.0f}ms)",
            queue_wait_values,
            unit="ms",
        )
        _stats_block("Segment End-to-Text Latency", all_segment_latencies, unit="ms")

        # Per-client breakdown
        print(f"\n  --- Per-Client Breakdown ---")
        print(f"  {'Client':>8} {'Time(s)':>8} {'Chunks':>8} {'Responses':>10} {'Avg Lat(ms)':>12} {'TTFR(ms)':>10} {'Segments':>10}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

        for cid in sorted(results.keys()):
            r = results[cid]
            if r["status"] != "ok":
                print(f"  {cid:>8} {'FAILED':>8}")
                continue
            avg_lat = statistics.mean(r["latencies_ms"]) if r["latencies_ms"] else 0
            n_seg = len(r["transcription_segments"])
            ttfr_str = f"{r['ttfr_ms']:.0f}" if r.get("ttfr_ms") is not None else "-"
            print(f"  {cid:>8} {r['total_time_s']:>8.2f} {r['chunks_sent']:>8} {r['responses']:>10} {avg_lat:>12.1f} {ttfr_str:>10} {n_seg:>10}")

    # GPU
    if gpu_before and gpu_after:
        print(f"\n  --- GPU Memory ---")
        print(f"  Before:             {gpu_before['gpu_memory_used_mib']} MiB / {gpu_before['gpu_memory_total_mib']} MiB ({gpu_before['gpu_utilization_pct']}% util)")
        print(f"  After:              {gpu_after['gpu_memory_used_mib']} MiB / {gpu_after['gpu_memory_total_mib']} MiB ({gpu_after['gpu_utilization_pct']}% util)")
        delta = gpu_after["gpu_memory_used_mib"] - gpu_before["gpu_memory_used_mib"]
        print(f"  Delta:              {'+' if delta >= 0 else ''}{delta} MiB")

    print("\n" + "=" * 70)


def _pct_or_none(values: list, pct: float):
    if not values:
        return None
    return float(np.percentile(values, pct))


def write_csv_row(
    csv_path: str,
    n_clients: int,
    wall_time: float,
    results: dict,
    audio_seconds_processed: float,
):
    """Append a single summary row to a CSV file (creating header if new)."""
    ok = [r for r in results.values() if r["status"] == "ok"]

    ttfr_values = [r["ttfr_ms"] for r in ok if r.get("ttfr_ms") is not None]
    seg_lat_values = []
    for r in ok:
        seg_lat_values.extend(r.get("segment_latencies_ms", []))

    ttfr_p50 = _pct_or_none(ttfr_values, 50)
    ttfr_p95 = _pct_or_none(ttfr_values, 95)
    seg_p50 = _pct_or_none(seg_lat_values, 50)
    seg_p95 = _pct_or_none(seg_lat_values, 95)

    rtf = (audio_seconds_processed / wall_time) if wall_time > 0 else 0.0
    success_rate = (len(ok) / n_clients) if n_clients > 0 else 0.0

    header = [
        "timestamp",
        "n_clients",
        "wall_time_s",
        "ttfr_p50_ms",
        "ttfr_p95_ms",
        "segment_lat_p50_ms",
        "segment_lat_p95_ms",
        "rtf",
        "success_rate",
    ]
    row = [
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        n_clients,
        f"{wall_time:.3f}",
        f"{ttfr_p50:.1f}" if ttfr_p50 is not None else "",
        f"{ttfr_p95:.1f}" if ttfr_p95 is not None else "",
        f"{seg_p50:.1f}" if seg_p50 is not None else "",
        f"{seg_p95:.1f}" if seg_p95 is not None else "",
        f"{rtf:.3f}",
        f"{success_rate:.3f}",
    ]

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


async def main():
    parser = argparse.ArgumentParser(description="TDVX WebSocket concurrency stress test")
    parser.add_argument("--clients", type=int, default=1, help="Number of concurrent WebSocket clients (default: 1)")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds of audio per client (default: 5)")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (WAV/MP3). If omitted, uses synthetic audio")
    parser.add_argument("--url", type=str, default="ws://localhost:8000/ws/transcribe", help="WebSocket URL")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Samples per WebSocket message (default: 4096)")
    parser.add_argument("--send-delay", type=float, default=0.01, help="Delay between chunk sends in seconds (default: 0.01). Use 'realtime' via --realtime flag")
    parser.add_argument("--realtime", action="store_true", help="Send at real-time pace (chunk_size/sample_rate delay)")
    parser.add_argument("--ramp-delay", type=float, default=0.0, help="Seconds between launching each client (0 = all at once)")
    parser.add_argument("--model", type=str, default=None, help="Model to use (tdv1, tdv1-balanced, tdv1-fast)")
    parser.add_argument("--expected-compute-ms", type=float, default=3000.0,
                        help="Expected Whisper compute time (ms) for first chunk. Used for queue_wait_ms. Default: 3000 (medium on Blackwell)")
    parser.add_argument("--csv", type=str, default=None,
                        help="If set, append a single summary row to this CSV file for cross-run comparison.")
    args = parser.parse_args()

    ws_url = args.url
    if args.model:
        ws_url += f"?model={args.model}"

    # Real-time pacing: delay = chunk_duration
    if args.realtime:
        args.send_delay = args.chunk_size / 16000  # seconds per chunk

    # Prepare audio
    if args.audio:
        audio_data = load_audio_file(args.audio)
        if audio_data is None:
            audio_data = generate_sine_audio(args.duration)
        # Trim/repeat to match requested duration
        target_samples = int(args.duration * 16000)
        if len(audio_data) > target_samples:
            audio_data = audio_data[:target_samples]
        elif len(audio_data) < target_samples:
            repeats = target_samples // len(audio_data) + 1
            audio_data = np.tile(audio_data, repeats)[:target_samples]
    else:
        audio_data = generate_sine_audio(args.duration)

    audio_duration = len(audio_data) / 16000

    print(f"\n  TDVX Stress Test")
    print(f"  ================")
    print(f"  URL:            {ws_url}")
    print(f"  Clients:        {args.clients}")
    print(f"  Audio:          {args.audio or 'synthetic sine'} ({audio_duration:.1f}s)")
    print(f"  Chunk size:     {args.chunk_size} samples ({args.chunk_size/16000*1000:.0f}ms)")
    print(f"  Send delay:     {args.send_delay*1000:.0f}ms")
    print(f"  Ramp delay:     {args.ramp_delay}s")
    print(f"  Expected cmp:   {args.expected_compute_ms:.0f}ms")
    print()

    gpu_before = get_gpu_stats()
    results = {}

    t_start = time.perf_counter()

    if args.ramp_delay > 0:
        # Staggered launch
        tasks = []
        for i in range(args.clients):
            print(f"  Launching client {i}...")
            task = asyncio.create_task(
                run_client(i, ws_url, audio_data, args.chunk_size, args.send_delay, results, args.expected_compute_ms)
            )
            tasks.append(task)
            if i < args.clients - 1:
                await asyncio.sleep(args.ramp_delay)
        await asyncio.gather(*tasks)
    else:
        # All at once
        print(f"  Launching all {args.clients} clients simultaneously...")
        tasks = [
            run_client(i, ws_url, audio_data, args.chunk_size, args.send_delay, results, args.expected_compute_ms)
            for i in range(args.clients)
        ]
        await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - t_start
    gpu_after = get_gpu_stats()

    audio_seconds_processed = audio_duration * args.clients

    print_report(
        results,
        wall_time,
        gpu_before,
        gpu_after,
        audio_seconds_processed=audio_seconds_processed,
        expected_compute_ms=args.expected_compute_ms,
    )

    if args.csv:
        try:
            write_csv_row(
                args.csv,
                n_clients=args.clients,
                wall_time=wall_time,
                results=results,
                audio_seconds_processed=audio_seconds_processed,
            )
            print(f"\n  CSV row appended to: {args.csv}")
        except Exception as e:
            print(f"\n  Failed to write CSV ({args.csv}): {e}")


if __name__ == "__main__":
    asyncio.run(main())
