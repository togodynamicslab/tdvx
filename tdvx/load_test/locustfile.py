"""
locustfile.py — Teste de carga para TDvX Transcription API

Uso:
  # Instalar: pip install locust

  # Interface web (http://localhost:8089):
  locust -f locustfile.py --host=http://localhost:8000

  # Headless — progressão de carga para encontrar ponto de quebra:
  locust -f locustfile.py --host=http://localhost:8000 --headless -u 1  --spawn-rate 1 --run-time 60s  --csv=results_1u
  locust -f locustfile.py --host=http://localhost:8000 --headless -u 5  --spawn-rate 1 --run-time 60s  --csv=results_5u
  locust -f locustfile.py --host=http://localhost:8000 --headless -u 10 --spawn-rate 2 --run-time 60s  --csv=results_10u
  locust -f locustfile.py --host=http://localhost:8000 --headless -u 20 --spawn-rate 2 --run-time 120s --csv=results_20u

Coloque arquivos WAV representativos (5–30s) em load_test/samples/.
Se a pasta estiver vazia, um WAV de silêncio sintético é gerado automaticamente.
"""

import io
import json
import os
import random
import struct
import time
import wave
from pathlib import Path

import numpy as np
from locust import HttpUser, between, events, task
from locust.exception import RescheduleTask

# ── Configurações ─────────────────────────────────────────────────────────────
SAMPLES_DIR = Path(__file__).parent / "samples"
LATENCY_SLO_S = 30.0       # p95 acima disto = degradação de serviço
FAILURE_RATE_THRESHOLD = 0.05  # 5% de falhas = ponto de quebra


# ── Geração de WAV sintético de fallback ──────────────────────────────────────

def _generate_silent_wav(duration_s: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Gera WAV mono 16-bit com ruído branco leve (evita VAD filtrar tudo)."""
    n_samples = int(duration_s * sample_rate)
    # Ruído branco baixo (simula fala distante — passa pelo VAD)
    noise = (np.random.randn(n_samples) * 500).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(noise.tobytes())
    return buf.getvalue()


def _load_samples() -> list[tuple[str, bytes]]:
    """
    Carrega WAVs da pasta samples/.
    Se vazia, gera 3 durações sintéticas (5s, 15s, 30s).
    Retorna lista de (nome, bytes).
    """
    samples = []
    if SAMPLES_DIR.exists():
        for f in SAMPLES_DIR.glob("*.wav"):
            samples.append((f.name, f.read_bytes()))

    if not samples:
        for dur in [5.0, 15.0, 30.0]:
            name = f"synthetic_{int(dur)}s.wav"
            samples.append((name, _generate_silent_wav(dur)))

    return samples


# Pré-carrega na inicialização do processo Locust
_SAMPLES: list[tuple[str, bytes]] = _load_samples()


# ── Métricas customizadas ─────────────────────────────────────────────────────

_latencies: list[float] = []

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    if exception is None and name == "/transcribe":
        _latencies.append(response_time / 1000.0)  # ms → s

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if not _latencies:
        return
    arr = sorted(_latencies)
    p50 = arr[len(arr) // 2]
    p95 = arr[int(len(arr) * 0.95)]
    p99 = arr[int(len(arr) * 0.99)]
    total = environment.stats.total
    fail_rate = total.fail_ratio

    print("\n" + "=" * 60)
    print("RESUMO DO TESTE DE CARGA")
    print("=" * 60)
    print(f"  Requisições:     {total.num_requests}")
    print(f"  Falhas:          {total.num_failures} ({fail_rate:.1%})")
    print(f"  RPS médio:       {total.current_rps:.1f}")
    print(f"  Latência p50:    {p50:.2f}s")
    print(f"  Latência p95:    {p95:.2f}s")
    print(f"  Latência p99:    {p99:.2f}s")
    print(f"  SLO (p95<{LATENCY_SLO_S}s): {'✓ OK' if p95 < LATENCY_SLO_S else '✗ VIOLADO'}")
    print(f"  Taxa de falha:   {'✓ OK' if fail_rate < FAILURE_RATE_THRESHOLD else '✗ ACIMA DE ' + str(FAILURE_RATE_THRESHOLD)}")
    print("=" * 60)


# ── Usuários de teste ─────────────────────────────────────────────────────────

class TranscriptionUser(HttpUser):
    """Simula usuário que envia arquivos para transcrição."""
    wait_time = between(1.0, 3.0)

    def on_start(self):
        if not _SAMPLES:
            raise RescheduleTask()
        # Verifica saúde do servidor antes de começar
        with self.client.get("/health", catch_response=True, name="/health") as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check falhou: {resp.status_code}")

    @task(5)
    def transcribe_random(self):
        """Envia arquivo aleatório — carga principal (peso 5)."""
        name, audio_bytes = random.choice(_SAMPLES)
        self._send_file(name, audio_bytes)

    @task(2)
    def transcribe_short(self):
        """Sempre usa o menor sample disponível — simula consulta rápida."""
        shortest = min(_SAMPLES, key=lambda x: len(x[1]))
        self._send_file(*shortest)

    @task(1)
    def health_check(self):
        """Mantém baseline de requisições leves."""
        with self.client.get("/health", catch_response=True, name="/health") as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    def _send_file(self, filename: str, audio_bytes: bytes):
        t0 = time.perf_counter()
        with self.client.post(
            "/transcribe",
            files={"file": (filename, io.BytesIO(audio_bytes), "audio/wav")},
            catch_response=True,
            name="/transcribe",
        ) as resp:
            elapsed = time.perf_counter() - t0

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    seg_count = len(data.get("segments", []))
                    # Falha de SLO — conta como erro de performance
                    if elapsed > LATENCY_SLO_S:
                        resp.failure(f"SLO violado: {elapsed:.1f}s > {LATENCY_SLO_S}s")
                    else:
                        resp.success()
                except Exception as exc:
                    resp.failure(f"JSON inválido: {exc}")
            elif resp.status_code == 413:
                resp.failure("Arquivo muito grande (413)")
            elif resp.status_code == 500:
                resp.failure(f"Erro interno: {resp.text[:200]}")
            else:
                resp.failure(f"HTTP {resp.status_code}")


class WebSocketUser(HttpUser):
    """
    Simula cliente WebSocket de transcrição live.
    Envia chunks de 500ms e aguarda respostas parciais.
    Peso menor — WebSocket é mais intensivo por conexão.
    """
    wait_time = between(5.0, 15.0)
    weight = 1  # 1 usuário WS para cada ~8 HTTP

    def on_start(self):
        # Pré-carrega áudio de teste (5s)
        _, wav_bytes = min(_SAMPLES, key=lambda x: len(x[1]))
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf) as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        self._test_audio = audio

    @task
    def stream_audio(self):
        """Simula streaming PCM via WebSocket."""
        import websocket as ws_lib  # pip install websocket-client

        audio = self._test_audio
        sample_rate = 16000
        chunk_samples = int(sample_rate * 0.5)  # 500ms

        ws_url = self.host.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/transcribe"

        received = []
        t0 = time.perf_counter()

        try:
            ws = ws_lib.create_connection(ws_url, timeout=30)

            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                ws.send_binary(chunk.tobytes())

                # Recebe respostas não-bloqueantes
                ws.settimeout(0.05)
                try:
                    msg = ws.recv()
                    received.append(json.loads(msg))
                except Exception:
                    pass

            # Sinaliza fim de stream
            ws.send_binary(b"")

            # Aguarda respostas finais
            ws.settimeout(10.0)
            while True:
                try:
                    msg = ws.recv()
                    received.append(json.loads(msg))
                except Exception:
                    break

            ws.close()
            elapsed = time.perf_counter() - t0
            self.environment.events.request.fire(
                request_type="WS",
                name="/ws/transcribe",
                response_time=elapsed * 1000,
                response_length=len(str(received)),
                exception=None,
                context={},
            )
        except Exception as exc:
            self.environment.events.request.fire(
                request_type="WS",
                name="/ws/transcribe",
                response_time=(time.perf_counter() - t0) * 1000,
                response_length=0,
                exception=exc,
                context={},
            )
