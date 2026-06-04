#!/usr/bin/env python3
"""
locustfile.py — Load test TDvX API com 150 usuários simultâneos
================================================================

Testa:
  - /transcribe          — transcrição simples (70% do tráfego)
  - /transcribe-diarize  — transcrição + diarização (25% do tráfego)
  - /health              — healthcheck (5% do tráfego)

Áudio: corpus pt-BR real (tests/corpus/pt-BR/*.wav) + geração TTS (edge-tts)

Uso:
    # UI web (recomendado)
    locust -f locustfile.py --host http://localhost:8000

    # Headless — 150 usuários, ramp 30s, duração 5min
    locust -f locustfile.py --host http://localhost:8000 \
        --headless --users 150 --spawn-rate 5 --run-time 5m \
        --html results/load_report.html --csv results/load
"""
from __future__ import annotations

import io
import os
import random
import time
from pathlib import Path

import numpy as np

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner, WorkerRunner

# ─────────────────────────────────────────────────────────────────────────────
# Áudio de teste
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_DIR = Path(__file__).parent / "tests" / "corpus" / "pt-BR"

FRASES_PT = [
    "Bom dia, como você está hoje?",
    "Precisamos agendar uma reunião para amanhã de manhã.",
    "O relatório financeiro do trimestre foi concluído com sucesso.",
    "A inteligência artificial está transformando o mercado de trabalho.",
    "Por favor, envie o documento por e-mail assim que possível.",
    "O sistema de reconhecimento de voz precisa ser calibrado.",
    "As transações foram processadas com sucesso pelo banco.",
    "Qual é o prazo de entrega para esse pedido urgente?",
    "Vamos revisar o contrato antes de assinar qualquer documento.",
    "Os dados foram atualizados no sistema automaticamente.",
]

_corpus_wavs: list[bytes] = []
_tts_cache:   list[bytes] = []


def _load_corpus() -> list[bytes]:
    wavs = []
    if CORPUS_DIR.exists():
        for f in sorted(CORPUS_DIR.glob("*.wav"))[:20]:  # limita a 20 arquivos
            wavs.append(f.read_bytes())
    return wavs


def _generate_tts_sync(text: str) -> bytes:
    """Gera áudio WAV via edge-tts (síncrono via asyncio.run)."""
    import asyncio
    import tempfile
    import wave

    try:
        import edge_tts
        import soundfile as sf
        import librosa

        async def _gen():
            communicate = edge_tts.Communicate(text, "pt-BR-FranciscaNeural")
            mp3_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes += chunk["data"]
            return mp3_bytes

        mp3 = asyncio.run(_gen())

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3)
            tmp = f.name

        audio, _ = librosa.load(tmp, sr=16_000, mono=True)
        os.unlink(tmp)

        buf = io.BytesIO()
        sf.write(buf, audio.astype("float32"), 16_000, format="WAV")
        return buf.getvalue()
    except Exception:
        # fallback: ruído sintético 2s
        audio = (np.random.randn(32_000) * 0.1).astype("float32")
        buf = io.BytesIO()
        try:
            import soundfile as sf
            sf.write(buf, audio, 16_000, format="WAV")
        except Exception:
            pass
        return buf.getvalue()


def _get_audio() -> bytes:
    """Retorna um arquivo de áudio aleatório (corpus ou TTS)."""
    pool = _corpus_wavs + _tts_cache
    if not pool:
        return _generate_tts_sync(random.choice(FRASES_PT))
    return random.choice(pool)


# ─────────────────────────────────────────────────────────────────────────────
# Hooks de setup
# ─────────────────────────────────────────────────────────────────────────────

@events.init.add_listener
def on_init(environment, **_):
    global _corpus_wavs, _tts_cache
    if isinstance(environment.runner, WorkerRunner):
        return

    print("[setup] Carregando corpus de áudio...")
    _corpus_wavs = _load_corpus()
    print(f"[setup] {len(_corpus_wavs)} arquivos WAV carregados do corpus")

    print("[setup] Gerando cache TTS (5 frases)...")
    for frase in random.sample(FRASES_PT, min(5, len(FRASES_PT))):
        wav = _generate_tts_sync(frase)
        if wav:
            _tts_cache.append(wav)
    print(f"[setup] {len(_tts_cache)} áudios TTS gerados")


@events.quitting.add_listener
def on_quitting(environment, **_):
    lats = [s.avg_response_time for s in environment.stats.entries.values() if s.num_requests > 0]
    errs = sum(s.num_failures for s in environment.stats.entries.values())
    total = sum(s.num_requests for s in environment.stats.entries.values())
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL — {total} requests | {errs} erros ({errs/max(1,total)*100:.1f}%)")
    if lats:
        print(f"Latência média: {sum(lats)/len(lats):.0f}ms")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Usuário
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(70)
    def transcribe(self):
        audio = _get_audio()
        with self.client.post(
            "/transcribe",
            files={"file": ("audio.wav", audio, "audio/wav")},
            params={"language": "pt"},
            catch_response=True,
            name="/transcribe",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(25)
    def transcribe_diarize(self):
        audio = _get_audio()
        with self.client.post(
            "/transcribe-diarize",
            files={"file": ("audio.wav", audio, "audio/wav")},
            params={"language": "pt"},
            catch_response=True,
            name="/transcribe-diarize",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(5)
    def health(self):
        self.client.get("/health", name="/health")
