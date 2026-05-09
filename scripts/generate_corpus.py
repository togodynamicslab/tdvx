"""
Generate one ~60s pt-BR base clip via ElevenLabs Multilingual v2.

The stochastic load test derives variable durations (3s–600s) from this single
clip by slicing and looping, so we only spend ~500 credits here instead of
generating separate clips per duration bucket.

Output: tests/corpus/pt-BR/base_60s.wav (16kHz mono PCM)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
CORPUS_DIR = REPO_ROOT / "tests" / "corpus" / "pt-BR"
MP3_OUT = CORPUS_DIR / "base_60s.mp3"
WAV_OUT = CORPUS_DIR / "base_60s.wav"

# Rachel's multilingual counterpart; any public voice works for load testing.
# Using the default "Rachel" voice ID — ElevenLabs supports it for all tiers.
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
MODEL_ID = "eleven_multilingual_v2"

# ~500 chars of natural pt-BR content — varied sentences with realistic
# wearable/dictation style (observations, quick notes, reminders).
SCRIPT_PT_BR = (
    "Hoje de manhã percebi que o café estava mais forte que de costume, "
    "talvez porque usei um pouco mais de pó. Preciso lembrar de comprar "
    "filtros novos antes do fim de semana. Ontem conversei com o João sobre "
    "o projeto de monitoramento e ele sugeriu que a gente revise os limites "
    "de alerta, porque estão disparando com muita frequência durante a "
    "madrugada. Marquei uma reunião para quinta-feira às duas da tarde. "
    "Também anotei aqui uma ideia: talvez valha a pena testar um novo modelo "
    "de transcrição na próxima sprint, só pra ver se a latência melhora em "
    "cenários de pico. O clima hoje está bem agradável, o suficiente pra "
    "correr no parque depois do almoço. Lembrar de pagar a conta de luz até "
    "o dia quinze."
)


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def tts_request(api_key: str, text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    payload = json.dumps(
        {
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def convert_to_wav(mp3_path: Path, wav_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ],
        check=True,
    )


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(out.strip())


def main() -> int:
    env = load_env(ENV_FILE)
    api_key = env.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not found in .env or environment", file=sys.stderr)
        return 1

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    chars = len(SCRIPT_PT_BR)
    print(f"Script: {chars} chars (cost ~{chars} credits on Multilingual v2)")

    if WAV_OUT.exists() and MP3_OUT.exists():
        dur = probe_duration(WAV_OUT)
        print(f"Corpus already exists: {WAV_OUT} ({dur:.2f}s) — skipping API call")
        return 0

    print(f"Calling ElevenLabs TTS (voice={VOICE_ID}, model={MODEL_ID})...")
    try:
        audio = tts_request(api_key, SCRIPT_PT_BR)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 2
    except urllib.error.URLError as e:
        print(f"Network error: {e}", file=sys.stderr)
        return 2

    MP3_OUT.write_bytes(audio)
    print(f"Saved MP3: {MP3_OUT} ({len(audio)} bytes)")

    convert_to_wav(MP3_OUT, WAV_OUT)
    dur = probe_duration(WAV_OUT)
    print(f"Saved WAV: {WAV_OUT} ({dur:.2f}s, 16kHz mono s16)")

    if dur < 30:
        print(f"WARNING: clip is {dur:.1f}s, under target 30s — looping will be needed for mid-range durations", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
