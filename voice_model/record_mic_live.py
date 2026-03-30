#!/usr/bin/env python3
"""
record_mic_live.py — Transcrição em Tempo Real com Diarização
=============================================================
Fluxo em duas fases para máxima responsividade:

  FASE 1 (enquanto grava):
    Audio → Whisper por chunks → texto aparece na tela imediatamente
    JSON é atualizado a cada chunk com user_id="..." (speaker pendente)

  FASE 2 (após pressionar ENTER):
    Audio completo → pyannote diarização → speaker IDs resolvidos
    JSON final atualizado com "Speaker 0", "Speaker 1", etc.

Uso:
    python record_mic_live.py                    # grava até ENTER
    python record_mic_live.py --duration 30      # grava 30s
    python record_mic_live.py --model medium     # modelo maior
    python record_mic_live.py --list-devices     # ver microfones
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from stt_pipeline import (
    AudioTranscriber, DiarizationEngine, PipelineConfig,
    SpeakerIndexer, STTEngine, TranscriptionSegment,
)

SAMPLE_RATE  = 16_000   # Whisper é treinado em 16 kHz
CHUNK_SEC    = 8        # Whisper processa a cada N segundos de áudio
MIN_SEC      = 1.5      # chunk mínimo para não desperdiçar chamadas
OVERLAP_SEC  = 0.5      # sobreposição entre chunks para não cortar palavras


# ══════════════════════════════════════════════════════
# Live Transcriber
# ══════════════════════════════════════════════════════

class LiveTranscriber:
    """
    Gerencia STT em tempo real (fase 1) e diarização ao final (fase 2).

    Fase 1 — chunks enquanto grava:
      • Cada chunk de CHUNK_SEC segundos é transcrito imediatamente
      • user_id = "..." até a diarização ser concluída

    Fase 2 — após ENTER:
      • Áudio completo → pyannote → SpeakerIndexer
      • Segmentos atualizados com speaker real
      • JSON final gravado
    """

    def __init__(self, cfg: PipelineConfig, output_path: str) -> None:
        self.cfg         = cfg
        self.output_path = output_path
        self.stt         = STTEngine(cfg)
        self.diarizer    = DiarizationEngine(cfg)
        self.indexer     = SpeakerIndexer(threshold=cfg.similarity_threshold)

        self._segments: List[TranscriptionSegment] = []
        self._language  = "?"
        self._offset    = 0.0   # acumulado de tempo dos chunks já processados
        self._lock      = threading.Lock()

        self._init_json()

    # ── Fase 1: STT por chunk ────────────────────────────

    def transcribe_chunk(self, audio: np.ndarray) -> None:
        """Transcreve um chunk de áudio e exibe imediatamente."""
        dur = len(audio) / SAMPLE_RATE
        if dur < MIN_SEC:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            tmp_path = tmp.name

        try:
            raw_segs, lang = self.stt.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

        with self._lock:
            self._language = lang
            for seg in raw_segs:
                text = seg["text"]
                if not text:
                    continue
                ts = TranscriptionSegment(
                    timestamp_start=round(seg["start"] + self._offset, 3),
                    timestamp_end=round(seg["end"]   + self._offset, 3),
                    user_id="...",   # placeholder — será resolvido na fase 2
                    text=text,
                    language_detected=lang,
                    confidence=round(seg["confidence"], 4),
                )
                self._segments.append(ts)
                self._print_live(ts)

            self._offset += dur - OVERLAP_SEC
            self._save_json(final=False)

    # ── Fase 2: Diarização no áudio completo ─────────────

    def diarize_full(self, full_audio: np.ndarray) -> None:
        """Roda diarização no áudio completo e atualiza speaker IDs."""
        if not self._segments:
            return

        print("\n  Rodando diarização no áudio completo... ", end="", flush=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, full_audio, SAMPLE_RATE)
            tmp_path = tmp.name

        try:
            turns = self.diarizer.diarize(tmp_path)

            # Embedding por speaker
            speaker_map: Dict[str, str] = {}
            by_speaker: Dict[str, list] = {}
            for t in turns:
                by_speaker.setdefault(t["raw_speaker"], []).append(t)

            for raw, turn_list in by_speaker.items():
                best = max(turn_list, key=lambda x: x["end"] - x["start"])
                try:
                    emb = self.diarizer.embed_segment(
                        tmp_path, best["start"], best["end"]
                    )
                    uid = self.indexer.identify(emb)
                except Exception:
                    uid = f"Speaker_Unknown_{raw}"
                speaker_map[raw] = uid

            # Atualizar user_id de cada segmento
            with self._lock:
                for seg in self._segments:
                    uid = self._dominant_speaker(
                        seg.timestamp_start, seg.timestamp_end,
                        turns, speaker_map,
                    )
                    seg.user_id = uid

        finally:
            os.unlink(tmp_path)

        print("concluída.\n")

        with self._lock:
            self._save_json(final=True)
            self._print_final_summary()

    # ── JSON ─────────────────────────────────────────────

    def _init_json(self) -> None:
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump({"status": "recording", "segments": []}, f,
                      ensure_ascii=False, indent=2)

    def _save_json(self, final: bool) -> None:
        """Sobrescreve o JSON com o estado atual (chamado com _lock)."""
        data = {
            "status": "complete" if final else "recording",
            "segments": [asdict(s) for s in self._segments],
            "speaker_registry": self.indexer.export() if final else {},
            "metadata": {
                "whisper_model":   self.cfg.whisper_model,
                "language":        self._language,
                "total_segments":  len(self._segments),
                "unique_speakers": self.indexer._counter if final else "?",
            },
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Display ──────────────────────────────────────────

    @staticmethod
    def _print_live(seg: TranscriptionSegment) -> None:
        print(
            f"\r  [{seg.timestamp_start:7.2f}s → {seg.timestamp_end:7.2f}s]"
            f"  {'...':<14}"
            f"  [{seg.language_detected}]"
            f"  {seg.text}"
        )

    def _print_final_summary(self) -> None:
        print("═" * 62)
        print("  TRANSCRIÇÃO FINAL COM SPEAKERS")
        print("═" * 62)
        for seg in self._segments:
            print(
                f"  [{seg.timestamp_start:7.2f}s → {seg.timestamp_end:7.2f}s]"
                f"  {seg.user_id:<14}"
                f"  [{seg.language_detected}]"
                f"  {seg.text}"
            )
        print("─" * 62)
        print(f"  Speakers  : {self.indexer._counter}")
        print(f"  Segmentos : {len(self._segments)}")
        print(f"  JSON      → {self.output_path}")
        print("═" * 62 + "\n")

    # ── Helper ───────────────────────────────────────────

    @staticmethod
    def _dominant_speaker(
        start: float, end: float,
        turns: list, speaker_map: Dict[str, str],
    ) -> str:
        overlap: Dict[str, float] = {}
        for t in turns:
            ov = max(0.0, min(end, t["end"]) - max(start, t["start"]))
            if ov > 0:
                uid = speaker_map.get(t["raw_speaker"], t["raw_speaker"])
                overlap[uid] = overlap.get(uid, 0.0) + ov
        return max(overlap, key=overlap.get) if overlap else "Speaker Unknown"


# ══════════════════════════════════════════════════════
# Worker: consome audio_queue e processa chunks
# ══════════════════════════════════════════════════════

def _chunk_worker(
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    lt: LiveTranscriber,
) -> None:
    chunk_samples   = int(CHUNK_SEC * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
    buffer = np.array([], dtype="float32")

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            data = audio_queue.get(timeout=0.05)
            buffer = np.append(buffer, data)
        except queue.Empty:
            continue

        # Processa quando buffer tem chunk completo
        while len(buffer) >= chunk_samples:
            chunk  = buffer[:chunk_samples]
            buffer = buffer[chunk_samples - overlap_samples:]
            lt.transcribe_chunk(chunk)

    # Processa resto do buffer ao parar
    if len(buffer) >= int(MIN_SEC * SAMPLE_RATE):
        lt.transcribe_chunk(buffer)


# ══════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════

def list_devices() -> None:
    default_in = sd.default.device[0]
    print("\n── Microfones disponíveis ──────────────────────────────────")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            marker = "  ◄ padrão" if i == default_in else ""
            print(f"  [{i:2d}]  {d['name']}{marker}")
    print()


def _build_config(args: argparse.Namespace, token: str) -> PipelineConfig:
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8_float16" if device == "cuda" else "int8"
    return PipelineConfig(
        device=device,
        compute_type=compute,
        whisper_model=args.model,
        hf_token=token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        vad_filter=True,
        vad_min_silence_duration_ms=400,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcrição em Tempo Real (STT live + diarização pós)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--duration", type=int, default=None,
                   metavar="SEG",
                   help="Gravar N segundos (padrão: grava até ENTER)")
    p.add_argument("--device", type=int, default=None,
                   metavar="ID",
                   help="ID do microfone (use --list-devices)")
    p.add_argument("--model", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="Modelo Whisper (tiny é o mais rápido para live)")
    p.add_argument("--output", default="transcription_live.json",
                   metavar="ARQUIVO",
                   help="JSON de saída (atualizado em tempo real)")
    p.add_argument("--hf-token", default="",
                   metavar="TOKEN",
                   help="Token HuggingFace (fallback: HF_TOKEN / .env)")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--list-devices", action="store_true",
                   help="Lista microfones e sai")
    p.add_argument("--keep-wav", action="store_true",
                   help="Salvar WAV completo em gravacao_live.wav")
    return p


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main() -> None:
    args  = _build_parser().parse_args()
    token = args.hf_token or os.getenv("HF_TOKEN", "")

    if args.list_devices:
        list_devices()
        sys.exit(0)

    list_devices()

    cfg = _build_config(args, token)

    # ── Pré-carrega modelos antes de gravar ──────────────
    print("  Carregando modelos (pode demorar na primeira vez)...")
    lt = LiveTranscriber(cfg, args.output)
    print("  Modelos prontos.\n")

    # ── Queues e eventos ─────────────────────────────────
    audio_queue = queue.Queue()
    stop_event  = threading.Event()
    full_audio_chunks: list[np.ndarray] = []  # acumula áudio completo para diarização

    def audio_callback(indata: np.ndarray, frames: int, t, status) -> None:
        if status:
            print(f"  [aviso] {status}", flush=True)
        chunk = indata.copy().flatten()
        audio_queue.put(chunk)
        full_audio_chunks.append(chunk)

    # ── Worker de STT em background ──────────────────────
    worker = threading.Thread(
        target=_chunk_worker,
        args=(audio_queue, stop_event, lt),
        daemon=True,
    )
    worker.start()

    # ── Gravar ───────────────────────────────────────────
    print("─" * 62)
    print("  FASE 1 — GRAVAÇÃO AO VIVO")
    print("  Texto aparece em tempo real  (speaker = '...')")
    if args.duration:
        print(f"  Gravando por {args.duration} segundos...")
    else:
        print("  Pressione  ENTER  para parar")
    print("─" * 62 + "\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=args.device,
        callback=audio_callback,
        blocksize=1024,
    ):
        if args.duration:
            time.sleep(args.duration)
        else:
            input()

    # ── Para worker e espera finalizar ───────────────────
    stop_event.set()
    worker.join(timeout=120)

    # ── Salva WAV completo se pedido ─────────────────────
    full_audio = np.concatenate(full_audio_chunks) if full_audio_chunks else np.array([])

    if args.keep_wav and len(full_audio) > 0:
        wav_path = Path(args.output).with_name("gravacao_live.wav")
        sf.write(str(wav_path), full_audio, SAMPLE_RATE)
        print(f"\n  WAV salvo → {wav_path}")

    # ── Fase 2: Diarização no áudio completo ─────────────
    if len(full_audio) / SAMPLE_RATE >= 1.0:
        print("\n─" * 31)
        print("  FASE 2 — DIARIZAÇÃO")
        print("  Identificando quem falou em cada trecho...")
        print("─" * 31)
        lt.diarize_full(full_audio)
    else:
        print("\n  Áudio muito curto para diarizar.")
        lt._save_json(final=True)


if __name__ == "__main__":
    main()
