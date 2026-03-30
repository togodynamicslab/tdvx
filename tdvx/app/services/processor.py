"""
processor.py — Orquestrador do pipeline de transcrição (upload de arquivo)

  STT → Diarização → renomeação de speakers → TranscriptionResponse
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

from app.config import settings
from app.models.response import TranscriptionResponse, TranscriptionSegment
from app.services.engine import DiarizationEngine, STTEngine, SpeakerIndexer

log = logging.getLogger(__name__)

# ── Filtro de alucinações do Whisper ──────────────────────────────────────────
_REPEAT_RE = re.compile(r'(.)\1{6,}')   # qualquer char repetido 7+ vezes
_REPEAT_WORD_RE = re.compile(r'\b(\w{2,})\b(?:\s+\1\b){4,}', re.IGNORECASE)


def _is_hallucination(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if _REPEAT_RE.search(t):
        return True
    if _REPEAT_WORD_RE.search(t):
        return True
    words = t.split()
    if len(words) >= 6 and len(set(w.lower() for w in words)) == 1:
        return True
    return False


# ── Singletons lazy dos engines ───────────────────────────────────────────────
_stt: Optional[STTEngine] = None
_diarizer: Optional[DiarizationEngine] = None
_indexer: Optional[SpeakerIndexer] = None


def _get_stt() -> STTEngine:
    global _stt
    if _stt is None:
        _stt = STTEngine(
            model_size=settings.whisper_model,
            cpu_threads=settings.cpu_threads,
        )
    return _stt


def _get_diarizer() -> DiarizationEngine:
    global _diarizer
    if _diarizer is None:
        _diarizer = DiarizationEngine(
            hf_token=settings.pyannote_auth_token,
            min_speakers=settings.pyannote_min_speakers if settings.pyannote_min_speakers > 0 else None,
            max_speakers=settings.pyannote_max_speakers,
            clustering_threshold=settings.pyannote_clustering_threshold,
        )
    return _diarizer


def _get_indexer() -> SpeakerIndexer:
    global _indexer
    if _indexer is None:
        _indexer = SpeakerIndexer(threshold=settings.speaker_similarity_threshold)
    return _indexer


def preload_engines() -> None:
    log.info("Pré-carregando STTEngine …")
    _get_stt()
    log.info("Pré-carregando DiarizationEngine …")
    try:
        _get_diarizer()
    except Exception as exc:
        log.error("Falha ao carregar DiarizationEngine: %s", exc)


def _to_waveform(audio: np.ndarray, sample_rate: int) -> dict:
    tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    return {"waveform": tensor, "sample_rate": sample_rate}


# ══════════════════════════════════════════════════════════════════════════════
# TranscriptionProcessor
# ══════════════════════════════════════════════════════════════════════════════

class TranscriptionProcessor:
    """Orquestra: STT → Diarização → Merge por overlap temporal."""

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResponse:
        t0 = time.time()
        duration = len(audio) / sample_rate
        waveform = _to_waveform(audio, sample_rate)

        # Passo 1 — Transcrição
        segments, language = _get_stt().transcribe(audio, sample_rate)

        if not segments:
            log.warning("Nenhum segmento de transcrição encontrado")
            return TranscriptionResponse(
                timestamp=datetime.now(),
                language=language or "unknown",
                duration=duration,
                segments=[],
            )

        # Passo 2 — Diarização (contexto completo do arquivo)
        try:
            turns = _get_diarizer().diarize(
                waveform,
                clustering_threshold=settings.pyannote_clustering_threshold,
            )
        except Exception as exc:
            log.warning("Diarização falhou: %s — usando speaker único", exc)
            turns = [{"start": 0.0, "end": duration, "raw_speaker": "SPEAKER_00"}]

        # Passo 3 — Mapeia labels brutos para IDs estáveis via embeddings
        speaker_map = self._map_speakers(waveform, turns)

        # Passo 4 — Merge por overlap temporal
        merged = self._merge(segments, turns, speaker_map)

        elapsed = time.time() - t0
        rtf = elapsed / duration if duration > 0 else 0
        log.info(
            "Processado %.2fs de áudio em %.2fs (RTF=%.3f) | idioma=%s | speakers=%d",
            duration, elapsed, rtf, language, len(speaker_map),
        )

        return TranscriptionResponse(
            timestamp=datetime.now(),
            language=language,
            duration=duration,
            segments=[
                TranscriptionSegment(
                    speaker=s["user_id"],
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    confidence=s["confidence"],
                )
                for s in merged
            ],
        )

    def _map_speakers(self, waveform: dict, turns: List[Dict]) -> Dict[str, str]:
        """Mapeia raw_speaker para IDs estáveis usando embedding de voz."""
        by_speaker: Dict[str, List[Dict]] = {}
        for t in turns:
            by_speaker.setdefault(t["raw_speaker"], []).append(t)

        mapping: Dict[str, str] = {}
        diarizer = _get_diarizer()
        indexer = _get_indexer()

        for raw, turn_list in by_speaker.items():
            best = max(turn_list, key=lambda x: x["end"] - x["start"])
            try:
                emb = diarizer.embed_segment(waveform, best["start"], best["end"])
                uid = indexer.identify(emb)
            except Exception as exc:
                log.warning("Embedding falhou para '%s': %s", raw, exc)
                uid = f"Speaker Unknown {raw}"
            mapping[raw] = uid

        return mapping

    def _merge(
        self,
        segments: List[Dict],
        turns: List[Dict],
        speaker_map: Dict[str, str],
    ) -> List[Dict]:
        out = []
        for seg in segments:
            if not seg["text"] or _is_hallucination(seg["text"]):
                continue
            uid = self._dominant_speaker(seg["start"], seg["end"], turns, speaker_map)
            out.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "user_id": uid,
                "text": seg["text"],
                "confidence": round(seg["confidence"], 4),
            })
        return out

    @staticmethod
    def _dominant_speaker(
        start: float,
        end: float,
        turns: List[Dict],
        speaker_map: Dict[str, str],
    ) -> str:
        overlap: Dict[str, float] = {}
        for t in turns:
            ov = max(0.0, min(end, t["end"]) - max(start, t["start"]))
            if ov > 0:
                uid = speaker_map.get(t["raw_speaker"], t["raw_speaker"])
                overlap[uid] = overlap.get(uid, 0.0) + ov
        return max(overlap, key=overlap.get) if overlap else "Speaker Unknown"


processor = TranscriptionProcessor()
