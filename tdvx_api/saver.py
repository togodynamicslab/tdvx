"""
tdvx_api/saver.py
=================
Responsável por persistir dados de transcrição + diarização para finetuning.

Estrutura gerada em `finetuning_data/`:

    finetuning_data/
    ├── audio/
    │   └── {session_id}/
    │       └── {speaker}_{start:.3f}_{end:.3f}.wav   ← segmento de áudio por fala
    ├── labels/
    │   └── {session_id}.json                          ← transcrição completa da sessão
    └── manifest.jsonl                                 ← uma linha JSON por segmento
                                                          (formato HuggingFace/NeMo/Whisper)

Formato de cada linha do manifest.jsonl:
    {
      "id": "abc123_Speaker0_1.200_3.800",
      "audio_filepath": "finetuning_data/audio/abc123/Speaker0_1.200_3.800.wav",
      "text": "Olá, como vai?",
      "speaker": "Speaker 0",
      "duration": 2.6,
      "confidence": 0.94,
      "language": "pt",
      "source": "entrevista.mp3",
      "session_id": "abc123",
      "timestamp": "2025-03-17T14:30:00"
    }
"""
from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger("tdvx_api.saver")


class FinetuningDatasetSaver:
    """
    Salva pares (segmento_de_áudio, label) no disco para finetuning.

    Cada segmento de áudio é cortado do array original nos timestamps
    fornecidos pela diarização — isso cria exemplos de treino isolados
    por speaker/fala, que é o formato ideal para ASR e diarização finetuning.
    """

    SAMPLE_RATE = 16000

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "audio"
        self.labels_dir = self.base_dir / "labels"
        self.manifest_path = self.base_dir / "manifest.jsonl"

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    # ── API pública ────────────────────────────────────────────────────────────

    def save(
        self,
        audio: np.ndarray,
        result: Any,            # TranscriptionResponse do tdvx
        source_name: str,
        original_path: str | None = None,
    ) -> list[dict]:
        """
        Salva todos os segmentos de uma transcrição no dataset.

        Retorna lista de dicts com metadados de cada entrada salva.
        """
        session_id = uuid.uuid4().hex[:12]
        session_audio_dir = self.audio_dir / session_id
        session_audio_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat()
        language = result.language or "unknown"
        duration_total = result.duration or (len(audio) / self.SAMPLE_RATE)

        saved_entries: list[dict] = []

        for seg in result.segments:
            # Corta o segmento do áudio original nos timestamps do Pyannote/Whisper
            wav_segment = self._slice_audio(audio, seg.start, seg.end)

            # Ignora segmentos muito curtos (< 0.3s) — ruído ou artefatos
            seg_duration = seg.end - seg.start
            if seg_duration < 0.3 or len(wav_segment) < int(0.3 * self.SAMPLE_RATE):
                logger.debug("Segmento ignorado (muito curto): %.3f–%.3f", seg.start, seg.end)
                continue

            # Nome de arquivo: Speaker0_1.200_3.800.wav
            speaker_safe = seg.speaker.replace(" ", "")
            wav_filename = f"{speaker_safe}_{seg.start:.3f}_{seg.end:.3f}.wav"
            wav_path = session_audio_dir / wav_filename

            self._write_wav(wav_segment, wav_path)

            entry_id = f"{session_id}_{speaker_safe}_{seg.start:.3f}_{seg.end:.3f}"

            # Caminho relativo ao base_dir para portabilidade
            rel_path = wav_path.relative_to(self.base_dir)

            entry = {
                "id": entry_id,
                "audio_filepath": str(rel_path).replace("\\", "/"),
                "text": seg.text,
                "speaker": seg.speaker,
                "duration": round(seg_duration, 4),
                "confidence": round(seg.confidence, 4),
                "language": language,
                "source": source_name,
                "session_id": session_id,
                "timestamp": timestamp,
            }
            if original_path:
                entry["original_path"] = original_path

            self._append_manifest(entry)
            saved_entries.append(entry)

        # Salva o label completo da sessão (transcrição inteira)
        self._save_session_label(
            session_id=session_id,
            result=result,
            source_name=source_name,
            original_path=original_path,
            duration_total=duration_total,
            timestamp=timestamp,
        )

        logger.info(
            "Sessão '%s' salva: %d segmentos de '%s'",
            session_id,
            len(saved_entries),
            source_name,
        )
        return saved_entries

    def stats(self) -> dict:
        """Retorna estatísticas do dataset acumulado."""
        entries = self._load_all_entries()
        if not entries:
            return {
                "total_entries": 0,
                "unique_speakers": 0,
                "unique_sources": 0,
                "total_duration_seconds": 0.0,
                "languages": {},
                "speakers": {},
            }

        speakers: dict[str, int] = defaultdict(int)
        sources: set[str] = set()
        languages: dict[str, int] = defaultdict(int)
        total_duration = 0.0

        for e in entries:
            speakers[e.get("speaker", "unknown")] += 1
            sources.add(e.get("source", "unknown"))
            languages[e.get("language", "unknown")] += 1
            total_duration += e.get("duration", 0.0)

        return {
            "total_entries": len(entries),
            "unique_speakers": len(speakers),
            "unique_sources": len(sources),
            "total_duration_seconds": round(total_duration, 2),
            "languages": dict(languages),
            "speakers": dict(speakers),
        }

    def remove_entry(self, entry_id: str) -> bool:
        """Remove uma entrada do manifest e apaga o arquivo WAV associado."""
        entries = self._load_all_entries()
        target = next((e for e in entries if e.get("id") == entry_id), None)

        if target is None:
            return False

        # Apaga o WAV
        wav_path = self.base_dir / target["audio_filepath"]
        if wav_path.exists():
            wav_path.unlink()
            logger.info("WAV removido: %s", wav_path)

        # Reescreve manifest sem a entrada removida
        remaining = [e for e in entries if e.get("id") != entry_id]
        self._write_manifest(remaining)
        return True

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _slice_audio(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        """Fatia o array de áudio pelos timestamps em segundos."""
        start_idx = max(0, int(start * self.SAMPLE_RATE))
        end_idx = min(len(audio), int(end * self.SAMPLE_RATE))
        return audio[start_idx:end_idx]

    def _write_wav(self, audio: np.ndarray, path: Path) -> None:
        """Escreve segmento de áudio como WAV 16kHz mono float32."""
        sf.write(str(path), audio, samplerate=self.SAMPLE_RATE, subtype="PCM_16")

    def _append_manifest(self, entry: dict) -> None:
        """Adiciona uma entrada ao manifest.jsonl (append)."""
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _write_manifest(self, entries: list[dict]) -> None:
        """Reescreve o manifest.jsonl completo (usado ao deletar entradas)."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load_all_entries(self) -> list[dict]:
        """Lê todas as entradas do manifest.jsonl."""
        if not self.manifest_path.exists():
            return []
        entries = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Linha inválida no manifest ignorada: %s", line[:80])
        return entries

    def _save_session_label(
        self,
        session_id: str,
        result: Any,
        source_name: str,
        original_path: str | None,
        duration_total: float,
        timestamp: str,
    ) -> None:
        """Salva o JSON completo da sessão em labels/{session_id}.json."""
        label_path = self.labels_dir / f"{session_id}.json"
        payload = {
            "session_id": session_id,
            "source": source_name,
            "original_path": original_path,
            "timestamp": timestamp,
            "language": result.language,
            "duration": round(duration_total, 4),
            "segments": [
                {
                    "speaker": seg.speaker,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "confidence": seg.confidence,
                }
                for seg in result.segments
            ],
        }
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
