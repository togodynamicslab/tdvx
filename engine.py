#!/usr/bin/env python3
"""
engine.py — Motor de transcrição com diarização
================================================
faster-whisper (CTranslate2) + NVIDIA NeMo SortFormer

Regras fixas:
  - task="transcribe" SEMPRE — nunca traduzir
  - language="pt" fixo — sem etapa de identificação de idioma
  - VAD + limiares anti-alucinação agressivos para silêncio/ruído
  - Diarização via SortFormer (nvidia/diar_sortformer_4spk-v1)
"""
from __future__ import annotations

import ast
import os
import zipfile
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Estruturas de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)

    def __str__(self) -> str:
        ts = f"[{_fmt(self.start)} → {_fmt(self.end)}]"
        return f"{self.speaker} {ts}  {self.text.strip()}"


@dataclass
class TranscriptionResult:
    segments: List[Segment]
    language: str
    duration: float

    @property
    def text(self) -> str:
        return "\n".join(str(s) for s in self.segments)

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 2),
            "segments": [
                {
                    "speaker": s.speaker,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text.strip(),
                }
                for s in self.segments
            ],
        }


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _read_hf_cached_token() -> str:
    """Lê o token salvo por `huggingface-cli login` ou pelo hf_hub."""
    candidates = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    for p in candidates:
        if p.exists():
            return p.read_text().strip()
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token() or ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Motor principal
# ─────────────────────────────────────────────────────────────────────────────

_SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"


class TranscriptionEngine:
    """
    Parâmetros:
        model_path      Caminho para o diretório CTranslate2 OU para o .zip do modelo.
        sortformer_name Modelo SortFormer NeMo (padrão: nvidia/diar_sortformer_4spk-v1).
        device          "cpu" | "cuda" | "mps" | "auto"
        compute_type    "int8" (padrão) | "int8_float16" | "float16"
        beam_size       Tamanho do feixe de busca (5 = boa qualidade).
    """

    def __init__(
        self,
        model_path: str,
        sortformer_name: str = _SORTFORMER_MODEL,
        device: str = "auto",
        compute_type: str = "int8",
        beam_size: int = 5,
    ) -> None:
        self._model_dir       = self._resolve_model(Path(model_path))
        self._sortformer_name = sortformer_name
        self._device          = self._pick_device(device)
        self._compute         = compute_type
        self._beam_size       = beam_size

        self._whisper    = None   # lazy load
        self._diarizer   = None   # lazy load

    # ── Inicialização lazy ────────────────────────────────────────────────────

    def _load_whisper(self):
        if self._whisper is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("pip install faster-whisper")

        log.info("Carregando faster-whisper de: %s (device=%s, compute=%s)",
                 self._model_dir, self._device, self._compute)
        self._whisper = WhisperModel(
            str(self._model_dir),
            device=self._device,
            compute_type=self._compute,
        )

    def _load_diarizer(self):
        if self._diarizer is not None:
            return
        try:
            from nemo.collections.asr.models.sortformer_diar_models import (
                SortformerEncLabelModel,
            )
        except ImportError:
            raise ImportError("pip install nemo_toolkit[asr]")

        log.info("Carregando SortFormer: %s ...", self._sortformer_name)
        try:
            self._diarizer = SortformerEncLabelModel.from_pretrained(
                self._sortformer_name
            )
            self._diarizer.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao carregar SortFormer ({exc}).\n"
                "Verifique: pip install nemo_toolkit[asr]"
            ) from exc

    # ── API pública ───────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> TranscriptionResult:
        """
        Transcreve um arquivo de áudio com diarização de falantes.

        Args:
            audio_path    Caminho para o arquivo de áudio.
            num_speakers  Número exato de falantes (None = auto).
            min_speakers  Mínimo de falantes esperado.
            max_speakers  Máximo de falantes esperado.

        Returns:
            TranscriptionResult com segmentos atribuídos a cada falante.
        """
        self._load_whisper()
        self._load_diarizer()

        # ── 1. Diarização ─────────────────────────────────────────────────────
        log.info("Diarizando: %s", audio_path)
        speaker_turns = _run_sortformer(
            self._diarizer,
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        log.info("Diarização: %d segmentos de voz detectados", len(speaker_turns))

        # ── 2. Transcrição com timestamps por palavra ─────────────────────────
        log.info("Transcrevendo: %s", audio_path)
        raw_segments, info = self._whisper.transcribe(
            audio_path,
            task="transcribe",
            language="pt",          # sem etapa de identificação; sem tradução
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            beam_size=self._beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                speech_pad_ms=150,
            ),
        )

        # Materializa o gerador
        words: List[Word] = []
        for seg in raw_segments:
            if seg.words:
                for w in seg.words:
                    # Descarta palavras com probabilidade muito baixa (ruído)
                    if w.probability >= 0.4:
                        words.append(Word(
                            start=w.start,
                            end=w.end,
                            text=w.word,
                            probability=w.probability,
                        ))

        log.info(
            "Transcrição: idioma=%s | %.1fs | %d palavras válidas",
            info.language, info.duration, len(words),
        )

        # ── 3. Atribuição palavra → falante ───────────────────────────────────
        segments = _assign_words_to_speakers(words, speaker_turns)

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            duration=info.duration,
        )

    def transcribe_only(self, audio_path: str) -> TranscriptionResult:
        """Transcreve sem diarização (falante único 'SPK_00')."""
        self._load_whisper()

        raw_segments, info = self._whisper.transcribe(
            audio_path,
            task="transcribe",
            language="pt",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            beam_size=self._beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                speech_pad_ms=150,
            ),
        )

        segments = []
        for seg in raw_segments:
            words = [
                Word(w.start, w.end, w.word, w.probability)
                for w in (seg.words or [])
                if w.probability >= 0.4
            ]
            if not words:
                continue
            segments.append(Segment(
                speaker="SPK_00",
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=words,
            ))

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            duration=info.duration,
        )

    # ── Helpers internos ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_model(path: Path) -> Path:
        """Extrai o zip se necessário e retorna o diretório do modelo CT2."""
        if path.is_dir():
            _check_ct2_dir(path)
            return path

        if path.suffix == ".zip":
            dest = path.parent / path.stem.split("-ct2")[0].split("_ct2")[0]
            # Procura pasta ct2 dentro do zip
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                roots = {n.split("/")[0] for n in names if "/" in n}
                ct2_root = next(
                    (r for r in roots if "ct2" in r.lower()),
                    next(iter(roots), None),
                )
                if ct2_root is None:
                    raise ValueError(f"Zip inválido: nenhuma pasta encontrada em {path}")

                dest = path.parent / ct2_root
                if not dest.is_dir():
                    log.info("Extraindo modelo de %s → %s", path.name, dest)
                    zf.extractall(path.parent)
                    log.info("Extração concluída.")
                else:
                    log.info("Modelo já extraído: %s", dest)

            _check_ct2_dir(dest)
            return dest

        raise FileNotFoundError(
            f"model_path deve ser um diretório CTranslate2 ou um .zip.\n"
            f"Recebido: {path}"
        )

    @staticmethod
    def _pick_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        # faster-whisper (CTranslate2) não suporta mps — usa cpu
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# SortFormer — diarização via NeMo 2.x (SortformerEncLabelModel)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sortformer(
    model,
    audio_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Tuple[float, float, str]]:
    """
    Executa o SortFormer num arquivo de áudio e retorna lista de
    (start, end, speaker_label) ordenada por tempo.

    O NeMo 2.x retorna List[List[str]] onde cada string tem formato
    "[begin_seconds, end_seconds, speaker_index]".
    SortFormer é end-to-end: determina falantes automaticamente.
    max_speakers limita via DiarizeConfig.max_num_of_spks.
    """
    from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig

    # SortFormer não aceita oracle_num_speakers; apenas max é suportado
    max_spk = max_speakers or num_speakers  # se usuário deu exato, usa como máximo
    cfg = DiarizeConfig(
        max_num_of_spks=max_spk,
        verbose=False,
        num_workers=0,
    )

    # model.diarize aceita path direto ou lista de paths
    results = model.diarize(
        audio=audio_path,
        batch_size=1,
        override_config=cfg,
    )

    # results é List[List[str]]; o primeiro item corresponde ao único arquivo
    raw_turns: List[str] = results[0] if results else []
    return _parse_sortformer_output(raw_turns)


def _parse_sortformer_output(turns: List[str]) -> List[Tuple[float, float, str]]:
    """
    Converte a saída do SortformerEncLabelModel para (start, end, speaker_label).
    NeMo 2.x retorna strings no formato: "start_sec end_sec speaker_label"
    Ex: "0.000 4.560 speaker_0"
    """
    parsed = []
    for entry in turns:
        parts = entry.strip().split()
        if len(parts) == 3:
            try:
                start = float(parts[0])
                end   = float(parts[1])
                label = parts[2].upper().replace("SPEAKER_", "SPK_")
                parsed.append((start, end, label))
                continue
            except ValueError:
                pass
        # Fallback: tenta como lista Python "[start, end, idx]"
        try:
            start, end, spk_idx = ast.literal_eval(entry)
            label = f"SPK_{int(spk_idx):02d}"
            parsed.append((float(start), float(end), label))
        except Exception:
            log.warning("SortFormer: entrada ignorada: %r", entry)
    parsed.sort(key=lambda t: t[0])
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Atribuição de falantes por sobreposição temporal de palavras
# ─────────────────────────────────────────────────────────────────────────────

def _speaker_at(t_start: float, t_end: float, turns: list) -> str:
    """Retorna o falante com maior sobreposição no intervalo [t_start, t_end]."""
    best_label   = "SPK_UNK"
    best_overlap = 0.0
    for (s, e, label) in turns:
        overlap = max(0.0, min(t_end, e) - max(t_start, s))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label   = label
    return best_label


def _assign_words_to_speakers(words: List[Word], turns: list) -> List[Segment]:
    """
    Agrupa palavras em segmentos por falante.
    Usa sobreposição de timestamp para decidir a quem cada palavra pertence.
    """
    if not words:
        return []

    segments: List[Segment] = []
    current_speaker = _speaker_at(words[0].start, words[0].end, turns)
    current_words: List[Word] = []

    for w in words:
        speaker = _speaker_at(w.start, w.end, turns)
        if speaker != current_speaker and current_words:
            segments.append(_make_segment(current_speaker, current_words))
            current_speaker = speaker
            current_words   = []
        current_words.append(w)

    if current_words:
        segments.append(_make_segment(current_speaker, current_words))

    return segments


def _make_segment(speaker: str, words: List[Word]) -> Segment:
    text = "".join(w.text for w in words)
    return Segment(
        speaker=speaker,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=words,
    )


def _check_ct2_dir(path: Path) -> None:
    required = {"model.bin", "config.json", "vocabulary.json"}
    found = {f.name for f in path.iterdir()} if path.is_dir() else set()
    missing = required - found
    if missing:
        raise FileNotFoundError(
            f"Diretório CTranslate2 incompleto em '{path}'.\n"
            f"Arquivos faltando: {missing}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI simples para teste
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    _MODEL_DEFAULT = str(
        Path(__file__).parent / "models" /
        "tdv3-cv-pt-v1-ct2-20260508T184337Z-3-001.zip"
    )

    p = argparse.ArgumentParser(description="Motor de transcrição TDvX")
    p.add_argument("audio",        help="Arquivo de áudio a transcrever")
    p.add_argument("--model",      default=_MODEL_DEFAULT, help="Caminho do modelo CT2 ou zip")
    p.add_argument("--sortformer", default=_SORTFORMER_MODEL, help="Modelo SortFormer NeMo")
    p.add_argument("--device",     default="auto", choices=["auto","cpu","cuda","mps"])
    p.add_argument("--compute",    default="int8", choices=["int8","int8_float16","float16"])
    p.add_argument("--speakers",   type=int, default=None, help="Número exato de falantes")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--no-diarize", action="store_true", help="Pula diarização")
    p.add_argument("--json",       action="store_true", help="Saída em JSON")
    args = p.parse_args()

    engine = TranscriptionEngine(
        model_path=args.model,
        sortformer_name=args.sortformer,
        device=args.device,
        compute_type=args.compute,
    )

    if args.no_diarize:
        result = engine.transcribe_only(args.audio)
    else:
        result = engine.transcribe(
            args.audio,
            num_speakers=args.speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )

    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"\nIdioma detectado: {result.language}")
        print(f"Duração: {result.duration:.1f}s\n")
        print(result.text)
