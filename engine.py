#!/usr/bin/env python3
"""
engine.py — Motor de transcrição com diarização
================================================
CTranslate2 direto + NVIDIA NeMo SortFormer

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
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_SAMPLE_RATE  = 16000
_CHUNK_SECS   = 30
_FRAME_MS     = 20   # VAD: tamanho de frame em ms


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


# ─────────────────────────────────────────────────────────────────────────────
# Motor principal
# ─────────────────────────────────────────────────────────────────────────────

_SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"
_MODEL_DEFAULT    = str(Path(__file__).parent / "models" / "tdv3-cv-pt-v1-ct2")


class TranscriptionEngine:
    """
    Parâmetros:
        model_path      Diretório CTranslate2 do modelo (tdv3-cv-pt-v1-ct2).
        sortformer_name Modelo SortFormer NeMo.
        device          "cpu" | "cuda" | "auto"
        compute_type    "int8" (padrão) | "int8_float16" | "float16"
        beam_size       Tamanho do feixe de busca.
    """

    def __init__(
        self,
        model_path: str = _MODEL_DEFAULT,
        sortformer_name: str = _SORTFORMER_MODEL,
        device: str = "auto",
        compute_type: str = "int8",
        beam_size: int = 5,
    ) -> None:
        self._model_dir       = _resolve_model(Path(model_path))
        self._sortformer_name = sortformer_name
        self._device          = _pick_device(device)
        self._compute         = compute_type
        self._beam_size       = beam_size

        self._ct2      = None   # ctranslate2.models.Whisper — lazy
        self._fe       = None   # WhisperFeatureExtractor     — lazy
        self._tok      = None   # WhisperTokenizer            — lazy
        self._ts_map   = None   # {token_id: seconds}         — lazy
        self._diarizer = None   # SortformerEncLabelModel     — lazy

    # ── Inicialização lazy ────────────────────────────────────────────────────

    def _load_whisper(self):
        if self._ct2 is not None:
            return

        import ctranslate2
        from transformers import WhisperFeatureExtractor, WhisperTokenizer

        log.info("Carregando CTranslate2 de: %s  (device=%s  compute=%s)",
                 self._model_dir, self._device, self._compute)

        self._ct2 = ctranslate2.models.Whisper(
            str(self._model_dir),
            device=self._device,
            compute_type=self._compute,
        )
        # O feature extractor e tokenizer base do whisper-medium
        # compartilham o mesmo vocabulário do modelo fine-tunado
        self._fe  = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
        self._tok = WhisperTokenizer.from_pretrained("openai/whisper-medium")

        # Monta mapa {token_id → segundos} para tokens de timestamp
        vocab = self._tok.get_vocab()
        self._ts_map: Dict[int, float] = {}
        for token, tid in vocab.items():
            if token.startswith("<|") and token.endswith("|>"):
                try:
                    self._ts_map[tid] = float(token[2:-2])
                except ValueError:
                    pass

        # Prompt fixo: PT, transcribe — sem detecção de idioma
        self._prompt_ids = self._tok.convert_tokens_to_ids([
            "<|startoftranscript|>", "<|pt|>", "<|transcribe|>",
        ])
        log.info("Modelo pronto.  Prompt: %s", self._prompt_ids)

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
        """Transcreve com diarização SortFormer."""
        self._load_whisper()
        self._load_diarizer()

        audio, duration = _load_audio(audio_path)

        # ── 1. Diarização ─────────────────────────────────────────────────────
        log.info("Diarizando: %s", audio_path)
        speaker_turns = _run_sortformer(
            self._diarizer, audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        log.info("Diarização: %d turnos detectados", len(speaker_turns))

        # ── 2. Transcrição ────────────────────────────────────────────────────
        log.info("Transcrevendo: %s", audio_path)
        words = self._transcribe_audio(audio, duration)
        log.info("Transcrição: %d palavras válidas  (%.1fs)", len(words), duration)

        # ── 3. Atribuição falante ─────────────────────────────────────────────
        segments = _assign_words_to_speakers(words, speaker_turns)
        return TranscriptionResult(segments=segments, language="pt", duration=duration)

    def transcribe_only(self, audio_path: str) -> TranscriptionResult:
        """Transcreve sem diarização (falante único SPK_00)."""
        self._load_whisper()

        audio, duration = _load_audio(audio_path)
        words = self._transcribe_audio(audio, duration)
        log.info("Transcrição: %d palavras  (%.1fs)", len(words), duration)

        segments = _words_to_single_speaker(words)
        return TranscriptionResult(segments=segments, language="pt", duration=duration)

    # ── Transcrição interna ───────────────────────────────────────────────────

    def _transcribe_audio(self, audio: np.ndarray, duration: float) -> List[Word]:
        """Aplica VAD e transcreve em chunks de 30s via CTranslate2."""
        speech_segs = _vad_segments(audio)
        if not speech_segs:
            log.info("VAD: nenhuma fala detectada.")
            return []

        removed = duration - sum(e - s for s, e in speech_segs)
        log.info("VAD: %.3fs de não-fala removidos", removed)

        words: List[Word] = []
        chunk = _CHUNK_SECS * _SAMPLE_RATE

        for seg_start, seg_end in speech_segs:
            i0, i1 = int(seg_start * _SAMPLE_RATE), int(seg_end * _SAMPLE_RATE)
            seg_audio = audio[i0:i1]

            # Divide em sub-chunks de 30s se necessário
            for sub_start in range(0, len(seg_audio), chunk):
                sub = seg_audio[sub_start:sub_start + chunk]
                time_offset = seg_start + sub_start / _SAMPLE_RATE
                chunk_words = self._transcribe_chunk(sub, time_offset)
                words.extend(chunk_words)

        return words

    def _transcribe_chunk(self, audio: np.ndarray, offset: float) -> List[Word]:
        import ctranslate2

        # Pad até 30s (requisito do encoder Whisper)
        target = _CHUNK_SECS * _SAMPLE_RATE
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]

        features = self._fe(
            audio.astype(np.float32),
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
        )
        storage = ctranslate2.StorageView.from_array(features.input_features)

        # Prompt sem <|notimestamps|> → o modelo gera tokens de timestamp
        # no_speech_prob e scores são checados manualmente (ctranslate2 não tem
        # os parâmetros de alto nível do faster-whisper)
        # Filtro de idioma — só processa PT; qualquer outra língua é descartada
        # ctranslate2 retorna tokens no formato "<|pt|>" — normaliza para "pt"
        lang_results = self._ct2.detect_language(storage)
        raw_lang, best_prob = lang_results[0][0]
        best_lang = raw_lang.strip("<|>")
        if best_lang != "pt":
            log.info("Chunk ignorado: idioma=%s (prob=%.2f) — só PT é processado",
                     best_lang, best_prob)
            return []

        result = self._ct2.generate(
            storage,
            [self._prompt_ids],
            beam_size=self._beam_size,
            return_no_speech_prob=True,
            return_scores=True,
            max_initial_timestamp_index=50,
        )[0]

        if result.no_speech_prob > 0.6:
            return []
        if result.scores and result.scores[0] < -1.0:
            return []

        return _parse_timestamp_tokens(
            result.sequences_ids[0], self._tok, self._ts_map, offset
        )


# ─────────────────────────────────────────────────────────────────────────────
# Áudio — carregamento e VAD por energia
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio(path: str) -> Tuple[np.ndarray, float]:
    import librosa
    audio, _ = librosa.load(path, sr=_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), len(audio) / _SAMPLE_RATE


def _vad_segments(
    audio: np.ndarray,
    threshold_db: float = -38.0,
    min_speech_ms: int  = 250,
    min_silence_ms: int = 500,
    pad_ms: int         = 150,
) -> List[Tuple[float, float]]:
    """
    VAD por energia RMS (frame a frame).
    Retorna lista de (start_s, end_s) com fala detectada.
    """
    frame_len = int(_SAMPLE_RATE * _FRAME_MS / 1000)
    n_frames  = len(audio) // frame_len

    is_speech = []
    for i in range(n_frames):
        frame = audio[i * frame_len:(i + 1) * frame_len]
        rms   = np.sqrt(np.mean(frame ** 2) + 1e-10)
        db    = 20.0 * np.log10(rms)
        is_speech.append(db > threshold_db)

    # Converte sequência booleana → intervalos
    raw: List[Tuple[float, float]] = []
    in_speech = False
    t_start   = 0.0
    for i, speech in enumerate(is_speech):
        t = i * _FRAME_MS / 1000
        if speech and not in_speech:
            t_start   = t
            in_speech = True
        elif not speech and in_speech:
            if (t - t_start) * 1000 >= min_speech_ms:
                raw.append((t_start, t))
            in_speech = False
    if in_speech:
        raw.append((t_start, len(audio) / _SAMPLE_RATE))

    # Funde segmentos próximos
    merged: List[Tuple[float, float]] = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) * 1000 < min_silence_ms:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Padding
    pad = pad_ms / 1000
    dur = len(audio) / _SAMPLE_RATE
    return [(max(0.0, s - pad), min(dur, e + pad)) for s, e in merged]


# ─────────────────────────────────────────────────────────────────────────────
# Parser de tokens de timestamp do Whisper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_timestamp_tokens(
    token_ids: List[int],
    tokenizer,
    ts_map: Dict[int, float],
    offset: float,
) -> List[Word]:
    """
    Converte a sequência de tokens (texto + timestamps) em Words com timing.
    Whisper gera: <|t0|> texto <|t1|> texto ...
    """
    words: List[Word] = []
    current_start: Optional[float] = None
    current_ids:   List[int]       = []
    all_special    = set(tokenizer.all_special_ids)

    for tid in token_ids:
        if tid in ts_map:
            t = ts_map[tid] + offset
            if current_start is not None and current_ids:
                text = tokenizer.decode(current_ids, skip_special_tokens=True).strip()
                if text:
                    words.append(Word(start=current_start, end=t, text=" " + text))
                current_ids = []
            current_start = t
        elif tid not in all_special:
            if current_start is not None:
                current_ids.append(tid)

    # Último segmento sem timestamp final
    if current_start is not None and current_ids:
        text = tokenizer.decode(current_ids, skip_special_tokens=True).strip()
        if text:
            end = current_start + max(len(text) * 0.07, 0.1)
            words.append(Word(start=current_start, end=end, text=" " + text))

    return words


# ─────────────────────────────────────────────────────────────────────────────
# SortFormer — diarização via NeMo 2.x
# ─────────────────────────────────────────────────────────────────────────────

def _run_sortformer(
    model,
    audio_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Tuple[float, float, str]]:
    from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig

    # DiarizeConfig só expõe max; min não é suportado pelo SortFormer
    if min_speakers is not None:
        log.warning("SortFormer: min_speakers ignorado (não suportado pelo modelo)")
    max_spk = max_speakers or num_speakers
    cfg = DiarizeConfig(max_num_of_spks=max_spk, verbose=False, num_workers=0)

    results  = model.diarize(audio=audio_path, batch_size=1, override_config=cfg)
    raw: List[str] = results[0] if results else []
    return _parse_sortformer_output(raw)


def _parse_sortformer_output(turns: List[str]) -> List[Tuple[float, float, str]]:
    parsed = []
    for entry in turns:
        parts = entry.strip().split()
        if len(parts) == 3:
            try:
                label = parts[2].upper().replace("SPEAKER_", "SPK_")
                parsed.append((float(parts[0]), float(parts[1]), label))
                continue
            except ValueError:
                pass
        try:
            start, end, spk_idx = ast.literal_eval(entry)
            parsed.append((float(start), float(end), f"SPK_{int(spk_idx):02d}"))
        except Exception:
            log.warning("SortFormer: entrada ignorada: %r", entry)
    parsed.sort(key=lambda t: t[0])
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Atribuição falante → palavras
# ─────────────────────────────────────────────────────────────────────────────

def _speaker_at(t_start: float, t_end: float, turns: list) -> str:
    best_label, best_overlap = "SPK_UNK", 0.0
    for (s, e, label) in turns:
        overlap = max(0.0, min(t_end, e) - max(t_start, s))
        if overlap > best_overlap:
            best_overlap, best_label = overlap, label
    return best_label


def _assign_words_to_speakers(words: List[Word], turns: list) -> List[Segment]:
    if not words:
        return []
    segments: List[Segment] = []
    cur_spk   = _speaker_at(words[0].start, words[0].end, turns)
    cur_words: List[Word] = []
    for w in words:
        spk = _speaker_at(w.start, w.end, turns)
        if spk != cur_spk and cur_words:
            segments.append(_make_segment(cur_spk, cur_words))
            cur_spk, cur_words = spk, []
        cur_words.append(w)
    if cur_words:
        segments.append(_make_segment(cur_spk, cur_words))
    return segments


def _words_to_single_speaker(words: List[Word]) -> List[Segment]:
    if not words:
        return []
    # Agrupa por pausas > 1s
    segments: List[Segment] = []
    cur: List[Word] = [words[0]]
    for w in words[1:]:
        if w.start - cur[-1].end > 1.0:
            segments.append(_make_segment("SPK_00", cur))
            cur = []
        cur.append(w)
    if cur:
        segments.append(_make_segment("SPK_00", cur))
    return segments


def _make_segment(speaker: str, words: List[Word]) -> Segment:
    return Segment(
        speaker=speaker,
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.text for w in words),
        words=words,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model(path: Path) -> Path:
    """Aceita diretório CT2 ou .zip; extrai se necessário."""
    if path.is_dir():
        _check_ct2_dir(path)
        return path
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            roots = {n.split("/")[0] for n in zf.namelist() if "/" in n}
            ct2_root = next((r for r in roots if "ct2" in r.lower()), next(iter(roots), None))
            if ct2_root is None:
                raise ValueError(f"Zip inválido: {path}")
            dest = path.parent / ct2_root
            if not dest.is_dir():
                log.info("Extraindo %s → %s", path.name, dest)
                zf.extractall(path.parent)
        _check_ct2_dir(dest)
        return dest
    raise FileNotFoundError(f"model_path inválido: {path}")


def _check_ct2_dir(path: Path) -> None:
    missing = {"model.bin", "config.json", "vocabulary.json"} - {f.name for f in path.iterdir()}
    if missing:
        raise FileNotFoundError(f"Modelo incompleto em '{path}': faltam {missing}")


def _pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Motor de transcrição TDvX")
    p.add_argument("audio")
    p.add_argument("--model",        default=_MODEL_DEFAULT)
    p.add_argument("--sortformer",   default=_SORTFORMER_MODEL)
    p.add_argument("--device",       default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute",      default="int8", choices=["int8", "int8_float16", "float16"])
    p.add_argument("--speakers",     type=int, default=None)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--no-diarize",   action="store_true")
    p.add_argument("--json",         action="store_true")
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
        print(f"\nIdioma: {result.language}  |  Duração: {result.duration:.1f}s\n")
        print(result.text)
