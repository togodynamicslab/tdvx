#!/usr/bin/env python3
"""
stt_pipeline.py — Local Speech-to-Text with Speaker Diarization
================================================================
Stack:
  • faster-whisper  (CTranslate2)   → int8 / int8_float16 quantisation
  • pyannote.audio  3.1             → diarisation + speaker embeddings
  • Optimised for bilingual PT-BR / English audio

Prerequisites (see install instructions at the bottom of this file):
  1. pip install the dependencies listed in requirements section
  2. Accept the pyannote model terms on HuggingFace:
       → https://hf.co/pyannote/speaker-diarization-3.1
       → https://hf.co/pyannote/embedding
  3. Export your HF token: export HF_TOKEN=hf_xxxxxxxx
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Load .env first, then apply offline/telemetry settings ───────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Local models/ folder — keep models alongside the code ────────────────────
# If HF_HOME is not already set (e.g. by .env), point it to the models/
# sub-directory next to this file so all HuggingFace models are stored locally.
_project_models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ.setdefault("HF_HOME", _project_models)

# Disable ALL HuggingFace Hub telemetry — no usage data is ever sent
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYANNOTE_AUDIO_TELEMETRY",  "0")

# If HF_HUB_OFFLINE=1 is set in .env, block every network call to HF Hub.
# Models must already be cached locally (run once with internet, then flip to 1).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torchaudio

# ── Compatibility patches ─────────────────────────────────────────────────────
# Must run BEFORE any pyannote / speechbrain import.
#
# 1. torchaudio >= 2.1 removed the audio-backend management API entirely.
#    speechbrain still calls these three functions at module init time,
#    even in versions that updated their own pyannote compat.
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# 2. numpy >= 2.0 removed np.NaN (renamed to np.nan).
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# 3. torchaudio >= 2.5 removed AudioMetaData from the top-level namespace.
#    pyannote/audio/core/io.py uses it as a type annotation.
if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio._torchaudio import AudioMetaData as _AudioMetaData
        torchaudio.AudioMetaData = _AudioMetaData
    except (ImportError, AttributeError):
        try:
            from torchaudio.backend.common import AudioMetaData as _AudioMetaData
            torchaudio.AudioMetaData = _AudioMetaData
        except (ImportError, AttributeError):
            from collections import namedtuple
            torchaudio.AudioMetaData = namedtuple(
                "AudioMetaData",
                ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
            )

# 4. PyTorch 2.6 changed torch.load default from weights_only=False to True.
#    pyannote model checkpoints contain many custom Python objects (TorchVersion,
#    Specifications, etc.) that are not in the safe-globals allowlist.
#    Restore weights_only=False as the default so pyannote loads normally.
#    This only affects calls that don't explicitly pass weights_only.
_orig_torch_load = torch.load

def _compat_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False   # force — lightning_fabric passes True explicitly
    return _orig_torch_load(*args, **kwargs)

torch.load = _compat_torch_load

# 5. huggingface_hub >= 0.22 removed `use_auth_token` from hf_hub_download.
#    pyannote.audio 3.x (including 3.4) still passes it internally.
#    Patch hf_hub_download BEFORE pyannote imports it so the shim is in place.
import huggingface_hub as _hf_hub
import huggingface_hub.file_download as _hf_fd
_orig_hf_hub_download = _hf_hub.hf_hub_download

def _compat_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs.setdefault("token", kwargs.pop("use_auth_token"))
    return _orig_hf_hub_download(*args, **kwargs)

_hf_hub.hf_hub_download = _compat_hf_hub_download
_hf_fd.hf_hub_download  = _compat_hf_hub_download
# ─────────────────────────────────────────────────────────────────────────────

from scipy.spatial.distance import cosine
from faster_whisper import WhisperModel
from pyannote.audio import Inference, Model, Pipeline

# ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stt_pipeline")


# ══════════════════════════════════════════════════════
# 1. Configuration
# ══════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """
    Central configuration — every tunable knob lives here.
    Pass an instance of this to AudioTranscriber.
    """

    # ── Whisper (faster-whisper / CTranslate2) ───────────────────────
    whisper_model: str = "medium"
    # "auto" detects GPU/CPU; override with "cuda" or "cpu"
    device: str = "auto"
    # "auto" picks int8_float16 on GPU, int8 on CPU — never float32
    compute_type: str = "auto"
    cpu_threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    num_workers: int = 2

    # Bilingual PT-BR / EN initial prompt.
    # Helps the model handle English technical terms inside Portuguese speech
    # without hallucinating or transcribing them phonetically.
    initial_prompt: str = (
        "Transcrição de reunião técnica em português do Brasil com termos técnicos "
        "em inglês. Vocabulário comum: API, endpoint, backend, frontend, deploy, "
        "pipeline, sprint, commit, pull request, machine learning, dataset, "
        "framework, bug, feature, refactor, merge, branch, container, Kubernetes, "
        "Docker, microservice, CI/CD, DevOps, SLA, SLO, latência."
    )

    # ── Voice Activity Detection (VAD) ───────────────────────────────
    vad_filter: bool = True
    # Minimum silence in ms to split a segment — increase to keep long segments
    vad_min_silence_duration_ms: int = 500

    # ── Diarisation (pyannote.audio 3.1) ─────────────────────────────
    # REQUIRED: your HuggingFace access token.
    # Set via env var HF_TOKEN or pass directly here.
    hf_token: str = field(default="")
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/embedding"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # ── Speaker Re-identification ─────────────────────────────────────
    # Cosine similarity threshold for matching an incoming embedding to a
    # known speaker. Higher → stricter matching (fewer false positives).
    similarity_threshold: float = 0.80


# ══════════════════════════════════════════════════════
# 2. Output Data Structure
# ══════════════════════════════════════════════════════

@dataclass
class TranscriptionSegment:
    """A single transcribed + attributed speech segment."""
    timestamp_start: float
    timestamp_end: float
    user_id: str            # e.g. "Speaker 0", "Speaker 1"
    text: str
    language_detected: str  # ISO 639-1 code, e.g. "pt" or "en"
    confidence: float       # Whisper avg log-prob converted to probability


# ══════════════════════════════════════════════════════
# 3. STT Engine — faster-whisper + CTranslate2
# ══════════════════════════════════════════════════════

class STTEngine:
    """
    Loads Whisper via faster-whisper (CTranslate2 backend).

    Quantisation strategy (minimises memory, maximises throughput):
      • GPU (CUDA) → int8_float16   (~2× less VRAM than float16, near-lossless)
      • CPU        → int8           (fastest CPU path, ~4× less RAM than float32)
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.device, self.compute_type = self._resolve_backend()
        log.info(
            "Loading Whisper '%s' | device=%s | quantisation=%s",
            cfg.whisper_model, self.device, self.compute_type,
        )
        self.model = WhisperModel(
            cfg.whisper_model,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cfg.cpu_threads,
            num_workers=cfg.num_workers,
        )

    def _resolve_backend(self) -> Tuple[str, str]:
        device = self.cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute = self.cfg.compute_type
        if compute == "auto":
            compute = "int8_float16" if device == "cuda" else "int8"

        return device, compute

    def transcribe(self, audio_path: str) -> Tuple[List[Dict], str]:
        """
        Transcribe the audio file.

        Returns
        -------
        segments : list of {start, end, text, confidence}
        language : ISO 639-1 detected language code
        """
        vad_params: Optional[Dict] = (
            {"min_silence_duration_ms": self.cfg.vad_min_silence_duration_ms}
            if self.cfg.vad_filter
            else None
        )

        segments_iter, info = self.model.transcribe(
            audio_path,
            task="transcribe",
            initial_prompt=self.cfg.initial_prompt,
            vad_filter=self.cfg.vad_filter,
            vad_parameters=vad_params,
            beam_size=5,
            best_of=5,
            temperature=0.0,                  # greedy-ish, avoids hallucinations
            condition_on_previous_text=True,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            word_timestamps=False,
        )

        lang = info.language
        log.info("Language: %s  (p=%.2f)", lang, info.language_probability)

        segments: List[Dict] = []
        for seg in segments_iter:             # generator — forces decoding here
            segments.append({
                "start": seg.start,
                "end":   seg.end,
                "text":  seg.text.strip(),
                "confidence": float(np.exp(seg.avg_logprob))
                               if seg.avg_logprob is not None else 0.0,
            })

        log.info("Whisper produced %d segments", len(segments))
        return segments, lang


# ══════════════════════════════════════════════════════
# 4. Speaker Indexer — embedding registry + cosine re-ID
# ══════════════════════════════════════════════════════

class SpeakerIndexer:
    """
    Maintains a growing dictionary of known speaker voice embeddings.

    Algorithm
    ---------
    1. L2-normalise every incoming embedding.
    2. Compute cosine similarity against all stored centroids.
    3. If the nearest centroid exceeds `threshold` → same speaker (running
       average update of the centroid keeps it adaptive).
    4. Otherwise → register as a new speaker (Speaker 0, Speaker 1, …).
    """

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self._registry: Dict[str, np.ndarray] = {}  # user_id → centroid
        self._counter: int = 0

    # ── Public API ───────────────────────────────────────

    def identify(self, embedding: np.ndarray) -> str:
        """Return an existing user_id or register a new speaker."""
        emb = self._normalise(embedding)
        best_id, best_sim = self._nearest(emb)

        if best_id is not None and best_sim >= self.threshold:
            self._update_centroid(best_id, emb)
            return best_id

        user_id = f"Speaker {self._counter}"
        self._counter += 1
        self._registry[user_id] = emb.copy()
        log.info("New speaker registered → %s  (registry size: %d)",
                 user_id, self._counter)
        return user_id

    def export(self) -> Dict[str, List[float]]:
        """Return a JSON-serialisable copy of the registry."""
        return {uid: emb.tolist() for uid, emb in self._registry.items()}

    # ── Internals ────────────────────────────────────────

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v

    def _nearest(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._registry:
            return None, 0.0
        best_id, best_sim = None, -1.0
        for uid, ref in self._registry.items():
            sim = 1.0 - float(cosine(emb, ref))
            if sim > best_sim:
                best_sim, best_id = sim, uid
        return best_id, best_sim

    def _update_centroid(self, uid: str, emb: np.ndarray) -> None:
        """Exponential running average keeps the centroid adaptive."""
        self._registry[uid] = (self._registry[uid] + emb) / 2.0


# ══════════════════════════════════════════════════════
# 5. Diarisation Engine — pyannote.audio 3.1
# ══════════════════════════════════════════════════════

class DiarizationEngine:
    """
    Wraps pyannote/speaker-diarization-3.1.

    Responsibilities
    ----------------
    • diarize()        → speaker-turn timeline [{start, end, raw_speaker}]
    • embed_segment()  → 192-d speaker embedding for a given time window
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        token = cfg.hf_token or os.getenv("HF_TOKEN", "")

        if not token:
            raise ValueError(
                "\n[ERROR] A Hugging Face token is required.\n"
                "  Step 1 — Accept terms at:\n"
                "    https://hf.co/pyannote/speaker-diarization-3.1\n"
                "    https://hf.co/pyannote/embedding\n"
                "  Step 2 — Generate a token at https://hf.co/settings/tokens\n"
                "  Step 3 — Set it:\n"
                "    export HF_TOKEN=hf_xxxxxxxx           (Linux/macOS)\n"
                "    set    HF_TOKEN=hf_xxxxxxxx           (Windows CMD)\n"
                "    $env:HF_TOKEN='hf_xxxxxxxx'           (PowerShell)\n"
                "    or pass hf_token='hf_…' to PipelineConfig."
            )

        self.torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info("Diarisation device: %s", self.torch_device)

        log.info("HF token prefix: %s…", token[:12])

        # ── Speaker-diarization pipeline ─────────────────
        log.info("Loading %s …", cfg.diarization_model)
        _diar = Pipeline.from_pretrained(
            cfg.diarization_model, use_auth_token=token
        )
        if _diar is None:
            raise RuntimeError(
                "\n[ERRO] Não foi possível carregar o modelo de diarização.\n"
                "  Causa mais comum: termos de uso NÃO aceitos no HuggingFace.\n\n"
                "  1. Acesse e clique em 'Agree and access repository':\n"
                "       https://hf.co/pyannote/speaker-diarization-3.1\n"
                "       https://hf.co/pyannote/embedding\n\n"
                "  2. Confirme que o token tem permissão 'read':\n"
                "       https://hf.co/settings/tokens\n\n"
                f"  Token usado: {token[:12]}…"
            )
        self.diar_pipeline = _diar.to(self.torch_device)

        # ── Embedding model (for our SpeakerIndexer) ─────
        log.info("Loading %s …", cfg.embedding_model)
        emb_model = Model.from_pretrained(
            cfg.embedding_model, use_auth_token=token
        )
        if emb_model is None:
            raise RuntimeError(
                "\n[ERRO] Não foi possível carregar pyannote/embedding.\n"
                "  Aceite os termos em: https://hf.co/pyannote/embedding"
            )
        self.inference = Inference(emb_model, window="whole")
        self.inference.to(self.torch_device)

    # ── Diarisation ──────────────────────────────────────

    def diarize(self, audio_path: str) -> List[Dict]:
        """Run the diarization pipeline; return speaker turns."""
        kwargs: Dict = {}
        if self.cfg.min_speakers:
            kwargs["min_speakers"] = self.cfg.min_speakers
        if self.cfg.max_speakers:
            kwargs["max_speakers"] = self.cfg.max_speakers

        # Pass waveform dict instead of file path to bypass torchaudio.io.AudioDecoder,
        # which is absent in CPU-only Windows builds of torchaudio.
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_input = {"waveform": waveform, "sample_rate": sr}

        log.info("Running diarisation …")
        annotation = self.diar_pipeline(audio_input, **kwargs)

        turns = [
            {"start": turn.start, "end": turn.end, "raw_speaker": label}
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
        unique_speakers = len({t["raw_speaker"] for t in turns})
        log.info(
            "Diarisation: %d unique speakers, %d turns",
            unique_speakers, len(turns),
        )
        return turns

    # ── Embedding extraction ─────────────────────────────

    def embed_segment(
        self, audio_path: str, start: float, end: float
    ) -> np.ndarray:
        """
        Extract a speaker embedding for the audio window [start, end].

        Uses torchaudio for precise, sample-accurate cropping, then
        runs pyannote's Inference engine on the mono waveform dict.
        """
        waveform, sr = torchaudio.load(audio_path)  # (C, T)

        # Stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Crop to segment; pad to ≥ 500 ms so pyannote doesn't crash
        s = int(start * sr)
        e = int(end   * sr)
        min_len = int(0.5 * sr)
        if (e - s) < min_len:
            e = min(s + min_len, waveform.shape[1])
        cropped = waveform[:, s:e]

        audio_input = {"waveform": cropped, "sample_rate": sr}
        embedding = self.inference(audio_input)
        return np.array(embedding).flatten()


# ══════════════════════════════════════════════════════
# 6. Main Orchestrator
# ══════════════════════════════════════════════════════

class AudioTranscriber:
    """
    Orchestrates the full pipeline:
      STTEngine → DiarizationEngine → SpeakerIndexer → structured JSON

    Typical usage
    -------------
    >>> cfg = PipelineConfig(hf_token="hf_xxxxx", max_speakers=3)
    >>> t   = AudioTranscriber(cfg)
    >>> out = t.process("meeting.wav")
    >>> AudioTranscriber.save_json(out, "result.json")
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg     = cfg
        self.stt     = STTEngine(cfg)
        self.diarizer = DiarizationEngine(cfg)
        self.indexer  = SpeakerIndexer(threshold=cfg.similarity_threshold)

    # ── Public ───────────────────────────────────────────

    def process(self, audio_path: str) -> Dict:
        """
        Full pipeline — returns a structured dict.

        Output schema
        -------------
        {
          "audio_file": str,
          "segments": [
            {
              "timestamp_start": float,
              "timestamp_end":   float,
              "user_id":         str,   # "Speaker 0", "Speaker 1", …
              "text":            str,
              "language_detected": str, # "pt", "en", …
              "confidence":      float
            },
            …
          ],
          "speaker_registry": {
            "Speaker 0": [192-d embedding as float list],
            …
          },
          "metadata": { … }
        }
        """
        audio_path = str(Path(audio_path).resolve())
        log.info("=== Processing: %s ===", audio_path)

        # Step 1 — Transcription (Whisper / CTranslate2)
        raw_segments, language = self.stt.transcribe(audio_path)

        # Step 2 — Diarisation (pyannote.audio 3.1)
        turns = self.diarizer.diarize(audio_path)

        # Step 3 — Build raw_label → user_id map (one embedding per speaker)
        speaker_map = self._map_speakers(audio_path, turns)

        # Step 4 — Align whisper segments to speaker turns
        segments = self._merge(raw_segments, language, turns, speaker_map)

        return {
            "audio_file": audio_path,
            "segments":   [asdict(s) for s in segments],
            "speaker_registry": self.indexer.export(),
            "metadata": {
                "whisper_model":       self.cfg.whisper_model,
                "compute_type":        self.stt.compute_type,
                "device":              self.stt.device,
                "diarization_model":   self.cfg.diarization_model,
                "total_segments":      len(segments),
                "unique_speakers":     self.indexer._counter,
                "language_detected":   language,
            },
        }

    @staticmethod
    def save_json(result: Dict, path: str) -> None:
        """Serialize and save the result dict as formatted UTF-8 JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        log.info("Saved → %s", path)

    # ── Internals ────────────────────────────────────────

    def _map_speakers(
        self, audio_path: str, turns: List[Dict]
    ) -> Dict[str, str]:
        """
        For each unique pyannote speaker label:
          1. pick the longest available turn (best embedding quality)
          2. extract the voice embedding
          3. resolve to a stable user_id via SpeakerIndexer
        """
        # Group turns by raw speaker label
        by_speaker: Dict[str, List[Dict]] = {}
        for t in turns:
            by_speaker.setdefault(t["raw_speaker"], []).append(t)

        mapping: Dict[str, str] = {}
        for raw, turn_list in by_speaker.items():
            # Use the longest turn for the highest-quality embedding
            best = max(turn_list, key=lambda x: x["end"] - x["start"])
            try:
                emb = self.diarizer.embed_segment(
                    audio_path, best["start"], best["end"]
                )
                uid = self.indexer.identify(emb)
            except Exception as exc:
                log.warning("Embedding failed for '%s': %s", raw, exc)
                uid = f"Speaker_Unknown_{raw}"
            mapping[raw] = uid

        return mapping

    def _merge(
        self,
        raw_segments: List[Dict],
        language: str,
        turns: List[Dict],
        speaker_map: Dict[str, str],
    ) -> List[TranscriptionSegment]:
        """
        Align whisper segments ↔ diarization turns using temporal overlap.
        Each segment is attributed to the speaker with the greatest overlap.
        """
        out: List[TranscriptionSegment] = []
        for seg in raw_segments:
            text = seg["text"]
            if not text:
                continue
            uid = self._dominant_speaker(seg["start"], seg["end"],
                                         turns, speaker_map)
            out.append(TranscriptionSegment(
                timestamp_start=round(seg["start"], 3),
                timestamp_end=round(seg["end"],   3),
                user_id=uid,
                text=text,
                language_detected=language,
                confidence=round(seg["confidence"], 4),
            ))
        return out

    @staticmethod
    def _dominant_speaker(
        start: float,
        end: float,
        turns: List[Dict],
        speaker_map: Dict[str, str],
    ) -> str:
        """Return the user_id with the most overlap in [start, end]."""
        overlap: Dict[str, float] = {}
        for t in turns:
            ov = max(0.0, min(end, t["end"]) - max(start, t["start"]))
            if ov > 0:
                uid = speaker_map.get(t["raw_speaker"], t["raw_speaker"])
                overlap[uid] = overlap.get(uid, 0.0) + ov
        return max(overlap, key=overlap.get) if overlap else "Speaker Unknown"


# ══════════════════════════════════════════════════════
# 7. CLI Entry Point
# ══════════════════════════════════════════════════════

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Local STT + Speaker Diarization  "
                    "(faster-whisper + pyannote.audio 3.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio",
                   help="Path to audio file (WAV / MP3 / FLAC / OGG …)")
    p.add_argument("--hf-token", default="",
                   metavar="TOKEN",
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--model", default="medium",
                   metavar="SIZE",
                   help="Whisper model: tiny|base|small|medium|large-v3")
    p.add_argument("--output", default="transcription.json",
                   metavar="FILE",
                   help="Output JSON path")
    p.add_argument("--min-speakers", type=int, default=None,
                   help="Minimum number of speakers (hint for pyannote)")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="Maximum number of speakers (hint for pyannote)")
    p.add_argument("--similarity-threshold", type=float, default=0.80,
                   metavar="FLOAT",
                   help="Cosine-similarity cutoff for speaker re-ID (0–1)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    cfg = PipelineConfig(
        whisper_model=args.model,
        hf_token=args.hf_token or os.getenv("HF_TOKEN", ""),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        similarity_threshold=args.similarity_threshold,
    )

    transcriber = AudioTranscriber(cfg)
    result = transcriber.process(args.audio)
    AudioTranscriber.save_json(result, args.output)

    # ── Terminal preview ──────────────────────────────────
    print("\n── Transcription Preview ────────────────────────────────────────────")
    for seg in result["segments"][:15]:
        print(
            f"  [{seg['timestamp_start']:7.2f}s → {seg['timestamp_end']:7.2f}s]"
            f"  {seg['user_id']:<14}"
            f"  [{seg['language_detected']}]"
            f"  {seg['text']}"
        )
    remaining = len(result["segments"]) - 15
    if remaining > 0:
        print(f"  … {remaining} more segments (see {args.output})")

    meta = result["metadata"]
    print(f"\n  Speakers found : {meta['unique_speakers']}")
    print(f"  Total segments : {meta['total_segments']}")
    print(f"  Language       : {meta['language_detected']}")
    print(f"  Device / quant : {meta['device']} / {meta['compute_type']}")
    print(f"  Full JSON      → {args.output}\n")


if __name__ == "__main__":
    main()
