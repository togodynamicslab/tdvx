"""
engine.py — Core STT + Diarization + Speaker Re-ID engine

Portado de stt_pipeline.py para a arquitetura tdvx.
Option B: tudo em memória (numpy arrays / waveform dicts) — sem temp files.

Stack:
  • faster-whisper (CTranslate2)  → int8_float16 (GPU) / int8 (CPU)
  • pyannote.audio 3.1            → diarização + speaker embeddings
  • SpeakerIndexer                → cosine re-ID entre chunks/sessões
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

# ── Env / telemetry ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYANNOTE_AUDIO_TELEMETRY", "0")

import numpy as np
from scipy.spatial.distance import cosine
import torch
import torchaudio

# ── Compatibility patches ─────────────────────────────────────────────────────
# Devem rodar ANTES de qualquer import de pyannote / speechbrain.

# 1. torchaudio >= 2.1 removeu a API de gerenciamento de backend de áudio.
#    speechbrain ainda chama essas três funções na inicialização do módulo.
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# 2. numpy >= 2.0 renomeou np.NaN → np.nan.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# 3. torchaudio >= 2.2 removeu o subpacote torchaudio.backend por completo.
#    speechbrain e pyannote tentam importar torchaudio.backend.* — criamos mocks
#    em sys.modules ANTES de qualquer import desses pacotes.
import sys
from types import ModuleType

if "torchaudio.backend" not in sys.modules:
    _backend_mock = ModuleType("torchaudio.backend")
    sys.modules["torchaudio.backend"] = _backend_mock
    torchaudio.backend = _backend_mock  # type: ignore[attr-defined]

for _sub in ("common", "soundfile_backend", "sox_io_backend", "no_backend"):
    _full = f"torchaudio.backend.{_sub}"
    if _full not in sys.modules:
        _m = ModuleType(_full)
        sys.modules[_full] = _m
        setattr(sys.modules["torchaudio.backend"], _sub, _m)

# 3b. torchaudio >= 2.5 removeu AudioMetaData do namespace raiz.
if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio._torchaudio import AudioMetaData as _AudioMetaData
        torchaudio.AudioMetaData = _AudioMetaData
    except (ImportError, AttributeError):
        from collections import namedtuple
        torchaudio.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )

# Expõe AudioMetaData também no mock torchaudio.backend.common (pyannote usa)
sys.modules["torchaudio.backend.common"].AudioMetaData = torchaudio.AudioMetaData  # type: ignore

# 4. PyTorch 2.6 mudou torch.load default weights_only=False → True.
#    checkpoints do pyannote contêm objetos Python customizados fora da allowlist.
_orig_torch_load = torch.load

def _compat_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _compat_torch_load

# 5. huggingface_hub >= 0.22 removeu use_auth_token de hf_hub_download.
#    pyannote.audio 3.x ainda o passa internamente.
import huggingface_hub as _hf_hub
import huggingface_hub.file_download as _hf_fd

_orig_hf_hub_download = _hf_hub.hf_hub_download

def _compat_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs.setdefault("token", kwargs.pop("use_auth_token"))
    return _orig_hf_hub_download(*args, **kwargs)

_hf_hub.hf_hub_download = _compat_hf_hub_download
_hf_fd.hf_hub_download = _compat_hf_hub_download

# ─────────────────────────────────────────────────────────────────────────────

from faster_whisper import WhisperModel
from pyannote.audio import Inference, Model, Pipeline

log = logging.getLogger(__name__)

# Prompt bilíngue PT-BR / EN — reduz alucinações com termos técnicos em inglês
_INITIAL_PROMPT = (
    "Transcrição de reunião técnica em português do Brasil com termos técnicos "
    "em inglês. Vocabulário comum: API, endpoint, backend, frontend, deploy, "
    "pipeline, sprint, commit, pull request, machine learning, dataset, "
    "framework, bug, feature, refactor, merge, branch, container, Kubernetes, "
    "Docker, microservice, CI/CD, DevOps, SLA, SLO, latência."
)


class SpeakerIndexer:
    """Speaker registry with cosine-based re-identification."""

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self._registry: Dict[str, np.ndarray] = {}
        self._counter = 0

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
        self._registry[uid] = (self._registry[uid] + emb) / 2.0

    def identify(self, embedding: np.ndarray) -> str:
        emb = self._normalise(embedding)
        best_id, best_sim = self._nearest(emb)

        if best_id is not None and best_sim >= self.threshold:
            self._update_centroid(best_id, emb)
            return best_id

        user_id = f"Speaker {self._counter}"
        self._counter += 1
        self._registry[user_id] = emb.copy()
        return user_id


# ══════════════════════════════════════════════════════════════════════════════
# STTEngine — Faster-Whisper + CTranslate2
# ══════════════════════════════════════════════════════════════════════════════

class STTEngine:
    """
    Transcrição via faster-whisper (CTranslate2).

    Quantização automática:
      • GPU (CUDA) → int8_float16  (~2× menos VRAM que float16, near-lossless)
      • CPU        → int8          (~4× menos RAM que float32)

    Aceita numpy arrays — sem I/O de arquivo.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: int = 0,
        num_workers: int = 2,
    ) -> None:
        self.device, self.compute_type = self._resolve(device, compute_type)
        if cpu_threads == 0:
            cpu_threads = os.cpu_count() or 4

        log.info(
            "Carregando Whisper '%s' | device=%s | compute=%s",
            model_size, self.device, self.compute_type,
        )
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        log.info("Whisper carregado com sucesso.")

    @staticmethod
    def _resolve(device: str, compute_type: str) -> Tuple[str, str]:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type == "auto":
            if device == "cuda":
                # int8_float16 requires Tensor Cores (compute capability >= 7.0)
                # Volta/Turing/Ampere/Ada: cc >= 7.0 → int8_float16
                # Pascal (GTX 10xx): cc 6.x → float16 only
                try:
                    major, _ = torch.cuda.get_device_capability()
                    compute_type = "int8_float16" if major >= 7 else "float16"
                    log.info("GPU compute capability: %d.x → compute_type=%s", major, compute_type)
                except Exception:
                    compute_type = "float16"
            else:
                compute_type = "int8"
        return device, compute_type

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 500,
        initial_prompt: Optional[str] = _INITIAL_PROMPT,
    ) -> Tuple[List[Dict], str]:
        """
        Transcreve audio em memória.

        Parâmetros
        ----------
        audio       : float32 numpy array, mono, sample_rate Hz
        sample_rate : taxa de amostragem (padrão 16 kHz)

        Retorna
        -------
        segments : [{start, end, text, confidence}, ...]
        language : código ISO 639-1 detectado
        """
        vad_params = {"min_silence_duration_ms": vad_min_silence_ms} if vad_filter else None

        segments_iter, info = self.model.transcribe(
            audio,
            task="transcribe",
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            word_timestamps=False,
        )

        lang = info.language
        log.info("Idioma detectado: %s (p=%.2f)", lang, info.language_probability)

        segments: List[Dict] = []
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "confidence": float(np.exp(seg.avg_logprob)) if seg.avg_logprob is not None else 0.0,
            })

        log.info("Whisper: %d segmentos produzidos", len(segments))
        return segments, lang


# ══════════════════════════════════════════════════════════════════════════════
# DiarizationEngine — pyannote.audio 3.1
# ══════════════════════════════════════════════════════════════════════════════

class DiarizationEngine:
    """
    Wraps pyannote/speaker-diarization-3.1.
    Aceita waveform dicts em memória — sem temp files.
    {"waveform": torch.Tensor shape (1, T), "sample_rate": int}
    """

    def __init__(
        self,
        hf_token: str,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        clustering_threshold: float = 0.5,
    ) -> None:
        if not hf_token:
            raise ValueError(
                "[ERRO] HF token obrigatório. Configure PYANNOTE_AUTH_TOKEN no .env\n"
                "  Aceite os termos em:\n"
                "    https://hf.co/pyannote/speaker-diarization-3.1"
            )

        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._current_threshold = clustering_threshold
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Diarização device: %s", self.torch_device)

        log.info("Carregando %s …", diarization_model)
        pipeline = Pipeline.from_pretrained(diarization_model, use_auth_token=hf_token)
        if pipeline is None:
            raise RuntimeError(
                f"[ERRO] Não foi possível carregar {diarization_model}.\n"
                "  Verifique se aceitou os termos de uso no HuggingFace."
            )

        try:
            pipeline.instantiate({
                "segmentation": {"min_duration_off": 0.0},
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 2,
                    "threshold": clustering_threshold,
                },
            })
            log.info("Clustering threshold configurado: %.3f", clustering_threshold)
        except Exception as exc:
            log.warning("Não foi possível configurar clustering threshold: %s. Usando padrão.", exc)

        emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        if emb_model is None:
            raise RuntimeError("Não foi possível carregar pyannote/embedding")
        self.inference = Inference(emb_model, window="whole")
        self.inference.to(self.torch_device)

        self.diar_pipeline = pipeline.to(self.torch_device)
        log.info("DiarizationEngine carregado com sucesso.")

    def diarize(
        self,
        waveform: dict,
        clustering_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Executa diarização em áudio em memória.
        Retorna [{start, end, raw_speaker}, ...]
        """
        if clustering_threshold is not None and clustering_threshold != self._current_threshold:
            try:
                self.diar_pipeline.instantiate({
                    "segmentation": {"min_duration_off": 0.0},
                    "clustering": {
                        "method": "centroid",
                        "min_cluster_size": 2,
                        "threshold": clustering_threshold,
                    },
                })
                self._current_threshold = clustering_threshold
            except Exception as exc:
                log.warning("Não foi possível atualizar clustering threshold: %s", exc)

        kwargs: Dict = {}
        if self.min_speakers:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers:
            kwargs["max_speakers"] = self.max_speakers

        log.info("Executando diarização …")
        annotation = self.diar_pipeline(waveform, **kwargs)

        turns = [
            {"start": turn.start, "end": turn.end, "raw_speaker": label}
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
        unique = len({t["raw_speaker"] for t in turns})
        log.info("Diarização: %d speakers únicos, %d turns", unique, len(turns))
        return turns

    def embed_segment(self, waveform: dict, start: float, end: float) -> np.ndarray:
        audio = waveform["waveform"].squeeze(0).detach().cpu().numpy()
        sample_rate = int(waveform["sample_rate"])
        n = len(audio)

        s = max(0, min(int(start * sample_rate), n))
        e = max(s + 1, min(int(end * sample_rate), n))
        chunk = audio[s:e]

        emb = self.inference(chunk)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-10 else emb
