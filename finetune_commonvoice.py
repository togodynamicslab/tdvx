#!/usr/bin/env python3
"""
finetune_commonvoice.py — TDvX v4 — Motor de transcrição pt-BR proprietário
===========================================================================

Fine-tuna o whisper-medium nos corpora:
  - Mozilla Common Voice (scripted speech)
  - CORAA v1.1 (290h pt-BR espontâneo, opcional)
  - LaPS BM — FalaBrasil (eval adicional, opcional)

Workflow:
  1. Extrai o corpus se necessário (tar.gz → clips/ + TSVs)
  2. Carrega amostras validadas (train.tsv para treino, dev.tsv para eval)
  3. Fine-tuning completo do whisper-medium (ou LoRA com --use-lora)
  4. Converte para CTranslate2 int8 pronto para produção
  5. Tracking via MLflow (opcional)

Uso rápido (8x GPU):
    torchrun --nproc_per_node=8 finetune_commonvoice.py \\
        --language pt --coraa --lapsbm-eval \\
        --no-gradient-checkpointing --torch-compile \\
        --run-name tdvx-v4

Teste com amostra limitada (recomendado antes do treino completo):
    python finetune_commonvoice.py --language pt --max-train-samples 5000 --dry-run

Retomar de checkpoint:
    python finetune_commonvoice.py --resume-from-checkpoint auto

Modo LoRA (menos VRAM, treino mais rápido):
    python finetune_commonvoice.py --use-lora --batch-size 16

─────────────────────────────────────────────────────────────────────────────
Modelo base:  openai/whisper-medium  (769M params, multilíngue)
Estratégia:   Full fine-tune (padrão) ou LoRA --use-lora
Saída:        HuggingFace Transformers + CTranslate2 int8
Idioma fixo:  pt (pt-BR — token <|pt|> fixado; áudio não-PT descartado)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
import unicodedata

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()


def _normalize_text(text: str) -> str:
    """Normaliza texto pt-BR para treino: Unicode NFC, espaços, pontuação."""
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    # colapsa múltiplos espaços
    text = re.sub(r"\s+", " ", text)
    # remove caracteres de controle
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Imports opcionais — o script falha de forma amigável se faltarem
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_DEPS_MISSING = False
try:
    import evaluate
    from datasets import Dataset
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )
except ImportError:
    _TRAIN_DEPS_MISSING = True

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass

_PEFT_AVAILABLE = False
try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    pass

_GDRIVE_AVAILABLE = False
try:
    from google.oauth2 import service_account as _gdrive_sa
    from google.oauth2.credentials import Credentials as _OAuthCredentials
    from googleapiclient.discovery import build as _gdrive_build
    from googleapiclient.http import MediaFileUpload as _MediaFileUpload
    _GDRIVE_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Constantes e configuração padrão
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent

_DEFAULT_CVSS_DIR   = _ROOT / "cvss"
_DEFAULT_OUTPUT_DIR = _ROOT / "models" / "tdvx-v4-cv-pt"
_DEFAULT_BASE_MODEL = "openai/whisper-medium"
_DEFAULT_LANGUAGE   = "pt"

# Mapeamento de código ISO para nome completo do WhisperTokenizer
_LANGUAGE_MAP: Dict[str, str] = {
    "pt": "portuguese",
    "en": "english",
    "portuguese": "portuguese",
    "english": "english",
    "es": "spanish",
    "fr": "french",
}

_MAX_DURATION_S = 28.0   # limite técnico da janela de contexto do Whisper

_CORAA_BASE_URL    = "https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/main"
_LAPSBM_DATASET_ID = "falabrasil/lapsbm"          # LaPS BM — usado só como eval


# ─────────────────────────────────────────────────────────────────────────────
# CORAA v1.1 — download e carregamento local
# Estrutura HuggingFace: CSV de metadados + zip (dev/test) ou RAR (train)
# Campos CSV: file_path, text, up_votes, down_votes, votes_for_noise_or_low_voice
# ─────────────────────────────────────────────────────────────────────────────

def _hf_download(url: str, dest: Path, token: Optional[str] = None,
                 desc: str = "") -> bool:
    """Download com progress bar simples. Retorna True se OK."""
    import urllib.request as _ur
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = _ur.Request(url, headers=headers)
    try:
        log.info("Baixando %s → %s", desc or url.split("/")[-1], dest)
        with _ur.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            chunk = 1 << 20  # 1 MB
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {desc or dest.name}: {pct}% ({done//1e6:.0f}/{total//1e6:.0f} MB)",
                          end="", flush=True)
            print()
        return True
    except Exception as exc:
        log.error("Falha ao baixar %s: %s", url, exc)
        return False


def _extract_zip(zip_path: Path, dest: Path) -> bool:
    import zipfile as _zf
    log.info("Extraindo %s → %s", zip_path.name, dest)
    try:
        with _zf.ZipFile(zip_path) as z:
            z.extractall(dest)
        return True
    except Exception as exc:
        log.error("Falha ao extrair zip: %s", exc)
        return False


def _extract_rars(rar_dir: Path, dest: Path) -> bool:
    """Extrai partes RAR usando rarfile (pip) ou unrar do sistema."""
    parts = sorted(rar_dir.glob("*.rar"))
    if not parts:
        log.error("Nenhum .rar encontrado em %s", rar_dir)
        return False
    first = parts[0]
    log.info("Extraindo %d parte(s) RAR → %s", len(parts), dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import rarfile as _rf
        with _rf.RarFile(first) as rf:
            rf.extractall(dest)
        return True
    except ImportError:
        pass
    try:
        subprocess.run(["unrar", "x", "-y", str(first), str(dest)], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.error(
            "Não foi possível extrair RAR.\n"
            "Instale: pip install rarfile  OU  apt install unrar"
        )
        return False


def setup_coraa(coraa_dir: Path, split: str = "dev",
                hf_token: Optional[str] = None) -> bool:
    """
    Baixa e extrai o CORAA v1.1 para coraa_dir se ainda não estiver lá.

    split: "dev" (~1.2 GB zip) | "test" (~2.4 GB zip) | "train" (~50 GB RAR)
    Estrutura gerada:
        coraa_dir/
            metadata_{split}_final.csv
            {split}/sp/*.wav   (arquivos extraídos)
    """
    coraa_dir.mkdir(parents=True, exist_ok=True)

    csv_dest = coraa_dir / f"metadata_{split}_final.csv"
    audio_marker = coraa_dir / split   # pasta criada após extração

    # CSV
    if not csv_dest.exists():
        ok = _hf_download(
            f"{_CORAA_BASE_URL}/metadata_{split}_final.csv",
            csv_dest, token=hf_token, desc=f"CORAA {split} CSV",
        )
        if not ok:
            return False

    # Áudio
    if not audio_marker.exists():
        if split in ("dev", "test"):
            zip_dest = coraa_dir / f"{split}.zip"
            if not zip_dest.exists():
                ok = _hf_download(
                    f"{_CORAA_BASE_URL}/{split}.zip",
                    zip_dest, token=hf_token, desc=f"CORAA {split} áudio",
                )
                if not ok:
                    return False
            if not _extract_zip(zip_dest, coraa_dir):
                return False
        else:  # train — RARs grandes
            rar_dir = coraa_dir / "train_rar"
            rar_dir.mkdir(exist_ok=True)
            for i in range(1, 6):
                part = rar_dir / f"train.part{i}.rar"
                if not part.exists():
                    ok = _hf_download(
                        f"{_CORAA_BASE_URL}/train_dividido/train.part{i}.rar",
                        part, token=hf_token, desc=f"CORAA train part{i}/5",
                    )
                    if not ok:
                        return False
            if not _extract_rars(rar_dir, coraa_dir):
                return False

    log.info("CORAA %s pronto em %s", split, coraa_dir)
    return True


class CORAALocalDataset(torch.utils.data.Dataset):
    """
    Carrega CORAA v1.1 a partir de arquivos locais (CSV + WAV).

    CSV colunas relevantes:
        file_path   — caminho relativo ao coraa_dir (ex.: dev/sp/58031_sp_.wav)
        text        — transcrição pt-BR
        up_votes / down_votes / votes_for_noise_or_low_voice
    """
    _MAX_LABEL_TOKENS = 448

    def __init__(
        self,
        coraa_dir: Path,
        split: str,
        processor: "WhisperProcessor",
        task: str,
        whisper_language: str,
        max_samples: Optional[int] = None,
        augment: bool = False,
        min_up_votes: int = 1,
    ) -> None:
        import librosa as _librosa
        self._librosa   = _librosa
        self._processor = processor
        self._augment   = augment
        self._rng       = np.random.default_rng(seed=3)
        self._coraa_dir = coraa_dir

        processor.tokenizer.set_prefix_tokens(language=whisper_language, task=task)

        csv_path = coraa_dir / f"metadata_{split}_final.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CORAA CSV não encontrado: {csv_path}\n"
                                    "Execute setup_coraa() primeiro ou passe --coraa-dir correto.")

        valid: List[Dict] = []
        skipped = 0
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                up   = int(row.get("up_votes", 0) or 0)
                down = int(row.get("down_votes", 0) or 0)
                noise = int(row.get("votes_for_noise_or_low_voice", 0) or 0)
                if up < min_up_votes or down > up or noise >= up:
                    skipped += 1
                    continue

                text = _normalize_text(row.get("text", ""))
                if not text or len(text.split()) < 2:
                    skipped += 1
                    continue

                ids = processor.tokenizer(text).input_ids
                if len(ids) > self._MAX_LABEL_TOKENS:
                    skipped += 1
                    continue

                audio_path = coraa_dir / row["file_path"]
                if not audio_path.exists():
                    skipped += 1
                    continue

                valid.append({"path": str(audio_path), "label_ids": ids})
                if max_samples and len(valid) >= max_samples:
                    break

        log.info("CORAA %s: %d válidas, %d ignoradas (CSV: %s)",
                 split, len(valid), skipped, csv_path.name)
        self._items = valid

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        item = self._items[idx]
        try:
            array, _ = self._librosa.load(item["path"], sr=16_000, mono=True)
        except Exception:
            array = np.zeros(16_000, dtype=np.float32)

        array = np.clip(array, -1.0, 1.0).astype(np.float32)
        if self._augment and len(item["label_ids"]) > 1:
            array = self._augment_audio(array)

        feats = self._processor.feature_extractor(array, sampling_rate=16_000, return_tensors="pt")
        return {"input_features": feats.input_features[0], "labels": item["label_ids"]}

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        rng = self._rng
        if rng.random() < 0.4:
            noise     = rng.standard_normal(len(audio)).astype(np.float32)
            sig_rms   = np.sqrt(np.mean(audio ** 2)) + 1e-9
            noise_rms = np.sqrt(np.mean(noise  ** 2)) + 1e-9
            audio     = audio + (sig_rms / (noise_rms * 10 ** (25 / 20))) * noise
        if rng.random() < 0.2:
            audio = self._librosa.effects.time_stretch(audio, rate=float(rng.uniform(0.9, 1.1)))
        return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Extração e localização do Common Voice corpus
# ─────────────────────────────────────────────────────────────────────────────

def _find_corpus_dir(cvss_dir: Path, lang_code: str) -> Optional[Path]:
    """Localiza o diretório do idioma dentro do corpus extraído."""
    candidates = [
        cvss_dir / lang_code,                          # cvss/pt/
        *list(cvss_dir.glob(f"cv-corpus-*/{lang_code}")),  # cvss/cv-corpus-25.0.../pt/
        *list(cvss_dir.glob(f"*/{lang_code}")),        # qualquer subpasta/pt/
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "validated.tsv").exists():
            return candidate
    return None


def extract_corpus(cvss_dir: Path, lang_code: str) -> Path:
    """
    Extrai o corpus do Common Voice se ainda não extraído.
    Retorna o caminho para o diretório do idioma (ex.: .../pt/).
    """
    corpus_dir = _find_corpus_dir(cvss_dir, lang_code)
    if corpus_dir:
        log.info("Corpus já extraído: %s", corpus_dir)
        return corpus_dir

    tarballs = sorted(cvss_dir.glob("*.tar.gz"))
    if not tarballs:
        raise FileNotFoundError(
            f"Nenhum arquivo .tar.gz encontrado em '{cvss_dir}'.\n"
            f"Baixe o corpus em https://commonvoice.mozilla.org/pt/datasets"
        )

    tarball = tarballs[0]
    size_gb = tarball.stat().st_size / 1e9
    log.info("Extraindo '%s' (%.1f GB) — pode levar vários minutos...", tarball.name, size_gb)

    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(cvss_dir)

    corpus_dir = _find_corpus_dir(cvss_dir, lang_code)
    if not corpus_dir:
        raise RuntimeError(
            f"Extração concluída mas diretório '{lang_code}' não encontrado.\n"
            f"Conteúdo de {cvss_dir}: {[p.name for p in cvss_dir.iterdir()]}"
        )

    log.info("Corpus extraído para: %s", corpus_dir)
    return corpus_dir


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento e filtragem dos dados Common Voice
# ─────────────────────────────────────────────────────────────────────────────

def load_cv_split(
    corpus_dir: Path,
    split: str,
    max_samples: Optional[int] = None,
    min_up_votes: int = 1,
) -> List[Dict]:
    """
    Carrega um split do Common Voice a partir do TSV.

    Common Voice já filtra por qualidade via votação (validated.tsv tem up_votes >= 2).
    train.tsv / dev.tsv são subconjuntos do validated.tsv.
    """
    tsv_path = corpus_dir / f"{split}.tsv"
    if not tsv_path.exists():
        available = [p.stem for p in corpus_dir.glob("*.tsv")]
        raise FileNotFoundError(
            f"Split '{split}.tsv' não encontrado. Disponíveis: {available}"
        )

    clips_dir = corpus_dir / "clips"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Diretório de áudio não encontrado: {clips_dir}")

    entries: List[Dict] = []
    skipped = 0

    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = _normalize_text(row.get("sentence", ""))
            # descarta frases muito curtas (< 3 palavras) — causam hallucination
            if not sentence or len(sentence.split()) < 3:
                skipped += 1
                continue

            audio_path = clips_dir / row.get("path", "")
            if not audio_path.exists():
                skipped += 1
                continue

            up_votes   = int(row.get("up_votes", 0)   or 0)
            down_votes = int(row.get("down_votes", 0) or 0)

            # Rejeita amostras com mais votos negativos que positivos
            if down_votes > up_votes or up_votes < min_up_votes:
                skipped += 1
                continue

            entries.append({
                "audio_filepath": str(audio_path),
                "sentence": sentence,
            })

            if max_samples and len(entries) >= max_samples:
                break

    log.info(
        "Split '%s': %d amostras carregadas, %d ignoradas (total lido: %d)",
        split, len(entries), skipped, len(entries) + skipped,
    )
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Amostras sintéticas de silêncio/ruído com label vazio
# Ensina o modelo a NÃO transcrever nada quando não há fala — anti-hallucination
# ─────────────────────────────────────────────────────────────────────────────

def generate_no_speech_entries(n: int, sample_rate: int = 16_000) -> List[Dict]:
    """
    Gera n entradas sintéticas de ruído/silêncio com transcrição vazia.
    Cada entrada tem o áudio inline em 'audio_array' (sem arquivo em disco).
    """
    rng = np.random.default_rng(seed=42)
    entries = []
    durations = [1.0, 2.0, 3.0, 4.0, 5.0]   # segundos variados

    noise_fns = [
        lambda n: rng.standard_normal(n).astype(np.float32) * 0.02,          # white noise fraco
        lambda n: rng.standard_normal(n).astype(np.float32) * 0.10,          # white noise forte
        lambda n: np.zeros(n, dtype=np.float32),                             # silêncio puro
        lambda n: (np.cumsum(rng.standard_normal(n)) * 0.005).astype(np.float32),  # ruído rosa
    ]

    for i in range(n):
        dur = durations[i % len(durations)]
        fn  = noise_fns[i % len(noise_fns)]
        length = int(dur * sample_rate)
        audio = np.clip(fn(length), -1.0, 1.0)
        entries.append({"audio_array": audio, "sentence": ""})

    log.info("Geradas %d amostras no-speech sintéticas", len(entries))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Dataset lazy (PyTorch) — evita serializar espectrogramas via IPC
# ─────────────────────────────────────────────────────────────────────────────

class CommonVoiceDataset(torch.utils.data.Dataset):
    """
    Carrega e processa cada amostra on-the-fly no __getitem__.

    Por que não usar datasets.map():
    - datasets 4.x usa multiprocess para map(), serializando TODOS os
      resultados via pickle/IPC → MemoryError com 22k mel-spectrogramas
    - Aqui, apenas batch_size amostras ficam em memória simultaneamente
    - Labels são pré-tokenizados no __init__ (CPU-only, rápido) para
      permitir filtrar amostras com texto inválido antes do treino
    """

    _MAX_LABEL_TOKENS = 448   # limite do decoder Whisper (generation_max_length)

    def __init__(
        self,
        entries: List[Dict],
        processor: "WhisperProcessor",
        task: str,
        whisper_language: str,
        label: str = "dataset",
        augment: bool = False,
    ) -> None:
        import librosa as _librosa
        self._librosa   = _librosa
        self._processor = processor
        self._augment   = augment
        self._rng       = np.random.default_rng(seed=0)

        processor.tokenizer.set_prefix_tokens(language=whisper_language, task=task)

        valid, skipped = [], 0
        for e in entries:
            sentence = e.get("sentence", "")
            ids = processor.tokenizer(sentence).input_ids
            if len(ids) <= self._MAX_LABEL_TOKENS:
                item = {"label_ids": ids}
                if "audio_array" in e:
                    item["audio_array"] = e["audio_array"]   # no-speech inline
                else:
                    item["path"] = e["audio_filepath"]
                valid.append(item)
            else:
                skipped += 1

        if skipped:
            log.warning(
                "%s: %d amostras ignoradas (labels > %d tokens)",
                label, skipped, self._MAX_LABEL_TOKENS,
            )
        log.info("%s: %d amostras válidas", label, len(valid))
        self._items = valid

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Augmentação leve aleatória para robustez a ruído e variações de fala."""
        rng = self._rng

        # 40% chance: adiciona ruído branco fraco (SNR ~25 dB)
        if rng.random() < 0.4:
            noise = rng.standard_normal(len(audio)).astype(np.float32)
            signal_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
            noise_rms  = np.sqrt(np.mean(noise ** 2)) + 1e-9
            snr_linear = 10 ** (25 / 20)
            audio = audio + (signal_rms / (noise_rms * snr_linear)) * noise

        # 20% chance: leve variação de velocidade (±10%)
        if rng.random() < 0.2:
            rate = float(rng.uniform(0.9, 1.1))
            audio = self._librosa.effects.time_stretch(audio, rate=rate)

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        item = self._items[idx]

        if "audio_array" in item:
            array = item["audio_array"]
        else:
            try:
                array, _ = self._librosa.load(item["path"], sr=16_000, mono=True)
            except Exception as exc:
                log.debug("Falha ao carregar %s: %s — substituindo por silêncio", item["path"], exc)
                array = np.zeros(16_000, dtype=np.float32)

        if self._augment and len(item["label_ids"]) > 1:
            # só aumenta amostras com fala real (não no-speech)
            array = self._augment_audio(array)

        feats = self._processor.feature_extractor(
            array, sampling_rate=16_000, return_tensors="pt"
        )
        return {
            "input_features": feats.input_features[0],
            "labels":         item["label_ids"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# HFAudioDataset — dataset lazy para qualquer HuggingFace audio dataset
# Suporta CORAA (campo "text" + "audio") e LaPS BM (campo "txt" + "wav")
# ─────────────────────────────────────────────────────────────────────────────

class HFAudioDataset(torch.utils.data.Dataset):
    """
    Carrega datasets de áudio do HuggingFace de forma lazy (sem OOM).

    Parâmetros:
        dataset_id    ID do dataset HuggingFace (ex: "gabrielrstan/CORAA-v1.1")
        split         Split a carregar ("train", "test", etc.)
        text_field    Nome do campo de transcrição ("text", "txt", "sentence")
        audio_field   Nome do campo de áudio ("audio", "wav")
        processor     WhisperProcessor já instanciado
        task          "transcribe"
        whisper_language  Nome por extenso do idioma (ex: "portuguese")
        max_samples   Limite de amostras (None = tudo)
        augment       Aplica augmentação de áudio
        hf_token      Token HuggingFace (para datasets restritos)
    """
    _MAX_LABEL_TOKENS = 448

    def __init__(
        self,
        dataset_id: str,
        split: str,
        text_field: str,
        audio_field: str,
        processor: "WhisperProcessor",
        task: str,
        whisper_language: str,
        max_samples: Optional[int] = None,
        augment: bool = False,
        hf_token: Optional[str] = None,
    ) -> None:
        from datasets import load_dataset, Audio as HFAudio

        log.info("HFAudioDataset: carregando %s (split=%s) ...", dataset_id, split)
        ds = load_dataset(dataset_id, split=split, token=hf_token, trust_remote_code=True)
        ds = ds.cast_column(audio_field, HFAudio(sampling_rate=16_000))
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        self._ds          = ds
        self._audio_field = audio_field
        self._processor   = processor
        self._augment     = augment
        self._rng         = np.random.default_rng(seed=2)

        processor.tokenizer.set_prefix_tokens(language=whisper_language, task=task)

        # Pré-valida labels; guarda apenas índices + token ids (não o áudio)
        valid: List[tuple] = []
        skipped = 0
        for i in range(len(ds)):
            raw_text = ds[i].get(text_field, "") or ""
            text = _normalize_text(raw_text)
            if not text or len(text.split()) < 3:
                skipped += 1
                continue
            ids = processor.tokenizer(text).input_ids
            if len(ids) <= self._MAX_LABEL_TOKENS:
                valid.append((i, ids))
            else:
                skipped += 1

        self._valid = valid
        log.info("%s/%s: %d amostras válidas, %d ignoradas", dataset_id, split, len(valid), skipped)

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> Dict:
        hf_idx, label_ids = self._valid[idx]
        sample = self._ds[hf_idx]

        audio_col = sample.get(self._audio_field) or {}
        array = np.array(audio_col.get("array", []), dtype=np.float32)
        if array.size == 0:
            array = np.zeros(16_000, dtype=np.float32)

        array = np.clip(array, -1.0, 1.0)

        if self._augment and len(label_ids) > 1:
            array = self._augment_audio(array)

        feats = self._processor.feature_extractor(array, sampling_rate=16_000, return_tensors="pt")
        return {"input_features": feats.input_features[0], "labels": label_ids}

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        import librosa as _lib
        rng = self._rng
        if rng.random() < 0.4:
            noise      = rng.standard_normal(len(audio)).astype(np.float32)
            sig_rms    = np.sqrt(np.mean(audio ** 2)) + 1e-9
            noise_rms  = np.sqrt(np.mean(noise  ** 2)) + 1e-9
            audio      = audio + (sig_rms / (noise_rms * 10 ** (25 / 20))) * noise
        if rng.random() < 0.2:
            audio = _lib.effects.time_stretch(audio, rate=float(rng.uniform(0.9, 1.1)))
        return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data collator (padding variável por batch)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove o decoder_start_token duplicado se já incluído nos labels
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Métrica WER
# ─────────────────────────────────────────────────────────────────────────────

def make_compute_metrics_fn(processor: "WhisperProcessor"):
    from jiwer import wer as _jiwer_wer
    import numpy as _np
    normalizer = getattr(processor.tokenizer, "normalize", None) or processor.tokenizer._normalize

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # predict_with_generate=False → predictions são logits [batch, seq, vocab]
        if isinstance(pred_ids, _np.ndarray) and pred_ids.ndim == 3:
            pred_ids = _np.argmax(pred_ids, axis=-1)

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str  = [normalizer(p) for p in pred_str]
        label_str = [normalizer(l) for l in label_str]

        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l]
        if not pairs:
            return {"wer": float("nan")}
        pred_f, label_f = zip(*pairs)
        wer = _jiwer_wer(list(label_f), list(pred_f))
        return {"wer": round(100 * wer, 2)}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# LoRA — fine-tuning eficiente em parâmetros (opcional)
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model: "WhisperForConditionalGeneration") -> "WhisperForConditionalGeneration":
    """
    Aplica LoRA no encoder e decoder do Whisper.
    Treina ~2% dos parâmetros (vs 100% no full fine-tune) —
    suficiente para especialização de idioma com menos VRAM.
    """
    if not _PEFT_AVAILABLE:
        raise ImportError(
            "peft não instalado. Execute: pip install peft\n"
            "Ou remova --use-lora para full fine-tuning."
        )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        "LoRA ativo: %d parâmetros treináveis de %d (%.1f%%)",
        trainable, total, 100 * trainable / total,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Conversão CTranslate2 (int8 — production-ready)
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_ctranslate2(hf_dir: Path, ct2_dir: Path, quantization: str = "int8") -> bool:
    """
    Converte o modelo fine-tunado para o formato CTranslate2 (faster-whisper).
    A quantização int8 reduz o modelo em ~4x e acelera 2-3x em CPU/GPU.

    Lida automaticamente com a incompatibilidade entre transformers>=5.0 e ctranslate2:
    faz downgrade temporário para 4.44.2, converte, e restaura a versão original.
    """
    import importlib.metadata as _meta

    ct2_dir.mkdir(parents=True, exist_ok=True)

    ct2_cmd = [sys.executable, "-m", "ctranslate2.converters.transformers",
               "--model", str(hf_dir), "--output_dir", str(ct2_dir),
               "--quantization", quantization, "--force"]

    def _run_conversion() -> bool:
        for cmd in [ct2_cmd, ["ct2-transformers-converter",
                               "--model", str(hf_dir), "--output_dir", str(ct2_dir),
                               "--quantization", quantization, "--force"]]:
            try:
                log.info("Convertendo para CTranslate2 (quantização: %s) ...", quantization)
                subprocess.run(cmd, check=True)
                log.info("Conversão concluída → %s", ct2_dir)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        return False

    # Verifica se transformers >= 5.0 (incompatível com ctranslate2 atual)
    try:
        tf_version = _meta.version("transformers")
        needs_downgrade = int(tf_version.split(".")[0]) >= 5
    except Exception:
        needs_downgrade = False

    if needs_downgrade:
        log.info("transformers %s detectado — downgrade temporário para 4.44.2 (CT2)", tf_version)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "transformers==4.44.2", "-q"],
                check=True,
            )
            ok = _run_conversion()
        finally:
            log.info("Restaurando transformers %s ...", tf_version)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"transformers=={tf_version}", "-q"],
                check=True,
            )
        return ok

    ok = _run_conversion()
    if not ok:
        log.warning(
            "Conversão automática falhou. Execute manualmente:\n"
            "  pip install transformers==4.44.2\n"
            "  ct2-transformers-converter --model %s --output_dir %s --quantization %s\n"
            "  pip install --upgrade transformers",
            hf_dir, ct2_dir, quantization,
        )
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Upload para Google Drive (modelos grandes → evita perda em VM efêmera)
# ─────────────────────────────────────────────────────────────────────────────

def upload_dir_to_gdrive(
    local_dir: Path,
    parent_folder_id: str,
    credentials_path: Path,
    folder_name: Optional[str] = None,
) -> Optional[str]:
    """
    Faz upload de um diretório inteiro para o Google Drive via upload resumível.

    Usa chunks de 100 MB para tolerar conexões lentas e permitir retomada
    automática em caso de falha de rede — adequado para modelos de vários GB.

    Requer Service Account com papel "Editor" (ou "Contributor") na pasta destino.
    Para criar uma Service Account e o JSON de credenciais, veja:
      https://developers.google.com/drive/api/guides/about-auth

    Args:
        local_dir:        Diretório local a enviar.
        parent_folder_id: ID da pasta destino no Drive (string após /folders/ na URL).
        credentials_path: Caminho para o JSON de Service Account.
        folder_name:      Nome da subpasta a criar no Drive (padrão: local_dir.name).

    Returns:
        URL da pasta criada no Drive, ou None se o upload falhar.
    """
    if not _GDRIVE_AVAILABLE:
        log.warning(
            "google-api-python-client não instalado — upload ignorado.\n"
            "  pip install google-api-python-client google-auth"
        )
        return None

    if not local_dir.exists():
        log.warning("Diretório local não existe, upload ignorado: %s", local_dir)
        return None

    _SCOPES     = ["https://www.googleapis.com/auth/drive"]
    _CHUNK_SIZE = 100 * 1024 * 1024  # 100 MB por chunk

    try:
        import json as _json
        key_data = _json.loads(credentials_path.read_text(encoding="utf-8"))
        if key_data.get("type") == "service_account":
            creds = _gdrive_sa.Credentials.from_service_account_file(
                str(credentials_path), scopes=_SCOPES
            )
        else:
            # token.json OAuth2 gerado pelo auth_gdrive.py
            creds = _OAuthCredentials.from_authorized_user_file(str(credentials_path), _SCOPES)
            if creds.expired and creds.refresh_token:
                from google.auth.transport.requests import Request as _Request
                creds.refresh(_Request())
        service = _gdrive_build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        log.error("Falha ao autenticar no Google Drive: %s", exc)
        return None

    folder_name = folder_name or local_dir.name

    def _make_folder(name: str, parent_id: str) -> str:
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        return service.files().create(body=meta, fields="id").execute()["id"]

    def _upload_file(file_path: Path, parent_id: str) -> None:
        size_mb = file_path.stat().st_size / 1e6
        log.info("  Enviando %-40s (%.1f MB) ...", file_path.name, size_mb)
        media   = _MediaFileUpload(str(file_path), resumable=True, chunksize=_CHUNK_SIZE)
        request = service.files().create(
            body={"name": file_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                log.info("    %s ... %d%%", file_path.name, int(status.progress() * 100))
        log.info("  ✓ %s", file_path.name)

    def _upload_dir(dir_path: Path, parent_id: str, top_level: bool = False) -> None:
        for item in sorted(dir_path.iterdir()):
            if item.is_dir():
                # Ignora checkpoints intermediários na raiz do modelo
                if top_level and item.name.startswith("checkpoint-"):
                    log.info("  Ignorando checkpoint: %s", item.name)
                    continue
                sub_id = _make_folder(item.name, parent_id)
                _upload_dir(item, sub_id)
            elif item.is_file():
                _upload_file(item, parent_id)

    try:
        log.info("Google Drive: criando pasta '%s' em folder_id=%s ...", folder_name, parent_folder_id)
        drive_folder_id = _make_folder(folder_name, parent_folder_id)
        drive_url       = f"https://drive.google.com/drive/folders/{drive_folder_id}"
        log.info("Google Drive: iniciando upload de '%s' → %s", local_dir, drive_url)
        _upload_dir(local_dir, drive_folder_id, top_level=True)
        log.info("Google Drive: upload concluído → %s", drive_url)
        return drive_url
    except Exception as exc:
        log.error("Falha no upload para Google Drive: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MLflow tracking (simplificado — sem Model Registry)
# ─────────────────────────────────────────────────────────────────────────────

class MlflowTracker:
    def __init__(self, args: argparse.Namespace) -> None:
        # Respeita ENABLE_MLFLOW=false do .env antes de qualquer coisa
        env_enabled = os.environ.get("ENABLE_MLFLOW", "true").lower()
        if env_enabled in ("false", "0", "no"):
            self._enabled = False
        else:
            self._enabled = _MLFLOW_AVAILABLE and bool(
                args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI")
            )
        self._args = args
        self.run_id = ""

    def __enter__(self) -> "MlflowTracker":
        if not self._enabled:
            return self

        uri = self._args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        experiment = self._args.mlflow_experiment or os.environ.get(
            "MLFLOW_FINETUNE_EXPERIMENT", "tdv1-commonvoice"
        )
        run_name = self._args.run_name or f"cv-pt-{datetime.now().strftime('%Y%m%d-%H%M')}"

        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self._run  = mlflow.start_run(run_name=run_name)
            self.run_id = self._run.info.run_id
            os.environ["MLFLOW_TRACKING_URI"]    = uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
            log.info("MLflow: run '%s' (id=%s) | UI: %s", run_name, self.run_id, uri)
        except Exception as exc:
            # Servidor HTTP inacessível → fallback silencioso para armazenamento local
            log.warning(
                "MLflow: falha ao conectar em '%s' (%s).\n"
                "  Fallback para armazenamento local: ./mlruns\n"
                "  Para desabilitar completamente: ENABLE_MLFLOW=false no .env",
                uri, type(exc).__name__,
            )
            local_uri = str(Path("./mlruns").resolve())
            try:
                mlflow.set_tracking_uri(local_uri)
                mlflow.set_experiment(experiment)
                self._run   = mlflow.start_run(run_name=run_name)
                self.run_id = self._run.info.run_id
                os.environ["MLFLOW_TRACKING_URI"]    = local_uri
                os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
                log.info("MLflow (local): run '%s' em %s", run_name, local_uri)
            except Exception:
                log.warning("MLflow: fallback local também falhou — tracking desabilitado.")
                self._enabled = False

        return self

    def __exit__(self, *_) -> None:
        if self._enabled and self.run_id:
            mlflow.end_run()

    def log_params(self, args: argparse.Namespace, train_size: int, eval_size: int) -> None:
        if not self._enabled:
            return
        mlflow.log_params({
            "base_model":         args.base_model,
            "language":           args.language,
            "use_lora":           args.use_lora,
            "learning_rate":      args.learning_rate,
            "batch_size":         args.batch_size,
            "grad_accum":         args.gradient_accumulation_steps,
            "warmup_steps":       args.warmup_steps,
            "num_epochs":         args.num_epochs,
            "max_steps":          args.max_steps or "—",
            "ct2_quantization":   args.ct2_quantization,
            "train_size":         train_size,
            "eval_size":          eval_size,
            "max_train_samples":  args.max_train_samples or "all",
            "max_eval_samples":   args.max_eval_samples  or "all",
        })
        mlflow.set_tags({"base_model": args.base_model, "language": args.language})

    def log_final(self, metrics: Dict) -> None:
        if not self._enabled:
            return
        mlflow.log_metrics({k.replace("eval_", ""): v for k, v in metrics.items()
                            if isinstance(v, (int, float))})

    @property
    def report_to(self) -> List[str]:
        reporters = ["mlflow"] if self._enabled else []
        if getattr(self._args, "tensorboard", False):
            reporters.append("tensorboard")
        return reporters or ["none"]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(args: argparse.Namespace) -> None:
    if _TRAIN_DEPS_MISSING:
        log.error(
            "Dependências de treino não instaladas.\n"
            "Execute: pip install -r requirements-finetune.txt"
        )
        sys.exit(1)

    # Reduz fragmentação de VRAM em multi-GPU — especialmente durante eval com logits grandes
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(42)

    lang_code      = args.language.lower()
    whisper_lang   = _LANGUAGE_MAP.get(lang_code, lang_code)
    cvss_dir       = Path(args.cvss_dir).resolve()
    output_dir     = Path(args.output_dir).resolve()
    ct2_dir        = output_dir.parent / (output_dir.name + "-ct2")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    log.info("═" * 65)
    log.info("TDvX v4 — Fine-tuning proprietário pt-BR (CV + CORAA + LaPS BM)")
    log.info("  Modelo base  : %s", args.base_model)
    log.info("  Idioma       : %s (%s)", lang_code, whisper_lang)
    log.info("  Modo         : %s", "LoRA" if args.use_lora else "Full fine-tune")
    log.info("  Corpus CV    : %s", cvss_dir)
    log.info("  CORAA        : %s", "sim" if args.coraa else "não")
    log.info("  LaPS BM eval : %s", "sim" if args.lapsbm_eval else "não")
    log.info("  Saída HF     : %s", output_dir)
    log.info("  Saída CT2    : %s", ct2_dir)
    log.info("  GPU          : %s", "CUDA" if torch.cuda.is_available() else "CPU (lento)")
    log.info("═" * 65)

    # ── 1. Extrair corpus ─────────────────────────────────────────────────────
    corpus_dir = extract_corpus(cvss_dir, lang_code)

    # ── 2. Carregar splits Common Voice ──────────────────────────────────────
    train_entries = load_cv_split(
        corpus_dir, "train",
        max_samples=args.max_train_samples,
        min_up_votes=args.min_up_votes,
    )
    eval_entries = load_cv_split(
        corpus_dir, "dev",
        max_samples=args.max_eval_samples,
        min_up_votes=args.min_up_votes,
    )

    log.info(
        "Common Voice: %d treino | %d avaliação",
        len(train_entries), len(eval_entries),
    )

    if args.dry_run:
        log.info("[dry-run] Dataset pronto. Nenhum treino executado.")
        return

    # ── 3. Processor (carregado antes do dataset para pré-tokenizar labels) ──
    log.info("Carregando processor: %s", args.base_model)
    processor = WhisperProcessor.from_pretrained(args.base_model)

    # ── 4. Amostras no-speech (anti-hallucination) ────────────────────────────
    n_no_speech = max(50, len(train_entries) // 20)
    no_speech_entries = generate_no_speech_entries(n_no_speech)
    train_entries_aug = train_entries + no_speech_entries
    import random as _random
    _random.seed(42)
    _random.shuffle(train_entries_aug)

    # ── 5. Dataset lazy Common Voice ─────────────────────────────────────────
    cv_train_ds = CommonVoiceDataset(
        train_entries_aug, processor, args.task, whisper_lang, "CV-treino",
        augment=not args.no_augment,
    )
    cv_eval_ds = CommonVoiceDataset(eval_entries, processor, args.task, whisper_lang, "CV-eval")

    # ── 6. CORAA v1.1 (opcional) ──────────────────────────────────────────────
    train_datasets = [cv_train_ds]
    eval_datasets  = [cv_eval_ds]

    if args.coraa:
        coraa_dir = Path(args.coraa_dir).resolve()

        # Download automático do dev set (sempre) e train (se não for só dev)
        dev_ok   = setup_coraa(coraa_dir, "dev",   hf_token=hf_token)
        train_ok = setup_coraa(coraa_dir, "train", hf_token=hf_token) if not args.coraa_dev_only else False

        if dev_ok:
            coraa_eval_ds = CORAALocalDataset(
                coraa_dir, "dev", processor, args.task, whisper_lang,
                max_samples=min(args.max_eval_samples or 2000, 2000),
            )
            eval_datasets.append(coraa_eval_ds)
            log.info("CORAA dev: %d amostras de avaliação", len(coraa_eval_ds))

        if train_ok:
            coraa_train_ds = CORAALocalDataset(
                coraa_dir, "train", processor, args.task, whisper_lang,
                max_samples=args.max_coraa_samples,
                augment=not args.no_augment,
            )
            train_datasets.append(coraa_train_ds)
            log.info("CORAA train: %d amostras de treino", len(coraa_train_ds))
        elif args.coraa_dev_only:
            log.info("CORAA: modo --coraa-dev-only — usando dev como treino adicional")
            if dev_ok:
                train_datasets.append(coraa_eval_ds)
        else:
            log.warning("CORAA train: download falhou ou RAR não extraído — treino só com CV.")

    # ── 7. LaPS BM (eval adicional, opcional) ─────────────────────────────────
    if args.lapsbm_eval:
        lapsbm_ds = HFAudioDataset(
            dataset_id=_LAPSBM_DATASET_ID,
            split="test",
            text_field="txt",
            audio_field="wav",
            processor=processor,
            task=args.task,
            whisper_language=whisper_lang,
            hf_token=hf_token,
        )
        eval_datasets.append(lapsbm_ds)
        log.info("LaPS BM eval: %d amostras", len(lapsbm_ds))

    # ── 8. Merge datasets ─────────────────────────────────────────────────────
    import torch.utils.data as _td
    train_ds = _td.ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    eval_ds  = _td.ConcatDataset(eval_datasets)  if len(eval_datasets)  > 1 else eval_datasets[0]

    log.info("Total: %d amostras treino | %d avaliação", len(train_ds), len(eval_ds))

    with MlflowTracker(args) as tracker:
        tracker.log_params(args, len(train_ds), len(eval_ds))

        # ── 5. Modelo ─────────────────────────────────────────────────────────
        log.info("Carregando modelo: %s", args.base_model)
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

        # Configura geração para o idioma alvo
        model.generation_config.language               = whisper_lang
        model.generation_config.task                   = args.task
        model.generation_config.forced_decoder_ids     = None
        # Anti-hallucination: não condiciona no texto anterior (evita snowball)
        model.generation_config.condition_on_previous_text = False
        # Suprime saída quando confiança é baixa (ruído, silêncio)
        model.generation_config.no_speech_threshold   = 0.6
        model.generation_config.logprob_threshold     = -1.0
        model.generation_config.compression_ratio_threshold = 2.4

        if args.use_lora:
            model = apply_lora(model)

        # ── 7. Collator + Métricas ────────────────────────────────────────────
        data_collator   = WhisperDataCollator(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        compute_metrics = make_compute_metrics_fn(processor)

        # ── 8. Training arguments ─────────────────────────────────────────────
        # BF16 auto-detect: ativa em qualquer GPU Ampere/Hopper/Blackwell (4090/5090/A100/H100)
        _has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_bf16  = torch.cuda.is_available() and (args.bf16 or _has_bf16) and not args.fp16
        use_fp16  = torch.cuda.is_available() and not use_bf16

        _n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        log.info(
            "Precisão: %s | GPUs: %d | batch efetivo: %d",
            "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
            _n_gpus,
            args.batch_size * args.gradient_accumulation_steps * _n_gpus,
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,   # sem gradiente → mesma memória
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps if args.max_steps else -1,
            num_train_epochs=args.num_epochs if not args.max_steps else 1,
            # gradient_checkpointing=False com --no-gradient-checkpointing:
            # usa mais VRAM mas elimina re-computação → mais rápido com 5090
            gradient_checkpointing=not args.no_gradient_checkpointing,
            fp16=use_fp16,
            bf16=use_bf16,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            logging_steps=50,
            report_to=tracker.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            prediction_loss_only=True,
            dataloader_num_workers=0 if sys.platform == "win32" else args.num_workers,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            label_smoothing_factor=0.0,
            max_grad_norm=1.0,
            # AdamW fusionado: kernel único no CUDA → ~5-10% mais rápido que adamw_torch
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            # Evita overhead de DDP verificar parâmetros não usados a cada step
            ddp_find_unused_parameters=False,
            torch_compile=getattr(args, "torch_compile", False),
        )

        # ── 9. Trainer ────────────────────────────────────────────────────────
        callbacks = []
        if args.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            ))
            log.info(
                "Early stopping: patience=%d, threshold=%.4f",
                args.early_stopping_patience, args.early_stopping_threshold,
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor.feature_extractor,
            callbacks=callbacks or None,
        )

        # ── 10. Treino (com suporte a retomada de checkpoint) ─────────────────
        log.info("Iniciando treinamento ...")
        resume = args.resume_from_checkpoint
        if resume == "auto":
            # Verifica se existe algum checkpoint salvo
            checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
            resume = str(checkpoints[-1]) if checkpoints else None
            if resume:
                log.info("Retomando de: %s", resume)

        trainer.train(resume_from_checkpoint=resume or None)

        # ── 11. Salva modelo + processor ──────────────────────────────────────
        log.info("Salvando modelo HuggingFace → %s", output_dir)
        trainer.save_model()
        processor.save_pretrained(str(output_dir))

        # Se usou LoRA, mescla os adapters no modelo base antes de salvar
        if args.use_lora:
            log.info("Mesclando LoRA adapters no modelo base ...")
            merged = model.merge_and_unload()
            merged.save_pretrained(str(output_dir))
            processor.save_pretrained(str(output_dir))

        # ── 12. Avaliação final ───────────────────────────────────────────────
        log.info("Avaliação final ...")
        metrics = trainer.evaluate()
        wer = metrics.get("eval_wer", float("nan"))
        log.info("WER final: %.2f%%", wer)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        tracker.log_final(metrics)

        # Passos pós-treino executados apenas pelo rank 0 (evita conflitos em multi-GPU)
        is_main = int(os.environ.get("RANK", 0)) == 0
        if is_main:
            # ── 13. Conversão CTranslate2 (int8 → produção) ───────────────────
            if args.convert_ct2:
                ok = convert_to_ctranslate2(output_dir, ct2_dir, quantization=args.ct2_quantization)
                if ok:
                    log.info(
                        "\nPara usar no tdvx, atualize engine.py:\n"
                        "  WhisperModel('%s', device='cuda', compute_type='%s')",
                        ct2_dir, args.ct2_quantization,
                    )

            # ── 14. Upload para Google Drive (opcional) ───────────────────────
            if args.gdrive_folder_id and args.gdrive_credentials:
                creds_path = Path(args.gdrive_credentials).resolve()
                if not creds_path.exists():
                    log.warning(
                        "Credenciais do Drive não encontradas: %s — upload ignorado.", creds_path
                    )
                else:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
                    run_label = args.run_name or f"cv-{lang_code}-{timestamp}"

                    hf_url = upload_dir_to_gdrive(
                        output_dir,
                        args.gdrive_folder_id,
                        creds_path,
                        folder_name=f"{run_label}-hf",
                    )
                    if hf_url:
                        log.info("Modelo HF no Drive: %s", hf_url)

                    if args.convert_ct2 and ct2_dir.exists():
                        ct2_url = upload_dir_to_gdrive(
                            ct2_dir,
                            args.gdrive_folder_id,
                            creds_path,
                            folder_name=f"{run_label}-ct2",
                        )
                        if ct2_url:
                            log.info("Modelo CT2 no Drive: %s", ct2_url)

    log.info("═" * 65)
    log.info("Fine-tuning concluído!")
    log.info("  WER pt-BR    : %.2f%%", wer)
    log.info("  Modelo HF    : %s", output_dir)
    if args.convert_ct2:
        log.info("  Modelo CT2   : %s", ct2_dir)
    if args.gdrive_folder_id:
        log.info("  Google Drive : https://drive.google.com/drive/folders/%s", args.gdrive_folder_id)
    log.info("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TDvX v4 — Fine-tuning proprietário pt-BR via Common Voice + CORAA + LaPS BM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Localização dos dados
    p.add_argument("--cvss-dir",   default=str(_DEFAULT_CVSS_DIR),
                   help="Pasta com o .tar.gz do Common Voice (ou corpus já extraído)")
    p.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR),
                   help="Destino do modelo HuggingFace fine-tunado")

    # Modelo
    p.add_argument("--base-model", default=_DEFAULT_BASE_MODEL,
                   help="Modelo Whisper base (HuggingFace ID ou caminho local)")
    p.add_argument("--language",   default=_DEFAULT_LANGUAGE,
                   help="Código ISO do idioma alvo: pt (padrão) | en | es | fr")
    p.add_argument("--task",       default="transcribe", choices=["transcribe", "translate"])

    # Dataset
    p.add_argument("--max-train-samples", type=int, default=0,
                   help="Limita amostras de treino (0 = usar tudo)")
    p.add_argument("--max-eval-samples",  type=int, default=2000,
                   help="Limita amostras de avaliação (0 = usar tudo do dev.tsv)")
    p.add_argument("--min-up-votes",      type=int, default=1,
                   help="Votos positivos mínimos (>=2 filtra mais agressivamente)")

    # Treino
    p.add_argument("--max-steps",                type=int,   default=0,
                   help="Steps máximos (0 = usa --num-epochs)")
    p.add_argument("--num-epochs",               type=int,   default=10)
    p.add_argument("--early-stopping-patience",  type=int,   default=3,
                   help="Para o treino se eval_loss nao melhorar por N avaliações (0 = desativado)")
    p.add_argument("--early-stopping-threshold", type=float, default=0.001,
                   help="Melhora mínima no eval_loss para contar como progresso")
    p.add_argument("--batch-size",               type=int,   default=64,
                   help="Amostras por GPU por step. 64 preenche ~20GB por GPU (5090 32GB)")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Batch efetivo = batch_size × grad_accum × n_gpus (ex.: 64×1×8=512)")
    p.add_argument("--learning-rate",            type=float, default=1e-5)
    p.add_argument("--warmup-steps",             type=int,   default=500)
    p.add_argument("--eval-steps",               type=int,   default=500,
                   help="Frequência de avaliação e checkpoint")

    # Precisão
    p.add_argument("--bf16", action="store_true",
                   help="Força bfloat16. Auto-ativado em GPUs que suportam (ex.: 4090/5090/A100)")
    p.add_argument("--fp16", action="store_true",
                   help="Força float16 (sobrepõe bf16 auto-detect)")

    # Multi-GPU / performance
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Desativa gradient checkpointing — usa mais VRAM mas é mais rápido (recomendado com 5090)")
    p.add_argument("--torch-compile", action="store_true",
                   help="Ativa torch.compile (PyTorch 2.x) — ~10-20%% mais rápido após warm-up")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Número de workers por GPU para o DataLoader")

    # LoRA
    p.add_argument("--use-lora", action="store_true",
                   help="LoRA: treina ~2%% dos parâmetros, menos VRAM (requer peft)")

    # CTranslate2
    p.add_argument("--convert-ct2",     action="store_true", default=True)
    p.add_argument("--ct2-quantization", default="int8",
                   choices=["int8", "int8_float16", "float16", "int4"],
                   help="int8 recomendado para CPU/GPU. int8_float16 para GPU pura.")

    # MLflow
    p.add_argument("--mlflow-uri",        default="",
                   help="URI do MLflow (ex.: http://localhost:5000 ou ./mlruns)")
    p.add_argument("--mlflow-experiment", default="")
    p.add_argument("--run-name",          default="",
                   help="Nome do run MLflow (ex.: tdv1-cv-pt-v1)")
    p.add_argument("--model-name",        default="",
                   help="Nome para MLflow Model Registry")
    p.add_argument("--tensorboard",       action="store_true")

    # Checkpoint
    p.add_argument("--resume-from-checkpoint", default="",
                   help="Caminho para checkpoint ou 'auto' para retomar o último")

    # Augmentação e anti-hallucination
    p.add_argument("--no-augment", action="store_true",
                   help="Desativa augmentação de áudio no treino (mais rápido, menos robusto)")

    # Google Drive — upload automático do modelo após o treino
    p.add_argument(
        "--gdrive-folder-id", default="",
        help=(
            "ID da pasta destino no Google Drive. Encontre na URL: "
            "https://drive.google.com/drive/folders/<ID>"
        ),
    )
    p.add_argument(
        "--gdrive-credentials", default="",
        help=(
            "Caminho para o JSON de Service Account do Google Drive. "
            "Crie em: console.cloud.google.com → IAM → Service Accounts → Keys."
        ),
    )

    # CORAA + LaPS BM
    p.add_argument("--coraa", action="store_true",
                   help="Inclui CORAA v1.1 (290h pt-BR espontâneo) no treino")
    p.add_argument("--coraa-dir", default="./coraa",
                   help="Pasta local onde CORAA será baixado/extraído")
    p.add_argument("--coraa-dev-only", action="store_true",
                   help="Usa só o dev do CORAA (~1.2 GB zip) — útil para teste rápido sem baixar os 50 GB de train")
    p.add_argument("--max-coraa-samples", type=int, default=0,
                   help="Limita amostras do CORAA (0 = usar tudo)")
    p.add_argument("--lapsbm-eval", action="store_true",
                   help="Adiciona LaPS BM como dataset de avaliação adicional")
    p.add_argument("--hf-token", default="",
                   help="Token HuggingFace para datasets restritos (ou use HF_TOKEN no .env)")

    # Utilitário
    p.add_argument("--dry-run", action="store_true",
                   help="Valida extração e carregamento sem treinar")

    args = p.parse_args()

    # Normaliza: 0 → None para max_samples
    if args.max_train_samples == 0:
        args.max_train_samples = None
    if args.max_eval_samples == 0:
        args.max_eval_samples = None
    if args.max_coraa_samples == 0:
        args.max_coraa_samples = None

    return args


if __name__ == "__main__":
    run_finetune(parse_args())
