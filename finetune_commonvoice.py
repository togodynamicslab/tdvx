#!/usr/bin/env python3
"""
finetune_commonvoice.py — Modelo proprietário TDv1 via Mozilla Common Voice
===========================================================================

Gera o primeiro modelo proprietário fine-tunando o whisper-medium no
Mozilla Common Voice (scripted speech), otimizado para pt-BR.

Workflow:
  1. Extrai o corpus se necessário (tar.gz → clips/ + TSVs)
  2. Carrega amostras validadas (train.tsv para treino, dev.tsv para eval)
  3. Fine-tuning completo do whisper-medium (ou LoRA com --use-lora)
  4. Converte para CTranslate2 int8 pronto para produção
  5. Tracking via MLflow (opcional)

Uso rápido:
    python finetune_commonvoice.py --language pt

Teste com amostra limitada (recomendado antes do treino completo):
    python finetune_commonvoice.py --language pt --max-train-samples 5000 --dry-run

Treino completo pt-BR com tracking:
    python finetune_commonvoice.py \\
        --language pt \\
        --run-name tdv1-cv-pt-v1 \\
        --model-name tdv1-pt-proprietario \\
        --num-epochs 3

Retomar de checkpoint:
    python finetune_commonvoice.py --resume-from-checkpoint auto

Modo LoRA (menos VRAM, treino mais rápido):
    python finetune_commonvoice.py --use-lora --batch-size 16

─────────────────────────────────────────────────────────────────────────────
Modelo base:  openai/whisper-medium  (769M params, multilíngue)
Estratégia:   Full fine-tune (padrão) ou LoRA --use-lora
Saída:        HuggingFace Transformers + CTranslate2 int8
Idioma padrão: pt (pt-BR — token <|pt|> fixado durante treino)
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

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

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
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
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


# ─────────────────────────────────────────────────────────────────────────────
# Constantes e configuração padrão
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent

_DEFAULT_CVSS_DIR   = _ROOT / "cvss"
_DEFAULT_OUTPUT_DIR = _ROOT / "models" / "tdv1-cv-pt"
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
            sentence = row.get("sentence", "").strip()
            if not sentence:
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
    ) -> None:
        import librosa as _librosa
        self._librosa   = _librosa
        self._processor = processor

        processor.tokenizer.set_prefix_tokens(language=whisper_language, task=task)

        # Pré-tokeniza para filtrar textos que excedem o limite do decoder
        valid, skipped = [], 0
        for e in entries:
            ids = processor.tokenizer(e["sentence"]).input_ids
            if len(ids) <= self._MAX_LABEL_TOKENS:
                valid.append({"path": e["audio_filepath"], "label_ids": ids})
            else:
                skipped += 1

        if skipped:
            log.warning(
                "%s: %d amostras ignoradas (labels > %d tokens)",
                label, skipped, self._MAX_LABEL_TOKENS,
            )
        log.info("%s: %d amostras válidas", label, len(valid))
        self._items = valid

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        item = self._items[idx]
        try:
            array, _ = self._librosa.load(item["path"], sr=16_000, mono=True)
        except Exception as exc:
            log.debug("Falha ao carregar %s: %s — substituindo por silêncio", item["path"], exc)
            array = np.zeros(16_000, dtype=np.float32)

        feats = self._processor.feature_extractor(
            array, sampling_rate=16_000, return_tensors="pt"
        )
        return {
            "input_features": feats.input_features[0],
            "labels":         item["label_ids"],
        }


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
    wer_metric = evaluate.load("wer")
    normalizer = getattr(processor.tokenizer, "normalize", None) or processor.tokenizer._normalize

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str  = [normalizer(p) for p in pred_str]
        label_str = [normalizer(l) for l in label_str]

        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l]
        if not pairs:
            return {"wer": float("nan")}
        pred_f, label_f = zip(*pairs)
        wer = wer_metric.compute(predictions=list(pred_f), references=list(label_f))
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
    """
    ct2_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-m", "ctranslate2.converters.transformers",
         "--model", str(hf_dir), "--output_dir", str(ct2_dir),
         "--quantization", quantization, "--force"],
        ["ct2-transformers-converter",
         "--model", str(hf_dir), "--output_dir", str(ct2_dir),
         "--quantization", quantization, "--force"],
    ]
    for cmd in commands:
        try:
            log.info("Convertendo para CTranslate2 (quantização: %s) ...", quantization)
            subprocess.run(cmd, check=True)
            log.info("Conversão concluída → %s", ct2_dir)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    log.warning(
        "Conversão automática falhou. Execute manualmente:\n"
        "  ct2-transformers-converter --model %s --output_dir %s --quantization %s",
        hf_dir, ct2_dir, quantization,
    )
    return False


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

    set_seed(42)

    lang_code      = args.language.lower()
    whisper_lang   = _LANGUAGE_MAP.get(lang_code, lang_code)
    cvss_dir       = Path(args.cvss_dir).resolve()
    output_dir     = Path(args.output_dir).resolve()
    ct2_dir        = output_dir.parent / (output_dir.name + "-ct2")

    log.info("═" * 65)
    log.info("TDv1 — Fine-tuning proprietário via Mozilla Common Voice")
    log.info("  Modelo base  : %s", args.base_model)
    log.info("  Idioma       : %s (%s)", lang_code, whisper_lang)
    log.info("  Modo         : %s", "LoRA" if args.use_lora else "Full fine-tune")
    log.info("  Corpus       : %s", cvss_dir)
    log.info("  Saída HF     : %s", output_dir)
    log.info("  Saída CT2    : %s", ct2_dir)
    log.info("  GPU          : %s", "CUDA" if torch.cuda.is_available() else "CPU (lento)")
    log.info("═" * 65)

    # ── 1. Extrair corpus ─────────────────────────────────────────────────────
    corpus_dir = extract_corpus(cvss_dir, lang_code)

    # ── 2. Carregar splits ────────────────────────────────────────────────────
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
        "Amostras: %d treino | %d avaliação",
        len(train_entries), len(eval_entries),
    )

    if args.dry_run:
        log.info("[dry-run] Dataset pronto. Nenhum treino executado.")
        return

    # ── 3. Processor (carregado antes do dataset para pré-tokenizar labels) ──
    log.info("Carregando processor: %s", args.base_model)
    processor = WhisperProcessor.from_pretrained(args.base_model)

    # ── 4. Dataset lazy (PyTorch) — sem datasets.map(), sem IPC, sem OOM ─────
    train_ds = CommonVoiceDataset(train_entries, processor, args.task, whisper_lang, "treino")
    eval_ds  = CommonVoiceDataset(eval_entries,  processor, args.task, whisper_lang, "avaliação")

    with MlflowTracker(args) as tracker:
        tracker.log_params(args, len(train_ds), len(eval_ds))

        # ── 5. Modelo ─────────────────────────────────────────────────────────
        log.info("Carregando modelo: %s", args.base_model)
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

        # Configura geração para o idioma alvo
        model.generation_config.language          = whisper_lang
        model.generation_config.task              = args.task
        model.generation_config.forced_decoder_ids = None

        if args.use_lora:
            model = apply_lora(model)

        # ── 7. Collator + Métricas ────────────────────────────────────────────
        data_collator   = WhisperDataCollator(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        compute_metrics = make_compute_metrics_fn(processor)

        # ── 8. Training arguments ─────────────────────────────────────────────
        use_fp16 = torch.cuda.is_available() and not args.bf16
        use_bf16 = args.bf16 and torch.cuda.is_available()

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=max(1, args.batch_size // 2),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps if args.max_steps else -1,
            num_train_epochs=args.num_epochs if not args.max_steps else 1,
            gradient_checkpointing=True,
            fp16=use_fp16,
            bf16=use_bf16,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            save_total_limit=3,
            logging_steps=50,
            report_to=tracker.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            predict_with_generate=True,
            generation_max_length=225,
            dataloader_num_workers=0,   # Windows exige 0; Linux pode usar 4
            remove_unused_columns=False,
        )

        # ── 9. Trainer ────────────────────────────────────────────────────────
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=processor.feature_extractor,
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

        # ── 13. Conversão CTranslate2 (int8 → produção) ───────────────────────
        if args.convert_ct2:
            ok = convert_to_ctranslate2(output_dir, ct2_dir, quantization=args.ct2_quantization)
            if ok:
                log.info(
                    "\nPara usar no tdvx, atualize engine.py:\n"
                    "  WhisperModel('%s', device='cuda', compute_type='%s')",
                    ct2_dir, args.ct2_quantization,
                )

        tracker.log_final(metrics)

    log.info("═" * 65)
    log.info("Fine-tuning concluído!")
    log.info("  WER pt-BR    : %.2f%%", wer)
    log.info("  Modelo HF    : %s", output_dir)
    if args.convert_ct2:
        log.info("  Modelo CT2   : %s", ct2_dir)
    log.info("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tuning proprietário do TDv1 via Mozilla Common Voice",
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
    p.add_argument("--num-epochs",               type=int,   default=3)
    p.add_argument("--batch-size",               type=int,   default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4,
                   help="Batch efetivo = batch_size × grad_accum (ex.: 8×4=32)")
    p.add_argument("--learning-rate",            type=float, default=1e-5)
    p.add_argument("--warmup-steps",             type=int,   default=500)
    p.add_argument("--eval-steps",               type=int,   default=500,
                   help="Frequência de avaliação e checkpoint")

    # Precisão
    p.add_argument("--bf16", action="store_true",
                   help="Usa bfloat16 (RTX 4090/A100). Padrão: float16 em GPU")

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

    # Utilitário
    p.add_argument("--dry-run", action="store_true",
                   help="Valida extração e carregamento sem treinar")

    args = p.parse_args()

    # Normaliza: 0 → None para max_samples
    if args.max_train_samples == 0:
        args.max_train_samples = None
    if args.max_eval_samples == 0:
        args.max_eval_samples = None

    return args


if __name__ == "__main__":
    run_finetune(parse_args())
