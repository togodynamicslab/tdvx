#!/usr/bin/env python3
"""
finetune.py — Fine-tuning do modelo proprietário TDv1 (Whisper)
===============================================================

Consome o dataset acumulado automaticamente pelo FinetuningDatasetSaver
durante a sprint e re-treina o modelo Whisper base com dados reais da empresa.

Uso básico (após fechamento da sprint):
    python finetune.py

Com MLflow e publicação no HuggingFace Hub:
    python finetune.py \\
        --run-name tdv1-sprint-4 \\
        --model-name tdv1-pt-en \\
        --hub-model-id minha-empresa/tdv1-pt-en-v2

Sobrescrevendo parâmetros comuns:
    python finetune.py --base-model openai/whisper-large-v3 \\
                       --data-dir /caminho/finetuning_data \\
                       --output-dir models/tdv1-finetuned \\
                       --max-steps 2000

─────────────────────────────────────────────────────────────────────────────
Modelo base:   openai/whisper-medium  (TDv1-Fast / TDv1-Balanced)
               openai/whisper-large-v3 (TDv1) via --base-model
Abordagem:     Fine-tuning multilíngue completo com HuggingFace Seq2SeqTrainer.
               Cada amostra carrega seu próprio token de idioma (<|pt|> / <|en|>)
               nos labels — o modelo aprende a transcrever ambos os idiomas.
Idiomas:       Português (pt) + Inglês (en) por padrão
Dados:         manifest.jsonl gerado pelo app (audio 16kHz WAV + transcrição)
Tracking:      MLflow (opcional) — métricas por step, params, Model Registry
─────────────────────────────────────────────────────────────────────────────

Dependências extras (não incluídas no requirements.txt base):
    pip install -r requirements-finetune.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

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
# Imports com mensagem de erro amigável se dependências de treino faltarem
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_DEPS_MISSING = False
try:
    import evaluate
    from datasets import Audio, Dataset
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
    import mlflow.transformers
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuração padrão
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent

_DEFAULT_DATA_DIR    = _ROOT / "finetuning_data"
_DEFAULT_OUTPUT_DIR  = _ROOT / "models" / "tdv1-finetuned"
_DEFAULT_BASE_MODEL  = "openai/whisper-medium"
_DEFAULT_TASK        = "transcribe"
_DEFAULT_LANGUAGES   = ["portuguese", "english"]

# Mapeamento ISO 639-1 → nome completo esperado pelo WhisperTokenizer
_ISO_TO_WHISPER: Dict[str, str] = {
    "pt": "portuguese",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
}

# Filtros de qualidade do dataset
_MIN_DURATION_S = 0.5
_MAX_DURATION_S = 28.0   # limite técnico do Whisper (janela de 30s)
_MIN_CONFIDENCE = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de idioma
# ─────────────────────────────────────────────────────────────────────────────

def normalise_language(raw: str) -> str:
    raw = raw.strip().lower()
    return _ISO_TO_WHISPER.get(raw, raw)


def build_allowed_set(languages: List[str]) -> Set[str]:
    allowed: Set[str] = set()
    iso_reverse = {v: k for k, v in _ISO_TO_WHISPER.items()}
    for lang in languages:
        norm = normalise_language(lang)
        allowed.add(norm)
        if norm in iso_reverse:
            allowed.add(iso_reverse[norm])
    return allowed


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento e filtragem do dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(data_dir: Path) -> List[Dict]:
    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"manifest.jsonl não encontrado em '{data_dir}'.\n"
            "Certifique-se de que o app coletou dados antes de rodar o fine-tuning."
        )

    entries, skipped = [], 0
    with open(manifest, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Linha %d inválida no manifest — ignorada", i)
                skipped += 1

    log.info("Manifest: %d entradas carregadas, %d ignoradas", len(entries), skipped)
    return entries


def filter_entries(
    entries: List[Dict],
    data_dir: Path,
    allowed_languages: Set[str],
    min_duration: float = _MIN_DURATION_S,
    max_duration: float = _MAX_DURATION_S,
    min_confidence: float = _MIN_CONFIDENCE,
) -> List[Dict]:
    filtered = []
    reasons: Dict[str, int] = {
        "audio_missing": 0, "duration": 0, "confidence": 0, "language": 0,
    }

    for e in entries:
        audio_path = data_dir / e.get("audio_filepath", "")
        if not audio_path.exists():
            reasons["audio_missing"] += 1
            continue
        if not (min_duration <= e.get("duration", 0.0) <= max_duration):
            reasons["duration"] += 1
            continue
        if e.get("confidence", 1.0) < min_confidence:
            reasons["confidence"] += 1
            continue

        raw_lang = e.get("language", "")
        norm_lang = normalise_language(raw_lang)
        if norm_lang not in allowed_languages and raw_lang not in allowed_languages:
            reasons["language"] += 1
            continue

        filtered.append({**e, "audio_filepath": str(audio_path), "language": norm_lang})

    kept = len(filtered)
    log.info("Filtro: %d/%d entradas mantidas | descartadas → %s", kept, len(entries), reasons)
    log.info("Distribuição de idiomas: %s", dict(Counter(e["language"] for e in filtered)))

    if kept == 0:
        raise ValueError(
            "Nenhuma entrada passou pelos filtros de qualidade.\n"
            f"Idiomas aceitos: {allowed_languages}\n"
            "Verifique --languages, --min-confidence ou --min-duration."
        )
    return filtered


def dataset_stats(entries: List[Dict]) -> Dict:
    lang_dist = dict(Counter(e["language"] for e in entries))
    total_duration_s = sum(e.get("duration", 0.0) for e in entries)
    return {
        "total_entries": len(entries),
        "total_duration_h": round(total_duration_s / 3600, 2),
        "lang_distribution": lang_dist,
    }


def build_hf_dataset(entries: List[Dict], eval_fraction: float = 0.1) -> tuple:
    ds = Dataset.from_dict({
        "audio":    [e["audio_filepath"] for e in entries],
        "sentence": [e.get("text", "").strip() for e in entries],
        "language": [e["language"] for e in entries],
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    split = ds.train_test_split(test_size=eval_fraction, seed=42)
    log.info("Dataset: %d treino | %d avaliação", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────────────────────
# Pré-processamento de features (bilíngue)
# ─────────────────────────────────────────────────────────────────────────────

def make_prepare_fn(processor: "WhisperProcessor", task: str):
    """
    Injeta o token de idioma correto (<|pt|> ou <|en|>) nos labels de cada
    amostra via set_prefix_tokens(). O processor global não tem idioma fixado,
    permitindo treino multilíngue dentro do mesmo batch.

    Nota: set_prefix_tokens modifica estado do tokenizer — manter num_proc=1.
    """
    def prepare(batch):
        audio = batch["audio"]
        inputs = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt",
        )
        batch["input_features"] = inputs.input_features[0]

        processor.tokenizer.set_prefix_tokens(
            language=batch.get("language", "portuguese"), task=task
        )
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    return prepare


# ─────────────────────────────────────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Métricas (WER)
# ─────────────────────────────────────────────────────────────────────────────

def make_compute_metrics_fn(processor: "WhisperProcessor"):
    wer_metric = evaluate.load("wer")
    normalizer = processor.tokenizer._normalize

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
# MLflow — tracking de experimentos e Model Registry
# ─────────────────────────────────────────────────────────────────────────────

class MlflowTracker:
    """
    Gerencia o ciclo de vida de um run MLflow para o fine-tuning.

    Uso:
        with MlflowTracker(args) as tracker:
            tracker.log_params(args, stats)
            trainer.train()
            tracker.log_final(metrics, output_dir, ct2_dir)
            tracker.register_model(output_dir, "tdv1-pt-en")

    Quando MLflow não está disponível ou desabilitado, todos os métodos
    são no-ops — o script funciona normalmente sem tracking.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._enabled = _MLFLOW_AVAILABLE and bool(args.mlflow_uri or
                        os.environ.get("MLFLOW_TRACKING_URI"))
        self._args = args
        self.run_id: str = ""

        if _MLFLOW_AVAILABLE and not self._enabled:
            log.info("MLflow disponível mas MLFLOW_TRACKING_URI não configurado — tracking desabilitado.")
        if not _MLFLOW_AVAILABLE and (args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI")):
            log.warning("mlflow não instalado. Adicione-o ao requirements-finetune.txt.")

    def __enter__(self) -> "MlflowTracker":
        if not self._enabled:
            return self

        uri = self._args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        experiment = (
            self._args.mlflow_experiment
            or os.environ.get("MLFLOW_FINETUNE_EXPERIMENT")
            or os.environ.get("MLFLOW_EXPERIMENT_NAME", "tdv1-finetune")
        )

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

        run_name = self._args.run_name or f"tdv1-{datetime.now().strftime('%Y%m%d-%H%M')}"
        self._run = mlflow.start_run(run_name=run_name)
        self.run_id = self._run.info.run_id

        # expõe para o callback nativo do transformers Trainer
        os.environ["MLFLOW_TRACKING_URI"]    = uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment

        log.info("MLflow run iniciado: %s (id=%s)", run_name, self.run_id)
        log.info("UI: %s/#/experiments | tracking URI: %s", uri, uri)
        return self

    def __exit__(self, *_) -> None:
        if self._enabled:
            mlflow.end_run()
            log.info("MLflow run finalizado: %s", self.run_id)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_params(self, args: argparse.Namespace, stats: Dict, train_size: int, eval_size: int) -> None:
        if not self._enabled:
            return

        params = {
            # modelo
            "base_model":    args.base_model,
            "task":          args.task,
            "languages":     "+".join(sorted(args.languages)),
            # treino
            "learning_rate": args.learning_rate,
            "batch_size":    args.batch_size,
            "grad_accum":    args.gradient_accumulation_steps,
            "warmup_steps":  args.warmup_steps,
            "num_epochs":    args.num_epochs if not args.max_steps else "—",
            "max_steps":     args.max_steps or "—",
            "eval_steps":    args.eval_steps,
            # qualidade do dataset
            "min_confidence":  args.min_confidence,
            "min_duration_s":  args.min_duration,
            "eval_split":      args.eval_split,
            # dataset
            "dataset_total":     stats["total_entries"],
            "dataset_duration_h": stats["total_duration_h"],
            "train_size":        train_size,
            "eval_size":         eval_size,
        }
        # log de distribuição por idioma
        for lang, count in stats["lang_distribution"].items():
            params[f"lang_{lang}"] = count

        mlflow.log_params(params)
        mlflow.set_tags({
            "base_model": args.base_model,
            "languages":  "+".join(sorted(args.languages)),
            "sprint":     args.run_name or "manual",
        })

    def log_final(self, metrics: Dict, output_dir: Path, ct2_dir: Path) -> None:
        if not self._enabled:
            return

        # métricas finais de avaliação
        final = {k.replace("eval_", ""): v for k, v in metrics.items()
                 if isinstance(v, (int, float))}
        mlflow.log_metrics(final)

        # artefatos leves
        eval_json = output_dir / "eval_results.json"
        if eval_json.exists():
            mlflow.log_artifact(str(eval_json), artifact_path="eval")

        ct2_config = ct2_dir / "config.json"
        if ct2_config.exists():
            mlflow.log_artifact(str(ct2_config), artifact_path="ct2")

        log.info("MLflow: métricas e artefatos registrados.")

    def register_model(self, output_dir: Path, model_name: str) -> None:
        """
        Registra o modelo no MLflow Model Registry com o nome fornecido.
        Cada chamada cria uma nova versão automaticamente.

        O modelo registrado pode ser promovido via UI:
            Staging → Production
        """
        if not self._enabled or not model_name:
            return

        model_uri = f"runs:/{self.run_id}/model"
        try:
            mlflow.transformers.log_model(
                transformers_model=str(output_dir),
                artifact_path="model",
            )
            result = mlflow.register_model(model_uri=model_uri, name=model_name)
            log.info(
                "Modelo registrado no MLflow Registry: '%s' v%s",
                model_name, result.version,
            )
        except Exception as exc:
            log.warning("Falha ao registrar modelo no Registry: %s", exc)

    @property
    def report_to(self) -> List[str]:
        """Lista para Seq2SeqTrainingArguments.report_to."""
        reporters = ["mlflow"] if self._enabled else []
        if getattr(self._args, "tensorboard", False):
            reporters.append("tensorboard")
        return reporters or ["none"]


# ─────────────────────────────────────────────────────────────────────────────
# Conversão para CTranslate2 (para uso com faster-whisper)
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_ctranslate2(hf_model_dir: Path, ct2_output_dir: Path, quantization: str = "int8") -> bool:
    ct2_output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-m", "ctranslate2.converters.transformers",
         "--model", str(hf_model_dir), "--output_dir", str(ct2_output_dir),
         "--quantization", quantization, "--force"],
        ["ct2-transformers-converter",
         "--model", str(hf_model_dir), "--output_dir", str(ct2_output_dir),
         "--quantization", quantization, "--force"],
    ]
    for cmd in commands:
        try:
            log.info("Convertendo para CTranslate2 …")
            subprocess.run(cmd, check=True)
            log.info("Conversão concluída → %s", ct2_output_dir)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    log.warning(
        "Conversão automática para CTranslate2 falhou.\n"
        "Execute manualmente:\n"
        "  ct2-transformers-converter --model %s --output_dir %s --quantization %s",
        hf_model_dir, ct2_output_dir, quantization,
    )
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Publicação no HuggingFace Hub (versionamento de binários)
# ─────────────────────────────────────────────────────────────────────────────

def push_to_hub(model_dir: Path, hub_model_id: str, metrics: Dict) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.warning("huggingface_hub não encontrado — push ignorado.")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("PYANNOTE_AUTH_TOKEN")
    if not token:
        log.warning("HF_TOKEN não configurado no .env — push ignorado.")
        return

    api = HfApi(token=token)
    try:
        api.repo_info(hub_model_id, repo_type="model")
    except Exception:
        log.info("Criando repositório privado: %s", hub_model_id)
        api.create_repo(hub_model_id, private=True, repo_type="model", exist_ok=True)

    wer = metrics.get("eval_wer", "N/A")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=hub_model_id,
        repo_type="model",
        commit_message=f"fine-tune: WER={wer}%  |  langs=pt+en  |  base=whisper-medium",
    )
    log.info("Modelo publicado em https://huggingface.co/%s", hub_model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(args: argparse.Namespace) -> None:
    if _TRAIN_DEPS_MISSING:
        log.error(
            "Dependências de treino não encontradas.\n"
            "Instale com: pip install -r requirements-finetune.txt"
        )
        sys.exit(1)

    set_seed(42)

    data_dir   = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ct2_dir    = output_dir.parent / (output_dir.name + "-ct2")

    languages   = [normalise_language(l) for l in args.languages]
    allowed_set = build_allowed_set(languages)

    log.info("═" * 60)
    log.info("TDv1 Fine-tuning (bilíngue)")
    log.info("  Modelo base  : %s", args.base_model)
    log.info("  Idiomas      : %s", " + ".join(languages))
    log.info("  Dados        : %s", data_dir)
    log.info("  Saída HF     : %s", output_dir)
    log.info("  Saída CT2    : %s", ct2_dir)
    log.info("  MLflow       : %s", args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI") or "desabilitado")
    log.info("═" * 60)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    entries = load_manifest(data_dir)
    entries = filter_entries(
        entries, data_dir=data_dir, allowed_languages=allowed_set,
        min_duration=args.min_duration, max_duration=args.max_duration,
        min_confidence=args.min_confidence,
    )

    stats = dataset_stats(entries)

    if args.dry_run:
        log.info("[dry-run] %d entradas prontas para treino. Encerrando.", len(entries))
        log.info("Stats: %s", stats)
        return

    train_ds, eval_ds = build_hf_dataset(entries, eval_fraction=args.eval_split)

    # ── Abre o run MLflow (no-op se não configurado) ──────────────────────────
    with MlflowTracker(args) as tracker:

        tracker.log_params(args, stats, len(train_ds), len(eval_ds))

        # ── 2. Processor (multilíngue — sem idioma fixo) ──────────────────────
        log.info("Carregando processor: %s", args.base_model)
        processor = WhisperProcessor.from_pretrained(args.base_model)

        # ── 3. Pré-processamento ──────────────────────────────────────────────
        prepare_fn = make_prepare_fn(processor, args.task)
        log.info("Pré-processando dataset de treino …")
        train_ds = train_ds.map(prepare_fn, remove_columns=train_ds.column_names, num_proc=args.num_proc)
        log.info("Pré-processando dataset de avaliação …")
        eval_ds  = eval_ds.map(prepare_fn, remove_columns=eval_ds.column_names,  num_proc=args.num_proc)

        # ── 4. Modelo ─────────────────────────────────────────────────────────
        log.info("Carregando modelo: %s", args.base_model)
        model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
        model.generation_config.language         = None
        model.generation_config.task             = args.task
        model.generation_config.forced_decoder_ids = None

        # ── 5. Collator + Métricas ────────────────────────────────────────────
        data_collator   = WhisperDataCollator(
            processor=processor, decoder_start_token_id=model.config.decoder_start_token_id,
        )
        compute_metrics = make_compute_metrics_fn(processor)

        # ── 6. Training arguments ─────────────────────────────────────────────
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps if args.max_steps else -1,
            num_train_epochs=args.num_epochs if not args.max_steps else 1,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            logging_steps=25,
            # o Trainer loga train/loss, eval/loss e eval/wer a cada step no MLflow
            report_to=tracker.report_to,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            predict_with_generate=True,
            generation_max_length=225,
            dataloader_num_workers=0,
        )

        # ── 7. Trainer ────────────────────────────────────────────────────────
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        # ── 8. Treino ─────────────────────────────────────────────────────────
        log.info("Iniciando treinamento …")
        trainer.train()

        # ── 9. Salva modelo + processor ───────────────────────────────────────
        log.info("Salvando modelo em %s …", output_dir)
        trainer.save_model()
        processor.save_pretrained(str(output_dir))

        # ── 10. Avaliação final ───────────────────────────────────────────────
        log.info("Avaliação final …")
        metrics = trainer.evaluate()
        log.info("WER final: %.2f%%", metrics.get("eval_wer", float("nan")))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # ── 11. Conversão para CTranslate2 ────────────────────────────────────
        if args.convert_ct2:
            convert_to_ctranslate2(output_dir, ct2_dir, quantization=args.ct2_quantization)
            log.info(
                "\nPara usar o modelo no tdvx, atualize engine.py:\n"
                "  WhisperModel('%s', device='cuda', compute_type='%s')",
                ct2_dir, args.ct2_quantization,
            )

        # ── 12. MLflow — métricas finais + Model Registry ─────────────────────
        tracker.log_final(metrics, output_dir, ct2_dir)
        if args.model_name:
            tracker.register_model(output_dir, args.model_name)

        # ── 13. HuggingFace Hub (binários versionados) ────────────────────────
        if args.hub_model_id:
            push_to_hub(output_dir, args.hub_model_id, metrics)

    log.info("Fine-tuning concluído.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning bilíngue do modelo proprietário TDv1 (Whisper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Localização
    parser.add_argument("--data-dir", default=str(_DEFAULT_DATA_DIR),
                        help="Diretório com manifest.jsonl gerado pelo app")
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR),
                        help="Saída do modelo HuggingFace fine-tunado")

    # Modelo base
    parser.add_argument("--base-model", default=_DEFAULT_BASE_MODEL,
                        help="'openai/whisper-medium' (TDv1-Fast) ou 'openai/whisper-large-v3' (TDv1)")

    # Dataset
    parser.add_argument("--languages", nargs="+", default=_DEFAULT_LANGUAGES, metavar="LANG",
                        help="Ex.: --languages portuguese english  |  --languages pt en")
    parser.add_argument("--task", default=_DEFAULT_TASK, choices=["transcribe", "translate"])
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--min-duration", type=float, default=_MIN_DURATION_S)
    parser.add_argument("--max-duration", type=float, default=_MAX_DURATION_S)
    parser.add_argument("--min-confidence", type=float, default=_MIN_CONFIDENCE,
                        help="Confiança mínima do Whisper para aceitar como ground-truth")
    parser.add_argument("--num-proc", type=int, default=1,
                        help="Processos para Dataset.map (manter 1 no Windows)")

    # Treino
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Steps de treino (0 = usa --num-epochs)")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="Frequência de avaliação e checkpoint")

    # Logging
    parser.add_argument("--tensorboard", action="store_true",
                        help="Habilita TensorBoard (combinável com MLflow)")

    # CTranslate2
    parser.add_argument("--convert-ct2", action="store_true", default=True,
                        help="Converte para CTranslate2 após o treino")
    parser.add_argument("--ct2-quantization", default="int8",
                        choices=["int8", "int8_float16", "float16", "int4"],
                        help="int8_float16 recomendado para GPU")

    # MLflow
    parser.add_argument("--mlflow-uri", default="",
                        help="URI do servidor MLflow (ex.: http://localhost:5000 ou ./mlruns). "
                             "Sobrescreve MLFLOW_TRACKING_URI do .env")
    parser.add_argument("--mlflow-experiment", default="",
                        help="Nome do experimento (padrão: tdv1-finetune). "
                             "Sobrescreve MLFLOW_FINETUNE_EXPERIMENT do .env")
    parser.add_argument("--run-name", default="",
                        help="Nome do run MLflow (ex.: tdv1-sprint-4). "
                             "Padrão: tdv1-YYYYMMDD-HHMM")
    parser.add_argument("--model-name", default="",
                        help="Nome para registrar no MLflow Model Registry "
                             "(ex.: tdv1-pt-en). Cria nova versão a cada run.")

    # HuggingFace Hub
    parser.add_argument("--hub-model-id", default="", metavar="ORG/REPO",
                        help="Publica no HF Hub (ex.: minha-empresa/tdv1-pt-en-v2). "
                             "Requer HF_TOKEN no .env")

    # Utilitário
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida o dataset sem treinar")

    return parser.parse_args()


if __name__ == "__main__":
    run_finetune(parse_args())
