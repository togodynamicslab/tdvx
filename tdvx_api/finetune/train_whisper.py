"""
train_whisper.py — Pipeline de fine-tuning do Whisper para gírias PT-BR

Pré-requisitos:
  pip install transformers datasets evaluate mlflow accelerate

Coleta de dados:
  Use tdvx_api para acumular segmentos em finetuning_data/manifest.jsonl
  Recomendado: >= 500 segmentos com confidence >= 0.70

Iniciar servidor MLflow (tracking de experimentos):
  mlflow server --host 0.0.0.0 --port 5000

Executar fine-tuning:
  cd tdvx_api
  python -m finetune.train_whisper

Acompanhar métricas:
  http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune")

# ── Configuração ──────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Dados
    manifest_path: str = "finetuning_data/manifest.jsonl"
    base_audio_dir: str = "finetuning_data"

    # Modelo base
    model_id: str = "openai/whisper-medium"
    language: str = "portuguese"
    task: str = "transcribe"

    # Saída
    output_dir: str = "finetuning_data/checkpoints"
    mlflow_uri: str = "http://localhost:5000"
    run_name: str = "whisper-medium-girias-ptbr"

    # Filtragem de dados
    min_confidence: float = 0.70
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0

    # Treino
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2    # batch efetivo = 16
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    max_steps: int = 3000
    eval_steps: int = 300
    save_steps: int = 300
    fp16: bool = True                       # desativar se CPU
    gradient_checkpointing: bool = True     # economiza VRAM
    test_size: float = 0.1
    seed: int = 42


CFG = TrainConfig()

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: str, base_dir: str, cfg: TrainConfig) -> list[dict]:
    """Lê manifest.jsonl e filtra por confiança e duração."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest não encontrado: {path}\n"
            "Execute a API tdvx_api para coletar dados primeiro."
        )

    entries = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue

            if e.get("confidence", 0) < cfg.min_confidence:
                skipped += 1
                continue
            dur = e.get("duration", 0)
            if dur < cfg.min_duration_s or dur > cfg.max_duration_s:
                skipped += 1
                continue
            if not e.get("text", "").strip():
                skipped += 1
                continue

            # Resolve caminho absoluto do áudio
            e["_audio_path"] = str(Path(base_dir) / e["audio_filepath"])
            entries.append(e)

    log.info("Manifest: %d entradas válidas, %d descartadas", len(entries), skipped)
    return entries


def build_hf_dataset(entries: list[dict], cfg: TrainConfig):
    """Converte lista de entradas para HuggingFace Dataset com áudio."""
    from datasets import Dataset, Audio as HFAudio

    records = [
        {"audio": e["_audio_path"], "text": e["text"], "speaker": e.get("speaker", "")}
        for e in entries
    ]
    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))
    return ds.train_test_split(test_size=cfg.test_size, seed=cfg.seed)


# ── Preprocessamento ──────────────────────────────────────────────────────────

def make_prepare_fn(processor):
    """Retorna função de pré-processamento compatível com datasets.map()."""

    def prepare(batch):
        audio = batch["audio"]["array"]
        sr = batch["audio"]["sampling_rate"]

        # Resample se necessário (segurança)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        batch["input_features"] = processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return prepare


def make_compute_metrics(processor):
    """Retorna função de métricas que calcula WER."""
    from evaluate import load as load_metric
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        score = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(score, 4)}

    return compute_metrics


# ── Data collator ─────────────────────────────────────────────────────────────

class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator padrão para Whisper fine-tuning."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove token de início de decodificação se presente
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── MLflow callback ───────────────────────────────────────────────────────────

class MLflowMetricsCallback:
    """Loga métricas de treino/eval no MLflow a cada step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        import mlflow
        step = state.global_step
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if metrics:
            mlflow.log_metrics(metrics, step=step)


# ── Treinamento principal ─────────────────────────────────────────────────────

def train(cfg: TrainConfig = CFG) -> str:
    """
    Executa fine-tuning completo e retorna o run_id do MLflow.

    Returns:
        run_id (str) para recuperar o modelo treinado via MLflow.
    """
    import mlflow
    import mlflow.pytorch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    # ── Setup MLflow ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment("whisper-finetune-ptbr")

    with mlflow.start_run(run_name=cfg.run_name) as run:
        run_id = run.info.run_id
        log.info("MLflow run iniciado: %s", run_id)

        mlflow.log_params({
            "base_model": cfg.model_id,
            "language": cfg.language,
            "manifest": cfg.manifest_path,
            "min_confidence": cfg.min_confidence,
            "learning_rate": cfg.learning_rate,
            "batch_size_effective": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
            "max_steps": cfg.max_steps,
            "fp16": cfg.fp16,
        })

        # ── Dados ─────────────────────────────────────────────────────────────
        entries = load_manifest(cfg.manifest_path, cfg.base_audio_dir, cfg)
        if len(entries) < 50:
            log.warning(
                "Apenas %d entradas — recomendado >= 500 para fine-tuning efetivo",
                len(entries)
            )
        mlflow.log_param("train_samples", int(len(entries) * (1 - cfg.test_size)))
        mlflow.log_param("eval_samples", int(len(entries) * cfg.test_size))

        splits = build_hf_dataset(entries, cfg)

        # ── Modelo e processor ────────────────────────────────────────────────
        log.info("Carregando %s …", cfg.model_id)
        processor = WhisperProcessor.from_pretrained(
            cfg.model_id, language=cfg.language, task=cfg.task
        )
        model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
        model.generation_config.language = cfg.language
        model.generation_config.task = cfg.task
        model.generation_config.forced_decoder_ids = None

        if cfg.gradient_checkpointing:
            model.config.use_cache = False

        # ── Pré-processamento ─────────────────────────────────────────────────
        log.info("Pré-processando dataset …")
        prepare = make_prepare_fn(processor)
        cols_to_remove = splits["train"].column_names

        splits = splits.map(
            prepare,
            remove_columns=cols_to_remove,
            num_proc=1,    # aumentar se CPU tiver muitos cores
        )

        # ── Treinamento ───────────────────────────────────────────────────────
        training_args = Seq2SeqTrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            fp16=cfg.fp16,
            evaluation_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_steps=cfg.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=25,
            report_to="none",   # MLflow via callback manual
            seed=cfg.seed,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=splits["train"],
            eval_dataset=splits["test"],
            tokenizer=processor.feature_extractor,
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
            compute_metrics=make_compute_metrics(processor),
            callbacks=[MLflowMetricsCallback()],
        )

        log.info("Iniciando treino — %d steps …", cfg.max_steps)
        trainer.train()

        # ── Métricas finais ───────────────────────────────────────────────────
        final_metrics = trainer.evaluate()
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
        log.info("WER final: %.4f", final_metrics.get("eval_wer", -1))

        # ── Salva artefatos ───────────────────────────────────────────────────
        best_ckpt = Path(cfg.output_dir)
        trainer.save_model(str(best_ckpt / "best"))
        processor.save_pretrained(str(best_ckpt / "best"))
        mlflow.log_artifacts(str(best_ckpt / "best"), artifact_path="model")

        log.info("Modelo salvo em: %s/best", cfg.output_dir)
        log.info("MLflow run_id: %s", run_id)
        log.info("Para usar: WHISPER_MODEL=%s/best no .env", cfg.output_dir)

    return run_id


# ── Utilitário: carregar modelo treinado ──────────────────────────────────────

def load_finetuned(checkpoint_dir: str):
    """
    Carrega o modelo fine-tunado para uso em produção.

    Exemplo de uso no .env:
        WHISPER_MODEL=finetuning_data/checkpoints/best

    Retorna (model, processor) prontos para inferência.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    log.info("Carregando modelo fine-tunado de: %s", checkpoint_dir)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
    processor = WhisperProcessor.from_pretrained(checkpoint_dir)
    return model, processor


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning Whisper para gírias PT-BR")
    parser.add_argument("--manifest", default=CFG.manifest_path)
    parser.add_argument("--model", default=CFG.model_id, help="ID HuggingFace ou path local")
    parser.add_argument("--steps", type=int, default=CFG.max_steps)
    parser.add_argument("--mlflow-uri", default=CFG.mlflow_uri)
    parser.add_argument("--output", default=CFG.output_dir)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        manifest_path=args.manifest,
        model_id=args.model,
        max_steps=args.steps,
        mlflow_uri=args.mlflow_uri,
        output_dir=args.output,
        fp16=not args.no_fp16,
    )

    run_id = train(cfg)
    print(f"\nConcluído! MLflow run_id: {run_id}")
    print(f"Para usar o modelo: defina WHISPER_MODEL={cfg.output_dir}/best no .env")
