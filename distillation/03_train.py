"""
Step 3: Knowledge distillation training.

Trains the student model using:
- KL divergence loss (student mimics teacher's output distribution)
- Cross-entropy loss (student matches pseudo-labels)
- Full model training (encoder + decoder, since student is randomly initialized)

The student has a different architecture from the teacher (small encoder vs
large encoder), so each runs its own full forward pass.

Usage:
    # Full training
    python 03_train.py

    # Resume from checkpoint
    python 03_train.py --resume

    # Quick test run (100 steps)
    python 03_train.py --test-run
"""

import argparse
import os
import logging
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STUDENT_MODEL = "./student-init"
TEACHER_MODEL = "openai/whisper-large-v3"
DATASET_DIR = "./dataset-pt-en/train_with_labels"
OUTPUT_DIR = "./distil-whisper-pt-en"

# Training hyperparameters
LEARNING_RATE = 1e-4
WARMUP_STEPS = 500
MAX_STEPS = 60_000
BATCH_SIZE = 4  # Reduced — teacher + student both in VRAM
GRADIENT_ACCUMULATION = 8  # Effective batch = 32
TEMPERATURE = 2.0
KL_WEIGHT = 0.5  # Balance CE and KL (0.5 each for fresh student)


@dataclass
class DataCollatorSpeechSeq2Seq:
    """Custom data collator that prepares batches for distillation."""
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract audio features
        audio_arrays = [f["audio"]["array"] for f in features]

        input_features = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).input_features

        # Tokenize labels (use teacher_text if available, else original sentence)
        label_texts = [f.get("teacher_text", f["sentence"]) for f in features]
        languages = [f["language"] for f in features]

        # Set language for tokenizer
        labels_batch = []
        for text, lang in zip(label_texts, languages):
            self.processor.tokenizer.set_prefix_tokens(language=lang, task="transcribe")
            label_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids[0]
            labels_batch.append(label_ids)

        # Pad labels
        max_label_len = max(len(l) for l in labels_batch)
        padded_labels = []
        for label_ids in labels_batch:
            padding = torch.full(
                (max_label_len - len(label_ids),),
                -100,
                dtype=label_ids.dtype,
            )
            padded_labels.append(torch.cat([label_ids, padding]))

        labels = torch.stack(padded_labels)

        return {
            "input_features": input_features,
            "labels": labels,
        }


class DistillationTrainer(Seq2SeqTrainer):
    """Custom trainer that adds KL divergence loss from teacher.

    Since the student has a different architecture (small encoder)
    from the teacher (large encoder), each model runs its own
    complete forward pass on the same input features.
    """

    def __init__(self, teacher_model=None, temperature=2.0, kl_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.kl_weight = kl_weight

        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Student forward pass
        outputs = model(**inputs)
        ce_loss = outputs.loss

        if self.teacher is None:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)

        # KL divergence loss — align on the shorter sequence length
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits

        min_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_len, :] / self.temperature
        teacher_logits = teacher_logits[:, :min_len, :] / self.temperature

        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Combined loss
        loss = (1 - self.kl_weight) * ce_loss + self.kl_weight * kl_loss

        return (loss, outputs) if return_outputs else loss


def train(args):
    logger.info("Loading dataset...")
    dataset = load_from_disk(DATASET_DIR)
    logger.info(f"Dataset: {len(dataset)} samples")

    # Split into train/eval
    split = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load student
    logger.info(f"Loading student model: {STUDENT_MODEL}")
    student = WhisperForConditionalGeneration.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = WhisperProcessor.from_pretrained(STUDENT_MODEL)

    # Do NOT freeze encoder — it's randomly initialized, needs training
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} (100% — fresh student)")

    # Load teacher (for KL loss)
    teacher = None
    if not args.no_teacher:
        logger.info(f"Loading teacher model: {TEACHER_MODEL}")
        teacher = WhisperForConditionalGeneration.from_pretrained(
            TEACHER_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        # Move teacher to same device as student (trainer will handle student)
        if torch.cuda.is_available():
            teacher = teacher.cuda()

    # Data collator
    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=student.config.decoder_start_token_id,
    )

    # Metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    max_steps = 100 if args.test_run else MAX_STEPS
    eval_steps = 25 if args.test_run else 1000
    save_steps = 50 if args.test_run else 1000

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=448,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        push_to_hub=False,
    )

    # Trainer
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=TEMPERATURE,
        kl_weight=KL_WEIGHT,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Train
    if args.resume:
        logger.info("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save final model
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    logger.info(f"Saved final model to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--test-run", action="store_true", help="Quick test (100 steps)")
    parser.add_argument("--no-teacher", action="store_true", help="Train without KL loss (CE only)")
    args = parser.parse_args()
    train(args)
