import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from textsummarizer.config import GenerationParams, TrainerConfig, TrainingParams
from textsummarizer.errors import UserFacingError
from textsummarizer.metrics import compute_rouge
from textsummarizer.storage import atomic_directory

logger = logging.getLogger(__name__)

VALIDATED = "Validation status: True"
SETTINGS_FILE = "run_settings.json"
# Settings that change what a run learns. Resuming with different values would silently
# mix two runs, so --resume refuses. Epoch count and step limit may change: extending a
# run is a normal reason to resume.
RESUME_SENSITIVE = (
    "learning_rate",
    "warmup_steps",
    "weight_decay",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "fp16",
    "gradient_checkpointing",
    "optim",
    "seed",
    "max_train_samples",
)


class ModelTrainer:
    """Fine-tune a seq2seq checkpoint on the tokenized, validated dataset."""

    def __init__(
        self,
        config: TrainerConfig,
        params: TrainingParams,
        generation: GenerationParams,
        callbacks: list | None = None,
    ):
        self.config = config
        self.params = params
        self.generation = generation
        self.callbacks = callbacks

    def train(self, resume: bool = False) -> Path:
        self._require_validated_data()
        checkpoint = self._checkpoint_to_resume(resume)
        dataset = load_from_disk(str(self.config.data_dir))
        train_set = _limit(dataset["train"], self.params.max_train_samples)
        eval_set = _limit(dataset["validation"], self.params.max_eval_samples)

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model)
        # The base checkpoint's own generation defaults (Pegasus forces min_length=32) would
        # otherwise steer validation and best-checkpoint selection, and be saved with the model.
        model.generation_config.update(**self.generation.model_dump())
        trainer = Seq2SeqTrainer(
            model=model,
            args=self._training_arguments(),
            train_dataset=train_set,
            eval_dataset=eval_set,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            processing_class=tokenizer,
            compute_metrics=_rouge_metrics(tokenizer),
            callbacks=self.callbacks,
        )

        result = trainer.train(resume_from_checkpoint=checkpoint)
        if checkpoint is not None and trainer.state.global_step == _checkpoint_step(checkpoint):
            logger.warning(
                "No training steps ran: %s is already at the final step of the configured "
                "schedule. Delete %s to train again.",
                Path(checkpoint).name,
                self.config.checkpoints_dir,
            )
        validation = trainer.evaluate()

        with atomic_directory(self.config.model_dir) as tmp:
            trainer.save_model(str(tmp))
            tokenizer.save_pretrained(str(tmp))

        best = trainer.state.best_model_checkpoint
        self._write_metrics(
            {
                "base_model": self.config.base_model,
                "train": result.metrics,
                "validation": validation,
                "best_checkpoint": Path(best).name if best else None,
                "train_samples": len(train_set),
                "eval_samples": len(eval_set),
                "training_params": self.params.model_dump(),
            }
        )
        logger.info("Saved the fine-tuned model to %s", self.config.model_dir)
        return self.config.model_dir

    def _require_validated_data(self) -> None:
        status = self.config.status_file
        lines = status.read_text().splitlines() if status.is_file() else []
        if not lines or lines[0].strip() != VALIDATED:
            raise UserFacingError(
                f"The data has not passed validation (see {status}). "
                "Run `textsummarizer validate` first."
            )

    def _training_arguments(self) -> Seq2SeqTrainingArguments:
        p = self.params
        if p.max_steps > 0:  # smoke runs: evaluate and save once, at the last step
            schedule = {
                "eval_strategy": "steps",
                "eval_steps": p.max_steps,
                "save_strategy": "steps",
                "save_steps": p.max_steps,
            }
        else:
            schedule = {"eval_strategy": "epoch", "save_strategy": "epoch"}
        return Seq2SeqTrainingArguments(
            output_dir=str(self.config.checkpoints_dir),
            num_train_epochs=p.num_train_epochs,
            max_steps=p.max_steps,
            learning_rate=p.learning_rate,
            warmup_steps=p.warmup_steps,
            weight_decay=p.weight_decay,
            optim=p.optim,
            per_device_train_batch_size=p.per_device_train_batch_size,
            per_device_eval_batch_size=p.per_device_eval_batch_size,
            gradient_accumulation_steps=p.gradient_accumulation_steps,
            fp16=p.fp16 and torch.cuda.is_available(),
            gradient_checkpointing=p.gradient_checkpointing,
            seed=p.seed,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=self.generation.max_length,
            generation_num_beams=p.eval_num_beams,
            logging_steps=10,
            report_to=p.report_to,
            **schedule,
        )

    def _checkpoint_to_resume(self, resume: bool) -> str | None:
        """Decide whether this run continues an earlier one, and refuse if that is unsafe."""
        directory = self.config.checkpoints_dir
        existing = get_last_checkpoint(str(directory)) if directory.is_dir() else None
        if existing is not None and not resume:
            raise UserFacingError(
                f"Checkpoints from an earlier run exist in {directory}. Pass --resume to "
                "continue that run, or delete the folder to start over."
            )
        if existing is None:
            if resume:
                logger.info("No checkpoint found in %s; starting from scratch", directory)
            self._record_settings()
            return None
        self._require_same_settings()
        logger.info("Resuming from %s", existing)
        return existing

    def _settings(self) -> dict:
        dump = self.params.model_dump()
        return {key: dump[key] for key in RESUME_SENSITIVE}

    def _record_settings(self) -> None:
        directory = self.config.checkpoints_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SETTINGS_FILE).write_text(json.dumps(self._settings(), indent=2))

    def _require_same_settings(self) -> None:
        directory = self.config.checkpoints_dir
        record = directory / SETTINGS_FILE
        if not record.is_file():
            logger.warning(
                "No %s in %s; cannot check the settings are unchanged", SETTINGS_FILE, directory
            )
            return
        recorded = json.loads(record.read_text())
        changed = sorted(
            key for key, value in self._settings().items() if recorded.get(key) != value
        )
        if changed:
            raise UserFacingError(
                f"The training settings changed since the checkpoints in {directory} were "
                f"written ({', '.join(changed)}). Restore them, or delete the folder to start over."
            )

    def _write_metrics(self, metrics: dict) -> None:
        self.config.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_file.write_text(json.dumps(metrics, indent=2, default=str))


def _checkpoint_step(checkpoint: str) -> int | None:
    state_file = Path(checkpoint) / "trainer_state.json"
    if not state_file.is_file():
        return None
    return json.loads(state_file.read_text()).get("global_step")


def _limit(dataset: Dataset, count: int | None) -> Dataset:
    return dataset if count is None else dataset.select(range(min(count, len(dataset))))


def _rouge_metrics(tokenizer):
    def compute_metrics(prediction) -> dict[str, float]:
        pad = tokenizer.pad_token_id
        predictions = np.where(prediction.predictions != -100, prediction.predictions, pad)
        labels = np.where(prediction.label_ids != -100, prediction.label_ids, pad)
        return compute_rouge(
            tokenizer.batch_decode(predictions, skip_special_tokens=True),
            tokenizer.batch_decode(labels, skip_special_tokens=True),
        )

    return compute_metrics
