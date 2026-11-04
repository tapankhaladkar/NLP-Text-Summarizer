import json
import logging

import pytest
from transformers import AutoModelForSeq2SeqLM

from textsummarizer.components.model_trainer import ModelTrainer
from textsummarizer.errors import UserFacingError


def trainer_for(cfg, **overrides):
    params = cfg.training.model_copy(update=overrides)
    return ModelTrainer(cfg.model_trainer, params, cfg.generation)


def test_train_refuses_when_validation_never_ran(prepared):
    prepared.model_trainer.status_file.unlink()
    with pytest.raises(UserFacingError, match="validate"):
        trainer_for(prepared).train()


def test_train_refuses_when_validation_failed(prepared):
    prepared.model_trainer.status_file.write_text("Validation status: False\n- train: bad\n")
    with pytest.raises(UserFacingError, match="validate"):
        trainer_for(prepared).train()


def test_train_saves_a_loadable_model_and_tokenizer(prepared):
    model_dir = trainer_for(prepared, max_steps=2).train()
    assert (model_dir / "config.json").is_file() and (model_dir / "tokenizer.json").is_file()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    assert model.generation_config.num_beams == prepared.generation.num_beams
    assert model.generation_config.max_length == prepared.generation.max_length
    assert model.generation_config.min_length == prepared.generation.min_length


def test_train_writes_metrics_with_validation_rouge(prepared):
    trainer_for(prepared, max_steps=2).train()
    metrics = json.loads(prepared.model_trainer.metrics_file.read_text())
    assert metrics["base_model"] == prepared.model_trainer.base_model
    assert {"eval_rouge1", "eval_rougeL", "eval_rougeLsum"} <= set(metrics["validation"])
    assert metrics["best_checkpoint"].startswith("checkpoint-")
    assert metrics["train_samples"] == 4 and metrics["eval_samples"] == 2


def test_each_epoch_is_checkpointed_and_only_two_are_kept(prepared):
    trainer_for(prepared, num_train_epochs=3).train()
    checkpoints = sorted(
        p.name for p in prepared.model_trainer.checkpoints_dir.glob("checkpoint-*")
    )
    assert len(checkpoints) == 2


def test_resume_continues_from_the_last_checkpoint(prepared, caplog):
    caplog.set_level(logging.INFO)
    trainer_for(prepared, num_train_epochs=1).train()
    trainer_for(prepared, num_train_epochs=2).train(resume=True)
    assert "Resuming from" in caplog.text
    assert (prepared.model_trainer.checkpoints_dir / "checkpoint-4").is_dir()


def test_resume_without_a_checkpoint_starts_from_scratch(prepared, caplog):
    caplog.set_level(logging.INFO)
    trainer_for(prepared, max_steps=2).train(resume=True)
    assert "No checkpoint found" in caplog.text


def test_sample_limits_shrink_the_data(prepared):
    trainer_for(prepared, max_steps=1, max_train_samples=2, max_eval_samples=1).train()
    metrics = json.loads(prepared.model_trainer.metrics_file.read_text())
    assert (metrics["train_samples"], metrics["eval_samples"]) == (2, 1)
