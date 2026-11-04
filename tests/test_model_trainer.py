import json
import logging

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSeq2SeqLM, TrainerCallback

from textsummarizer.components.model_trainer import ModelTrainer
from textsummarizer.errors import UserFacingError


class Recorder(TrainerCallback):
    """Remembers what the Trainer looked like at the moment training began."""

    def __init__(self):
        self.global_steps, self.min_lengths, self.shared_weights = [], [], []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.global_steps.append(state.global_step)
        self.min_lengths.append(model.generation_config.min_length)
        self.shared_weights.append(model.model.shared.weight.detach().clone())


def trainer_for(cfg, callbacks=None, **overrides):
    params = cfg.training.model_copy(update=overrides)
    return ModelTrainer(cfg.model_trainer, params, cfg.generation, callbacks)


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


def test_training_and_validation_use_the_configured_generation_settings(prepared):
    # The base checkpoint's own min_length (5 in the tiny fixture, 32 for real Pegasus)
    # must not leak into validation or best-checkpoint selection.
    recorder = Recorder()
    trainer_for(prepared, [recorder], max_steps=2).train()
    assert recorder.min_lengths == [prepared.generation.min_length]


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
    first, second = Recorder(), Recorder()
    trainer_for(prepared, [first], num_train_epochs=1).train()
    checkpoints = prepared.model_trainer.checkpoints_dir
    saved = load_file(str(checkpoints / "checkpoint-2" / "model.safetensors"))
    trainer_for(prepared, [second], num_train_epochs=2).train(resume=True)
    assert (first.global_steps, second.global_steps) == ([0], [2])
    assert torch.equal(second.shared_weights[0], saved["model.shared.weight"])
    assert "Resuming from" in caplog.text
    assert (checkpoints / "checkpoint-4").is_dir()


def test_resume_without_a_checkpoint_starts_from_scratch(prepared, caplog):
    caplog.set_level(logging.INFO)
    trainer_for(prepared, max_steps=2).train(resume=True)
    assert "No checkpoint found" in caplog.text


def test_train_without_resume_refuses_to_reuse_existing_checkpoints(prepared):
    trainer_for(prepared, max_steps=2).train()
    with pytest.raises(UserFacingError, match="--resume"):
        trainer_for(prepared, max_steps=2).train()


def test_resume_refuses_when_the_training_settings_changed(prepared):
    trainer_for(prepared, max_steps=2).train()
    with pytest.raises(UserFacingError, match="settings changed"):
        trainer_for(prepared, max_steps=2, learning_rate=5e-4).train(resume=True)


def test_resuming_a_finished_run_says_that_no_steps_will_run(prepared, caplog):
    caplog.set_level(logging.INFO)
    trainer_for(prepared, num_train_epochs=1).train()
    recorder = Recorder()
    trainer_for(prepared, [recorder], num_train_epochs=1).train(resume=True)
    assert "already at the final step" in caplog.text
    assert recorder.global_steps == [2]


def test_sample_limits_shrink_the_data(prepared):
    trainer_for(prepared, max_steps=1, max_train_samples=2, max_eval_samples=1).train()
    metrics = json.loads(prepared.model_trainer.metrics_file.read_text())
    assert (metrics["train_samples"], metrics["eval_samples"]) == (2, 1)


def test_training_arguments_follow_the_params(prepared, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = trainer_for(
        prepared, optim="adafactor", warmup_steps=0.05, fp16=True
    )._training_arguments()
    assert args.optim == "adafactor" and args.warmup_steps == 0.05
    assert args.fp16 is False  # requested, but there is no CUDA device
    assert args.metric_for_best_model == "rougeL" and args.save_total_limit == 2
    assert args.output_dir == str(prepared.model_trainer.checkpoints_dir)
    assert args.eval_strategy == "epoch"
    assert trainer_for(prepared, max_steps=2)._training_arguments().eval_strategy == "steps"
