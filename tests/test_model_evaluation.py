import json

import pytest

from textsummarizer.components.model_evaluation import ModelEvaluation
from textsummarizer.components.model_trainer import ModelTrainer
from textsummarizer.errors import UserFacingError
from textsummarizer.metrics import ROUGE_KEYS


def evaluation_for(cfg, bertscore_fn=None, **overrides):
    params = cfg.evaluation.model_copy(update=overrides)
    return ModelEvaluation(cfg.model_evaluation, cfg.generation, params, bertscore_fn)


@pytest.fixture
def trained(prepared):
    params = prepared.training.model_copy(update={"max_steps": 2})
    ModelTrainer(prepared.model_trainer, params, prepared.generation).train()
    return prepared


def test_metrics_cover_baseline_fine_tuned_and_improvement(trained):
    path = evaluation_for(trained).evaluate()
    metrics = json.loads(path.read_text())
    assert metrics["num_examples"] == 2
    for section in ("baseline", "fine_tuned"):
        assert all(0 <= metrics[section][key] <= 100 for key in ROUGE_KEYS)
    for key in ROUGE_KEYS:
        gap = metrics["fine_tuned"][key] - metrics["baseline"][key]
        assert metrics["improvement"][key] == pytest.approx(gap, abs=0.01)


def test_both_models_use_the_recorded_generation_settings(trained):
    metrics = json.loads(evaluation_for(trained).evaluate().read_text())
    assert metrics["generation"] == trained.generation.model_dump()


def test_examples_file_lists_the_requested_number_of_examples(trained):
    evaluation_for(trained, num_examples=1).evaluate()
    text = trained.model_evaluation.examples_file.read_text()
    assert text.count("### Example") == 1


def test_sample_limit_shrinks_the_evaluation(trained):
    metrics = json.loads(evaluation_for(trained, max_test_samples=1).evaluate().read_text())
    assert metrics["num_examples"] == 1


def test_bertscore_is_used_only_when_a_model_is_configured(trained):
    calls = []

    def fake(predictions, references):
        calls.append(len(predictions))
        return 87.5

    off = json.loads(evaluation_for(trained, fake, bertscore_model=None).evaluate().read_text())
    assert off["baseline"]["bertscore_f1"] is None and calls == []
    on = json.loads(evaluation_for(trained, fake, bertscore_model="x").evaluate().read_text())
    assert on["fine_tuned"]["bertscore_f1"] == 87.5 and calls == [2, 2]


def test_missing_fine_tuned_model_points_at_the_train_command(prepared):
    with pytest.raises(UserFacingError, match="train"):
        evaluation_for(prepared).evaluate()
