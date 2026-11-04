import json
import logging

import pytest

from textsummarizer.components import model_publisher
from textsummarizer.components.model_publisher import ModelPublisher, render_model_card
from textsummarizer.config import PublishConfig
from textsummarizer.errors import UserFacingError

ROUGE = {"rouge1": 30.0, "rouge2": 10.0, "rougeL": 25.0, "rougeLsum": 25.5}
BETTER = {k: v + 15.0 for k, v in ROUGE.items()}
METRICS = {
    "num_examples": 819,
    "generation": {"num_beams": 8, "max_length": 128, "min_length": 0, "length_penalty": 0.8},
    "baseline": {"model": "base", **ROUGE, "bertscore_f1": None},
    "fine_tuned": {"model": "tuned", **BETTER, "bertscore_f1": None},
    "improvement": {k: 15.0 for k in ROUGE},
}


class FakeApi:
    def __init__(self, token=None):
        self.token, self.calls = token, []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))


@pytest.fixture
def ready(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps(METRICS))
    return PublishConfig(
        model_dir=model_dir, metrics_file=metrics_file, repo_id="me/pegasus-samsum", private=False
    )


def test_model_card_has_results_and_the_license_note(ready):
    card = render_model_card(METRICS, ready, "google/pegasus-cnn_dailymail")
    assert card.startswith("---\n") and "base_model: google/pegasus-cnn_dailymail" in card
    assert "45.00" in card and "30.00" in card and "CC BY-NC-ND 4.0" in card


def test_publish_creates_the_repo_and_uploads_the_model_folder(ready):
    api = FakeApi()
    assert ModelPublisher(ready, "base", api).publish() == "me/pegasus-samsum"
    names = [name for name, _ in api.calls]
    assert names == ["create_repo", "upload_folder"]
    assert api.calls[0][1] == {"repo_id": "me/pegasus-samsum", "private": False, "exist_ok": True}
    assert api.calls[1][1]["folder_path"] == str(ready.model_dir)
    assert (ready.model_dir / "README.md").read_text().startswith("---")


def test_publish_refuses_a_model_that_does_not_beat_the_baseline(ready):
    worse = {**METRICS, "improvement": {**METRICS["improvement"], "rouge2": -1.0}}
    ready.metrics_file.write_text(json.dumps(worse))
    with pytest.raises(UserFacingError, match="--force"):
        ModelPublisher(ready, "base", FakeApi()).publish()


def test_force_publishes_anyway(ready):
    worse = {**METRICS, "improvement": {k: -1.0 for k in ROUGE}}
    ready.metrics_file.write_text(json.dumps(worse))
    assert ModelPublisher(ready, "base", FakeApi()).publish(force=True) == "me/pegasus-samsum"


def test_publish_needs_trained_and_evaluated_artifacts(ready):
    ready.metrics_file.unlink()
    with pytest.raises(UserFacingError, match="evaluate"):
        ModelPublisher(ready, "base", FakeApi()).publish()


def test_publish_needs_a_token_when_no_api_is_injected(ready, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(UserFacingError, match="HF_TOKEN"):
        ModelPublisher(ready, "base").publish()


def test_the_token_reaches_the_api_but_never_the_card_or_the_logs(ready, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET_VALUE")
    made = []
    monkeypatch.setattr(
        model_publisher, "HfApi", lambda token=None: made.append(token) or FakeApi(token)
    )
    ModelPublisher(ready, "base").publish()
    assert made == ["hf_SECRET_VALUE"]
    assert "hf_SECRET_VALUE" not in (ready.model_dir / "README.md").read_text()
    assert "hf_SECRET_VALUE" not in caplog.text
