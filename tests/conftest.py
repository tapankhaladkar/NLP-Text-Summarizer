import logging
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from textsummarizer.config import (
    AppConfig,
    EvaluationConfig,
    EvaluationParams,
    GenerationParams,
    IngestionConfig,
    PublishConfig,
    TokenizationParams,
    TrainerConfig,
    TrainingParams,
    TransformationConfig,
    ValidationConfig,
)

TINY_WORDS = (
    "hannah amanda betty larry hey do you have number lemme check sorry find it ask he called"
).split()


def _words(start: int, count: int) -> str:
    return " ".join(TINY_WORDS[(start + k) % len(TINY_WORDS)] for k in range(count))


def _split(rows: int) -> Dataset:
    return Dataset.from_dict(
        {
            "id": [str(i) for i in range(rows)],
            "dialogue": [_words(i, 3 + 2 * (i % 4)) for i in range(rows)],
            "summary": [_words(i + 7, 5 + (i % 2)) for i in range(rows)],
        }
    )


@pytest.fixture(autouse=True)
def reset_package_logging():
    """Remove handlers that configure_logging installed, so tests don't leak them."""
    yield
    package_logger = logging.getLogger("textsummarizer")
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
        handler.close()


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory) -> Path:
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(TINY_WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    path = tmp_path_factory.mktemp("tiny_tokenizer")
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def make_dataset_dict() -> Callable[[dict[str, int] | None], DatasetDict]:
    def build(rows: dict[str, int] | None = None) -> DatasetDict:
        rows = rows or {"train": 4, "validation": 2, "test": 2}
        return DatasetDict({split: _split(count) for split, count in rows.items()})

    return build


@pytest.fixture
def app_config(tmp_path, tiny_tokenizer_dir) -> AppConfig:
    artifacts = tmp_path / "artifacts"
    dataset_dir = artifacts / "data_ingestion" / "samsum_dataset"
    tokenized_dir = artifacts / "data_transformation" / "samsum_dataset"
    status_file = artifacts / "data_validation" / "status.txt"
    model_dir = artifacts / "model_trainer" / "model"
    metrics_file = artifacts / "model_evaluation" / "metrics.json"
    return AppConfig(
        artifacts_root=artifacts,
        data_ingestion=IngestionConfig(
            root_dir=artifacts / "data_ingestion",
            dataset_name="x/y",
            revision="0" * 40,
            output_dir=dataset_dir,
        ),
        data_validation=ValidationConfig(
            root_dir=artifacts / "data_validation",
            data_dir=dataset_dir,
            status_file=status_file,
            required_columns=["id", "dialogue", "summary"],
            expected_rows={"train": 4, "validation": 2, "test": 2},
        ),
        data_transformation=TransformationConfig(
            root_dir=artifacts / "data_transformation",
            data_dir=dataset_dir,
            output_dir=tokenized_dir,
            tokenizer_name=str(tiny_tokenizer_dir),
        ),
        model_trainer=TrainerConfig(
            root_dir=artifacts / "model_trainer",
            data_dir=tokenized_dir,
            status_file=status_file,
            base_model="tiny/base",
            checkpoints_dir=artifacts / "model_trainer" / "checkpoints",
            model_dir=model_dir,
            metrics_file=artifacts / "model_trainer" / "train_metrics.json",
        ),
        model_evaluation=EvaluationConfig(
            root_dir=artifacts / "model_evaluation",
            data_dir=tokenized_dir,
            base_model="tiny/base",
            model_dir=model_dir,
            metrics_file=metrics_file,
            examples_file=artifacts / "model_evaluation" / "examples.md",
        ),
        publish=PublishConfig(
            model_dir=model_dir,
            metrics_file=metrics_file,
            repo_id="me/pegasus-samsum",
            private=False,
        ),
        tokenization=TokenizationParams(max_input_length=6, max_target_length=4),
        training=TrainingParams(
            num_train_epochs=1,
            learning_rate=1e-3,
            warmup_steps=0.1,
            weight_decay=0.0,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            fp16=False,
            gradient_checkpointing=False,
            eval_num_beams=2,
            seed=42,
            max_steps=-1,
            max_train_samples=None,
            max_eval_samples=None,
            report_to="none",
        ),
        generation=GenerationParams(num_beams=2, max_length=8, min_length=0, length_penalty=1.0),
        evaluation=EvaluationParams(
            batch_size=2, max_test_samples=None, num_examples=3, bertscore_model=None
        ),
    )


@pytest.fixture
def config_files(app_config, tmp_path) -> tuple[Path, Path]:
    """config.yaml and params.yaml on disk that load back into `app_config`."""
    dumped = app_config.model_dump(mode="json")
    params = {
        name: dumped.pop(name) for name in ("tokenization", "training", "generation", "evaluation")
    }
    config_path, params_path = tmp_path / "config.yaml", tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(dumped))
    params_path.write_text(yaml.safe_dump(params))
    return config_path, params_path
