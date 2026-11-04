import pytest
import yaml
from pydantic import ValidationError

from textsummarizer.config import AppConfig, load_config
from textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


def test_load_config_reads_repo_files():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.data_ingestion.dataset_name == "knkarthick/samsum"
    assert config.data_ingestion.revision == "6b929ff10edec703164e3ddb2e94aae058c9ab5f"
    assert config.data_validation.expected_rows == {"train": 14731, "validation": 818, "test": 819}
    assert config.data_validation.required_columns == ["id", "dialogue", "summary"]
    assert config.data_transformation.tokenizer_name == "google/pegasus-cnn_dailymail"
    assert config.tokenization.max_input_length == 512
    assert config.tokenization.max_target_length == 128


def test_stages_agree_on_the_dataset_directory():
    config = load_config()
    assert (
        config.data_ingestion.output_dir
        == config.data_validation.data_dir
        == config.data_transformation.data_dir
    )


def test_load_config_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    load_config()
    assert list(tmp_path.iterdir()) == []


def test_unknown_config_key_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG_FILE_PATH.read_text())
    raw["data_ingestion"]["source_url"] = "https://example.com/x.zip"
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_config(bad, PARAMS_FILE_PATH)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", PARAMS_FILE_PATH)


def test_relative_paths_are_anchored_at_the_repo_root_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    repo_root = CONFIG_FILE_PATH.parent.parent
    assert config.data_ingestion.output_dir == repo_root / "artifacts/data_ingestion/samsum_dataset"
    assert config.model_trainer.model_dir.is_absolute()


def test_absolute_paths_are_left_alone(tmp_path):
    raw = yaml.safe_load(CONFIG_FILE_PATH.read_text())
    raw["data_ingestion"]["output_dir"] = str(tmp_path / "elsewhere")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    assert load_config(config_file, PARAMS_FILE_PATH).data_ingestion.output_dir == (
        tmp_path / "elsewhere"
    )


def test_section_defined_in_both_files_is_rejected(tmp_path):
    params = yaml.safe_load(PARAMS_FILE_PATH.read_text())
    params["data_ingestion"] = {}
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(params))
    with pytest.raises(ValueError, match="defined in both"):
        load_config(CONFIG_FILE_PATH, params_file)


def test_new_stage_sections_load_from_repo_files():
    config = load_config()
    training = config.training
    assert config.model_trainer.base_model == "google/pegasus-cnn_dailymail"
    assert (training.num_train_epochs, training.learning_rate) == (3, 5e-5)
    assert training.per_device_train_batch_size * training.gradient_accumulation_steps == 16
    assert training.warmup_steps == 0.05
    assert config.generation.model_dump() == {
        "num_beams": 8,
        "max_length": 128,
        "min_length": 0,
        "length_penalty": 0.8,
    }
    assert config.publish.repo_id == "tapankhaladkar/pegasus-samsum"
    assert config.evaluation.bertscore_model == "roberta-large"


def test_cross_stage_settings_agree():
    c = load_config()
    assert c.model_trainer.status_file == c.data_validation.status_file
    assert c.model_trainer.data_dir == c.model_evaluation.data_dir
    assert c.model_trainer.data_dir == c.data_transformation.output_dir
    assert c.model_trainer.base_model == c.model_evaluation.base_model
    assert c.model_trainer.model_dir == c.model_evaluation.model_dir == c.publish.model_dir
    assert c.model_evaluation.metrics_file == c.publish.metrics_file
