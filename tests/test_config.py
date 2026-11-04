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
