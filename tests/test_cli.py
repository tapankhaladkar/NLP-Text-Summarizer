import subprocess
import sys

from datasets import load_from_disk
from typer.testing import CliRunner

from textsummarizer.cli import app
from textsummarizer.pipeline import (
    stage_01_data_ingestion,
    stage_04_model_trainer,
    stage_06_publish,
)
from textsummarizer.storage import save_dataset_dict

runner = CliRunner()


def invoke(config_files, *args):
    config, params = config_files
    return runner.invoke(app, ["--config", str(config), "--params", str(params), *args])


def stub_ingestion(monkeypatch, dataset_dict):
    def run(config):
        save_dataset_dict(dataset_dict, config.data_ingestion.output_dir)
        return config.data_ingestion.output_dir

    monkeypatch.setattr(stage_01_data_ingestion, "run", run)


def test_help_lists_the_stage_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "validate", "transform", "all", "train", "evaluate", "publish"):
        assert command in result.output


def test_validate_passes_on_good_data(app_config, config_files, make_dataset_dict):
    make_dataset_dict().save_to_disk(str(app_config.data_validation.data_dir))
    assert invoke(config_files, "validate").exit_code == 0
    assert app_config.data_validation.status_file.read_text().startswith("Validation status: True")


def test_validate_exits_nonzero_on_bad_data(app_config, config_files, make_dataset_dict):
    make_dataset_dict({"train": 3, "validation": 2, "test": 2}).save_to_disk(
        str(app_config.data_validation.data_dir)
    )
    assert invoke(config_files, "validate").exit_code == 1
    assert app_config.data_validation.status_file.read_text().startswith("Validation status: False")


def test_all_runs_the_three_stages(monkeypatch, app_config, config_files, make_dataset_dict):
    stub_ingestion(monkeypatch, make_dataset_dict())
    assert invoke(config_files, "all").exit_code == 0
    out = load_from_disk(str(app_config.data_transformation.output_dir))
    assert "labels" in out["train"].column_names


def test_all_stops_when_validation_fails(monkeypatch, app_config, config_files, make_dataset_dict):
    stub_ingestion(monkeypatch, make_dataset_dict({"train": 3, "validation": 2, "test": 2}))
    assert invoke(config_files, "all").exit_code == 1
    assert not app_config.data_transformation.output_dir.exists()


def test_importing_the_cli_does_not_load_heavy_libraries():
    code = (
        "import sys, textsummarizer.cli;"
        "print(sorted(m for m in ('datasets', 'transformers', 'torch') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_missing_config_file_is_a_one_line_error(tmp_path):
    result = runner.invoke(app, ["--config", str(tmp_path / "nope.yaml"), "validate"])
    assert result.exit_code == 1
    assert "nope.yaml" in result.output
    assert "Traceback" not in result.output


def test_invalid_config_is_reported_without_a_traceback(config_files):
    config, _ = config_files
    config.write_text(config.read_text() + "surprise: 1\n")
    result = invoke(config_files, "validate")
    assert result.exit_code == 1
    assert "surprise" in result.output
    assert "Traceback" not in result.output


def test_validation_failure_lists_the_problems_without_a_traceback(
    app_config, config_files, make_dataset_dict
):
    make_dataset_dict({"train": 3, "validation": 2, "test": 2}).save_to_disk(
        str(app_config.data_validation.data_dir)
    )
    result = invoke(config_files, "validate")
    assert result.exit_code == 1
    assert "train" in result.output
    assert "Traceback" not in result.output


def test_unexpected_errors_keep_their_traceback(monkeypatch, config_files):
    def boom(config):
        raise RuntimeError("boom")

    monkeypatch.setattr(stage_01_data_ingestion, "run", boom)
    result = invoke(config_files, "ingest")
    assert result.exit_code == 1
    assert "Traceback" in result.output and "boom" in result.output


def test_unknown_log_level_is_a_usage_error():
    assert runner.invoke(app, ["--log-level", "verbose", "validate"]).exit_code == 2


def test_train_forwards_the_resume_flag(monkeypatch, config_files):
    seen = {}
    monkeypatch.setattr(
        stage_04_model_trainer, "run", lambda config, resume=False: seen.update(resume=resume)
    )
    assert invoke(config_files, "train", "--resume").exit_code == 0
    assert seen == {"resume": True}


def test_publish_forwards_the_force_flag(monkeypatch, config_files):
    seen = {}
    monkeypatch.setattr(
        stage_06_publish, "run", lambda config, force=False: seen.update(force=force)
    )
    assert invoke(config_files, "publish", "--force").exit_code == 0
    assert seen == {"force": True}
