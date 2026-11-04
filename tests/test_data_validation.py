import pytest

from textsummarizer.components.data_validation import DataValidation, DataValidationError
from textsummarizer.pipeline import stage_02_data_validation


def check(app_config, dataset_dict):
    dataset_dict.save_to_disk(str(app_config.data_validation.data_dir))
    return DataValidation(app_config.data_validation).validate()


def test_good_dataset_passes(app_config, make_dataset_dict):
    report = check(app_config, make_dataset_dict())
    assert report.ok and report.problems == []


def test_wrong_row_count_names_the_split(app_config, make_dataset_dict):
    report = check(app_config, make_dataset_dict({"train": 3, "validation": 2, "test": 2}))
    assert not report.ok
    assert any("train" in p for p in report.problems)


def test_missing_split_is_reported(app_config, make_dataset_dict):
    dd = make_dataset_dict()
    del dd["test"]
    assert any("test" in p for p in check(app_config, dd).problems)


def test_unexpected_split_is_reported(app_config, make_dataset_dict):
    dd = make_dataset_dict()
    dd["extra"] = dd["test"]
    assert any("extra" in p for p in check(app_config, dd).problems)


def test_missing_column_names_split_and_column(app_config, make_dataset_dict):
    dd = make_dataset_dict()
    dd["train"] = dd["train"].remove_columns("summary")
    assert any("train" in p and "summary" in p for p in check(app_config, dd).problems)


@pytest.mark.parametrize(
    "column,value", [("summary", ""), ("dialogue", "  \n "), ("dialogue", None)]
)
def test_blank_or_null_text_is_reported(app_config, make_dataset_dict, column, value):
    dd = make_dataset_dict()
    dd["train"] = dd["train"].map(lambda row: {column: value} if row["id"] == "0" else {})
    report = check(app_config, dd)
    assert not report.ok
    assert any("train" in p and column in p for p in report.problems)


def test_missing_data_dir_is_a_problem_not_a_crash(app_config):
    report = DataValidation(app_config.data_validation).validate()
    assert not report.ok


def test_stage_writes_status_and_raises_on_bad_data(app_config, make_dataset_dict):
    bad = make_dataset_dict({"train": 3, "validation": 2, "test": 2})
    bad.save_to_disk(str(app_config.data_validation.data_dir))
    with pytest.raises(DataValidationError):
        stage_02_data_validation.run(app_config)
    status = app_config.data_validation.status_file.read_text().splitlines()
    assert status[0] == "Validation status: False" and status[1].startswith("- ")


def test_stage_writes_true_status_for_good_data(app_config, make_dataset_dict):
    make_dataset_dict().save_to_disk(str(app_config.data_validation.data_dir))
    stage_02_data_validation.run(app_config)
    assert app_config.data_validation.status_file.read_text().splitlines() == [
        "Validation status: True"
    ]


def test_non_string_text_column_is_reported(app_config, make_dataset_dict):
    dd = make_dataset_dict()
    dd["train"] = dd["train"].map(lambda row: {"dialogue": 7})
    report = check(app_config, dd)
    assert not report.ok
    assert any("train" in p and "dialogue" in p for p in report.problems)


def test_stage_removes_a_stale_status_file_when_validation_crashes(app_config, monkeypatch):
    status = app_config.data_validation.status_file
    status.parent.mkdir(parents=True)
    status.write_text("Validation status: True\n")

    def crash(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(DataValidation, "validate", crash)
    with pytest.raises(RuntimeError):
        stage_02_data_validation.run(app_config)
    assert not status.exists()
