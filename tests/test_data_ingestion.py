from pathlib import Path

import pytest
from datasets import load_from_disk

from textsummarizer.components.data_ingestion import DataIngestion


def test_download_loads_the_pinned_revision_and_saves(app_config, make_dataset_dict):
    calls = []

    def loader(name, revision):
        calls.append((name, revision))
        return make_dataset_dict()

    path = DataIngestion(app_config.data_ingestion, loader=loader).download()
    assert calls == [("x/y", "0" * 40)]
    assert path == app_config.data_ingestion.output_dir
    assert {k: v.num_rows for k, v in load_from_disk(str(path)).items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }


def test_existing_output_is_not_downloaded_again(app_config, make_dataset_dict):
    DataIngestion(
        app_config.data_ingestion, loader=lambda name, revision: make_dataset_dict()
    ).download()

    def must_not_run(name, revision):
        raise AssertionError("loader called a second time")

    DataIngestion(app_config.data_ingestion, loader=must_not_run).download()


def test_failed_load_leaves_no_output(app_config):
    def offline(name, revision):
        raise ConnectionError("offline")

    with pytest.raises(ConnectionError):
        DataIngestion(app_config.data_ingestion, loader=offline).download()
    assert not app_config.data_ingestion.output_dir.exists()


def test_interrupted_save_leaves_no_output_and_the_next_run_recovers(app_config, make_dataset_dict):
    class DiesMidSave:
        def save_to_disk(self, path):
            Path(path).mkdir(parents=True)
            raise OSError("disk died")

    output_dir = app_config.data_ingestion.output_dir
    with pytest.raises(OSError):
        DataIngestion(
            app_config.data_ingestion, loader=lambda n, revision: DiesMidSave()
        ).download()
    assert not output_dir.exists()

    DataIngestion(
        app_config.data_ingestion, loader=lambda name, revision: make_dataset_dict()
    ).download()
    assert (output_dir / "dataset_dict.json").is_file()


def test_output_from_another_revision_is_replaced(app_config, make_dataset_dict):
    revisions = []

    def loader(name, revision):
        revisions.append(revision)
        return make_dataset_dict()

    DataIngestion(app_config.data_ingestion, loader=loader).download()
    newer = app_config.data_ingestion.model_copy(update={"revision": "1" * 40})
    DataIngestion(newer, loader=loader).download()
    assert revisions == ["0" * 40, "1" * 40]


def test_output_without_a_marker_is_replaced(app_config, make_dataset_dict):
    make_dataset_dict().save_to_disk(str(app_config.data_ingestion.output_dir))  # old pipeline
    revisions = []

    def loader(name, revision):
        revisions.append(revision)
        return make_dataset_dict()

    DataIngestion(app_config.data_ingestion, loader=loader).download()
    assert revisions == ["0" * 40]
