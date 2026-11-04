import pytest
from datasets import load_from_disk
from transformers import AutoTokenizer

from textsummarizer.components.data_transformation import DataTransformation


@pytest.fixture
def transformed(app_config, make_dataset_dict):
    make_dataset_dict().save_to_disk(str(app_config.data_transformation.data_dir))
    out = DataTransformation(app_config.data_transformation, app_config.tokenization).convert()
    return load_from_disk(str(out))


def test_source_directory_is_left_untouched(app_config, make_dataset_dict):
    source = app_config.data_transformation.data_dir
    make_dataset_dict().save_to_disk(str(source))
    before = sorted(p.relative_to(source) for p in source.rglob("*"))
    DataTransformation(app_config.data_transformation, app_config.tokenization).convert()
    assert sorted(p.relative_to(source) for p in source.rglob("*")) == before


def test_every_split_gets_model_features_and_keeps_its_columns(transformed):
    assert set(transformed) == {"train", "validation", "test"}
    for split in transformed.values():
        assert {"id", "dialogue", "summary", "input_ids", "attention_mask", "labels"} <= set(
            split.column_names
        )
    assert transformed["train"].num_rows == 4


def test_inputs_and_labels_are_truncated_to_the_configured_lengths(transformed):
    rows = list(transformed["train"])
    assert max(len(r["input_ids"]) for r in rows) == 6
    assert max(len(r["labels"]) for r in rows) == 4
    assert all(len(r["input_ids"]) <= 6 and len(r["labels"]) <= 4 for r in rows)


def test_sequences_are_not_padded(transformed):
    assert len({len(r["input_ids"]) for r in transformed["train"]}) > 1


def test_labels_are_the_tokenized_summary_not_the_dialogue(app_config, transformed):
    tokenizer = AutoTokenizer.from_pretrained(app_config.data_transformation.tokenizer_name)
    row = transformed["train"][0]
    expected = tokenizer(text_target=row["summary"], max_length=4, truncation=True)["input_ids"]
    assert row["labels"] == expected
    assert row["labels"] != row["input_ids"][:4]
