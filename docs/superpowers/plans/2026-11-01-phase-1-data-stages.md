# Phase 1: Data Stages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three data stages (ingestion, validation, transformation) with typed-config, tested, import-safe components driven by a `textsummarizer` CLI that runs end to end on CPU.

**Architecture:** Pydantic models loaded from `config/config.yaml` + `params.yaml` replace `ConfigManager`, the entity dataclasses, `python-box` and `ensure`. Each stage is a component class (logic; its constructor takes only its own config slice) plus a `run(config)` function in `pipeline/` (wiring). A Typer CLI imports stages lazily, so importing config or the CLI never loads `datasets`/`transformers`. Datasets are written atomically so an interrupted run never leaves output the next run trusts. Tests use a tiny offline tokenizer and tiny datasets; only the final manual run touches the network.

**Tech Stack:** pydantic 2, typer, datasets 5, transformers 5, pytest, ruff, uv.

**Spec:** `docs/superpowers/plans/2026-10-19-text-summarizer-roadmap.md` (Phase 1)

## Verified facts (dry-run 2026-11-01)

- **Hub dataset** `knkarthick/samsum` at revision `6b929ff10edec703164e3ddb2e94aae058c9ab5f`: CSV splits train/validation/test = 14,731 / 818 / 819 rows, columns `id`, `dialogue`, `summary` (all strings), no nulls, no empty text, `\n` line endings only. License tag `cc-by-nc-nd-4.0`. After whitespace normalization its text is identical to the zip in `data/`; the zip differs only in `\r\n` endings (12,844 train, 710 validation, 724 test rows) and one empty-dialogue train row (id 13828807) that the Hub copy dropped. `load_dataset(..., revision=<sha>)`, `save_to_disk` and `load_from_disk` round-trip with datasets 5.
- **Pegasus token lengths** (real tokenizer, all splits): dialogue p99 about 480, only 0.4-0.7% above 512 tokens, one train example above 1024; summaries max 82 tokens, none above 128. So 512 / 128 truncation is safe.
- **datasets 5:** `dataset["column"]` returns a lazy `Column`, not a list, and tokenizers reject it. Inside a batched `map` the batch values are plain lists. Outside `map`, wrap with `list(...)`.
- **Tiny test tokenizer:** a `tokenizers` WordLevel model with a Whitespace pre-tokenizer wrapped in `PreTrainedTokenizerFast`, saved with `save_pretrained`, reloads through `AutoTokenizer.from_pretrained(dir)` (the class is named `TokenizersBackend` in transformers 5; never assert the class name). `tok(text_target=..., max_length=n, truncation=True)` truncates correctly.

## Global Constraints

- Python `>=3.11`, tested on 3.11 and 3.12; type hints use `X | None`.
- `ruff check .` and `ruff format --check .` stay green (line length 100); typer options use `Annotated[...]` defaults (a `typer.Option(...)` default trips ruff `B008`).
- Every dependency change is followed by `uv lock`, and `uv.lock` is committed; `uv sync --locked --extra train` must succeed.
- Tests never touch the network and never write into the repo directory (use `tmp_path` and `monkeypatch.chdir`).
- Dataset: `knkarthick/samsum` pinned to revision `6b929ff10edec703164e3ddb2e94aae058c9ab5f`; tokenizer `google/pegasus-cnn_dailymail`; `max_input_length` 512, `max_target_length` 128 (in `params.yaml`).
- Do not edit `research/` notebooks or the empty `app.py` / `Dockerfile`; leave `data/summarizer-data.zip` in place.
- Work on branch `claude/practical-davinci-q2qix4`; no pull request unless asked.
- Commit messages end with the attribution trailers the session specifies.

## Review Focus

- An interrupted download or save (a Colab disconnect) leaves a half-written output directory that the next run skips as "already done". Expected: no output directory after a failure, and the next run recovers. Pinned in Task 3.
- A mistyped or stale key in `config.yaml` is silently ignored. Expected: loading fails loudly. Pinned in Task 1.
- Validation fails but the pipeline keeps going (today's behavior: the status file is written, then transformation runs anyway). Expected: `all` stops with a non-zero exit and produces no transformed output. Pinned in Task 6.
- Importing the CLI or config in a lean install pulls in `datasets` / `transformers`, or importing any module creates a `logs/` directory in the caller's working directory. Pinned in Tasks 6 and 7.
- The upstream Hub copy changes or loses rows. Expected: validation reports the row-count mismatch by split name instead of training on different data. Pinned in Task 4.

## File Structure

| Path | Responsibility |
|---|---|
| `src/textsummarizer/config/models.py` (new) | Pydantic models: one per stage plus `TokenizationParams` and `AppConfig` |
| `src/textsummarizer/config/loader.py` (new) | `load_config`: pure YAML to `AppConfig`, no side effects |
| `src/textsummarizer/log.py` (new) | `configure_logging`, called only from entrypoints |
| `src/textsummarizer/storage.py` (new) | `save_dataset_dict`: atomic `save_to_disk` |
| `src/textsummarizer/components/data_{ingestion,validation,transformation}.py` (rewritten) | Stage logic |
| `src/textsummarizer/pipeline/stage_0{1,2,3}_*.py` (rewritten) | `run(config)` wiring per stage |
| `src/textsummarizer/cli.py`, `__main__.py` (new) | Typer app and `python -m textsummarizer` |
| `tests/conftest.py` (new) | Shared fixtures (created in Task 2, extended in Tasks 3 and 6) |
| Removed in Task 7 | `main.py`, `template.py`, `structure.txt`, `config/configuration.py`, `entity/`, `utils/`, `logging/` |

---

### Task 1: Typed config

**Files:**
- Create: `src/textsummarizer/config/models.py`, `src/textsummarizer/config/loader.py`, `tests/test_config.py`
- Modify: `src/textsummarizer/config/__init__.py`, `config/config.yaml`, `params.yaml`, `pyproject.toml`, `uv.lock`, `tests/test_smoke.py`

**Interfaces:**
- Produces (`textsummarizer.config`, all models `frozen=True, extra="forbid"`):
  - `IngestionConfig(root_dir: Path, dataset_name: str, revision: str, output_dir: Path)`
  - `ValidationConfig(root_dir: Path, data_dir: Path, status_file: Path, required_columns: list[str], expected_rows: dict[str, int])`
  - `TransformationConfig(root_dir: Path, data_dir: Path, output_dir: Path, tokenizer_name: str)`
  - `TokenizationParams(max_input_length: int, max_target_length: int)`
  - `AppConfig(artifacts_root: Path, data_ingestion: IngestionConfig, data_validation: ValidationConfig, data_transformation: TransformationConfig, tokenization: TokenizationParams)`
  - `load_config(config_path: Path = CONFIG_FILE_PATH, params_path: Path = PARAMS_FILE_PATH) -> AppConfig`

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL at collection, `ImportError: cannot import name 'AppConfig' from 'textsummarizer.config'`.

- [ ] **Step 3: Add the dependency.** In `pyproject.toml` add `"pydantic>=2.7,<3"` to `[project].dependencies`, then run `uv lock && uv sync --locked --extra train`.

- [ ] **Step 4: Replace `config/config.yaml` and `params.yaml`** with exactly:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  dataset_name: knkarthick/samsum
  revision: 6b929ff10edec703164e3ddb2e94aae058c9ab5f
  output_dir: artifacts/data_ingestion/samsum_dataset

data_validation:
  root_dir: artifacts/data_validation
  data_dir: artifacts/data_ingestion/samsum_dataset
  status_file: artifacts/data_validation/status.txt
  required_columns: [id, dialogue, summary]
  expected_rows: {train: 14731, validation: 818, test: 819}

data_transformation:
  root_dir: artifacts/data_transformation
  data_dir: artifacts/data_ingestion/samsum_dataset
  output_dir: artifacts/data_transformation/samsum_dataset
  tokenizer_name: google/pegasus-cnn_dailymail
```

```yaml
tokenization:
  max_input_length: 512
  max_target_length: 128
```

(The old `params.yaml` placeholders `model:` / `parameters:` were read by nothing and contradict the training design; Phase 2 adds training and generation params.)

- [ ] **Step 5: Implement `config/models.py` and `config/loader.py`** to the signatures above. `load_config` reads both YAML files with `yaml.safe_load`, merges the two top-level dicts, and returns `AppConfig.model_validate(...)`. It must not create directories. Re-export `AppConfig`, the stage models and `load_config` from `config/__init__.py`.

- [ ] **Step 6: Retire the superseded smoke test.** In `tests/test_smoke.py` delete the `ConfigManager` import and `test_config_manager_builds_ingestion_config` (the old `ConfigManager` reads `config.yaml` keys that no longer exist; it is deleted in Task 7).

- [ ] **Step 7: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `7 passed` (5 config + 2 smoke), `All checks passed!`, `already formatted`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(config): typed pydantic config for the data stages"
```

---

### Task 2: Import-safe logging

**Files:**
- Create: `src/textsummarizer/log.py`, `tests/test_log.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `configure_logging(level: str = "INFO", log_file: Path | None = None) -> None`. It manages handlers on the `"textsummarizer"` logger only (never the root logger): one stdout `StreamHandler`, plus a `FileHandler` when `log_file` is given (parent directory created). Format `[%(asctime)s: %(levelname)s: %(message)s]`. Calling it again replaces the handlers it installed earlier. Nothing is written to disk unless `log_file` is given.

- [ ] **Step 1: Write the failing tests** in `tests/test_log.py`, and create `tests/conftest.py` with an autouse fixture `reset_package_logging` that removes and closes every handler on the `"textsummarizer"` logger after each test (otherwise later CLI tests leave a handler bound to a closed stream)

```python
import logging

from textsummarizer.log import configure_logging

PACKAGE_LOGGER = logging.getLogger("textsummarizer")


def test_repeated_calls_do_not_duplicate_handlers():
    configure_logging()
    configure_logging()
    assert len(PACKAGE_LOGGER.handlers) == 1


def test_log_file_receives_messages(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    configure_logging(log_file=log_file)
    logging.getLogger("textsummarizer.test").info("hello pipeline")
    assert "hello pipeline" in log_file.read_text()
    assert len(PACKAGE_LOGGER.handlers) == 2


def test_nothing_is_written_without_a_log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configure_logging()
    logging.getLogger("textsummarizer.test").info("stdout only")
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_log.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'textsummarizer.log'`.

- [ ] **Step 3: Implement `configure_logging`** in `src/textsummarizer/log.py` as specified. The old `textsummarizer/logging/` package stays until Task 7.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `10 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(log): configure_logging for entrypoints, no import side effects"
```

---

### Task 3: Ingestion from the Hub (with atomic saves)

**Files:**
- Create: `src/textsummarizer/storage.py`, `tests/test_data_ingestion.py`
- Modify: `tests/conftest.py` (add the fixtures below)
- Rewrite: `src/textsummarizer/components/data_ingestion.py`, `src/textsummarizer/pipeline/stage_01_data_ingestion.py`

**Interfaces:**
- Consumes: Task 1 models.
- Produces:
  - `tests/conftest.py` fixtures:
    - `tiny_tokenizer_dir` (session scope, `-> Path`): vocabulary `<pad>`=0, `</s>`=1, `<unk>`=2, then `TINY_WORDS = "hannah amanda betty larry hey do you have number lemme check sorry find it ask he called".split()` from id 3; WordLevel + Whitespace; wrapped in `PreTrainedTokenizerFast(pad_token="<pad>", eos_token="</s>", unk_token="<unk>")` and `save_pretrained` into a `tmp_path_factory` directory.
    - `make_dataset_dict` (`-> Callable[[dict[str, int] | None], DatasetDict]`): default rows `{"train": 4, "validation": 2, "test": 2}`; row `i` has `id=str(i)`, a dialogue of `3 + 2*(i % 4)` words and a summary of `5 + (i % 2)` words, the dialogue cycling through `TINY_WORDS` from index `i` and the summary from index `i + 7` (so every word is in the vocabulary, dialogues vary in length, summaries exceed 4 tokens, and a summary never starts with the same tokens as its dialogue).
    - `app_config` (`-> AppConfig`): every path under `tmp_path / "artifacts"` mirroring `config.yaml`'s layout (dataset directory `data_ingestion/samsum_dataset` shared by all three stages); `dataset_name="x/y"`, `revision="0" * 40`, `required_columns=["id","dialogue","summary"]`, `expected_rows={"train": 4, "validation": 2, "test": 2}`, `tokenizer_name=str(tiny_tokenizer_dir)`, `max_input_length=6`, `max_target_length=4`.
  - `save_dataset_dict(dataset_dict: DatasetDict, target: Path) -> None`: calls `dataset_dict.save_to_disk(str(tmp))` on the sibling `<target.name>.tmp` (removing a stale one first), then renames it to `target`. On any failure the `.tmp` directory is removed and `target` is untouched.
  - `DataIngestion(config: IngestionConfig, loader: Callable[..., DatasetDict] | None = None)` with `download() -> Path`. `loader` defaults to `datasets.load_dataset`, imported inside the method. It is called as `loader(config.dataset_name, revision=config.revision)`. If `config.output_dir` already exists, log and return it without calling the loader; otherwise save with `save_dataset_dict` and return `output_dir`.
  - `stage_01_data_ingestion.run(config: AppConfig) -> Path`.

- [ ] **Step 1: Add the fixtures to `tests/conftest.py` and write the failing tests** in `tests/test_data_ingestion.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_data_ingestion.py -q`
Expected: FAIL (the old `DataIngestion` takes no `loader` and imports the deleted-key config; collection or `TypeError`).

- [ ] **Step 3: Implement** `storage.py`, the `DataIngestion` rewrite, and `stage_01_data_ingestion.run` to the interfaces above (the stage creates `root_dir` and delegates to `DataIngestion(config.data_ingestion).download()`). Use `logging.getLogger(__name__)`, not the old logger package.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `14 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(ingestion): load pinned SAMSum revision from the Hub, save atomically"
```

---

### Task 4: Real data validation

**Files:**
- Create: `tests/test_data_validation.py`
- Rewrite: `src/textsummarizer/components/data_validation.py`, `src/textsummarizer/pipeline/stage_02_data_validation.py`

**Interfaces:**
- Consumes: Task 3 fixtures (`app_config`, `make_dataset_dict`).
- Produces:
  - `ValidationReport` (frozen dataclass: `ok: bool`, `problems: list[str]`; `ok` is true exactly when `problems` is empty) and `DataValidationError(Exception)` in `components/data_validation.py`.
  - `DataValidation(config: ValidationConfig)` with `validate() -> ValidationReport`. Checks, each problem naming the split (and column where relevant): `data_dir` missing or unreadable by `load_from_disk` (a problem, not an exception); split names differ from `expected_rows` keys (missing or unexpected); per split, column names differ from `required_columns`; row count differs from `expected_rows[split]`; any `dialogue` or `summary` that is null, empty or whitespace-only.
  - `stage_02_data_validation.run(config: AppConfig) -> ValidationReport`: creates `root_dir`, validates, writes `status_file` (first line `Validation status: True` or `Validation status: False`, then one `- <problem>` line each), and raises `DataValidationError` (message: the problems joined with `; `) when not ok.

- [ ] **Step 1: Write the failing tests** in `tests/test_data_validation.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_data_validation.py -q`
Expected: FAIL at collection, `ImportError: cannot import name 'DataValidationError'`.

- [ ] **Step 3: Implement** the component and the stage to the interfaces above. Load with `datasets.load_from_disk` (imported at module top is fine here; this stage needs the `train` extra). Result must not depend on directory listing order.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `25 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(validation): schema, row-count and blank-text checks; stage fails loudly"
```

---

### Task 5: Transformation on transformers 5

**Files:**
- Create: `tests/test_data_transformation.py`
- Rewrite: `src/textsummarizer/components/data_transformation.py`, `src/textsummarizer/pipeline/stage_03_data_transformation.py`

**Interfaces:**
- Consumes: Task 3 fixtures and `save_dataset_dict`.
- Produces:
  - `DataTransformation(config: TransformationConfig, params: TokenizationParams)`; the tokenizer is `AutoTokenizer.from_pretrained(config.tokenizer_name)`.
  - `convert_examples_to_features(batch: dict[str, list]) -> dict[str, list]`: `input_ids` / `attention_mask` from `dialogue` truncated to `max_input_length`, `labels` from `tokenizer(text_target=summary, ...)` truncated to `max_target_length`. No padding (the Phase 2 data collator pads per batch).
  - `convert() -> Path`: `load_from_disk(config.data_dir)`, batched `map`, original columns kept, `save_dataset_dict(..., config.output_dir)`.
  - `stage_03_data_transformation.run(config: AppConfig) -> Path`.

- [ ] **Step 1: Write the failing tests** in `tests/test_data_transformation.py`

```python
import pytest
from datasets import load_from_disk
from transformers import AutoTokenizer

from textsummarizer.components.data_transformation import DataTransformation


@pytest.fixture
def transformed(app_config, make_dataset_dict):
    make_dataset_dict().save_to_disk(str(app_config.data_transformation.data_dir))
    out = DataTransformation(app_config.data_transformation, app_config.tokenization).convert()
    return load_from_disk(str(out))


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_data_transformation.py -q`
Expected: FAIL (the old class takes no `params` argument and calls `as_target_tokenizer`).

- [ ] **Step 3: Implement** the component and the stage to the interfaces above. Do not pass `padding`. Do not subscript a `Dataset` column outside `map` (datasets 5 returns a lazy `Column`).

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `29 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(transformation): transformers-5 tokenization, 512/128 truncation, no padding"
```

---

### Task 6: CLI

**Files:**
- Create: `src/textsummarizer/cli.py`, `src/textsummarizer/__main__.py`, `tests/test_cli.py`
- Modify: `pyproject.toml` (+ `typer>=0.12` dependency, `[project.scripts] textsummarizer = "textsummarizer.cli:app"`), `uv.lock`, `tests/conftest.py` (new fixture)

**Interfaces:**
- Consumes: the three stage `run` functions, `configure_logging`, `load_config`.
- Produces:
  - `cli.app` (a `typer.Typer`). Global options on the callback: `--config PATH` (default `CONFIG_FILE_PATH`), `--params PATH` (default `PARAMS_FILE_PATH`), `--log-level TEXT` (default `INFO`). Commands: `ingest`, `validate`, `transform`, `all` (in that order; `all` stops at the first failing stage).
  - Each command calls `configure_logging`, `load_config`, then `stage_0X.run(config)` looked up on the module at call time (`from textsummarizer.pipeline import stage_01_data_ingestion` inside the command, then `stage_01_data_ingestion.run(...)`), so tests can patch it and so `datasets` / `transformers` load only when a command runs. On any exception it logs the traceback and exits with code 1.
  - `tests/conftest.py` fixture `config_files(app_config, tmp_path) -> tuple[Path, Path]`: writes `config.yaml` (`app_config.model_dump(mode="json")` without `tokenization`) and `params.yaml` (`{"tokenization": ...}`).

- [ ] **Step 1: Write the failing tests** in `tests/test_cli.py`

```python
import subprocess
import sys

from datasets import load_from_disk
from typer.testing import CliRunner

from textsummarizer.cli import app
from textsummarizer.pipeline import stage_01_data_ingestion
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
    for command in ("ingest", "validate", "transform", "all"):
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL at collection, `ModuleNotFoundError: No module named 'textsummarizer.cli'`.

- [ ] **Step 3: Add the dependency and entry point** in `pyproject.toml`, then `uv lock && uv sync --locked --extra train`. Add the `config_files` fixture to `tests/conftest.py`.

- [ ] **Step 4: Implement** `cli.py` (typer `Annotated` options) and `__main__.py` (`from textsummarizer.cli import app`, then `if __name__ == "__main__": app()`; the import-everything smoke tests import this module, so it must not run the CLI on import).

- [ ] **Step 5: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run textsummarizer --help`
Expected: `35 passed`, lint and format clean, `--help` lists `ingest`, `validate`, `transform`, `all`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(cli): textsummarizer ingest|validate|transform|all"
```

---

### Task 7: Remove the legacy layer and run the real pipeline

**Files:**
- Delete: `main.py`, `template.py`, `structure.txt`, `src/textsummarizer/config/configuration.py`, `src/textsummarizer/entity/`, `src/textsummarizer/utils/`, `src/textsummarizer/logging/`
- Modify: `pyproject.toml` (remove `python-box`, `ensure`), `uv.lock`, `README.md`, `tests/test_smoke.py`, the roadmap

**Interfaces:**
- Consumes: Tasks 1-6 (nothing may import the removed modules).

- [ ] **Step 1: Write the failing guard test** in `tests/test_smoke.py`

```python
import subprocess
import sys


def test_importing_every_module_creates_no_files(tmp_path):
    code = (
        "import importlib, pkgutil, textsummarizer;"
        "[importlib.import_module(m.name) for m in "
        "pkgutil.walk_packages(textsummarizer.__path__, 'textsummarizer.')]"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_smoke.py::test_importing_every_module_creates_no_files -q`
Expected: FAIL (the legacy `textsummarizer.logging` creates `logs/` on import).

- [ ] **Step 3: Remove the legacy code.** `git rm -r` the paths listed under Delete; remove `python-box` and `ensure` from `pyproject.toml`; run `uv lock && uv sync --locked --extra train`.

- [ ] **Step 4: Update the README.** Replace the "Workflow" checklist with a "Running the pipeline" section: `uv run textsummarizer all` (or `ingest`, `validate`, `transform`), where outputs land (`artifacts/`), and one sentence that the notebooks in `research/` show the original pre-refactor API and are kept as history.

- [ ] **Step 5: Verify tests, lint, and that nothing still references the legacy layer**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && grep -rnE --include='*.py' "ConfigManager|ConfigBox|ensure_annotations|textsummarizer\.(utils|entity|logging)" src tests; echo "grep-exit=$?"`
Expected: `36 passed`, lint and format clean, no grep matches (`grep-exit=1`).

- [ ] **Step 6: Run the real pipeline** (needs network; not part of CI)

Run: `rm -rf artifacts && uv run textsummarizer all`
Expected: exit 0. Then:
`uv run python -c "from datasets import load_from_disk; d=load_from_disk('artifacts/data_transformation/samsum_dataset'); print({k:v.num_rows for k,v in d.items()}, max(len(x) for x in d['train']['input_ids']), max(len(x) for x in d['train']['labels']))"`
prints `{'train': 14731, 'validation': 818, 'test': 819}` followed by two numbers with the first at most 512 and the second at most 128. `head -1 artifacts/data_validation/status.txt` prints `Validation status: True`. Run `uv run textsummarizer ingest` again: the log says the output already exists and nothing is downloaded.

- [ ] **Step 7: Commit, push, update the roadmap**

```bash
git add -A
git commit -m "refactor: remove legacy config/logging/utils layer; document the CLI"
git push
```
Then in the roadmap's Phase 1 row, set the Plan cell to `2026-11-01-phase-1-data-stages.md` and note the CI status; commit and push.
