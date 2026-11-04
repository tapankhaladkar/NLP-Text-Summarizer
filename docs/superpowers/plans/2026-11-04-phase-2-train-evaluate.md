# Phase 2: Train, Evaluate, Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `google/pegasus-cnn_dailymail` on the validated, tokenized SAMSum data from Phase 1, compare it fairly against the zero-shot model on the full test split, and publish the model with its metrics to the Hugging Face Hub, through CLI commands that a Colab notebook runs. Also fix the Phase 1 review findings that the new stages depend on.

**Architecture:** Three new stages (`model_trainer`, `model_evaluation`, `publish`) follow the Phase 1 pattern: a component class holding the logic plus a `stage_0N_*.run(config)` function, wired to new CLI commands `train`, `evaluate`, `publish`. A tiny randomly initialized Pegasus, built from a config in a fixture, lets every stage run in tests on CPU. The long training run happens on Colab's GPU through a notebook that only calls the CLI.

**Tech Stack:** transformers 5 (`Seq2SeqTrainer`), torch, `rouge-score`, `bert-score`, `huggingface_hub`, pydantic, typer, pytest, ruff, uv.

**Spec:** `docs/superpowers/plans/2026-10-19-text-summarizer-roadmap.md` (Phase 2 training design and the carried-over review findings)

## Verified facts (dry-run 2026-11-04, transformers 5.22.1, CPU)

- **Tiny Pegasus works end to end.** `PegasusConfig(vocab_size=len(tokenizer), d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32, pad_token_id=<pad>, eos_token_id=<eos>, decoder_start_token_id=<pad>, forced_eos_token_id=<eos>)` trains with `Seq2SeqTrainer` (4 steps), evaluates with generation and ROUGE, picks the best checkpoint, saves, resumes, reloads and generates, all in about 3 seconds.
- **transformers 5 differences:** `Seq2SeqTrainingArguments` has no `warmup_ratio`; `warmup_steps` takes a float in [0, 1) as a fraction of total steps. `processing_class=` replaces `tokenizer=`.
- **Resume restores every tensor.** The "missing keys" warning on resume is benign: only `model.shared.weight` is stored and the three tied copies equal it after loading.
- **ROUGE:** `rouge_score` with stemming gives rouge1/rouge2/rougeL/rougeLsum. Pegasus emits `<n>` as its newline marker; leaving it in drops rouge2 on one example from 0.59 to 0.44, and splitting both texts into sentences is what makes rougeLsum different from rougeL. For `"Amanda can't find the number.<n>Larry called Betty."` against `"Amanda can't find Betty's number. Larry called her."` the scores are rouge1 0.842, rouge2 0.588, rougeL 0.737, rougeLsum 0.842.
- **BERTScore** (`bert-score`) works on transformers 5.22 (`distilbert-base-uncased`, 18 s including the download).
- **Hub API:** `HfApi.create_repo(repo_id, private=, exist_ok=)` and `HfApi.upload_folder(repo_id=, folder_path=, commit_message=)` exist.
- **Path anchoring:** a pydantic `field_validator("*")` plus `model_validate(..., context={"base_dir": ...})` makes nested relative `Path` fields absolute and leaves absolute ones alone.
- **Generation length:** the checkpoint's own generation config has `min_length=32` (tuned for CNN/DailyMail). SAMSum reference summaries have a median of about 18 words, so forcing 32 tokens would over-generate for most examples. This plan sets `min_length` to 0 for both models, which differs from the roadmap's draft.
- **Real Pegasus on CPU** (from the roadmap check): loads in about 30 s, 2 training steps at batch size 1 take about 18 s, generation takes about 14 s per example at 8 beams.
- **Not verified here:** Colab itself (`pip install -e ".[train]"` there, GPU memory, fp16 stability), and the Hub upload.

## Decisions made for the carried-over findings

- **Newlines:** no preprocessing of dialogue text. Turns are flattened by Pegasus's tokenizer and speaker prefixes keep the structure. This matches the community baseline, and inference needs no special handling. A `\n` to `<n>` experiment is an optional Phase 5 item.
- **Paths:** relative paths in `config.yaml` resolve against the directory above the config file (the repo root), never the current directory.
- **Config overlap:** a section defined in both `config.yaml` and `params.yaml` is an error.
- **Gating:** `train` refuses unless `status.txt` says `Validation status: True`. `transform` stays ungated, and `all` still means ingest, validate, transform.
- **Assumption to confirm:** the Hub repo id defaults to `tapankhaladkar/pegasus-samsum`. Change `publish.repo_id` in `config/config.yaml` if the Hugging Face username differs.

## Global Constraints

- Python `>=3.11`, tested on 3.11 and 3.12; type hints use `X | None`.
- `ruff check .` and `ruff format --check .` stay green. This ruff also formats Python code blocks inside Markdown, so run `uv run ruff format .` after editing any `.md` file that contains Python.
- Every dependency change is followed by `uv lock`, and `uv.lock` is committed; `uv sync --locked --extra train` must succeed.
- Tests run on CPU, offline, and never write into the repo directory; they use the tiny tokenizer and the tiny Pegasus fixtures, never a real checkpoint.
- Training hyperparameters: effective batch size 16 (4 per device times 4 accumulation steps), learning rate 5e-5, `warmup_steps` 0.05, weight decay 0.01, 3 epochs, seed 42, fp16 only when CUDA is available. Generation: `num_beams` 8, `max_length` 128, `min_length` 0, `length_penalty` 0.8, identical for baseline and fine-tuned model.
- The Hugging Face token is read only from the `HF_TOKEN` environment variable. It must never be written to a file, a model card, a log line or a notebook output.
- Do not edit `research/`, `app.py` or `Dockerfile`; do not commit anything under `artifacts/`.
- Work on branch `claude/practical-davinci-q2qix4`; no pull request unless asked.
- Commit messages end with the attribution trailers the session specifies.

## Review Focus

- A Colab disconnect loses work or resumes wrongly. Expected: `train --resume` continues from the last checkpoint with weights restored, and a half-written model directory is never left behind. Pinned in Tasks 2 and 5.
- Training on data that was never validated, or whose validation failed. Expected: `train` refuses with a message naming `textsummarizer validate`. Pinned in Task 5.
- Publishing a broken model (near-zero ROUGE, or worse than the baseline). Expected: `publish` refuses unless `--force` is given. Pinned in Task 7.
- Baseline and fine-tuned models scored under different settings, or mis-scored (the `<n>` bug that produced ROUGE 0.02 in the notebook). Expected: identical generation settings recorded in `metrics.json`, and `<n>` treated as a sentence break. Pinned in Tasks 4 and 6.
- The Hugging Face token leaking, or the Colab notebook drifting away from the CLI. Expected: the token never appears in the card or logs, and the notebook calls only existing commands. Pinned in Tasks 7 and 8.
- Running from a different working directory silently creates a second `artifacts/` tree. Expected: paths anchor at the repo root. Pinned in Task 1.

## File Structure

| Path | Responsibility |
|---|---|
| `src/textsummarizer/errors.py` (new, Task 3) | `UserFacingError`: expected failures the CLI reports as one line |
| `src/textsummarizer/storage.py` (Task 2) | `atomic_directory` context manager; `save_dataset_dict` built on it |
| `src/textsummarizer/metrics.py` (new, Task 4) | `compute_rouge`, sentence splitting, `bertscore_f1` |
| `src/textsummarizer/components/model_trainer.py` (new) | `ModelTrainer` |
| `src/textsummarizer/components/model_evaluation.py` (new) | `ModelEvaluation` |
| `src/textsummarizer/components/model_publisher.py` (new) | `render_model_card`, `ModelPublisher` |
| `src/textsummarizer/pipeline/stage_04_model_trainer.py`, `stage_05_model_evaluation.py`, `stage_06_publish.py` (new) | `run(config, ...)` wiring |
| `notebooks/colab_train.ipynb` (new, Task 8) | Colab runner that only calls the CLI |

---

### Task 1: Config: repo-root paths, no silent overrides, new sections

**Files:**
- Modify: `src/textsummarizer/config/models.py`, `src/textsummarizer/config/loader.py`, `src/textsummarizer/config/__init__.py`, `config/config.yaml`, `params.yaml`, `pyproject.toml`, `uv.lock`, `tests/conftest.py`, `tests/test_config.py`, `README.md`

**Interfaces:**
- Produces, added to `textsummarizer.config` (all `frozen=True, extra="forbid"`):
  - `TrainerConfig(root_dir: Path, data_dir: Path, status_file: Path, base_model: str, checkpoints_dir: Path, model_dir: Path, metrics_file: Path)`
  - `EvaluationConfig(root_dir: Path, data_dir: Path, base_model: str, model_dir: Path, metrics_file: Path, examples_file: Path)`
  - `PublishConfig(model_dir: Path, metrics_file: Path, repo_id: str, private: bool)`
  - `TrainingParams(num_train_epochs: float, learning_rate: float, warmup_steps: float, weight_decay: float, per_device_train_batch_size: int, per_device_eval_batch_size: int, gradient_accumulation_steps: int, fp16: bool, gradient_checkpointing: bool, eval_num_beams: int, seed: int, max_steps: int, max_train_samples: int | None, max_eval_samples: int | None, report_to: str)`
  - `GenerationParams(num_beams: int, max_length: int, min_length: int, length_penalty: float)`
  - `EvaluationParams(batch_size: int, max_test_samples: int | None, num_examples: int, bertscore_model: str | None)`
  - `AppConfig` gains `model_trainer: TrainerConfig`, `model_evaluation: EvaluationConfig`, `publish: PublishConfig`, `training: TrainingParams`, `generation: GenerationParams`, `evaluation: EvaluationParams`.
  - `load_config(config_path, params_path)` now returns absolute paths (relative ones anchored at `config_path.resolve().parent.parent`) and raises `ValueError` naming any top-level section present in both files.
- Produces, `tests/conftest.py`: `app_config` gains the new sections with tiny values (`training`: 1 epoch, learning rate 1e-3, `warmup_steps` 0.1, weight decay 0.0, batch sizes 2, accumulation 1, `fp16` false, `eval_num_beams` 2, seed 42, `max_steps` -1, sample limits `None`, `report_to` "none"; `generation`: 2 beams, `max_length` 8, `min_length` 0, `length_penalty` 1.0; `evaluation`: batch size 2, `num_examples` 3, `bertscore_model` `None`; all new paths under `tmp_path / "artifacts"`, laid out like `config.yaml`: `model_trainer.data_dir` and `model_evaluation.data_dir` equal `data_transformation.output_dir`, `model_trainer.status_file` equals `data_validation.status_file`, `model_dir` is shared by trainer, evaluation and publish, and `metrics_file` by evaluation and publish; `publish.repo_id` is `"me/pegasus-samsum"`; `base_model` is the placeholder `"tiny/base"` until Task 5). `config_files` writes the four parameter sections (`tokenization`, `training`, `generation`, `evaluation`) to `params.yaml` and the rest to `config.yaml`.

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py` (imports already present from Phase 1)

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: 4 failures (`AttributeError` on the new sections, and the anchoring test). `test_absolute_paths_are_left_alone` already passes; it pins behavior the anchoring must keep.

- [ ] **Step 3: Add the dependencies.** In `pyproject.toml`, add `"rouge-score>=0.1.2"` and `"bert-score>=0.3.13"` to the `train` extra, then `uv lock && uv sync --locked --extra train`.
Expected: both resolve (`rouge-score` builds from source); `uv.lock` still has no `nvidia` packages.

- [ ] **Step 4: Append to `config/config.yaml`** exactly:

```yaml

model_trainer:
  root_dir: artifacts/model_trainer
  data_dir: artifacts/data_transformation/samsum_dataset
  status_file: artifacts/data_validation/status.txt
  base_model: google/pegasus-cnn_dailymail
  checkpoints_dir: artifacts/model_trainer/checkpoints
  model_dir: artifacts/model_trainer/model
  metrics_file: artifacts/model_trainer/train_metrics.json

model_evaluation:
  root_dir: artifacts/model_evaluation
  data_dir: artifacts/data_transformation/samsum_dataset
  base_model: google/pegasus-cnn_dailymail
  model_dir: artifacts/model_trainer/model
  metrics_file: artifacts/model_evaluation/metrics.json
  examples_file: artifacts/model_evaluation/examples.md

publish:
  model_dir: artifacts/model_trainer/model
  metrics_file: artifacts/model_evaluation/metrics.json
  repo_id: tapankhaladkar/pegasus-samsum
  private: false
```

and append to `params.yaml` exactly:

```yaml

training:
  num_train_epochs: 3
  learning_rate: 5.0e-5
  warmup_steps: 0.05  # a float in [0, 1) is a fraction of total steps (transformers 5)
  weight_decay: 0.01
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  fp16: true  # applied only when CUDA is available
  gradient_checkpointing: false
  eval_num_beams: 4
  seed: 42
  max_steps: -1  # a positive value overrides the epoch count (smoke runs)
  max_train_samples: null
  max_eval_samples: null
  report_to: none

generation:
  num_beams: 8
  max_length: 128
  min_length: 0  # the checkpoint's 32 is tuned for CNN/DailyMail; SAMSum summaries are shorter
  length_penalty: 0.8

evaluation:
  batch_size: 16
  max_test_samples: null
  num_examples: 20
  bertscore_model: roberta-large  # null disables BERTScore
```

- [ ] **Step 5: Implement** the models above in `models.py`, and in `loader.py`: raise `ValueError(f"sections defined in both config and params: {names}")` on overlap; validate with `context={"base_dir": config_path.resolve().parent.parent}`; give the shared base model a `field_validator("*", mode="after")` that turns relative `Path` values into `base_dir / value` when a `base_dir` context is present. Re-export the new names from `config/__init__.py`.

- [ ] **Step 6: Update `tests/conftest.py`** (`app_config`, `config_files`) as described under Interfaces.

- [ ] **Step 7: README.** Replace the sentence about running from the repo root, if any, with: "Relative paths in `config/config.yaml` resolve against the repository root, so the commands work from any directory."

- [ ] **Step 8: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `42 passed`, `All checks passed!`, `already formatted`.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(config): repo-root paths, reject overlapping sections, training/evaluation/publish settings"
```

---

### Task 2: Data-stage hardening

**Files:**
- Modify: `src/textsummarizer/storage.py`, `src/textsummarizer/components/data_ingestion.py`, `src/textsummarizer/components/data_validation.py`, `src/textsummarizer/pipeline/stage_02_data_validation.py`, `tests/test_data_ingestion.py`, `tests/test_data_validation.py`, `tests/test_data_transformation.py`
- Create: `tests/test_storage.py`

**Interfaces:**
- Produces `storage.atomic_directory(target: Path) -> Iterator[Path]` (a context manager). It yields `<target.name>.tmp` next to `target` (a stale one is removed first, and it does not exist yet). If the block raises, the `.tmp` directory is removed and `target` is untouched. On clean exit, an existing `target` is renamed to `<name>.old`, `.tmp` is renamed to `target`, then `.old` is removed; if the second rename fails, `.old` is renamed back and the error is re-raised. Afterwards `target` is the only sibling left. Use `Path.rename` for all three renames (the swap-failure test patches it).
- `save_dataset_dict(dataset_dict, target)` keeps its signature and is implemented with `atomic_directory`.
- `DataIngestion.download()` writes `ingestion.json` (`{"dataset_name": ..., "revision": ...}`) into `output_dir` after saving, and skips the download only when that file exists and matches the config; otherwise it logs a warning and downloads again, replacing the directory.
- `DataValidation.validate()` reports a problem (not an exception) when `dialogue` or `summary` is not a string column; `stage_02_data_validation.run` deletes the old status file before validating.

- [ ] **Step 1: Write the failing tests.** `tests/test_storage.py`:

```python
from pathlib import Path

import pytest

from textsummarizer.storage import atomic_directory


def old_target(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "old.txt").write_text("old")
    return target


def write_new(tmp):
    tmp.mkdir()
    (tmp / "new.txt").write_text("new")


def test_atomic_directory_publishes_on_success(tmp_path):
    target = tmp_path / "out"
    with atomic_directory(target) as tmp:
        write_new(tmp)
    assert (target / "new.txt").read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_atomic_directory_replaces_an_existing_target_without_leftovers(tmp_path):
    target = old_target(tmp_path)
    with atomic_directory(target) as tmp:
        write_new(tmp)
    assert [p.name for p in target.iterdir()] == ["new.txt"]
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_failure_inside_the_block_keeps_the_old_target(tmp_path):
    target = old_target(tmp_path)
    with pytest.raises(RuntimeError):
        with atomic_directory(target) as tmp:
            write_new(tmp)
            raise RuntimeError("boom")
    assert (target / "old.txt").read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_failed_swap_restores_the_old_target(tmp_path, monkeypatch):
    target = old_target(tmp_path)
    real_rename = Path.rename

    def flaky(self, destination):
        if self.name.endswith(".tmp"):
            raise OSError("rename failed")
        return real_rename(self, destination)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(OSError):
        with atomic_directory(target) as tmp:
            write_new(tmp)
    assert (target / "old.txt").read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["out"]
```

Add to `tests/test_data_ingestion.py`:

```python
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
```

Add to `tests/test_data_validation.py`:

```python
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
```

Add to `tests/test_data_transformation.py`:

```python
def test_convert_can_be_run_twice(app_config, make_dataset_dict):
    make_dataset_dict().save_to_disk(str(app_config.data_transformation.data_dir))
    transformation = DataTransformation(app_config.data_transformation, app_config.tokenization)
    transformation.convert()
    out = transformation.convert()
    assert load_from_disk(str(out))["train"].num_rows == 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_storage.py tests/test_data_ingestion.py tests/test_data_validation.py tests/test_data_transformation.py -q`
Expected: `tests/test_storage.py` fails at collection (`cannot import name 'atomic_directory'`), the two ingestion tests fail (existing directory skipped), the two validation tests fail (`AttributeError` on `.strip()`; stale status file remains). `test_convert_can_be_run_twice` already passes; it guards the replace path through the refactor.

- [ ] **Step 3: Implement** `atomic_directory` and rebuild `save_dataset_dict` on it; the ingestion marker and skip rule; the string-type check (a `Value` feature whose dtype is `string` or `large_string`); and `settings.status_file.unlink(missing_ok=True)` at the top of the validation stage.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `51 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix: atomic directory swaps, revision-aware ingestion, stricter validation"
```

---

### Task 3: Friendly CLI errors

**Files:**
- Create: `src/textsummarizer/errors.py`
- Modify: `src/textsummarizer/cli.py`, `src/textsummarizer/components/data_validation.py`, `tests/test_cli.py`

**Interfaces:**
- Produces `errors.UserFacingError(Exception)`; `DataValidationError` now subclasses it.
- `cli._run_stages` handles errors in two places, each logged as one error line without a traceback, then exit code 1. Around `load_config`: `FileNotFoundError` (logs `Config file not found:` plus the path) and `ValueError` (this includes pydantic's `ValidationError` and the section-overlap error; logs `Invalid configuration:` plus the details). Around the stages: `UserFacingError` (logs the message), `FileNotFoundError` (logs the missing path) and `ImportError` (logs the message plus `Install the training extras with: uv sync --extra train`). Any other exception keeps `logger.exception` (traceback) and exit code 1.
- `--log-level` becomes a choice of `DEBUG`, `INFO`, `WARNING`, `ERROR` (a `str` `Enum`); anything else is a usage error (exit code 2).

- [ ] **Step 1: Write the failing tests** in `tests/test_cli.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli.py -q`
Expected: 4 failures (tracebacks printed, `verbose` accepted); `test_unexpected_errors_keep_their_traceback` already passes and guards the other branch.

- [ ] **Step 3: Implement** to the interfaces above. `errors.py` must import nothing heavy (the lean-import test must keep passing).

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `56 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cli): one-line errors for expected failures, validated --log-level"
```

---

### Task 4: Metrics

**Files:**
- Create: `src/textsummarizer/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")`
  - `split_sentences(text: str) -> list[str]`: splits on newlines and on whitespace that follows `.`, `!` or `?`; strips; drops empty pieces.
  - `prepare_for_rouge(text: str) -> str`: replaces Pegasus's `<n>` with a newline, then joins `split_sentences` with `"\n"`.
  - `compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]`: mean F-measure times 100, rounded to 2 decimals, with stemming, after `prepare_for_rouge` on both sides; `ValueError` if the lists are empty or differ in length.
  - `bertscore_f1(predictions: list[str], references: list[str], model_type: str) -> float`: imports `bert_score` lazily, uses `lang="en"`, `device` cuda when available, returns the mean F1 times 100.

- [ ] **Step 1: Write the failing tests** in `tests/test_metrics.py`

```python
import pytest

from textsummarizer.metrics import ROUGE_KEYS, compute_rouge, prepare_for_rouge, split_sentences


def test_split_sentences_handles_newlines_and_terminal_punctuation():
    assert split_sentences("A b. C d!  E?\nF g") == ["A b.", "C d!", "E?", "F g"]


def test_prepare_for_rouge_turns_pegasus_newline_markers_into_sentence_breaks():
    assert prepare_for_rouge("One two.<n>Three four.") == "One two.\nThree four."


def test_identical_texts_score_100():
    text = ["Hannah needs Betty's number."]
    assert compute_rouge(text, text) == {key: 100.0 for key in ROUGE_KEYS}


def test_disjoint_texts_score_0():
    assert compute_rouge(["alpha beta"], ["gamma delta"]) == {key: 0.0 for key in ROUGE_KEYS}


def test_known_example_matches_hand_checked_scores():
    scores = compute_rouge(
        ["Amanda can't find the number.<n>Larry called Betty."],
        ["Amanda can't find Betty's number. Larry called her."],
    )
    expected = {"rouge1": 84.2, "rouge2": 58.8, "rougeL": 73.7, "rougeLsum": 84.2}
    assert scores == pytest.approx(expected, abs=0.1)


def test_mismatched_or_empty_inputs_are_rejected():
    with pytest.raises(ValueError):
        compute_rouge(["a"], [])
    with pytest.raises(ValueError):
        compute_rouge([], [])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: FAIL at collection, `ModuleNotFoundError: No module named 'textsummarizer.metrics'`.

- [ ] **Step 3: Implement** to the interfaces above (`rouge_score.rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)`).

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `62 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(metrics): ROUGE with sentence-aware Lsum, BERTScore helper"
```

---

### Task 5: Train stage

**Files:**
- Create: `src/textsummarizer/components/model_trainer.py`, `src/textsummarizer/pipeline/stage_04_model_trainer.py`, `tests/test_model_trainer.py`
- Modify: `src/textsummarizer/cli.py`, `tests/conftest.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 1 models, Task 2 `atomic_directory`, Task 3 `UserFacingError`, Task 4 `compute_rouge`.
- Produces:
  - `tests/conftest.py`: `tiny_model_dir` (session scope, `-> Path`; the tiny Pegasus from "Verified facts" saved with `save_pretrained`); `app_config` now sets both `base_model` fields to `str(tiny_model_dir)`; `prepared` (`-> AppConfig`; saves the raw tiny dataset, runs `DataTransformation` with the tiny tokenizer, writes `Validation status: True` to the status file).
  - `ModelTrainer(config: TrainerConfig, params: TrainingParams, generation: GenerationParams)` with `train(resume: bool = False) -> Path` returning `config.model_dir`.
    - Raises `UserFacingError` (message mentions `textsummarizer validate`) unless the status file's first line is `Validation status: True`.
    - Loads the tokenized dataset from `data_dir`; `max_train_samples` / `max_eval_samples` take the first N rows of `train` / `validation`.
    - Builds `Seq2SeqTrainer` from `AutoModelForSeq2SeqLM.from_pretrained(base_model)`, `DataCollatorForSeq2Seq`, `processing_class=tokenizer`, `predict_with_generate=True`, `generation_max_length=generation.max_length`, `generation_num_beams=params.eval_num_beams`, `fp16=params.fp16 and torch.cuda.is_available()`, `save_total_limit=2`, `load_best_model_at_end=True`, `metric_for_best_model="rougeL"`, `output_dir=checkpoints_dir`, `report_to=[params.report_to]` (or `"none"`).
    - Evaluates and saves once per epoch; when `params.max_steps > 0` it evaluates and saves once at step `max_steps` instead.
    - `compute_metrics` decodes predictions (replacing `-100` with the pad id) and labels with `skip_special_tokens=True` and returns `compute_rouge(...)`.
    - With `resume=True` it continues from `get_last_checkpoint(checkpoints_dir)` and logs `Resuming from <path>`; if there is none it logs `No checkpoint found; starting from scratch`.
    - Saves the best model and tokenizer with `atomic_directory(model_dir)`, and writes `metrics_file` as JSON with keys `base_model`, `train` (the training metrics), `validation` (final evaluation of the best model), `best_checkpoint`, `train_samples`, `eval_samples`, `training_params`.
  - `stage_04_model_trainer.run(config: AppConfig, resume: bool = False) -> Path` (creates `root_dir`).
  - CLI command `train` with `--resume/--no-resume` (default off); `_run_stages` forwards keyword arguments to each stage's `run`.

- [ ] **Step 1: Add the fixtures and write the failing tests** in `tests/test_model_trainer.py`

```python
import json
import logging

import pytest
from transformers import AutoModelForSeq2SeqLM

from textsummarizer.components.model_trainer import ModelTrainer
from textsummarizer.errors import UserFacingError


def trainer_for(cfg, **overrides):
    params = cfg.training.model_copy(update=overrides)
    return ModelTrainer(cfg.model_trainer, params, cfg.generation)


def test_train_refuses_when_validation_never_ran(prepared):
    prepared.model_trainer.status_file.unlink()
    with pytest.raises(UserFacingError, match="validate"):
        trainer_for(prepared).train()


def test_train_refuses_when_validation_failed(prepared):
    prepared.model_trainer.status_file.write_text("Validation status: False\n- train: bad\n")
    with pytest.raises(UserFacingError, match="validate"):
        trainer_for(prepared).train()


def test_train_saves_a_loadable_model_and_tokenizer(prepared):
    model_dir = trainer_for(prepared, max_steps=2).train()
    assert (model_dir / "config.json").is_file() and (model_dir / "tokenizer.json").is_file()
    assert AutoModelForSeq2SeqLM.from_pretrained(model_dir) is not None


def test_train_writes_metrics_with_validation_rouge(prepared):
    trainer_for(prepared, max_steps=2).train()
    metrics = json.loads(prepared.model_trainer.metrics_file.read_text())
    assert metrics["base_model"] == prepared.model_trainer.base_model
    assert {"eval_rouge1", "eval_rougeL", "eval_rougeLsum"} <= set(metrics["validation"])
    assert metrics["best_checkpoint"].startswith("checkpoint-")
    assert metrics["train_samples"] == 4 and metrics["eval_samples"] == 2


def test_each_epoch_is_checkpointed_and_only_two_are_kept(prepared):
    trainer_for(prepared, num_train_epochs=3).train()
    checkpoints = sorted(
        p.name for p in prepared.model_trainer.checkpoints_dir.glob("checkpoint-*")
    )
    assert len(checkpoints) == 2


def test_resume_continues_from_the_last_checkpoint(prepared, caplog):
    caplog.set_level(logging.INFO)
    trainer_for(prepared, num_train_epochs=1).train()
    trainer_for(prepared, num_train_epochs=2).train(resume=True)
    assert "Resuming from" in caplog.text
    assert (prepared.model_trainer.checkpoints_dir / "checkpoint-4").is_dir()


def test_resume_without_a_checkpoint_starts_from_scratch(prepared, caplog):
    caplog.set_level(logging.INFO)
    trainer_for(prepared, max_steps=2).train(resume=True)
    assert "No checkpoint found" in caplog.text


def test_sample_limits_shrink_the_data(prepared):
    trainer_for(prepared, max_steps=1, max_train_samples=2, max_eval_samples=1).train()
    metrics = json.loads(prepared.model_trainer.metrics_file.read_text())
    assert (metrics["train_samples"], metrics["eval_samples"]) == (2, 1)
```

and in `tests/test_cli.py` (and extend `test_help_lists_the_stage_commands` to include `train`):

```python
def test_train_forwards_the_resume_flag(monkeypatch, config_files):
    seen = {}
    monkeypatch.setattr(
        stage_04_model_trainer, "run", lambda config, resume=False: seen.update(resume=resume)
    )
    assert invoke(config_files, "train", "--resume").exit_code == 0
    assert seen == {"resume": True}
```

(import `stage_04_model_trainer` from `textsummarizer.pipeline` in that file).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_model_trainer.py tests/test_cli.py -q`
Expected: FAIL at collection, `ModuleNotFoundError: No module named 'textsummarizer.components.model_trainer'`.

- [ ] **Step 3: Implement** the component, stage and CLI command to the interfaces above. Use `logging.getLogger(__name__)`; do not print or log the token or any environment variable.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `71 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(train): Seq2SeqTrainer stage with validation gate, resume and atomic model save"
```

---

### Task 6: Evaluate stage

**Files:**
- Create: `src/textsummarizer/components/model_evaluation.py`, `src/textsummarizer/pipeline/stage_05_model_evaluation.py`, `tests/test_model_evaluation.py`
- Modify: `src/textsummarizer/cli.py`, `tests/test_cli.py` (extend the help test with `evaluate`)

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 5 (the `prepared` fixture and a trained or tiny model directory).
- Produces:
  - `ModelEvaluation(config: EvaluationConfig, generation: GenerationParams, params: EvaluationParams, bertscore_fn: Callable[[list[str], list[str]], float] | None = None)` with `evaluate() -> Path` returning `config.metrics_file`.
    - Raises `UserFacingError` (message mentions `textsummarizer train`) if `model_dir` does not exist.
    - Loads the `test` split of the tokenized dataset (first `max_test_samples` rows if set). For the baseline (`config.base_model`) and then the fine-tuned model (`config.model_dir`) it generates from the stored `input_ids` (padded per batch, `torch.no_grad`, on CUDA when available) with exactly `generation`'s four settings, decodes with `skip_special_tokens=True`, and scores with `compute_rouge` against the stored `summary` column. It frees each model before loading the next.
    - BERTScore: when `params.bertscore_model` is set, the score comes from `bertscore_fn` if given, else `metrics.bertscore_f1(preds, refs, params.bertscore_model)` (predictions with `<n>` replaced by a space); otherwise the value is `None`.
    - `metrics_file` JSON: `{"num_examples": N, "generation": {four settings}, "baseline": {"model": str, four ROUGE keys, "bertscore_f1": float | null}, "fine_tuned": {same shape}, "improvement": {four ROUGE keys: fine_tuned minus baseline, rounded to 2 decimals}}`.
    - `examples_file` Markdown: one `### Example <i>` section per example for the first `params.num_examples` rows, each listing dialogue, reference, baseline and fine-tuned text.
  - `stage_05_model_evaluation.run(config: AppConfig) -> Path` (creates `root_dir`); CLI command `evaluate`.

- [ ] **Step 1: Write the failing tests** in `tests/test_model_evaluation.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_model_evaluation.py -q`
Expected: FAIL at collection, `ModuleNotFoundError: No module named 'textsummarizer.components.model_evaluation'`.

- [ ] **Step 3: Implement** to the interfaces above.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `77 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(evaluate): baseline vs fine-tuned ROUGE/BERTScore on the full test split"
```

---

### Task 7: Publish stage

**Files:**
- Create: `src/textsummarizer/components/model_publisher.py`, `src/textsummarizer/pipeline/stage_06_publish.py`, `tests/test_model_publisher.py`
- Modify: `src/textsummarizer/cli.py`, `tests/test_cli.py` (extend the help test with `publish`; add the force test)

**Interfaces:**
- Consumes: Tasks 1, 3 (`UserFacingError`), 6 (the `metrics.json` shape).
- Produces:
  - `render_model_card(metrics: dict, config: PublishConfig, base_model: str) -> str`: YAML front matter (`license: other`, `language: en`, `base_model`, `datasets: [knkarthick/samsum]`, `tags: [summarization]`, `metrics: [rouge]`) followed by a Markdown body with a baseline vs fine-tuned ROUGE table (two decimals), the generation settings, and a note that the training data is SAMSum under CC BY-NC-ND 4.0 and the model is for non-commercial demo use.
  - `ModelPublisher(config: PublishConfig, base_model: str, api: object | None = None)` with `publish(force: bool = False) -> str` returning the repo id.
    - Raises `UserFacingError` naming `textsummarizer train` / `textsummarizer evaluate` when `model_dir` or `metrics_file` is missing.
    - Raises `UserFacingError` mentioning `--force` unless every value in `metrics["improvement"]` is positive, or `force` is true.
    - With no `api` it reads `HF_TOKEN` from the environment and raises `UserFacingError` mentioning `HF_TOKEN` if absent, then builds `huggingface_hub.HfApi(token=token)` (the class is looked up as `HfApi` in this module so tests can replace it).
    - Writes `README.md` (the card) and a copy of the metrics as `eval_metrics.json` into `model_dir`, calls `api.create_repo(repo_id=..., private=config.private, exist_ok=True)` then `api.upload_folder(repo_id=..., folder_path=str(model_dir), commit_message=...)`.
  - `stage_06_publish.run(config: AppConfig, force: bool = False) -> str`; CLI command `publish` with `--force/--no-force`.

- [ ] **Step 1: Write the failing tests** in `tests/test_model_publisher.py`

```python
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
```

and in `tests/test_cli.py`:

```python
def test_publish_forwards_the_force_flag(monkeypatch, config_files):
    seen = {}
    monkeypatch.setattr(
        stage_06_publish, "run", lambda config, force=False: seen.update(force=force)
    )
    assert invoke(config_files, "publish", "--force").exit_code == 0
    assert seen == {"force": True}
```

(import `stage_06_publish` from `textsummarizer.pipeline`).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_model_publisher.py tests/test_cli.py -q`
Expected: FAIL at collection, `ModuleNotFoundError: No module named 'textsummarizer.components.model_publisher'`.

- [ ] **Step 3: Implement** to the interfaces above. `ModelPublisher` takes `base_model` because `PublishConfig` does not carry it; the stage passes `config.model_trainer.base_model`.

- [ ] **Step 4: Verify**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `85 passed`, lint and format clean.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(publish): model card, quality gate and Hub upload"
```

---

### Task 8: Colab notebook, README, and a real CPU smoke run

**Files:**
- Create: `notebooks/colab_train.ipynb`, `tests/test_notebook.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: the CLI commands `ingest`, `validate`, `transform`, `all`, `train`, `evaluate`, `publish`.
- Produces: a notebook (nbformat 4, no stored outputs) whose cells, in order:
  1. Markdown: purpose, "use a GPU runtime", expected run time 1.5 to 3 hours on a T4, the `HF_TOKEN` Colab secret, and the instruction to restart the session if `pip` asks.
  2. Code: `!nvidia-smi`.
  3. Code: `BRANCH = "claude/practical-davinci-q2qix4"  # change to "main" once merged`, `!git clone --branch $BRANCH https://github.com/tapankhaladkar/NLP-Text-Summarizer.git`, `%cd NLP-Text-Summarizer`, `!pip install -q -e ".[train]"` (this keeps Colab's preinstalled CUDA torch).
  4. Code (optional, commented): mount Google Drive and symlink `artifacts` to a Drive folder so checkpoints survive a disconnect.
  5. Code: read the Colab secret into `os.environ["HF_TOKEN"]`; never print it.
  6. Code: `!textsummarizer all`.
  7. Code: `!textsummarizer train --resume`.
  8. Code: `!textsummarizer evaluate`, then display `artifacts/model_evaluation/metrics.json` and the head of `examples.md`.
  9. Code: `!textsummarizer publish`.

- [ ] **Step 1: Write the failing tests** in `tests/test_notebook.py`

```python
import json
import re
from pathlib import Path

from typer.testing import CliRunner

from textsummarizer.cli import app

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "colab_train.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def test_notebook_is_clean_and_holds_no_secrets():
    text = NOTEBOOK.read_text()
    assert json.loads(text)["nbformat"] == 4
    assert all(c.get("outputs", []) == [] for c in cells() if c["cell_type"] == "code")
    assert not re.search(r"hf_[A-Za-z0-9]{10,}", text)


def test_notebook_calls_only_commands_the_cli_has():
    listed = CliRunner().invoke(app, ["--help"]).output
    used = set()
    for cell in cells():
        for line in "".join(cell["source"]).splitlines():
            match = re.match(r"!textsummarizer\s+(?:--\S+\s+\S+\s+)*([a-z]+)", line)
            if match:
                used.add(match.group(1))
    assert {"all", "train", "evaluate", "publish"} <= used
    assert all(command in listed for command in used)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_notebook.py -q`
Expected: FAIL (`FileNotFoundError` for `notebooks/colab_train.ipynb`).

- [ ] **Step 3: Create the notebook** to the cell list above, as JSON with empty `outputs`, `execution_count: null`, and the `kernelspec`/`language_info` of a Python 3 notebook.

- [ ] **Step 4: README.** In the Setup section add one sentence saying to install `uv` first (https://docs.astral.sh/uv/) and that `uv sync --python 3.11 --extra train` selects Python 3.11 instead of the pinned 3.12. Then add a "Training on Colab" section: open `notebooks/colab_train.ipynb` in Colab with a T4 runtime, add the `HF_TOKEN` secret (a Hugging Face token with write access), run the cells; if fp16 produces NaN losses set `training.fp16: false` in `params.yaml`; if memory runs out lower `per_device_train_batch_size` and raise `gradient_accumulation_steps` to keep 16, or set `gradient_checkpointing: true`. Document the `train`, `evaluate` and `publish` commands, `--resume`, `--force`, and the `HF_TOKEN` variable.

- [ ] **Step 5: Verify the suite**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `87 passed`, lint and format clean.

- [ ] **Step 6: Real CPU smoke run** with the actual Pegasus checkpoint on a few examples (needs the network; not part of CI; peak memory is about 10 GB, so close other work)

```bash
SMOKE="$SCRATCH/params_smoke.yaml"   # SCRATCH: any directory outside the repo
uv run python - "$SMOKE" <<'EOF'
import sys

import yaml

p = yaml.safe_load(open("params.yaml"))
p["training"].update(
    max_steps=2,
    max_train_samples=8,
    max_eval_samples=8,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    eval_num_beams=2,
)
p["evaluation"].update(
    max_test_samples=4, batch_size=2, num_examples=2, bertscore_model="distilbert-base-uncased"
)
yaml.safe_dump(p, open(sys.argv[1], "w"))
EOF
rm -rf artifacts/model_trainer artifacts/model_evaluation
uv run textsummarizer --params $SMOKE all && uv run textsummarizer --params $SMOKE train && uv run textsummarizer --params $SMOKE evaluate
env -u HF_TOKEN uv run textsummarizer --params $SMOKE publish; echo "publish-exit=$?"
```

Expected (about 10 minutes): `all` re-downloads the dataset first, because the Phase 1 output has no `ingestion.json` marker, and then the first three commands exit 0. `artifacts/model_trainer/model/config.json` exists. `artifacts/model_evaluation/metrics.json` shows `num_examples` 4, a `baseline` and a `fine_tuned` section with four ROUGE values and numeric `bertscore_f1`, and the same four `generation` settings. `examples.md` holds 2 examples. `publish` exits 1 with a one-line message (either the `--force` quality gate, since two training steps will not beat the baseline reliably, or the missing `HF_TOKEN`) and no traceback. Afterwards run `rm -rf artifacts/model_trainer artifacts/model_evaluation` so smoke output can never be resumed as real training.

- [ ] **Step 7: Commit, push, update the roadmap**

```bash
git add -A
git commit -m "feat: Colab training notebook, README, smoke-tested train/evaluate/publish"
git push
```

Then in the roadmap's Phase 2 row set the Plan cell to `2026-11-04-phase-2-train-evaluate.md` with the status "code implemented; Colab training run pending", and commit and push.

---

## The real training run (done by the user on Colab, not an executor task)

1. Create a Hugging Face access token with write permission and store it as the Colab secret `HF_TOKEN`; confirm `publish.repo_id` in `config/config.yaml` matches the account.
2. Open `notebooks/colab_train.ipynb` on a T4 runtime and run the cells. Expect 1.5 to 3 hours; if the session disconnects, rerun the notebook from the clone cell on; `train --resume` continues from the last epoch checkpoint (mount Drive in cell 4 first if you want checkpoints to survive).
3. Success means all of: `improvement` is positive for all four ROUGE values (the `publish` gate checks this), the model repo exists on the Hub with a model card, and `examples.md` shows coherent summaries on a hand check of ten examples. From memory, published fine-tuned seq2seq models score roughly 40 to 55 ROUGE-1 on SAMSum and the zero-shot CNN/DailyMail model scores noticeably lower; treat a fine-tuned ROUGE-1 far below about 35, or a NaN loss in `train_metrics.json`, as a sign to investigate (try `fp16: false`) before publishing.
4. Bring back the Hub repo link and the contents of `metrics.json`; they go into the roadmap and the Phase 5 README.
