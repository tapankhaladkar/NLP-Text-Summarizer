# Phase 0: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repo a reproducible `uv` project with linting, tests and CI, without changing any pipeline behavior.

**Architecture:** Replace `setup.py` + `requirements.txt` with one `pyproject.toml` and a committed `uv.lock`. Heavy training deps live in a `train` extra so a later serving image can skip them. Add three smoke tests over the existing code, lint/format it with `ruff`, and run all of it in GitHub Actions on Python 3.11 and 3.12.

**Tech Stack:** `uv`, `hatchling`, `pytest`, `ruff`, GitHub Actions. All steps were dry-run on a copy of the repo (Python 3.11 and 3.12, uv 0.11, transformers 5.20, torch 2.14 CPU).

**Spec:** `docs/superpowers/plans/2026-10-19-text-summarizer-roadmap.md` (Phase 0)

## Global Constraints

- Python `>=3.11`, tested on 3.11 and 3.12.
- `transformers>=5.0,<6`, `datasets>=4.0,<6`, `torch>=2.6` from `https://download.pytorch.org/whl/cpu`; `uv.lock` must contain no `nvidia` packages.
- `ruff` line length 100; `research/` is excluded from lint.
- No behavior changes to pipeline code: only formatting, import fixes, and packaging.
- Work on branch `claude/practical-davinci-q2qix4`; no pull request.
- Commit messages end with the attribution trailers the session specifies.

## Review Focus

- Running tests leaves the working tree dirty (`logs/` is created on import, `artifacts/` by `ConfigManager`). Expected: `git status --short` shows nothing after `pytest`. Pinned in Task 1, Step 8.
- Python 3.11 behaves differently from 3.12. Pinned by the CI matrix in Task 3.
- `pyproject.toml` edited without re-locking. Expected: CI fails rather than silently resolving. Pinned in Task 3, Step 4.
- Ruff autofix (`UP`, `I`) changes the behavior of the `@ensure_annotations`-checked functions. Pinned by `test_config_manager_builds_ingestion_config`, re-run in Task 2, Step 5.
- A non-editable install cannot find `config/config.yaml` (`PROJECT_ROOT` is derived from `__file__`). Not fixed here; Phase 4 owns it. `test_config_paths_point_at_repo_files` pins the editable-install case only.

---

### Task 1: Installable project with smoke tests

**Files:**
- Create: `tests/test_smoke.py`, `pyproject.toml`, `.python-version`, `uv.lock` (generated)
- Modify: `.gitignore`, `README.md`
- Delete: `setup.py`, `requirements.txt`; untrack `.DS_Store`, `research/.DS_Store`, `src/.DS_Store`, `src/textsummarizer/.DS_Store`

**Interfaces:**
- Produces: `uv sync --locked --extra train` yields a working environment; `uv run pytest` runs `tests/`; the `dev` group provides `pytest` and `ruff`.

- [ ] **Step 1: Write the failing tests** in `tests/test_smoke.py`

```python
import importlib
import pkgutil

import textsummarizer
from textsummarizer.config.configuration import ConfigManager
from textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


def test_every_module_imports():
    failures = {}
    for module in pkgutil.walk_packages(textsummarizer.__path__, prefix="textsummarizer."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:
            failures[module.name] = repr(exc)
    assert not failures, failures


def test_config_paths_point_at_repo_files():
    assert CONFIG_FILE_PATH.is_file()
    assert PARAMS_FILE_PATH.is_file()


def test_config_manager_builds_ingestion_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # ConfigManager creates relative artifact dirs
    config = ConfigManager().get_data_ingestion_config()
    assert config.source_url.endswith("summarizer-data.zip")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest`
Expected: non-zero exit (no project environment yet; `pytest` or `textsummarizer` not found).

- [ ] **Step 3: Create `pyproject.toml`** with exactly this content

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "textsummarizer"
version = "0.1.0"
description = "Dialogue summarization with a fine-tuned Pegasus model"
readme = "README.md"
requires-python = ">=3.11"
license = { file = "LICENSE" }
authors = [{ name = "tapankhaladkar", email = "tapankhaladkar@gmail.com" }]
dependencies = [
    "pyyaml>=6.0",
    "python-box>=7.2",
    "ensure>=1.0.4",
]

[project.optional-dependencies]
train = [
    "torch>=2.6",
    "transformers[sentencepiece,torch]>=5.0,<6",
    "datasets>=4.0,<6",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "ruff>=0.8",
]

# uv only redirects *direct* dependencies, so torch must be listed in `train` above.
[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
extend-exclude = ["research"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 4: Pin Python and lock**

Run: `echo 3.12 > .python-version && uv lock && grep -c nvidia uv.lock`
Expected: `Resolved 67 packages` (±2), and the grep prints `0`. `grep -A3 '^name = "torch"' uv.lock` shows `source = { registry = "https://download.pytorch.org/whl/cpu" }`.

- [ ] **Step 5: Sync and check the environment**

Run: `uv sync --locked --extra train && uv run python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"`
Expected: prints a `+cpu` torch version, `False`, and a transformers version starting `5.`. The install is about 1.2 GB.

- [ ] **Step 6: Remove old packaging and untracked-worthy files**

Run: `git rm setup.py requirements.txt && git rm --cached .DS_Store research/.DS_Store src/.DS_Store src/textsummarizer/.DS_Store`
Then append three lines to `.gitignore`: `.DS_Store`, `logs/`, `wandb/`. (`.venv`, `artifacts/*` and `.ruff_cache/` are already ignored.)

- [ ] **Step 7: Add a Setup section to `README.md`** (below the title, above "Workflow"), containing these commands in a fenced block: `uv sync --extra train`, `uv run pytest`, `uv run ruff check . && uv run ruff format --check .`; plus one sentence: "Needs Python 3.11+. Installs CPU-only PyTorch; GPU training instructions arrive with Phase 2."

- [ ] **Step 8: Run tests and check the tree stays clean**

Run: `uv run pytest -q && git status --short`
Expected: `3 passed`. `git status` lists only the intended changes (new files, deletions, `.gitignore`, `README.md`); no `logs/` or `artifacts/`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .python-version tests .gitignore README.md
git commit -m "build: move to uv + pyproject, add smoke tests"
```

---

### Task 2: Lint and format baseline

**Files:**
- Modify: every `.py` file ruff touches; by hand: `src/textsummarizer/config/configuration.py:1`, `src/textsummarizer/logging/__init__.py:1`

**Interfaces:**
- Consumes: Task 1's `dev` group (`ruff`) and the `[tool.ruff]` config.
- Produces: a tree where `ruff check .` and `ruff format --check .` pass.

- [ ] **Step 1: Run lint to see it fail**

Run: `uv run ruff check .`
Expected: FAIL, about 39 errors, about 27 auto-fixable (E501, UP015, I001, F401, F403/F405, among others).

- [ ] **Step 2: Apply the automatic fixes**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: about 14 files reformatted, and 8 errors remain: `F403` + six `F405` in `configuration.py` (star import), `F401` in `logging/__init__.py`.

- [ ] **Step 3: Replace the star import in `configuration.py`** with `from pathlib import Path` and `from textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH` (it only uses those three names).

- [ ] **Step 4: Mark the logger re-export** in `logging/__init__.py`: `from .logger import logger as logger`.

- [ ] **Step 5: Verify lint, format and behavior**

Run: `uv run ruff check . --fix -q; uv run ruff format -q . ; uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
Expected: `All checks passed!`, `already formatted`, `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "style: ruff baseline (format, import order, explicit imports)"
```

---

### Task 3: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Delete: `.github/workflows/.gitkeep`
- Modify: roadmap status (Step 6)

**Interfaces:**
- Consumes: Tasks 1-2 (`uv.lock`, passing lint and tests).
- Produces: a check that runs on every push and pull request.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  check:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    env:
      UV_PYTHON: ${{ matrix.python-version }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv sync --locked --extra train
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest
```

`UV_PYTHON` overrides the committed `.python-version`; verified locally that `UV_PYTHON=3.11` keeps the 3.11 venv.

- [ ] **Step 2: Run the CI commands locally**

Run: `git rm -q .github/workflows/.gitkeep; uv sync --locked --extra train && uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
Expected: all pass, `3 passed`.

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint and test on Python 3.11 and 3.12"
git push -u origin claude/practical-davinci-q2qix4
```

- [ ] **Step 4: Prove `--locked` catches drift**

Change `pyyaml>=6.0` to `pyyaml>=6.0.1` in `pyproject.toml` without re-locking, run `uv sync --locked --extra train`.
Expected: fails with a message that the lockfile needs to be updated. Revert with `git checkout pyproject.toml`.

- [ ] **Step 5: Confirm CI is green on GitHub**

Use the GitHub Actions tools (`actions_list` for workflow runs on this branch; `get_job_logs` on failure) to check the run for the pushed commit.
Expected: both matrix jobs (`3.11`, `3.12`) conclude `success`. If a job fails because an action tag does not resolve, bump that action to its current major version and push again.

- [ ] **Step 6: Mark Phase 0 done** in the roadmap table (add `done` to the Phase 0 row's Plan cell), then commit and push:

```bash
git add docs/superpowers/plans/2026-10-19-text-summarizer-roadmap.md
git commit -m "docs: mark phase 0 done"
git push
```
