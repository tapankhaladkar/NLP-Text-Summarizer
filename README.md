## Text Summarizer using NLP

## Setup

```bash
uv sync --extra train
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Needs Python 3.11+. Installs CPU-only PyTorch; GPU training instructions arrive with Phase 2.

## Running the pipeline

```bash
uv run textsummarizer all          # ingest, validate, transform; stops at the first failure
uv run textsummarizer ingest       # or run one stage at a time:
uv run textsummarizer validate     #   validate, transform
uv run textsummarizer transform
```

Settings live in `config/config.yaml` and `params.yaml`. Outputs land in `artifacts/`:
the downloaded dataset in `artifacts/data_ingestion/`, the validation result in
`artifacts/data_validation/status.txt`, and the tokenized dataset in
`artifacts/data_transformation/`. Re-running skips a stage's download if its output
already exists; delete the folder to redo it. Relative paths in `config/config.yaml`
resolve against the repository root, so the commands work from any directory.

The data comes from the Hugging Face Hub copy of SAMSum (`knkarthick/samsum`, pinned
to a revision), which is licensed CC BY-NC-ND 4.0: non-commercial use only.

The notebooks in `research/` show the original, pre-refactor code and are kept as
history; they no longer run against the current package API.