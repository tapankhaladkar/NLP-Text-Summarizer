## Text Summarizer using NLP

## Setup

```bash
uv sync --extra train
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Install [uv](https://docs.astral.sh/uv/) first. Needs Python 3.11+ (`uv sync` picks the
pinned 3.12; use `uv sync --python 3.11 --extra train` for 3.11). This installs CPU-only
PyTorch, which is enough for development and the tests; training uses a GPU on Colab (see
below).

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

## Training, evaluating and publishing

```bash
uv run textsummarizer train            # fine-tune; add --resume to continue after an interruption
uv run textsummarizer evaluate         # zero-shot baseline vs fine-tuned, on the full test split
uv run textsummarizer publish          # upload to the Hugging Face Hub (needs HF_TOKEN)
```

`train` refuses to run until `validate` has passed. `evaluate` writes
`artifacts/model_evaluation/metrics.json` (ROUGE for both models with identical
generation settings, the improvement, and optional BERTScore) and side-by-side
`examples.md`. `publish` refuses to upload a model that does not beat the baseline on
every ROUGE variant unless you pass `--force`; the token comes only from the `HF_TOKEN`
environment variable. Training settings are in `params.yaml`, and the Hub repo id is
`publish.repo_id` in `config/config.yaml`.

### Training on Colab

Open `notebooks/colab_train.ipynb` in Google Colab with a **T4 GPU** runtime, add a
Colab secret named `HF_TOKEN` (a Hugging Face token with write access), and run the cells.
The full run takes roughly 1.5 to 3 hours. If something goes wrong:

- **Loss turns into NaN:** set `training.fp16: false` in `params.yaml`, delete
  `artifacts/model_trainer/checkpoints`, and run `train` again. (Without `--resume`, `train`
  refuses to reuse old checkpoints, and `--resume` refuses when the settings changed.)
- **Out of GPU memory:** lower `per_device_train_batch_size` and raise
  `gradient_accumulation_steps` to keep their product at 16, or set
  `gradient_checkpointing: true`.
- **The session disconnected:** run the notebook again; `train --resume` continues from the
  last epoch checkpoint if it is still there. The optional Drive cell keeps checkpoints across
  disconnects, but with the default AdamW optimizer each checkpoint is roughly 7 GB and two
  or three exist at once, which does not fit a free 15 GB Drive. Set `training.optim:
  adafactor` (roughly 2.5 GB per checkpoint) before using it, or use a larger Drive plan.

The notebooks in `research/` show the original, pre-refactor code and are kept as
history; they no longer run against the current package API.