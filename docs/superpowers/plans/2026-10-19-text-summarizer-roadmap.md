# Text Summarizer Roadmap

Roadmap and spec for taking this repo from a half-built pipeline to a deployed demo. Each phase gets its own detailed plan, written just before that phase starts, because later phases depend on what earlier ones teach us (the Hub dataset's real columns, real CPU latency, real ROUGE).

**Goal:** A public demo where a user pastes a chat dialogue and gets a short summary from a Pegasus model fine-tuned on SAMSum. Pushing to `main` runs lint and tests, builds a container, and updates the live demo.

## Decisions

- **Demo project, not a product.** SAMSum is believed to be CC BY-NC-ND 4.0 (verify before launch); the README states non-commercial, demo use.
- **Training stack:** `transformers` 5.x + PyTorch, `Seq2SeqTrainer`. Verified: transformers 5.20.2 is PyTorch-only, `Trainer` takes `processing_class`, `as_target_tokenizer` is removed.
- **Model:** `google/pegasus-cnn_dailymail` first. Comparison against `flan-t5-base`/`bart-base` and an LLM baseline is optional, after the pipeline is complete.
- **Tooling:** `uv` + `pyproject.toml`, Pydantic config, `typer` CLI, `ruff`, `pytest`, GitHub Actions.
- **Hosting:** model on the Hugging Face Hub; demo on a Hugging Face Docker Space; the image must also run elsewhere (Cloud Run, ECS).
- **Where things run:** training on the user's Colab/Kaggle GPU. Local dev, CI and the demo are CPU-only.

## Open items (defaults apply unless the user says otherwise)

- Hub username, model repo name, Space name: needed in Phase 2 and 4.
- Model repo public (default) or private.
- AWS target wanted (default: no).

## Global Constraints

- Python `>=3.11`, tested on 3.11 and 3.12.
- `transformers>=5.0,<6`, `datasets>=4.0,<6`, `torch>=2.6` from the PyTorch CPU index.
- `ruff` line length 100; `research/` notebooks are excluded from lint.
- Work on branch `claude/practical-davinci-q2qix4`. No pull request unless the user asks.
- Commit messages end with the attribution trailers the session specifies.

## Phases

| # | Phase | Done when | Plan |
|---|---|---|---|
| 0 | Foundation: `pyproject.toml`/`uv`, ruff, pytest, CI | A clean checkout passes `uv sync --locked`, lint and tests in CI on 3.11 and 3.12 | `2026-10-19-phase-0-foundation.md` |
| 1 | Refactor stages 1-3: Pydantic config, import-safe logging, `textsummarizer` CLI, real data validation, Hub dataset ingestion, transformers-5 tokenization with dynamic padding and 512 max length | All three stages run end to end on CPU with tests on a tiny model | not yet written |
| 2 | Train and evaluate: `Seq2SeqTrainer` stage, zero-shot baseline, ROUGE bug fix, full-test ROUGE + BERTScore, push model and card to the Hub | Hub repo exists; `metrics.json` shows fine-tuned beating the baseline with sane ROUGE | not yet written |
| 3 | Predict and API: `Summarizer` class, FastAPI `/summarize` + `/health`, simple UI, API tests, CPU latency recorded | Local `curl` returns a summary | not yet written |
| 4 | Container and deploy: multi-stage Dockerfile with CPU torch, container smoke test in CI, GHCR push, auto-deploy to the Space | Push to `main` produces a live URL with a passing health check | not yet written |
| 5 | Polish: README with architecture, metrics, demo link, license note; optional model comparison | A stranger can understand and try it in two minutes | not yet written |

## Risks

- **CPU latency.** Pegasus with 8 beams may be slow on a free CPU Space. Measure in Phase 3; fall back to fewer beams or a smaller model.
- **fp16 training.** May produce NaN on a T4; fall back to fp32, or bf16 on an L4/A100.
- **Non-editable installs.** `constants.PROJECT_ROOT` assumes an editable install and breaks inside a built image. Phase 4 must fix it (config path via argument or env var).
