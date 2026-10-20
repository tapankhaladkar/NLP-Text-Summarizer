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

## Phase 2 training design (draft; the Phase 2 plan finalizes it)

**Verified on CPU with transformers 5.20.2 / torch 2.14:**
- `google/pegasus-cnn_dailymail` (571M params) loads. The load report lists `embed_positions.weight` as MISSING. That is benign: Pegasus uses fixed sinusoidal positions, and zero-shot output was coherent.
- A 2-step `Seq2SeqTrainer` run works with `processing_class=` and `text_target=` labels.
- Zero-shot output contains literal `<n>` tokens (Pegasus's newline marker). The notebook's evaluation line was meant to be `replace("<n>", " ")`; as written (`replace("", " ")`) it spaced out every character, which is why ROUGE was about 0.02.
- CPU generation for one 122-token input on 4 cores: about 14 s with 8 beams, about 3 s with 4 beams (8-beam run went first, so it includes some warm-up). A free 2-vCPU Space will be slower.

**Design:**
- **Data:** SAMSum 14,732 train / 818 validation / 819 test. Drop the one train row with an empty field. Inputs truncated to 512 tokens, labels to 128 (summaries are at most 64 words).
- **Generation defaults** come from the checkpoint (`num_beams=8`, `max_length=128`, `min_length=32`, `length_penalty=0.8`) and move into `params.yaml`. The serving beam count is chosen in Phase 3 from measured latency.
- **Trainer:** `Seq2SeqTrainer` with `predict_with_generate=True`, ROUGE as `compute_metrics`, evaluation on the validation split, best checkpoint kept by ROUGE-L, `save_total_limit=2`.
- **Starting hyperparameters** (tune with at most two further runs): effective batch size 16, learning rate 5e-5, warmup about 5% of steps (the notebook's fixed 500 warmup steps covered over half of its 921-step run), weight decay 0.01, 3 epochs (the notebook ran 1).
- **Hardware:** Colab gave a Tesla T4 (15 GB, no bf16). Use fp16 with a per-device batch of 4 and 4 gradient-accumulation steps. If memory runs out, enable gradient checkpointing or `optim="adafactor"`. If the loss goes NaN, fall back to fp32. The notebook's fp32, batch-size-1 run took about 46 minutes per epoch, so 3 epochs is a multi-hour session at worst.
- **Colab disconnects:** checkpoint to the Hub or Drive and support `resume_from_checkpoint`.
- **How it runs:** a thin notebook clones the repo, runs `pip install -e ".[train]"` (keeping Colab's preinstalled CUDA torch; the CPU-index pin only affects `uv sync`), reads `HF_TOKEN` from Colab secrets, runs `textsummarizer train`, and pushes model, tokenizer, `metrics.json` and a model card to the Hub.
- **Evaluation:** a zero-shot baseline and the fine-tuned model scored on the full test split with identical generation settings. Split sentences on `<n>`/newlines for ROUGE-Lsum. Add BERTScore and a handful of hand-checked examples. Success means the fine-tuned model clearly beats the baseline. I expect a well-trained model to land in the high 40s to low 50s ROUGE-1 on SAMSum (from memory; check against published numbers).
- **Tests without a GPU or a download:** unit tests use a tiny randomly initialized Pegasus built from a config, never the real checkpoint.

## Risks

- **CPU latency.** Measured above: 8 beams is too slow for a free CPU Space. Phase 3 picks the beam count from measurements, and falls back to a smaller model if 4 beams is still too slow.
- **fp16 training.** May produce NaN on a T4; fall back to fp32, or bf16 on an L4/A100.
- **Non-editable installs.** `constants.PROJECT_ROOT` assumes an editable install and breaks inside a built image. Phase 4 must fix it (config path via argument or env var).
