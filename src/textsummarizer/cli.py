import importlib
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from textsummarizer.config import load_config
from textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textsummarizer.errors import UserFacingError
from textsummarizer.log import configure_logging

logger = logging.getLogger(__name__)

app = typer.Typer(help="Run the dialogue-summarization data pipeline.", no_args_is_help=True)

INGESTION = "stage_01_data_ingestion"
VALIDATION = "stage_02_data_validation"
TRANSFORMATION = "stage_03_data_transformation"
TRAINER = "stage_04_model_trainer"
EVALUATION = "stage_05_model_evaluation"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option(help="Path to config.yaml.")] = CONFIG_FILE_PATH,
    params: Annotated[Path, typer.Option(help="Path to params.yaml.")] = PARAMS_FILE_PATH,
    log_level: Annotated[
        LogLevel, typer.Option(case_sensitive=False, help="Logging level.")
    ] = LogLevel.INFO,
) -> None:
    configure_logging(log_level.value)
    ctx.obj = (config, params)


def _fail(message: str, *args) -> NoReturn:
    logger.error(message, *args)
    raise typer.Exit(code=1) from None


def _run_stages(ctx: typer.Context, *stages: str, **stage_kwargs) -> None:
    """Run pipeline stages in order, stopping at the first failure.

    Stage modules are imported here, not at the top of the file, so that importing
    the CLI never loads datasets or transformers. `module.run` is looked up at call
    time so tests can replace it; `stage_kwargs` are passed to every stage's `run`.
    Failures the user can fix are logged as one line; anything unexpected keeps its
    traceback.
    """
    config_path, params_path = ctx.obj
    try:
        config = load_config(config_path, params_path)
    except FileNotFoundError as exc:
        _fail("Config file not found: %s", exc.filename or exc)
    except ValueError as exc:  # includes pydantic.ValidationError
        _fail("Invalid configuration: %s", exc)

    try:
        for stage in stages:
            module = importlib.import_module(f"textsummarizer.pipeline.{stage}")
            logger.info(">>>>> %s started <<<<<", stage)
            module.run(config, **stage_kwargs)
            logger.info(">>>>> %s completed <<<<<", stage)
    except UserFacingError as exc:
        _fail("%s", exc)
    except FileNotFoundError as exc:
        _fail("File not found: %s", exc.filename or exc)
    except ImportError as exc:
        _fail(
            "Missing dependency: %s. Install the training extras with: uv sync --extra train", exc
        )
    except Exception:
        logger.exception("Pipeline failed")
        raise typer.Exit(code=1) from None


@app.command()
def ingest(ctx: typer.Context) -> None:
    """Download the pinned dataset revision from the Hugging Face Hub."""
    _run_stages(ctx, INGESTION)


@app.command()
def validate(ctx: typer.Context) -> None:
    """Check the ingested dataset's splits, columns, row counts and text."""
    _run_stages(ctx, VALIDATION)


@app.command()
def transform(ctx: typer.Context) -> None:
    """Tokenize the dataset into model features."""
    _run_stages(ctx, TRANSFORMATION)


@app.command()
def train(
    ctx: typer.Context,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Continue from the last checkpoint.")
    ] = False,
) -> None:
    """Fine-tune the model on the validated, tokenized data (needs a GPU for the full run)."""
    _run_stages(ctx, TRAINER, resume=resume)


@app.command()
def evaluate(ctx: typer.Context) -> None:
    """Score the zero-shot baseline and the fine-tuned model on the full test split."""
    _run_stages(ctx, EVALUATION)


@app.command("all")
def run_all(ctx: typer.Context) -> None:
    """Run ingest, validate and transform in order; stop at the first failure."""
    _run_stages(ctx, INGESTION, VALIDATION, TRANSFORMATION)
