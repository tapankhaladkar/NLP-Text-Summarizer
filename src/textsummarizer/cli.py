import importlib
import logging
from pathlib import Path
from typing import Annotated

import typer

from textsummarizer.config import load_config
from textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textsummarizer.log import configure_logging

logger = logging.getLogger(__name__)

app = typer.Typer(help="Run the dialogue-summarization data pipeline.", no_args_is_help=True)

INGESTION = "stage_01_data_ingestion"
VALIDATION = "stage_02_data_validation"
TRANSFORMATION = "stage_03_data_transformation"


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option(help="Path to config.yaml.")] = CONFIG_FILE_PATH,
    params: Annotated[Path, typer.Option(help="Path to params.yaml.")] = PARAMS_FILE_PATH,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    configure_logging(log_level)
    ctx.obj = (config, params)


def _run_stages(ctx: typer.Context, *stages: str) -> None:
    """Run pipeline stages in order, stopping at the first failure.

    Stage modules are imported here, not at the top of the file, so that importing
    the CLI never loads datasets or transformers. `module.run` is looked up at call
    time so tests can replace it.
    """
    config_path, params_path = ctx.obj
    try:
        config = load_config(config_path, params_path)
        for stage in stages:
            module = importlib.import_module(f"textsummarizer.pipeline.{stage}")
            logger.info(">>>>> %s started <<<<<", stage)
            module.run(config)
            logger.info(">>>>> %s completed <<<<<", stage)
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


@app.command("all")
def run_all(ctx: typer.Context) -> None:
    """Run ingest, validate and transform in order; stop at the first failure."""
    _run_stages(ctx, INGESTION, VALIDATION, TRANSFORMATION)
