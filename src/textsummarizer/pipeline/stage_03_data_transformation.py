from pathlib import Path

from textsummarizer.components.data_transformation import DataTransformation
from textsummarizer.config import AppConfig


def run(config: AppConfig) -> Path:
    config.data_transformation.root_dir.mkdir(parents=True, exist_ok=True)
    return DataTransformation(config.data_transformation, config.tokenization).convert()
