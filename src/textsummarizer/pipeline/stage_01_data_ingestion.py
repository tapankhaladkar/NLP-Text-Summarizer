from pathlib import Path

from textsummarizer.components.data_ingestion import DataIngestion
from textsummarizer.config import AppConfig


def run(config: AppConfig) -> Path:
    config.data_ingestion.root_dir.mkdir(parents=True, exist_ok=True)
    return DataIngestion(config.data_ingestion).download()
