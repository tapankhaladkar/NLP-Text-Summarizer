import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from textsummarizer.config import IngestionConfig
from textsummarizer.storage import save_dataset_dict

logger = logging.getLogger(__name__)


class DataIngestion:
    """Download the pinned dataset revision from the Hugging Face Hub to disk."""

    def __init__(self, config: IngestionConfig, loader: Callable[..., Any] | None = None):
        self.config = config
        self._loader = loader

    def download(self) -> Path:
        output_dir = self.config.output_dir
        if output_dir.exists():
            logger.info("Dataset already present at %s; skipping download", output_dir)
            return output_dir

        loader = self._loader
        if loader is None:
            from datasets import load_dataset

            loader = load_dataset
        logger.info("Loading %s at revision %s", self.config.dataset_name, self.config.revision)
        dataset_dict = loader(self.config.dataset_name, revision=self.config.revision)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_dict(dataset_dict, output_dir)
        logger.info("Saved dataset to %s", output_dir)
        return output_dir
