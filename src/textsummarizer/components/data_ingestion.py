import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from textsummarizer.config import IngestionConfig
from textsummarizer.storage import save_dataset_dict

logger = logging.getLogger(__name__)

MARKER_FILE = "ingestion.json"


class DataIngestion:
    """Download the pinned dataset revision from the Hugging Face Hub to disk."""

    def __init__(self, config: IngestionConfig, loader: Callable[..., Any] | None = None):
        self.config = config
        self._loader = loader

    def download(self) -> Path:
        output_dir = self.config.output_dir
        if self._is_current(output_dir):
            logger.info("Dataset already present at %s; skipping download", output_dir)
            return output_dir
        if output_dir.exists():
            logger.warning(
                "Replacing %s: it was not produced from revision %s",
                output_dir,
                self.config.revision,
            )

        loader = self._loader
        if loader is None:
            from datasets import load_dataset

            loader = load_dataset
        logger.info("Loading %s at revision %s", self.config.dataset_name, self.config.revision)
        dataset_dict = loader(self.config.dataset_name, revision=self.config.revision)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_dict(dataset_dict, output_dir)
        (output_dir / MARKER_FILE).write_text(json.dumps(self._marker()))
        logger.info("Saved dataset to %s", output_dir)
        return output_dir

    def _marker(self) -> dict[str, str]:
        return {"dataset_name": self.config.dataset_name, "revision": self.config.revision}

    def _is_current(self, output_dir: Path) -> bool:
        """True only if `output_dir` was written by this pipeline for the configured revision."""
        try:
            return json.loads((output_dir / MARKER_FILE).read_text()) == self._marker()
        except (OSError, ValueError):
            return False
