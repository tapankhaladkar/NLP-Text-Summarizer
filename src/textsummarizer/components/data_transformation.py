import logging
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from textsummarizer.config import TokenizationParams, TransformationConfig
from textsummarizer.storage import save_dataset_dict

logger = logging.getLogger(__name__)


class DataTransformation:
    """Tokenize dialogues and summaries into model features for seq2seq training."""

    def __init__(self, config: TransformationConfig, params: TokenizationParams):
        self.config = config
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, batch: dict[str, list]) -> dict[str, list]:
        inputs = self.tokenizer(
            batch["dialogue"], max_length=self.params.max_input_length, truncation=True
        )
        labels = self.tokenizer(
            text_target=batch["summary"], max_length=self.params.max_target_length, truncation=True
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
        }

    def convert(self) -> Path:
        dataset_dict = load_from_disk(str(self.config.data_dir))
        # keep_in_memory: otherwise datasets writes cache-*.arrow files next to the source data
        features = dataset_dict.map(
            self.convert_examples_to_features, batched=True, keep_in_memory=True
        )
        output_dir = self.config.output_dir
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_dict(features, output_dir)
        logger.info("Saved tokenized dataset to %s", output_dir)
        return output_dir
