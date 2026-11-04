import gc
import json
import logging
from collections.abc import Callable
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textsummarizer.config import EvaluationConfig, EvaluationParams, GenerationParams
from textsummarizer.errors import UserFacingError
from textsummarizer.metrics import ROUGE_KEYS, bertscore_f1, compute_rouge

logger = logging.getLogger(__name__)


class ModelEvaluation:
    """Score the zero-shot base model and the fine-tuned model on the test split.

    Both models see the same stored inputs and the same generation settings, so the
    difference between them is the effect of fine-tuning.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        generation: GenerationParams,
        params: EvaluationParams,
        bertscore_fn: Callable[[list[str], list[str]], float] | None = None,
    ):
        self.config = config
        self.generation = generation
        self.params = params
        self._bertscore_fn = bertscore_fn

    def evaluate(self) -> Path:
        if not self.config.model_dir.is_dir():
            raise UserFacingError(
                f"No fine-tuned model found at {self.config.model_dir}. "
                "Run `textsummarizer train` first."
            )
        test = load_from_disk(str(self.config.data_dir))["test"]
        if self.params.max_test_samples is not None:
            test = test.select(range(min(self.params.max_test_samples, len(test))))
        references = list(test["summary"])

        sections, predictions = {}, {}
        for name, model_path in (
            ("baseline", self.config.base_model),
            ("fine_tuned", str(self.config.model_dir)),
        ):
            logger.info("Generating %d summaries with %s (%s)", len(test), name, model_path)
            predictions[name] = self._generate(model_path, test)
            sections[name] = {
                "model": model_path,
                **compute_rouge(predictions[name], references),
                "bertscore_f1": self._bertscore(predictions[name], references),
            }

        improvement = {
            key: round(sections["fine_tuned"][key] - sections["baseline"][key], 2)
            for key in ROUGE_KEYS
        }
        metrics = {
            "num_examples": len(test),
            "generation": self.generation.model_dump(),
            **sections,
            "improvement": improvement,
        }
        self.config.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_file.write_text(json.dumps(metrics, indent=2))
        self._write_examples(test, predictions)
        logger.info("Wrote %s", self.config.metrics_file)
        return self.config.metrics_file

    def _generate(self, model_path: str, dataset: Dataset) -> list[str]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
        g = self.generation
        outputs: list[str] = []
        size = self.params.batch_size
        for start in range(0, len(dataset), size):
            batch = dataset[start : start + size]
            features = [
                {"input_ids": ids, "attention_mask": mask}
                for ids, mask in zip(batch["input_ids"], batch["attention_mask"], strict=True)
            ]
            padded = tokenizer.pad(features, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **padded,
                    num_beams=g.num_beams,
                    max_length=g.max_length,
                    min_length=g.min_length,
                    length_penalty=g.length_penalty,
                )
            outputs += tokenizer.batch_decode(generated, skip_special_tokens=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return outputs

    def _bertscore(self, predictions: list[str], references: list[str]) -> float | None:
        model_type = self.params.bertscore_model
        if model_type is None:
            return None
        flat = [_flatten(text) for text in predictions]
        if self._bertscore_fn is not None:
            return self._bertscore_fn(flat, references)
        return bertscore_f1(flat, references, model_type)

    def _write_examples(self, test: Dataset, predictions: dict[str, list[str]]) -> None:
        lines = ["# Example outputs", ""]
        for i in range(min(self.params.num_examples, len(test))):
            row = test[i]
            lines += [
                f"### Example {i + 1}",
                "",
                "**Dialogue**",
                "",
                row["dialogue"],
                "",
                "**Reference**",
                "",
                row["summary"],
                "",
                "**Baseline**",
                "",
                _flatten(predictions["baseline"][i]),
                "",
                "**Fine-tuned**",
                "",
                _flatten(predictions["fine_tuned"][i]),
                "",
            ]
        self.config.examples_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.examples_file.write_text("\n".join(lines))


def _flatten(text: str) -> str:
    """Pegasus's `<n>` newline marker as a space, for display and BERTScore."""
    return text.replace("<n>", " ")
