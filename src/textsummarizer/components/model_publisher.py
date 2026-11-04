import json
import logging
import os
import shutil

from huggingface_hub import HfApi

from textsummarizer.config import PublishConfig
from textsummarizer.errors import UserFacingError
from textsummarizer.metrics import ROUGE_KEYS

logger = logging.getLogger(__name__)

DATASET_ID = "knkarthick/samsum"


def render_model_card(metrics: dict, config: PublishConfig, base_model: str) -> str:
    """The README.md that goes on the Hub: front matter, results table, license note."""
    baseline, tuned, change = metrics["baseline"], metrics["fine_tuned"], metrics["improvement"]
    table = [
        f"| {key} | {baseline[key]:.2f} | {tuned[key]:.2f} | {change[key]:+.2f} |"
        for key in ROUGE_KEYS
    ]
    settings = ", ".join(f"{name}={value}" for name, value in metrics["generation"].items())
    lines = [
        "---",
        "license: other",
        "language: en",
        f"base_model: {base_model}",
        "datasets:",
        f"- {DATASET_ID}",
        "tags:",
        "- summarization",
        "metrics:",
        "- rouge",
        "---",
        "",
        f"# {config.repo_id}",
        "",
        f"`{base_model}` fine-tuned to summarize chat dialogues on SAMSum.",
        "",
        f"## Results on {metrics['num_examples']} test examples (ROUGE F1 x 100)",
        "",
        "| Metric | Zero-shot base | Fine-tuned | Change |",
        "|---|---|---|---|",
        *table,
        "",
        f"Both models were scored with identical generation settings: {settings}.",
        "",
        "## Intended use and license",
        "",
        "This is a demo model for non-commercial use. It was trained on the SAMSum corpus,",
        "which is licensed CC BY-NC-ND 4.0, starting from the base model named above;",
        "check that model's terms as well before any other use.",
        "",
    ]
    return "\n".join(lines)


class ModelPublisher:
    """Upload the fine-tuned model, a model card and the metrics to the Hugging Face Hub."""

    def __init__(self, config: PublishConfig, base_model: str, api: object | None = None):
        self.config = config
        self.base_model = base_model
        self._api = api

    def publish(self, force: bool = False) -> str:
        config = self.config
        if not config.model_dir.is_dir():
            raise UserFacingError(
                f"No trained model at {config.model_dir}. Run `textsummarizer train` first."
            )
        if not config.metrics_file.is_file():
            raise UserFacingError(
                f"No evaluation results at {config.metrics_file}. "
                "Run `textsummarizer evaluate` first."
            )
        metrics = json.loads(config.metrics_file.read_text())
        not_better = sorted(key for key, gain in metrics["improvement"].items() if gain <= 0)
        if not_better and not force:
            raise UserFacingError(
                f"The fine-tuned model does not beat the baseline on {', '.join(not_better)}, "
                "so it will not be published. Use --force to publish it anyway."
            )

        api = self._api or self._make_api()
        card = render_model_card(metrics, config, self.base_model)
        (config.model_dir / "README.md").write_text(card)
        shutil.copyfile(config.metrics_file, config.model_dir / "eval_metrics.json")
        api.create_repo(repo_id=config.repo_id, private=config.private, exist_ok=True)
        api.upload_folder(
            repo_id=config.repo_id,
            folder_path=str(config.model_dir),
            commit_message="Upload fine-tuned model",
        )
        logger.info("Published %s", config.repo_id)
        return config.repo_id

    @staticmethod
    def _make_api() -> HfApi:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise UserFacingError(
                "Set the HF_TOKEN environment variable to a Hugging Face token with write "
                "access (in Colab, add it under Secrets)."
            )
        return HfApi(token=token)
