from pathlib import Path

from textsummarizer.components.model_evaluation import ModelEvaluation
from textsummarizer.config import AppConfig


def run(config: AppConfig) -> Path:
    config.model_evaluation.root_dir.mkdir(parents=True, exist_ok=True)
    evaluation = ModelEvaluation(config.model_evaluation, config.generation, config.evaluation)
    return evaluation.evaluate()
