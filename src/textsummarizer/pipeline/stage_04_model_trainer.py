from pathlib import Path

from textsummarizer.components.model_trainer import ModelTrainer
from textsummarizer.config import AppConfig


def run(config: AppConfig, resume: bool = False) -> Path:
    config.model_trainer.root_dir.mkdir(parents=True, exist_ok=True)
    trainer = ModelTrainer(config.model_trainer, config.training, config.generation)
    return trainer.train(resume=resume)
