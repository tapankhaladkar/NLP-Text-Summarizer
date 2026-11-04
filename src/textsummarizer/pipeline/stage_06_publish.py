from textsummarizer.components.model_publisher import ModelPublisher
from textsummarizer.config import AppConfig


def run(config: AppConfig, force: bool = False) -> str:
    publisher = ModelPublisher(config.publish, config.model_trainer.base_model)
    return publisher.publish(force=force)
