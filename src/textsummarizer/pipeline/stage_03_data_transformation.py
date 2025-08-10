from textsummarizer.config.configuration import ConfigManager
from textsummarizer.components.data_transformation import DataTransformation
from textsummarizer.logging import logger

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()