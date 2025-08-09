from textsummarizer.config.configuration import ConfigManager
from textsummarizer.components.data_validation import DataValidation
from textsummarizer.logging import logger

class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)

        data_validation.validate_all_files_exist()