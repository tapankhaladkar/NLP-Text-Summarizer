from textsummarizer.config.configuration import ConfigManager
from textsummarizer.components.data_ingestion import DataIngestion
from textsummarizer.logging import logger

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)

        data_ingestion.download_data()
        data_ingestion.extract_zip_file()