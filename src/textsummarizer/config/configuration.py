from textsummarizer.constants import *
from textsummarizer.utils.common import read_yaml, create_directories
from textsummarizer.entity import data_ingestionConfig
from textsummarizer.entity import DataValidationConfig
from textsummarizer.entity import DataTransformationConfig

class ConfigManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> data_ingestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = data_ingestionConfig(
            source_url=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            root_dir=Path(config.root_dir)
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
            root_dir = Path(config.root_dir),
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Returns the data transformation configuration."""
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            tokenizer_name=config.tokenizer_name,
            data_path=config.data_path,
            root_dir=config.root_dir
        )
        return data_transformation_config