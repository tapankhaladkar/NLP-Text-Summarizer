import os
import urllib.request as request
import zipfile
from textsummarizer.logging import logger
from textsummarizer.utils.common import get_size
from pathlib import Path
from textsummarizer.entity import data_ingestionConfig

class DataIngestion:
    def __init__(self, config: data_ingestionConfig):
        self.config = config


    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(self.config.source_url, self.config.local_data_file)
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}") 

    def extract_zip_file(self):
        if not os.path.exists(self.config.unzip_dir):
            os.makedirs(self.config.unzip_dir)
            logger.info(f"Created directory: {self.config.unzip_dir}")

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
            logger.info(f"Extracted files to {self.config.unzip_dir}")

