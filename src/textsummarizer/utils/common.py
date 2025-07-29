import os 
from box.exceptions import BoxValueError
import yaml
from textsummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the YAML file.
        
    Returns:
        ConfigBox: Contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except FileNotFoundError as e:
        logger.error(f"File not found: {path_to_yaml}")
        raise e
    except BoxValueError as e:
        logger.error(f"Error reading YAML file: {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """
    Creates directories if they do not exist.
    
    Args:
        path_to_directories (list[str]): List of directory paths to create.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}.")
        #logger.info(f"Directory {directory} created or already exists.")
            

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file or directory.
    
    Args:
        path (Path): Path to the file or directory.
        
    Returns:
        str: Size in Kb.
    """

    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} Kb"
     