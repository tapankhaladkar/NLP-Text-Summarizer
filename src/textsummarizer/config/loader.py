from pathlib import Path

import yaml

from textsummarizer.config.models import AppConfig
from textsummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


def load_config(
    config_path: Path = CONFIG_FILE_PATH, params_path: Path = PARAMS_FILE_PATH
) -> AppConfig:
    """Read config.yaml and params.yaml into one validated AppConfig.

    Relative paths resolve against the repository root (the directory above the config
    file's directory), never the current working directory. A section that appears in
    both files is an error. Pure: reads two files and creates nothing on disk.
    """
    config_data, params_data = _read_yaml(config_path), _read_yaml(params_path)
    overlap = sorted(set(config_data) & set(params_data))
    if overlap:
        raise ValueError(
            f"sections defined in both {Path(config_path).name} and "
            f"{Path(params_path).name}: {', '.join(overlap)}"
        )
    base_dir = Path(config_path).resolve().parent.parent
    return AppConfig.model_validate({**config_data, **params_data}, context={"base_dir": base_dir})


def _read_yaml(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}
