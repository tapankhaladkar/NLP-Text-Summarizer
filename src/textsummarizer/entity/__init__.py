from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class data_ingestionConfig:
    source_url: str
    local_data_file: Path
    unzip_dir: Path
    root_dir: Path = Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation."""
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list[str]
    root_dir: Path = Path