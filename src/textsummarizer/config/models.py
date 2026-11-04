from pathlib import Path

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IngestionConfig(_Frozen):
    root_dir: Path
    dataset_name: str
    revision: str
    output_dir: Path


class ValidationConfig(_Frozen):
    root_dir: Path
    data_dir: Path
    status_file: Path
    required_columns: list[str]
    expected_rows: dict[str, int]


class TransformationConfig(_Frozen):
    root_dir: Path
    data_dir: Path
    output_dir: Path
    tokenizer_name: str


class TokenizationParams(_Frozen):
    max_input_length: int
    max_target_length: int


class AppConfig(_Frozen):
    artifacts_root: Path
    data_ingestion: IngestionConfig
    data_validation: ValidationConfig
    data_transformation: TransformationConfig
    tokenization: TokenizationParams
