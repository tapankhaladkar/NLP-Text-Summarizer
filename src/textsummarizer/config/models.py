from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _anchor_relative_paths(cls, value, info: ValidationInfo):
        """Make relative Path fields absolute when load_config supplies a base directory."""
        base_dir = (info.context or {}).get("base_dir")
        if base_dir is not None and isinstance(value, Path) and not value.is_absolute():
            return base_dir / value
        return value


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


class TrainerConfig(_Frozen):
    root_dir: Path
    data_dir: Path
    status_file: Path
    base_model: str
    checkpoints_dir: Path
    model_dir: Path
    metrics_file: Path


class EvaluationConfig(_Frozen):
    root_dir: Path
    data_dir: Path
    base_model: str
    model_dir: Path
    metrics_file: Path
    examples_file: Path


class PublishConfig(_Frozen):
    model_dir: Path
    metrics_file: Path
    repo_id: str
    private: bool


class TokenizationParams(_Frozen):
    max_input_length: int
    max_target_length: int


class TrainingParams(_Frozen):
    num_train_epochs: float
    learning_rate: float
    warmup_steps: float
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    fp16: bool
    gradient_checkpointing: bool
    eval_num_beams: int
    seed: int
    max_steps: int
    max_train_samples: int | None
    max_eval_samples: int | None
    report_to: str


class GenerationParams(_Frozen):
    num_beams: int
    max_length: int
    min_length: int
    length_penalty: float


class EvaluationParams(_Frozen):
    batch_size: int
    max_test_samples: int | None
    num_examples: int
    bertscore_model: str | None


class AppConfig(_Frozen):
    artifacts_root: Path
    data_ingestion: IngestionConfig
    data_validation: ValidationConfig
    data_transformation: TransformationConfig
    model_trainer: TrainerConfig
    model_evaluation: EvaluationConfig
    publish: PublishConfig
    tokenization: TokenizationParams
    training: TrainingParams
    generation: GenerationParams
    evaluation: EvaluationParams
