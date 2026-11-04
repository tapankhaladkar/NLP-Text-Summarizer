import logging
from dataclasses import dataclass, field

from datasets import Dataset, DatasetDict, Value, load_from_disk

from textsummarizer.config import ValidationConfig

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ("dialogue", "summary")


class DataValidationError(Exception):
    """Raised by the validation stage when the dataset fails any check."""


@dataclass(frozen=True)
class ValidationReport:
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


class DataValidation:
    """Check that the ingested dataset has the expected splits, columns, rows and text."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate(self) -> ValidationReport:
        data_dir = self.config.data_dir
        try:
            loaded = load_from_disk(str(data_dir))
        except Exception as exc:
            return ValidationReport([f"cannot load a dataset from {data_dir}: {exc}"])
        if not isinstance(loaded, DatasetDict):
            return ValidationReport([f"{data_dir} holds a single dataset, expected named splits"])

        expected = self.config.expected_rows
        problems = [f"{split}: split is missing" for split in sorted(set(expected) - set(loaded))]
        problems += [f"{split}: unexpected split" for split in sorted(set(loaded) - set(expected))]
        for split, rows in expected.items():
            if split in loaded:
                problems += self._check_split(split, loaded[split], rows)
        return ValidationReport(problems)

    def _check_split(self, split: str, dataset: Dataset, rows: int) -> list[str]:
        problems = []
        columns, required = set(dataset.column_names), set(self.config.required_columns)
        problems += [f"{split}: missing column '{name}'" for name in sorted(required - columns)]
        problems += [f"{split}: unexpected column '{name}'" for name in sorted(columns - required)]
        if dataset.num_rows != rows:
            problems.append(f"{split}: expected {rows} rows, found {dataset.num_rows}")
        for name in TEXT_COLUMNS:
            if name not in columns:
                continue
            if not _is_text(dataset.features[name]):
                found = getattr(dataset.features[name], "dtype", type(dataset.features[name]))
                problems.append(f"{split}: column '{name}' must be text, found {found}")
                continue
            blanks = sum(1 for text in dataset[name] if text is None or not text.strip())
            if blanks:
                problems.append(f"{split}: {blanks} row(s) with blank {name}")
        return problems


def _is_text(feature) -> bool:
    return isinstance(feature, Value) and feature.dtype in ("string", "large_string")
