from textsummarizer.components.data_validation import (
    DataValidation,
    DataValidationError,
    ValidationReport,
)
from textsummarizer.config import AppConfig


def run(config: AppConfig) -> ValidationReport:
    settings = config.data_validation
    settings.root_dir.mkdir(parents=True, exist_ok=True)
    report = DataValidation(settings).validate()

    lines = [f"Validation status: {report.ok}", *(f"- {problem}" for problem in report.problems)]
    settings.status_file.parent.mkdir(parents=True, exist_ok=True)
    settings.status_file.write_text("\n".join(lines) + "\n")

    if not report.ok:
        raise DataValidationError("; ".join(report.problems))
    return report
