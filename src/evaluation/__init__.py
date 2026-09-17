"""Public evaluation contracts, dataset loader, and deterministic metrics."""

from src.evaluation.dataset import (
    ALLOWED_CATEGORIES,
    ALLOWED_SPLITS,
    SUPPORTED_SCHEMA_VERSION,
    DatasetValidationError,
    is_valid_source_ref,
    load_dataset,
    parse_dataset,
    source_document,
)
from src.evaluation.metrics import EvaluationError, evaluate_predictions
from src.evaluation.models import (
    EvaluationCase,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationMetrics,
    EvaluationPrediction,
    EvaluationReport,
    EvaluationTurn,
)
from src.evaluation.reporting import load_report, report_to_markdown, write_report
from src.evaluation.runner import AnswerProvider, EvaluationRunError, run_evaluation

__all__ = [
    "ALLOWED_CATEGORIES",
    "ALLOWED_SPLITS",
    "SUPPORTED_SCHEMA_VERSION",
    "DatasetValidationError",
    "EvaluationCase",
    "EvaluationConfig",
    "EvaluationDataset",
    "EvaluationError",
    "EvaluationMetrics",
    "EvaluationPrediction",
    "EvaluationReport",
    "EvaluationTurn",
    "AnswerProvider",
    "EvaluationRunError",
    "evaluate_predictions",
    "is_valid_source_ref",
    "load_dataset",
    "parse_dataset",
    "load_report",
    "report_to_markdown",
    "run_evaluation",
    "source_document",
    "write_report",
]
