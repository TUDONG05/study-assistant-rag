"""Strict JSON loader for versioned evaluation datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.evaluation.models import EvaluationCase, EvaluationDataset, EvaluationTurn

ALLOWED_SPLITS = frozenset({"development", "holdout"})
SUPPORTED_SCHEMA_VERSION = "1.0"
SHA256_VERSION_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ALLOWED_CATEGORIES = frozenset(
    {
        "direct_fact",
        "exact_term",
        "cross_chunk",
        "cross_document",
        "unanswerable",
        "adversarial",
        "conversational",
        "paraphrase",
        "ambiguous",
        "multi_query",
    }
)
SOURCE_REF_PATTERN = re.compile(
    r"^(?P<document>[^:\s](?:[^:]*[^:\s])?)::"
    r"(?P<label>Đoạn \d+(?:–\d+)?|Bảng \d+|Trang \d+|Slide \d+)$"
)


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset violates its stable contract."""


def load_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate one UTF-8 JSON evaluation dataset."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Cannot load evaluation dataset: {exc}") from exc
    return parse_dataset(raw)


def parse_dataset(raw: Any) -> EvaluationDataset:
    data = _mapping(raw, "dataset")
    schema_version = _text(data.get("schema_version"), "schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise DatasetValidationError(f"Unsupported schema_version: {schema_version!r}")
    documents = _text_tuple(data.get("documents"), "documents", allow_empty=False)
    _require_unique(documents, "documents")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DatasetValidationError("cases must be a non-empty list")
    cases = tuple(_parse_case(item, documents, index) for index, item in enumerate(raw_cases))
    _require_unique(tuple(case.case_id for case in cases), "case IDs")
    corpus_version = _text(data.get("corpus_version"), "corpus_version")
    if not SHA256_VERSION_PATTERN.fullmatch(corpus_version):
        raise DatasetValidationError("corpus_version must be a sha256 digest")
    return EvaluationDataset(
        schema_version=schema_version,
        dataset_id=_text(data.get("dataset_id"), "dataset_id"),
        dataset_version=_text(data.get("dataset_version"), "dataset_version"),
        corpus_id=_text(data.get("corpus_id"), "corpus_id"),
        corpus_version=corpus_version,
        context_version=_text(data.get("context_version"), "context_version"),
        documents=documents,
        cases=cases,
    )


def is_valid_source_ref(value: object) -> bool:
    """Return whether a source uses the portable ``document::source_label`` format."""

    return isinstance(value, str) and SOURCE_REF_PATTERN.fullmatch(value) is not None


def source_document(source_ref: str) -> str:
    if not is_valid_source_ref(source_ref):
        raise DatasetValidationError(f"Malformed source reference: {source_ref!r}")
    return source_ref.split("::", 1)[0]


def _parse_case(raw: Any, documents: tuple[str, ...], index: int) -> EvaluationCase:
    data = _mapping(raw, f"cases[{index}]")
    case_id = _text(data.get("case_id"), f"cases[{index}].case_id")
    split = _text(data.get("split"), f"{case_id}.split")
    if split not in ALLOWED_SPLITS:
        raise DatasetValidationError(f"{case_id}.split must be development or holdout")
    history = _parse_history(data.get("history"), case_id)
    required_terms = _unique_text_tuple(data.get("required_terms"), f"{case_id}.required_terms")
    relevant_sources = _unique_text_tuple(
        data.get("relevant_sources"), f"{case_id}.relevant_sources"
    )
    for source_ref in relevant_sources:
        document = source_document(source_ref)
        if document not in documents:
            raise DatasetValidationError(f"{case_id} references unknown document {document!r}")
    categories = _unique_text_tuple(data.get("categories"), f"{case_id}.categories")
    if not categories or not set(categories) <= ALLOWED_CATEGORIES:
        raise DatasetValidationError(f"{case_id}.categories contains an unsupported category")
    case_documents = _unique_text_tuple(
        data.get("document_file_names"), f"{case_id}.document_file_names"
    )
    if not case_documents or not set(case_documents) <= set(documents):
        raise DatasetValidationError(f"{case_id}.document_file_names is invalid")
    if any(source_document(ref) not in case_documents for ref in relevant_sources):
        raise DatasetValidationError(f"{case_id} source is outside document_file_names")
    answerable = data.get("answerable")
    if not isinstance(answerable, bool):
        raise DatasetValidationError(f"{case_id}.answerable must be a boolean")
    expected_facts = _unique_text_tuple(data.get("expected_facts"), f"{case_id}.expected_facts")
    if answerable and (not relevant_sources or not expected_facts):
        raise DatasetValidationError(
            f"Answerable case {case_id!r} requires relevant_sources and expected_facts"
        )
    if not answerable and (relevant_sources or expected_facts):
        raise DatasetValidationError(
            f"Unanswerable case {case_id!r} cannot define relevant_sources or expected_facts"
        )
    if answerable == ("unanswerable" in categories):
        raise DatasetValidationError(f"{case_id} has inconsistent unanswerable category")
    return EvaluationCase(
        case_id=case_id,
        split=split,
        question=_text(data.get("question"), f"{case_id}.question"),
        history=history,
        intent=_text(data.get("intent"), f"{case_id}.intent"),
        required_terms=required_terms,
        relevant_sources=relevant_sources,
        expected_facts=expected_facts,
        answerable=answerable,
        categories=categories,
        document_file_names=case_documents,
    )


def _parse_history(raw: Any, case_id: str) -> tuple[EvaluationTurn, ...]:
    if not isinstance(raw, list):
        raise DatasetValidationError(f"{case_id}.history must be a list")
    turns: list[EvaluationTurn] = []
    for index, item in enumerate(raw):
        turn = _mapping(item, f"{case_id}.history[{index}]")
        role = _text(turn.get("role"), f"{case_id}.history[{index}].role")
        if role not in {"user", "assistant"} or (turns and turns[-1].role == role):
            raise DatasetValidationError(f"{case_id}.history has invalid role ordering")
        turns.append(EvaluationTurn(role, _text(turn.get("content"), f"{case_id}.history.content")))
    if turns and turns[0].role != "user":
        raise DatasetValidationError(f"{case_id}.history must start with user")
    return tuple(turns)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DatasetValidationError(f"{field} must be trimmed, non-empty text")
    return value


def _text_tuple(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "non-empty " if not allow_empty else ""
        raise DatasetValidationError(f"{field} must be a {qualifier}list")
    return tuple(_text(item, f"{field}[{index}]") for index, item in enumerate(value))


def _unique_text_tuple(value: Any, field: str) -> tuple[str, ...]:
    items = _text_tuple(value, field)
    _require_unique(items, field)
    return items


def _require_unique(items: tuple[str, ...], field: str) -> None:
    if len(items) != len(set(items)):
        raise DatasetValidationError(f"{field} must not contain duplicates")
