from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.dataset import DatasetValidationError, load_dataset, parse_dataset


def _dataset() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_id": "nhom3-ttcsn",
        "dataset_version": "1.0.0",
        "corpus_id": "nhom3",
        "corpus_version": "sha256:" + "a" * 64,
        "context_version": "ingestion-v3-test",
        "documents": ["Nhóm3_TTCSN.docx"],
        "cases": [
            {
                "case_id": "direct-001",
                "split": "holdout",
                "question": "Mục tiêu của hệ thống là gì?",
                "history": [
                    {"role": "user", "content": "Tài liệu nào?"},
                    {"role": "assistant", "content": "Bài tập lớn nhóm 3."},
                ],
                "intent": "Nêu mục tiêu hệ thống",
                "required_terms": ["hệ thống"],
                "relevant_sources": ["Nhóm3_TTCSN.docx::Đoạn 1–3"],
                "expected_facts": ["Hỗ trợ học tập"],
                "answerable": True,
                "categories": ["direct_fact", "conversational"],
                "document_file_names": ["Nhóm3_TTCSN.docx"],
            },
            {
                "case_id": "unanswerable-001",
                "split": "development",
                "question": "Thông tin không có là gì?",
                "history": [],
                "intent": "Từ chối câu ngoài tài liệu",
                "required_terms": [],
                "relevant_sources": [],
                "expected_facts": [],
                "answerable": False,
                "categories": ["unanswerable"],
                "document_file_names": ["Nhóm3_TTCSN.docx"],
            },
        ],
    }


def test_load_dataset_builds_immutable_versioned_contract(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_dataset(), ensure_ascii=False), encoding="utf-8")

    dataset = load_dataset(path)

    assert dataset.dataset_id == "nhom3-ttcsn"
    assert dataset.cases[0].history[1].role == "assistant"
    assert dataset.as_dict()["cases"][0]["case_id"] == "direct-001"
    with pytest.raises(FrozenInstanceError):
        dataset.dataset_version = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("categories", []),
        ("categories", ["unknown"]),
        ("history", "not-a-list"),
        ("history", [{"role": "system", "content": "x"}]),
        (
            "history",
            [
                {"role": "user", "content": "x"},
                {"role": "user", "content": "y"},
            ],
        ),
        ("relevant_sources", ["Nhóm3_TTCSN.docx#Đoạn 1"]),
        ("relevant_sources", ["Khác.docx::Đoạn 1"]),
    ],
)
def test_parse_dataset_rejects_invalid_case_fields(field: str, value: object) -> None:
    raw = _dataset()
    raw["cases"][0][field] = value

    with pytest.raises(DatasetValidationError):
        parse_dataset(raw)


def test_parse_dataset_rejects_duplicate_case_ids() -> None:
    raw = _dataset()
    duplicate = deepcopy(raw["cases"][0])
    raw["cases"].append(duplicate)

    with pytest.raises(DatasetValidationError, match="case IDs"):
        parse_dataset(raw)


@pytest.mark.parametrize("missing_field", ["relevant_sources", "expected_facts"])
def test_answerable_case_requires_gold_evidence(missing_field: str) -> None:
    raw = _dataset()
    raw["cases"][0][missing_field] = []

    with pytest.raises(DatasetValidationError, match="Answerable case"):
        parse_dataset(raw)


def test_unanswerable_case_rejects_gold_facts() -> None:
    raw = _dataset()
    raw["cases"][1]["expected_facts"] = ["Không hợp lệ"]

    with pytest.raises(DatasetValidationError, match="Unanswerable case"):
        parse_dataset(raw)


def test_load_dataset_wraps_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Cannot load"):
        load_dataset(path)


def test_parse_dataset_rejects_unknown_schema_version() -> None:
    raw = _dataset()
    raw["schema_version"] = "2.0"

    with pytest.raises(DatasetValidationError, match="Unsupported schema_version"):
        parse_dataset(raw)


@pytest.mark.parametrize(
    "field,value",
    [("corpus_version", "sha256:abc"), ("context_version", "")],
)
def test_parse_dataset_requires_versioned_corpus_and_pipeline(field: str, value: str) -> None:
    raw = _dataset()
    raw[field] = value

    with pytest.raises(DatasetValidationError):
        parse_dataset(raw)
