from __future__ import annotations

import json
from datetime import datetime, tzinfo
from pathlib import Path

import pytest

from src.evaluation.models import (
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationPrediction,
    EvaluationReport,
)
from src.evaluation.reporting import load_report, report_to_markdown, write_report


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _FixedDateTime:
        return cls(2026, 9, 13, 1, 2, 3, tzinfo=tz)


def _report() -> EvaluationReport:
    metrics = EvaluationMetrics(
        hit_rate_at_k=1.0,
        recall_at_k=0.75,
        mrr=0.5,
        ndcg_at_k=0.625,
        answerability_precision=1.0,
        answerability_recall=0.5,
        answerability_f1=2 / 3,
        citation_precision=0.8,
        citation_recall=0.6,
        citation_coverage=0.5,
        expected_fact_coverage=0.75,
        required_term_retention=1.0,
        query_drift_rate=0.0,
        retrieval_latency_p50_ms=12.0,
        retrieval_latency_p95_ms=20.0,
        end_to_end_latency_p50_ms=100.0,
        end_to_end_latency_p95_ms=150.0,
        model_calls_total=2,
        input_tokens_total=30,
        output_tokens_total=10,
        estimated_cost_usd_total=0.0001,
    )
    return EvaluationReport(
        schema_version="1.0",
        dataset_id="nhóm-3-eval",
        dataset_version="1.0.0",
        corpus_id="Nhóm3_TTCSN.docx",
        corpus_version="sha256:" + "a" * 64,
        config=EvaluationConfig(strategy_id="dense", top_k=8),
        evaluated_case_count=1,
        metrics=metrics,
        predictions=(
            EvaluationPrediction(
                case_id="case-1",
                retrieved_sources=("Nhóm3_TTCSN.docx::Đoạn 1",),
                answer_text="Câu trả lời",
            ),
        ),
    )


def test_write_load_and_render_report_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.evaluation.reporting.datetime", _FixedDateTime)

    json_path, markdown_path = write_report(_report(), tmp_path / "nested", split="holdout")
    loaded = load_report(json_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert json_path.name == "latest.json"
    assert markdown_path.name == "latest.md"
    assert loaded["dataset_id"] == "nhóm-3-eval"
    assert loaded["split"] == "holdout"
    assert loaded["generated_at"] == "2026-09-13T01:02:03+00:00"
    assert loaded["config"]["top_k"] == 8
    assert loaded["predictions"][0]["answer_text"] == "Câu trả lời"
    assert json.loads(json_path.read_text(encoding="utf-8")) == loaded
    assert report_to_markdown(loaded) == markdown
    assert "# Benchmark nhóm-3-eval" in markdown
    assert "| Hit Rate@8 | 1.0000 |" in markdown
    assert "| Answerability F1 | 0.6667 |" in markdown
    assert "Nhóm3_TTCSN.docx" in markdown
    assert markdown.endswith("retrieval xuyên tài liệu.\n")


@pytest.mark.parametrize("payload", [[], {}, {"metrics": []}, {"metrics": None}])
def test_load_report_rejects_invalid_shapes(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="không hợp lệ"):
        load_report(path)
