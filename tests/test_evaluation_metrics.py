from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import EvaluationError, evaluate_predictions
from src.evaluation.models import (
    EvaluationCase,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationPrediction,
)

DOC = "Nhóm3_TTCSN.docx"
S1 = f"{DOC}::Đoạn 1"
S2 = f"{DOC}::Bảng 1"
S3 = f"{DOC}::Đoạn 3"


def _case(
    case_id: str,
    *,
    answerable: bool = True,
    sources: tuple[str, ...] = (S1,),
    facts: tuple[str, ...] = ("sự kiện",),
    terms: tuple[str, ...] = (),
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        split="holdout",
        question="Câu hỏi",
        history=(),
        intent="Kiểm tra",
        required_terms=terms,
        relevant_sources=sources if answerable else (),
        expected_facts=facts if answerable else (),
        answerable=answerable,
        categories=("direct_fact",) if answerable else ("unanswerable",),
        document_file_names=(DOC,),
    )


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        schema_version="1.0",
        dataset_id="dataset",
        dataset_version="1.2.0",
        corpus_id="corpus",
        corpus_version="sha256:" + "a" * 64,
        context_version="ingestion-v3-test",
        documents=(DOC,),
        cases=(
            _case("one", sources=(S1, S2), facts=("alpha", "beta"), terms=("Mã X1", "Nguyễn")),
            _case("two", sources=(S3,), facts=("gamma",)),
            _case("three", answerable=False),
        ),
    )


def _predictions() -> list[EvaluationPrediction]:
    return [
        EvaluationPrediction(
            case_id="one",
            retrieved_sources=(f"{DOC}::Đoạn 9", S1, S2),
            effective_queries=("Mã X1 dùng để làm gì?",),
            answer_text="Có alpha.",
            cited_sources=(S1, f"{DOC}::Đoạn 9"),
            retrieval_latency_ms=10,
            end_to_end_latency_ms=100,
            model_calls=1,
            input_tokens=10,
            output_tokens=5,
            estimated_cost_usd=0.01,
        ),
        EvaluationPrediction(
            case_id="two",
            retrieved_sources=(S3,),
            effective_queries=("Câu hỏi",),
            answer_text="Có gamma.",
            cited_sources=(S3,),
            retrieval_latency_ms=20,
            end_to_end_latency_ms=200,
            model_calls=2,
            input_tokens=20,
            output_tokens=10,
            estimated_cost_usd=0.02,
        ),
        EvaluationPrediction(
            case_id="three",
            refused=True,
            retrieval_latency_ms=30,
            end_to_end_latency_ms=300,
        ),
    ]


def test_evaluate_predictions_computes_quality_latency_and_cost() -> None:
    report = evaluate_predictions(
        _dataset(), _predictions(), EvaluationConfig(strategy_id="dense", top_k=2)
    )
    metrics = report.metrics
    case_one_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))

    assert metrics.hit_rate_at_k == 1.0
    assert metrics.recall_at_k == 0.75
    assert metrics.mrr == 0.75
    assert metrics.ndcg_at_k == pytest.approx((case_one_ndcg + 1) / 2)
    assert metrics.answerability_precision == 1.0
    assert metrics.answerability_recall == 1.0
    assert metrics.answerability_f1 == 1.0
    assert metrics.citation_precision == pytest.approx(2 / 3)
    assert metrics.citation_recall == pytest.approx(2 / 3)
    assert metrics.citation_coverage == 1.0
    assert metrics.expected_fact_coverage == 0.75
    assert metrics.required_term_retention == 0.5
    assert metrics.query_drift_rate == 1.0
    assert metrics.retrieval_latency_p50_ms == 20
    assert metrics.retrieval_latency_p95_ms == 29
    assert metrics.end_to_end_latency_p50_ms == 200
    assert metrics.end_to_end_latency_p95_ms == 290
    assert metrics.model_calls_total == 3
    assert metrics.input_tokens_total == 30
    assert metrics.output_tokens_total == 15
    assert metrics.estimated_cost_usd_total == pytest.approx(0.03)
    assert report.as_dict()["metrics"]["recall_at_k"] == 0.75


def test_answerability_counts_false_positive_and_false_negative() -> None:
    predictions = _predictions()
    predictions[0] = EvaluationPrediction(case_id="one", refused=True)
    predictions[2] = EvaluationPrediction(case_id="three", answer_text="Bịa câu trả lời")

    metrics = evaluate_predictions(
        _dataset(), predictions, EvaluationConfig(strategy_id="dense", top_k=2)
    ).metrics

    assert metrics.answerability_precision == 0.5
    assert metrics.answerability_recall == 0.5
    assert metrics.answerability_f1 == 0.5


@pytest.mark.parametrize(
    "predictions,error",
    [
        ([EvaluationPrediction(case_id="one"), EvaluationPrediction(case_id="one")], "Duplicate"),
        ([EvaluationPrediction(case_id="missing")], "Unknown"),
        (
            [EvaluationPrediction(case_id="one", cited_sources=("bad-ref",))],
            "malformed",
        ),
        ([EvaluationPrediction(case_id="one", model_calls=-1)], "negative"),
    ],
)
def test_evaluate_predictions_rejects_invalid_predictions(
    predictions: list[EvaluationPrediction], error: str
) -> None:
    with pytest.raises(EvaluationError, match=error):
        evaluate_predictions(
            _dataset(), predictions, EvaluationConfig(strategy_id="dense", top_k=2)
        )


def test_duplicate_retrieval_does_not_inflate_ndcg_or_recall() -> None:
    dataset = _dataset()
    prediction = EvaluationPrediction(case_id="one", retrieved_sources=(S1, S1))

    metrics = evaluate_predictions(
        dataset, [prediction], EvaluationConfig(strategy_id="dense", top_k=2)
    ).metrics

    assert metrics.recall_at_k == 0.5
    assert metrics.ndcg_at_k == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_config_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        EvaluationConfig(strategy_id="dense", top_k=0)


def test_config_rejects_invalid_cost_or_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="token costs"):
        EvaluationConfig(strategy_id="dense", top_k=8, input_cost_per_million=-1)
    with pytest.raises(ValueError, match="chunk_overlap"):
        EvaluationConfig(strategy_id="dense", top_k=8, chunk_size=100, chunk_overlap=100)
