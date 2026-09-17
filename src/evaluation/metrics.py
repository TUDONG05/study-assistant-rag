"""Deterministic, dependency-free metrics for offline RAG evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.evaluation.dataset import is_valid_source_ref
from src.evaluation.metric_primitives import divide, mean, normalize, percentile, ranking_scores
from src.evaluation.models import (
    EvaluationCase,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationMetrics,
    EvaluationPrediction,
    EvaluationReport,
)


class EvaluationError(ValueError):
    """Raised when predictions cannot be compared safely with a dataset."""


def evaluate_predictions(
    dataset: EvaluationDataset,
    predictions: Sequence[EvaluationPrediction],
    config: EvaluationConfig,
) -> EvaluationReport:
    pairs = _validated_pairs(dataset, predictions)
    retrieval = _retrieval_metrics(pairs, config.top_k)
    answerability = _answerability_metrics(pairs)
    citations = _citation_metrics(pairs)
    facts = _fact_coverage(pairs)
    retention, drift = _query_metrics(pairs)
    prediction_values = [prediction for _, prediction in pairs]
    metrics = EvaluationMetrics(
        **retrieval,
        **answerability,
        **citations,
        expected_fact_coverage=facts,
        required_term_retention=retention,
        query_drift_rate=drift,
        retrieval_latency_p50_ms=percentile(
            [item.retrieval_latency_ms for item in prediction_values], 0.50
        ),
        retrieval_latency_p95_ms=percentile(
            [item.retrieval_latency_ms for item in prediction_values], 0.95
        ),
        end_to_end_latency_p50_ms=percentile(
            [item.end_to_end_latency_ms for item in prediction_values], 0.50
        ),
        end_to_end_latency_p95_ms=percentile(
            [item.end_to_end_latency_ms for item in prediction_values], 0.95
        ),
        model_calls_total=sum(item.model_calls for item in prediction_values),
        input_tokens_total=sum(item.input_tokens for item in prediction_values),
        output_tokens_total=sum(item.output_tokens for item in prediction_values),
        estimated_cost_usd_total=math.fsum(
            item.estimated_cost_usd for item in prediction_values
        ),
    )
    return EvaluationReport(
        schema_version="1.0",
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        corpus_id=dataset.corpus_id,
        corpus_version=dataset.corpus_version,
        config=config,
        evaluated_case_count=len(pairs),
        metrics=metrics,
        predictions=tuple(prediction for _, prediction in pairs),
    )


def _validated_pairs(
    dataset: EvaluationDataset, predictions: Sequence[EvaluationPrediction]
) -> list[tuple[EvaluationCase, EvaluationPrediction]]:
    if not predictions:
        raise EvaluationError("At least one prediction is required")
    case_by_id = {case.case_id: case for case in dataset.cases}
    seen: set[str] = set()
    pairs: list[tuple[EvaluationCase, EvaluationPrediction]] = []
    for prediction in predictions:
        if prediction.case_id in seen:
            raise EvaluationError(f"Duplicate prediction ID: {prediction.case_id!r}")
        seen.add(prediction.case_id)
        if prediction.case_id not in case_by_id:
            raise EvaluationError(f"Unknown prediction case ID: {prediction.case_id!r}")
        _validate_prediction(prediction)
        pairs.append((case_by_id[prediction.case_id], prediction))
    return pairs


def _validate_prediction(prediction: EvaluationPrediction) -> None:
    refs = (*prediction.retrieved_sources, *prediction.cited_sources)
    if any(not is_valid_source_ref(ref) for ref in refs):
        raise EvaluationError(f"Prediction {prediction.case_id!r} has malformed source references")
    numeric = (
        prediction.retrieval_latency_ms,
        prediction.end_to_end_latency_ms,
        prediction.model_calls,
        prediction.input_tokens,
        prediction.output_tokens,
        prediction.estimated_cost_usd,
    )
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in numeric):
        raise EvaluationError(f"Prediction {prediction.case_id!r} has non-numeric counters")
    if any(not math.isfinite(float(value)) or value < 0 for value in numeric):
        raise EvaluationError(f"Prediction {prediction.case_id!r} has negative/non-finite counters")
    integer_counts = (prediction.model_calls, prediction.input_tokens, prediction.output_tokens)
    if any(not isinstance(value, int) for value in integer_counts):
        raise EvaluationError(f"Prediction {prediction.case_id!r} has non-integer counters")


def _retrieval_metrics(
    pairs: Sequence[tuple[EvaluationCase, EvaluationPrediction]], top_k: int
) -> dict[str, float]:
    values: list[tuple[float, float, float, float]] = []
    for case, prediction in pairs:
        if not case.relevant_sources:
            continue
        gold = set(case.relevant_sources)
        ranked = prediction.retrieved_sources[:top_k]
        values.append(ranking_scores(gold, ranked, top_k))
    columns = tuple(zip(*values, strict=True)) if values else ((), (), (), ())
    return {
        "hit_rate_at_k": mean(columns[0]),
        "recall_at_k": mean(columns[1]),
        "mrr": mean(columns[2]),
        "ndcg_at_k": mean(columns[3]),
    }


def _answerability_metrics(
    pairs: Sequence[tuple[EvaluationCase, EvaluationPrediction]],
) -> dict[str, float]:
    decisions = [(case.answerable, not pred.refused and pred.error is None) for case, pred in pairs]
    true_positive = sum(expected and actual for expected, actual in decisions)
    false_positive = sum(not expected and actual for expected, actual in decisions)
    false_negative = sum(expected and not actual for expected, actual in decisions)
    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    return {
        "answerability_precision": precision,
        "answerability_recall": recall,
        "answerability_f1": divide(2 * precision * recall, precision + recall),
    }


def _citation_metrics(
    pairs: Sequence[tuple[EvaluationCase, EvaluationPrediction]],
) -> dict[str, float]:
    correct = 0
    cited = 0
    relevant = 0
    covered = 0
    answerable_count = 0
    for case, prediction in pairs:
        unique_citations = set(prediction.cited_sources)
        gold = set(case.relevant_sources)
        correct_here = len(unique_citations & gold)
        correct += correct_here
        cited += len(unique_citations)
        if case.answerable:
            answerable_count += 1
            relevant += len(gold)
            covered += bool(correct_here)
    return {
        "citation_precision": divide(correct, cited),
        "citation_recall": divide(correct, relevant),
        "citation_coverage": divide(covered, answerable_count),
    }


def _fact_coverage(pairs: Sequence[tuple[EvaluationCase, EvaluationPrediction]]) -> float:
    scores: list[float] = []
    for case, prediction in pairs:
        if not case.expected_facts:
            continue
        answer = normalize(prediction.answer_text)
        matches = sum(normalize(fact) in answer for fact in case.expected_facts)
        scores.append(matches / len(case.expected_facts))
    return mean(scores)


def _query_metrics(
    pairs: Sequence[tuple[EvaluationCase, EvaluationPrediction]],
) -> tuple[float, float]:
    scores: list[float] = []
    for case, prediction in pairs:
        if not case.required_terms:
            continue
        queries = [normalize(query) for query in prediction.effective_queries]
        retained = sum(
            any(normalize(term) in query for query in queries) for term in case.required_terms
        )
        scores.append(retained / len(case.required_terms))
    return mean(scores), mean([float(score < 1.0) for score in scores])
