"""Immutable contracts for reproducible offline RAG evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationTurn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    split: str
    question: str
    history: tuple[EvaluationTurn, ...]
    intent: str
    required_terms: tuple[str, ...]
    relevant_sources: tuple[str, ...]
    expected_facts: tuple[str, ...]
    answerable: bool
    categories: tuple[str, ...]
    document_file_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    schema_version: str
    dataset_id: str
    dataset_version: str
    corpus_id: str
    corpus_version: str
    context_version: str
    documents: tuple[str, ...]
    cases: tuple[EvaluationCase, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationPrediction:
    case_id: str
    retrieved_sources: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    effective_queries: tuple[str, ...] = ()
    answer_text: str = ""
    cited_sources: tuple[str, ...] = ()
    refused: bool = False
    retrieval_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    strategy_id: str
    top_k: int
    seed: int = 0
    temperature: float = 0.0
    answer_model: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    score_threshold: float | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    context_mode: str | None = None
    pipeline_version: str | None = None
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id must not be empty")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.embedding_dimension is not None and (
            isinstance(self.embedding_dimension, bool)
            or not isinstance(self.embedding_dimension, int)
            or self.embedding_dimension < 1
        ):
            raise ValueError("embedding_dimension must be positive")
        if self.chunk_size is not None and (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap is not None and (
            isinstance(self.chunk_overlap, bool)
            or not isinstance(self.chunk_overlap, int)
            or self.chunk_overlap < 0
        ):
            raise ValueError("chunk_overlap must not be negative")
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, int | float)
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if self.score_threshold is not None and (
            isinstance(self.score_threshold, bool)
            or not isinstance(self.score_threshold, int | float)
            or not math.isfinite(self.score_threshold)
            or not -1 <= self.score_threshold <= 1
        ):
            raise ValueError("score_threshold must be between -1 and 1")
        if self.context_mode is not None and self.context_mode not in {
            "deterministic",
            "gemini",
        }:
            raise ValueError("context_mode must be deterministic or gemini")
        if self.pipeline_version is not None and not self.pipeline_version.strip():
            raise ValueError("pipeline_version must not be empty")
        costs = (self.input_cost_per_million, self.output_cost_per_million)
        if any(
            isinstance(cost, bool)
            or not isinstance(cost, int | float)
            or not math.isfinite(cost)
            or cost < 0
            for cost in costs
        ):
            raise ValueError("token costs must not be negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    hit_rate_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    answerability_precision: float
    answerability_recall: float
    answerability_f1: float
    citation_precision: float
    citation_recall: float
    citation_coverage: float
    expected_fact_coverage: float
    required_term_retention: float
    query_drift_rate: float
    retrieval_latency_p50_ms: float
    retrieval_latency_p95_ms: float
    end_to_end_latency_p50_ms: float
    end_to_end_latency_p95_ms: float
    model_calls_total: int
    input_tokens_total: int
    output_tokens_total: int
    estimated_cost_usd_total: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: str
    dataset_id: str
    dataset_version: str
    corpus_id: str
    corpus_version: str
    config: EvaluationConfig
    evaluated_case_count: int
    metrics: EvaluationMetrics
    predictions: tuple[EvaluationPrediction, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
