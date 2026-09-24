"""End-to-end evaluation runner over the public retrieval and chat contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from src.chat import ChatError, GroundedAnswer
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.models import (
    EvaluationCase,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationPrediction,
    EvaluationReport,
)
from src.ingestion.models import DocumentRecord
from src.retrieval import ConversationTurn, RetrievalError, RetrievalRequest, Retriever
from src.retrieval.models import RetrievedChunk


class EvaluationRunError(RuntimeError):
    """Raised when the benchmark cannot run against the requested corpus."""


class AnswerProvider(Protocol):
    def answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        history: Sequence[ConversationTurn] = (),
    ) -> GroundedAnswer: ...


def run_evaluation(
    dataset: EvaluationDataset,
    *,
    split: str,
    workspace_id: str,
    records: Sequence[DocumentRecord],
    retriever: Retriever,
    answer_provider: AnswerProvider,
    config: EvaluationConfig,
) -> EvaluationReport:
    """Run one strategy on one immutable dataset split and aggregate its metrics."""

    selected_dataset = _select_split(dataset, split)
    records_by_name = {record.file_name: record for record in records}
    _validate_benchmark_corpus(selected_dataset, records_by_name)

    predictions = [
        _run_case(
            case,
            workspace_id=workspace_id,
            records_by_name=records_by_name,
            retriever=retriever,
            answer_provider=answer_provider,
            config=config,
        )
        for case in selected_dataset.cases
    ]
    return evaluate_predictions(selected_dataset, predictions, config)


def _validate_benchmark_corpus(
    dataset: EvaluationDataset, records_by_name: dict[str, DocumentRecord]
) -> None:
    """Reject reports whose labels cannot be tied to the active indexed corpus."""

    missing = sorted(set(dataset.documents) - records_by_name.keys())
    if missing:
        raise EvaluationRunError(
            "Corpus chưa được lập chỉ mục trong workspace: " + ", ".join(missing)
        )
    if len(dataset.documents) != 1:
        raise EvaluationRunError("Schema benchmark hiện chỉ hỗ trợ một tài liệu có hash xác thực.")
    record = records_by_name[dataset.documents[0]]
    expected_hash = dataset.corpus_version.removeprefix("sha256:")
    if record.content_hash != expected_hash:
        raise EvaluationRunError("Corpus đang active không khớp phiên bản đã gán nhãn.")
    if record.context_version != dataset.context_version:
        raise EvaluationRunError("Pipeline ingestion đang active không khớp phiên bản đã gán nhãn.")


def _select_split(dataset: EvaluationDataset, split: str) -> EvaluationDataset:
    cases = tuple(case for case in dataset.cases if case.split == split)
    if not cases:
        raise EvaluationRunError(f"Dataset không có ca thuộc split {split!r}.")
    return replace(dataset, cases=cases)


def _run_case(
    case: EvaluationCase,
    *,
    workspace_id: str,
    records_by_name: dict[str, DocumentRecord],
    retriever: Retriever,
    answer_provider: AnswerProvider,
    config: EvaluationConfig,
) -> EvaluationPrediction:
    history = tuple(ConversationTurn(turn.role, turn.content) for turn in case.history)
    document_ids = tuple(records_by_name[name].document_id for name in case.document_file_names)
    started = perf_counter()
    retrieval_finished = started
    try:
        retrieval = retriever.retrieve(
            RetrievalRequest(
                question=case.question,
                workspace_id=workspace_id,
                selected_document_ids=document_ids,
                history=history,
            )
        )
        retrieval_finished = perf_counter()
        answer = answer_provider.answer(case.question, retrieval.chunks, history=history)
        finished = perf_counter()
    except (RetrievalError, ChatError) as exc:
        raise EvaluationRunError(
            f"Không thể hoàn tất benchmark ở ca {case.case_id!r}: {type(exc).__name__}."
        ) from exc
    except Exception as exc:
        raise EvaluationRunError(
            f"Không thể hoàn tất benchmark ở ca {case.case_id!r}: {type(exc).__name__}."
        ) from exc

    sources = tuple(_source_ref(chunk) for chunk in retrieval.chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in retrieval.chunks}
    cited_sources = tuple(
        _source_ref(chunks_by_id[citation.chunk_id])
        for citation in answer.citations
        if citation.chunk_id in chunks_by_id
    )
    input_chars = len(case.question) + sum(len(turn.content) for turn in history)
    input_chars += sum(len(chunk.original_text) for chunk in retrieval.chunks)
    input_tokens = _estimated_tokens(input_chars)
    output_tokens = _estimated_tokens(len(answer.answer_markdown))
    estimated_cost = (
        input_tokens * config.input_cost_per_million
        + output_tokens * config.output_cost_per_million
    ) / 1_000_000
    return EvaluationPrediction(
        case_id=case.case_id,
        retrieved_sources=sources,
        retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in retrieval.chunks),
        effective_queries=(case.question,),
        answer_text=answer.answer_markdown,
        cited_sources=cited_sources,
        refused=answer.refused,
        retrieval_latency_ms=(retrieval_finished - started) * 1_000,
        end_to_end_latency_ms=(finished - started) * 1_000,
        model_calls=1 + int(bool(retrieval.chunks)),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
    )


def _source_ref(chunk: RetrievedChunk) -> str:
    return f"{chunk.file_name}::{chunk.source}"


def _estimated_tokens(character_count: int) -> int:
    """Use a documented deterministic estimate when provider usage is unavailable."""

    return math.ceil(character_count / 4) if character_count else 0
