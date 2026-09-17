from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from src import retrieval
from src.chat import ChatError, Citation, GroundedAnswer
from src.evaluation import models as eval_models
from src.evaluation.runner import EvaluationRunError, run_evaluation
from src.ingestion.models import DocumentRecord

DOC = "Nhóm3_TTCSN.docx"


def _case(case_id: str, split: str) -> eval_models.EvaluationCase:
    return eval_models.EvaluationCase(
        case_id=case_id, split=split, question="12345678",
        history=(eval_models.EvaluationTurn(role="user", content="abcd"),),
        intent="Kiểm tra runner", required_terms=(),
        relevant_sources=(f"{DOC}::Đoạn 1",),
        expected_facts=("1234",), answerable=True, categories=("direct_fact",),
        document_file_names=(DOC,),
    )


def _dataset() -> eval_models.EvaluationDataset:
    return eval_models.EvaluationDataset(
        schema_version="1.0", dataset_id="runner-dataset", dataset_version="1.0.0",
        corpus_id="ttcsn", corpus_version="sha256:" + "a" * 64,
        context_version="ingestion-v3-evaluation", documents=(DOC,),
        cases=(_case("development-case", "development"), _case("holdout-case", "holdout")),
    )


def _record() -> DocumentRecord:
    return DocumentRecord(
        workspace_id="workspace-1", document_id="document-1", version_id="version-1",
        file_name=DOC, normalized_name="nhom3_ttcsn.docx", title="Nhóm 3 TTCSN",
        file_type="docx", content_hash="a" * 64, chunk_count=2, extracted_chars=12,
        context_version="ingestion-v3-evaluation",
        created_at="2026-09-13T00:00:00+00:00",
    )


def _chunk(chunk_id: str, source: str, text: str, rank: int) -> retrieval.RetrievedChunk:
    return retrieval.RetrievedChunk(
        source_id=f"S{rank}", rank=rank, score=1 / rank, chunk_id=chunk_id,
        workspace_id="workspace-1", document_id="document-1", version_id="version-1",
        file_name=DOC,
        document_title="Nhóm 3 TTCSN",
        source=source, section=None, page_number=None, slide_number=None, original_text=text,
    )


def _result(chunks: tuple[retrieval.RetrievedChunk, ...]) -> retrieval.RetrievalResult:
    return retrieval.RetrievalResult(
        chunks=chunks,
        trace=retrieval.RetrievalTrace(
            trace_id="trace-1", strategy_id="dense", question="12345678",
            selected_document_ids=("document-1",), history_turns=1,
            candidate_count=len(chunks),
        ),
    )


@dataclass
class _Retriever:
    result: retrieval.RetrievalResult | Exception
    requests: list[retrieval.RetrievalRequest] = field(default_factory=list)

    def retrieve(self, request: retrieval.RetrievalRequest) -> retrieval.RetrievalResult:
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@dataclass
class _AnswerProvider:
    result: GroundedAnswer | Exception
    calls: list[tuple[str, Sequence[object], Sequence[object]]] = field(default_factory=list)

    def answer(
        self,
        question: str,
        chunks: Sequence[retrieval.RetrievedChunk],
        *,
        history: Sequence[retrieval.ConversationTurn] = (),
    ) -> GroundedAnswer:
        self.calls.append((question, chunks, history))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _run(
    retriever: _Retriever,
    answer_provider: _AnswerProvider,
    *,
    split: str = "holdout",
    records: Sequence[DocumentRecord] | None = None,
    config: eval_models.EvaluationConfig | None = None,
) -> eval_models.EvaluationReport:
    return run_evaluation(
        _dataset(), split=split, workspace_id="workspace-1",
        records=(_record(),) if records is None else records,
        retriever=retriever, answer_provider=answer_provider,
        config=config or eval_models.EvaluationConfig(strategy_id="dense", top_k=2),
    )


def test_run_evaluation_selects_split_and_builds_complete_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _chunk("chunk-1", "Đoạn 1", "abcdefgh", 1)
    second = _chunk("chunk-2", "Bảng 1", "ijkl", 2)
    unknown = _chunk("not-retrieved", "Đoạn 9", "unused", 3)
    retriever = _Retriever(_result((first, second)))
    answer_provider = _AnswerProvider(
        GroundedAnswer(
            answer_markdown="12345678",
            citations=(Citation.from_chunk(second), Citation.from_chunk(unknown)),
            refused=False,
        )
    )
    ticks = iter((10.0, 10.025, 10.100))
    monkeypatch.setattr("src.evaluation.runner.perf_counter", ticks.__next__)

    report = _run(
        retriever, answer_provider,
        config=eval_models.EvaluationConfig(
            strategy_id="dense",
            top_k=2,
            input_cost_per_million=2.0,
            output_cost_per_million=3.0,
        ),
    )

    assert report.evaluated_case_count == 1
    prediction = report.predictions[0]
    assert prediction.case_id == "holdout-case"
    assert prediction.retrieved_sources == (f"{DOC}::Đoạn 1", f"{DOC}::Bảng 1")
    assert prediction.retrieved_chunk_ids == ("chunk-1", "chunk-2")
    assert prediction.cited_sources == (f"{DOC}::Bảng 1",)
    assert prediction.effective_queries == ("12345678",)
    assert prediction.answer_text == "12345678"
    assert (prediction.model_calls, prediction.input_tokens, prediction.output_tokens) == (2, 6, 2)
    assert prediction.estimated_cost_usd == pytest.approx(0.000018)
    assert prediction.retrieval_latency_ms == pytest.approx(25)
    assert prediction.end_to_end_latency_ms == pytest.approx(100)
    request = retriever.requests[0]
    assert request.selected_document_ids == ("document-1",)
    assert request.history == (retrieval.ConversationTurn(role="user", content="abcd"),)
    assert answer_provider.calls[0] == ("12345678", (first, second), request.history)


def test_run_evaluation_rejects_unknown_split_missing_or_mismatched_corpus() -> None:
    inert_retriever = _Retriever(_result(()))
    inert_answer = _AnswerProvider(GroundedAnswer.refusal("no evidence"))

    with pytest.raises(EvaluationRunError, match="split 'test'"):
        _run(inert_retriever, inert_answer, split="test", records=())
    with pytest.raises(EvaluationRunError, match="Nhóm3_TTCSN.docx"):
        _run(inert_retriever, inert_answer, records=())
    wrong_content = DocumentRecord(
        **{**_record().payload(), "content_hash": "b" * 64}
    )
    with pytest.raises(EvaluationRunError, match="khớp phiên bản"):
        _run(inert_retriever, inert_answer, records=(wrong_content,))
    assert inert_retriever.requests == []


@pytest.mark.parametrize(
    "error", [retrieval.RetrievalError("vector unavailable"), ChatError("model busy")]
)
def test_operational_errors_fail_the_benchmark(error: Exception) -> None:
    chunks = (_chunk("chunk-1", "Đoạn 1", "abcdefgh", 1),)
    retriever = _Retriever(
        error if isinstance(error, retrieval.RetrievalError) else _result(chunks)
    )
    answer_result = error if isinstance(error, ChatError) else GroundedAnswer.refusal("")
    answer_provider = _AnswerProvider(answer_result)
    with pytest.raises(EvaluationRunError, match=type(error).__name__):
        _run(retriever, answer_provider)


def test_unexpected_error_fails_without_leaking_exception_message() -> None:
    retriever = _Retriever(RuntimeError("secret endpoint and credential"))

    with pytest.raises(EvaluationRunError, match="RuntimeError") as exc_info:
        _run(retriever, _AnswerProvider(GroundedAnswer.refusal("")))
    assert "secret" not in str(exc_info.value)
