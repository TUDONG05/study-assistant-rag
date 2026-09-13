from __future__ import annotations

from typing import Any

from src.chat import Citation, GroundedAnswer
from src.ingestion.models import DocumentRecord
from src.ingestion.versioning import PIPELINE_VERSION
from src.retrieval import RetrievalTrace
from src.ui.chat_view import (
    _disabled_reason,
    answer_message,
    compatible_records,
    conversation_history,
    reconcile_selected_document_ids,
)


def _record(document_id: str, context_version: str) -> DocumentRecord:
    return DocumentRecord(
        workspace_id="workspace-a",
        document_id=document_id,
        version_id=f"version-{document_id}",
        file_name=f"{document_id}.pdf",
        normalized_name=f"{document_id}.pdf",
        title=document_id,
        file_type="pdf",
        content_hash=f"hash-{document_id}",
        chunk_count=1,
        extracted_chars=10,
        context_version=context_version,
        created_at="2026-09-09T00:00:00+00:00",
    )


def test_document_scope_excludes_old_pipeline_and_stale_selection() -> None:
    current = _record("current", f"{PIPELINE_VERSION}-abc")
    old = _record("old", "ingestion-v1-abc")

    compatible = compatible_records([current, old])
    selected = reconcile_selected_document_ids(["old", "missing", "current"], compatible)

    assert compatible == [current]
    assert selected == ["current"]


def test_disabled_reason_prioritizes_key_documents_then_quota() -> None:
    current = _record("current", f"{PIPELINE_VERSION}-abc")

    missing_key = _disabled_reason(
        api_key=None, compatible_records=[], request_count=0, request_limit=40
    )
    missing_documents = _disabled_reason(
        api_key="key", compatible_records=[], request_count=0, request_limit=40
    )
    exhausted = _disabled_reason(
        api_key="key", compatible_records=[current], request_count=40, request_limit=40
    )
    assert missing_key and missing_key.startswith("Nhập Gemini")
    assert missing_documents and missing_documents.startswith("Hãy lập chỉ mục")
    assert exhausted and exhausted.startswith("Phiên này")
    assert (
        _disabled_reason(
            api_key="key", compatible_records=[current], request_count=0, request_limit=40
        )
        is None
    )


def test_answer_message_is_a_serializable_citation_snapshot() -> None:
    citation = Citation(
        source_id="S1",
        chunk_id="chunk-a",
        document_id="document-a",
        version_id="version-a",
        file_name="lesson.pdf",
        title="Lesson",
        location="Trang 2",
        supporting_quote="Nội dung nguồn chính xác.",
        score=0.75,
    )
    trace = RetrievalTrace(
        trace_id="trace-123",
        strategy_id="dense",
        question="Câu hỏi",
        selected_document_ids=("document-a",),
        history_turns=2,
        candidate_count=1,
    )

    message = answer_message(
        GroundedAnswer(
            answer_markdown="Nội dung. [S1]",
            citations=(citation,),
            refused=False,
        ),
        retrieval_trace=trace,
    )

    assert message["content"] == "Nội dung. [S1]"
    assert message["citations"] == [citation.as_dict()]
    assert message["retrieval"] == trace.as_dict()
    assert not message["refused"]
    assert not message["error"]


def test_conversation_history_is_bounded_and_skips_operational_errors() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "cũ"},
        {"role": "assistant", "content": "lỗi provider", "error": True},
        {"role": "assistant", "content": "trả lời trước"},
        {"role": "user", "content": "câu mới rất dài"},
    ]

    history = conversation_history(messages, max_turns=2, max_chars=20)

    assert [(turn.role, turn.content) for turn in history] == [
        ("user", "câu mới rất dài"),
    ]
