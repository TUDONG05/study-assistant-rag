"""Streamlit workflow for dense grounded chat."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import streamlit as st

from src.chat import ChatError, GroundedAnswer, GroundedAnswerService
from src.clients import get_gemini_client, get_qdrant_client
from src.config import AppSettings
from src.ingestion.embeddings import GeminiEmbeddingProvider
from src.ingestion.models import DocumentRecord
from src.ingestion.versioning import is_current_pipeline_version
from src.retrieval import (
    ConversationTurn,
    DenseRetriever,
    RetrievalError,
    RetrievalRequest,
    RetrievalTrace,
)
from src.storage import DocumentStore, QdrantCollections, VectorStore, ensure_collections
from src.ui.session_state import (
    MESSAGES,
    REQUEST_COUNT,
    SELECTED_DOCUMENT_IDS,
    active_gemini_api_key,
    current_workspace_id,
)

_MAX_HISTORY_TURNS = 6
_MAX_HISTORY_CHARS = 6_000


def render(settings: AppSettings) -> None:
    st.subheader("Chat có trích dẫn")
    st.caption(f"Model: `{settings.chat_model}` · Retrieval: Dense baseline")
    workspace_id = current_workspace_id()

    try:
        client = get_qdrant_client(settings, workspace_id)
        collections = QdrantCollections(
            chunks=settings.chunks_collection,
            documents=settings.documents_collection,
        )
        ensure_collections(client, collections, embedding_dimension=settings.embedding_dimension)
        document_store = DocumentStore(client, collections.documents)
        vector_store = VectorStore(client, collections.chunks)
        records = document_store.list_active(workspace_id)
    except Exception:
        st.error("Không thể mở kho tài liệu lúc này.")
        return

    compatible = compatible_records(records)
    incompatible_count = len(records) - len(compatible)
    if incompatible_count:
        st.warning(
            f"{incompatible_count} tài liệu dùng định dạng embedding cũ. "
            "Hãy lập chỉ mục lại ở mục Tài liệu trước khi chat."
        )

    selected_ids = _render_document_scope(compatible)
    _render_history(st.session_state[MESSAGES])
    api_key = active_gemini_api_key()
    request_count = int(st.session_state[REQUEST_COUNT])
    disabled_reason = _disabled_reason(
        api_key=api_key,
        compatible_records=compatible,
        request_count=request_count,
        request_limit=settings.max_requests_per_session,
    )
    if disabled_reason:
        st.info(disabled_reason)

    question = st.chat_input("Đặt câu hỏi về tài liệu…", disabled=disabled_reason is not None)
    if not question or not api_key:
        return
    normalized = question.strip()
    if not normalized or len(normalized) > settings.max_question_chars:
        st.warning(f"Câu hỏi phải từ 1 đến {settings.max_question_chars:,} ký tự.")
        return

    history = conversation_history(st.session_state[MESSAGES])
    st.session_state[REQUEST_COUNT] = request_count + 1
    st.session_state[MESSAGES].append({"role": "user", "content": normalized})
    with st.chat_message("user"):
        st.markdown(normalized)

    with st.chat_message("assistant"), st.spinner("Đang tìm bằng chứng…"):
        try:
            gemini = get_gemini_client(api_key, settings.request_timeout_ms)
            retriever = DenseRetriever(
                document_store=document_store,
                vector_store=vector_store,
                embedding_provider=GeminiEmbeddingProvider(
                    gemini,
                    model=settings.embedding_model,
                    dimension=settings.embedding_dimension,
                    batch_size=settings.embedding_batch_size,
                    max_retries=settings.embedding_max_retries,
                ),
                top_k=settings.retrieval_top_k,
                score_threshold=settings.retrieval_score_threshold,
                max_question_chars=settings.max_question_chars,
            )
            retrieval = retriever.retrieve(
                RetrievalRequest(
                    question=normalized,
                    history=history,
                    workspace_id=workspace_id,
                    selected_document_ids=tuple(selected_ids),
                )
            )
            answer = GroundedAnswerService(gemini, model=settings.chat_model).answer(
                normalized,
                retrieval.chunks,
                history=history,
            )
            message = answer_message(answer, retrieval_trace=retrieval.trace)
            st.session_state[MESSAGES].append(message)
            st.markdown(answer.answer_markdown)
            _render_citations(message["citations"])
            _render_retrieval_trace(message["retrieval"])
        except (ChatError, RetrievalError) as exc:
            # Operational failures remain retryable and do not consume the session allowance.
            st.session_state[REQUEST_COUNT] = request_count
            message = {
                "role": "assistant",
                "content": str(exc),
                "citations": [],
                "refused": False,
                "error": True,
            }
            st.session_state[MESSAGES].append(message)
            st.error(message["content"])


def compatible_records(records: Sequence[DocumentRecord]) -> list[DocumentRecord]:
    return [record for record in records if is_current_pipeline_version(record.context_version)]


def reconcile_selected_document_ids(
    selected_ids: Sequence[str], records: Sequence[DocumentRecord]
) -> list[str]:
    allowed = {record.document_id for record in records}
    return [document_id for document_id in selected_ids if document_id in allowed]


def answer_message(
    answer: GroundedAnswer,
    *,
    retrieval_trace: RetrievalTrace | None = None,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": answer.answer_markdown,
        "citations": [citation.as_dict() for citation in answer.citations],
        "retrieval": retrieval_trace.as_dict() if retrieval_trace else None,
        "refused": answer.refused,
        "error": False,
    }


def conversation_history(
    messages: Sequence[dict[str, Any]],
    *,
    max_turns: int = _MAX_HISTORY_TURNS,
    max_chars: int = _MAX_HISTORY_CHARS,
) -> tuple[ConversationTurn, ...]:
    """Return recent non-error turns within deterministic prompt limits."""

    selected: list[ConversationTurn] = []
    remaining = max_chars
    for message in reversed(messages):
        if len(selected) >= max_turns or remaining <= 0:
            break
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content or message.get("error"):
            continue
        if len(content) > remaining:
            if not selected:
                selected.append(ConversationTurn(role=role, content=content[:remaining]))
            break
        selected.append(ConversationTurn(role=role, content=content))
        remaining -= len(content)
    return tuple(reversed(selected))


def _render_document_scope(records: Sequence[DocumentRecord]) -> list[str]:
    current = reconcile_selected_document_ids(st.session_state[SELECTED_DOCUMENT_IDS], records)
    if current != st.session_state[SELECTED_DOCUMENT_IDS]:
        st.session_state[SELECTED_DOCUMENT_IDS] = current
    labels = {record.document_id: record.file_name for record in records}
    st.multiselect(
        "Phạm vi tài liệu",
        options=list(labels),
        format_func=labels.__getitem__,
        key=SELECTED_DOCUMENT_IDS,
        help="Không chọn tài liệu nghĩa là tìm trong tất cả tài liệu tương thích.",
        disabled=not records,
    )
    return list(st.session_state[SELECTED_DOCUMENT_IDS])


def _disabled_reason(
    *,
    api_key: str | None,
    compatible_records: Sequence[DocumentRecord],
    request_count: int,
    request_limit: int,
) -> str | None:
    if not api_key:
        return "Nhập Gemini API key ở sidebar để bật chat."
    if not compatible_records:
        return "Hãy lập chỉ mục ít nhất một tài liệu tương thích trước khi đặt câu hỏi."
    if request_count >= request_limit:
        return "Phiên này đã đạt giới hạn câu hỏi. Hãy đặt lại phiên để tiếp tục."
    return None


def _render_history(messages: Sequence[dict[str, Any]]) -> None:
    for message in messages:
        with st.chat_message(str(message["role"])):
            if message.get("error"):
                st.error(str(message["content"]))
            else:
                st.markdown(str(message["content"]))
                _render_citations(message.get("citations", []))
                _render_retrieval_trace(message.get("retrieval"))


def _render_citations(citations: Sequence[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Nguồn tham khảo ({len(citations)})"):
        for citation in citations:
            st.markdown(f"**[{citation['source_id']}]**")
            st.text(
                f"{citation['file_name']} · {citation['location']} · "
                f"điểm truy xuất {float(citation['score']):.3f}"
            )
            supporting_quote = str(citation.get("supporting_quote", "")).strip()
            if supporting_quote:
                st.caption("Đoạn nguồn")
                st.text(supporting_quote)


def _render_retrieval_trace(trace: dict[str, Any] | None) -> None:
    if not trace:
        return
    strategy_id = str(trace.get("strategy_id", "unknown"))
    trace_id = str(trace.get("trace_id", "unknown"))[:8]
    candidate_count = int(trace.get("candidate_count", 0))
    history_turns = int(trace.get("history_turns", 0))
    with st.expander("Chi tiết truy xuất"):
        st.caption(
            f"Strategy: {strategy_id} · Trace: {trace_id} · "
            f"Candidates: {candidate_count} · History turns: {history_turns}"
        )
