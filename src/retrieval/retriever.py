"""Active-version dense retrieval orchestration."""

from __future__ import annotations

from typing import Protocol
from uuid import uuid4

from src.ingestion.embeddings import EmbeddingError
from src.ingestion.versioning import is_current_pipeline_version
from src.retrieval.models import (
    RetrievalError,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    RetrievedChunk,
)
from src.storage.document_store import DocumentStore
from src.storage.vector_store import VectorStore


class QueryEmbeddingProvider(Protocol):
    def embed_query(self, question: str) -> list[float]: ...


class DenseRetriever:
    strategy_id = "dense"

    def __init__(
        self,
        *,
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_provider: QueryEmbeddingProvider,
        top_k: int,
        score_threshold: float,
        max_question_chars: int,
    ) -> None:
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_question_chars = max_question_chars

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        normalized = request.question.strip()
        if not normalized:
            raise RetrievalError("Câu hỏi không được để trống.")
        if len(normalized) > self.max_question_chars:
            raise RetrievalError(f"Câu hỏi vượt quá giới hạn {self.max_question_chars:,} ký tự.")
        workspace_id = request.workspace_id.strip()
        if not workspace_id:
            raise RetrievalError("Workspace không được để trống.")

        records = [
            record
            for record in self.document_store.list_active(workspace_id)
            if is_current_pipeline_version(record.context_version)
        ]
        selected = {value.strip() for value in request.selected_document_ids if value.strip()}
        if selected:
            records = [record for record in records if record.document_id in selected]
        if not records:
            return self._result(request, normalized, ())

        try:
            vector = self.embedding_provider.embed_query(normalized)
            points = self.vector_store.query(
                vector,
                workspace_id=workspace_id,
                version_ids=[record.version_id for record in records],
                document_ids=[record.document_id for record in records],
                limit=self.top_k,
                score_threshold=self.score_threshold,
            )
        except EmbeddingError as exc:
            raise RetrievalError("Không thể tạo embedding cho câu hỏi.") from exc
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("Không thể truy xuất tài liệu lúc này.") from exc

        allowed_documents = {(record.document_id, record.version_id) for record in records}
        chunks: list[RetrievedChunk] = []
        for rank, point in enumerate(points, start=1):
            chunk = RetrievedChunk.from_payload(
                dict(point.payload or {}),
                rank=rank,
                score=float(point.score),
            )
            if (
                chunk.workspace_id != workspace_id
                or (chunk.document_id, chunk.version_id) not in allowed_documents
            ):
                raise RetrievalError("Kho vector trả về chunk ngoài phạm vi cho phép.")
            chunks.append(chunk)
        return self._result(request, normalized, tuple(chunks))

    def _result(
        self,
        request: RetrievalRequest,
        normalized_question: str,
        chunks: tuple[RetrievedChunk, ...],
    ) -> RetrievalResult:
        trace = RetrievalTrace(
            trace_id=uuid4().hex,
            strategy_id=self.strategy_id,
            question=normalized_question,
            selected_document_ids=tuple(
                dict.fromkeys(
                    value.strip() for value in request.selected_document_ids if value.strip()
                )
            ),
            history_turns=len(request.history),
            candidate_count=len(chunks),
            effective_queries=(normalized_question,),
        )
        return RetrievalResult(chunks=chunks, trace=trace)
