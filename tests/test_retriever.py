from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from src.ingestion.models import DocumentRecord
from src.ingestion.versioning import make_pipeline_version
from src.retrieval import (
    ConversationTurn,
    DenseRetriever,
    HybridRetriever,
    RetrievalError,
    RetrievalRequest,
    Retriever,
)
from src.storage import DocumentStore, QdrantCollections, VectorStore, ensure_collections


class _QueryEmbedder:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.questions.append(question)
        return [1.0, 0.0, 0.0]


def _pipeline_version() -> str:
    return str(
        make_pipeline_version(
            context_mode="deterministic",
            context_model=None,
            embedding_model="gemini-embedding-2",
            embedding_dimension=3,
            chunk_size_chars=100,
            chunk_overlap_chars=10,
            context_prefix_chars=20,
            max_enriched_chunks=0,
        )
    )


def _record(
    workspace: str,
    *,
    name: str,
    version: str,
    context_version: str | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        workspace_id=workspace,
        document_id=str(uuid4()),
        version_id=version,
        file_name=f"{name}.pdf",
        normalized_name=f"{name}.pdf",
        title=name,
        file_type="pdf",
        content_hash=f"hash-{name}",
        chunk_count=1,
        extracted_chars=100,
        context_version=context_version or _pipeline_version(),
        created_at="2026-09-09T00:00:00+00:00",
    )


def _payload(record: DocumentRecord, *, text: str) -> dict[str, object]:
    return {
        "chunk_id": str(uuid4()),
        "workspace_id": record.workspace_id,
        "document_id": record.document_id,
        "version_id": record.version_id,
        "file_name": record.file_name,
        "document_title": record.title,
        "source": "Trang 1",
        "section": None,
        "page_number": 1,
        "slide_number": None,
        "original_text": text,
    }


def _setup() -> tuple[QdrantClient, QdrantCollections, DocumentStore, VectorStore, _QueryEmbedder]:
    client = QdrantClient(location=":memory:")
    collections = QdrantCollections()
    ensure_collections(client, collections, embedding_dimension=3)
    return (
        client,
        collections,
        DocumentStore(client, collections.documents),
        VectorStore(client, collections.chunks),
        _QueryEmbedder(),
    )


def _upsert_chunk(
    client: QdrantClient,
    collection: str,
    payload: dict[str, object],
    vector: list[float],
) -> None:
    client.upsert(
        collection_name=collection,
        points=[models.PointStruct(id=payload["chunk_id"], vector=vector, payload=payload)],
        wait=True,
    )


def _retriever(
    documents: DocumentStore,
    vectors: VectorStore,
    embedder: _QueryEmbedder,
) -> DenseRetriever:
    return DenseRetriever(
        document_store=documents,
        vector_store=vectors,
        embedding_provider=embedder,
        top_k=6,
        score_threshold=0.5,
        max_question_chars=100,
    )


def test_retrieval_excludes_stale_orphan_incompatible_and_other_workspace_chunks() -> None:
    client, collections, documents, vectors, embedder = _setup()
    active = _record("workspace-a", name="active", version="version-active")
    incompatible = _record(
        "workspace-a", name="old", version="version-old", context_version="ingestion-v1-old"
    )
    other = _record("workspace-b", name="other", version="version-other")
    for record in (active, incompatible, other):
        documents.commit(record)
    candidates = (
        (active, "allowed", [1.0, 0.0, 0.0]),
        (replace(active, version_id="version-stale"), "stale", [1.0, 0.0, 0.0]),
        (replace(active, version_id="version-orphan"), "orphan", [1.0, 0.0, 0.0]),
        (incompatible, "old", [1.0, 0.0, 0.0]),
        (other, "cross-tenant", [1.0, 0.0, 0.0]),
    )
    for record, text, vector in candidates:
        _upsert_chunk(client, collections.chunks, _payload(record, text=text), vector)

    result = _retriever(documents, vectors, embedder).retrieve(
        RetrievalRequest(question="Nội dung nào hợp lệ?", workspace_id="workspace-a")
    )

    assert [chunk.original_text for chunk in result.chunks] == ["allowed"]
    assert result.chunks[0].source_id == "S1"
    assert result.trace.strategy_id == "dense"
    assert result.trace.candidate_count == 1
    assert result.trace.trace_id
    assert embedder.questions == ["Nội dung nào hợp lệ?"]


def test_selected_documents_limit_scope_and_empty_selection_means_all() -> None:
    client, collections, documents, vectors, embedder = _setup()
    first = _record("workspace-a", name="first", version="version-first")
    second = _record("workspace-a", name="second", version="version-second")
    for record in (first, second):
        documents.commit(record)
        _upsert_chunk(
            client,
            collections.chunks,
            _payload(record, text=record.title),
            [1.0, 0.0, 0.0],
        )

    retriever = _retriever(documents, vectors, embedder)
    selected = retriever.retrieve(
        RetrievalRequest(
            question="Câu hỏi",
            workspace_id="workspace-a",
            selected_document_ids=(second.document_id,),
        )
    )
    all_chunks = retriever.retrieve(
        RetrievalRequest(question="Câu hỏi", workspace_id="workspace-a")
    )

    assert [chunk.document_id for chunk in selected.chunks] == [second.document_id]
    assert {chunk.document_id for chunk in all_chunks.chunks} == {
        first.document_id,
        second.document_id,
    }
    assert selected.trace.selected_document_ids == (second.document_id,)


def test_no_compatible_active_scope_skips_embedding() -> None:
    _client, _collections, documents, vectors, embedder = _setup()
    documents.commit(
        _record(
            "workspace-a",
            name="old",
            version="version-old",
            context_version="ingestion-v1-old",
        )
    )

    result = _retriever(documents, vectors, embedder).retrieve(
        RetrievalRequest(
            question="Câu hỏi",
            workspace_id="workspace-a",
            history=(ConversationTurn(role="user", content="Lượt trước"),),
        )
    )

    assert result.chunks == ()
    assert result.trace.history_turns == 1
    assert embedder.questions == []


def test_question_validation_and_malformed_payload_are_safe() -> None:
    client, collections, documents, vectors, embedder = _setup()
    active = _record("workspace-a", name="active", version="version-active")
    documents.commit(active)
    payload = _payload(active, text="valid")
    payload.pop("original_text")
    _upsert_chunk(client, collections.chunks, payload, [1.0, 0.0, 0.0])
    retriever = _retriever(documents, vectors, embedder)

    with pytest.raises(RetrievalError, match="không được để trống"):
        retriever.retrieve(RetrievalRequest(question="   ", workspace_id="workspace-a"))
    with pytest.raises(RetrievalError, match="Payload chunk"):
        retriever.retrieve(RetrievalRequest(question="Câu hỏi", workspace_id="workspace-a"))


def test_retrieval_rejects_chunk_from_unapproved_document_version_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _client, _collections, documents, vectors, embedder = _setup()
    active = _record("workspace-a", name="active", version="version-active")
    documents.commit(active)
    foreign = replace(active, document_id=str(uuid4()))
    monkeypatch.setattr(
        vectors,
        "query",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                payload=_payload(foreign, text="wrong document"),
                score=1.0,
            )
        ],
    )
    retriever = _retriever(documents, vectors, embedder)

    with pytest.raises(RetrievalError, match="ngoài phạm vi"):
        retriever.retrieve(RetrievalRequest(question="Câu hỏi", workspace_id="workspace-a"))


def test_dense_retriever_implements_shared_protocol_and_requires_workspace() -> None:
    _client, _collections, documents, vectors, embedder = _setup()
    retriever = _retriever(documents, vectors, embedder)

    assert isinstance(retriever, Retriever)
    with pytest.raises(RetrievalError, match="Workspace"):
        retriever.retrieve(RetrievalRequest(question="Câu hỏi", workspace_id="  "))


def test_hybrid_reranks_dense_candidates_using_lexical_evidence() -> None:
    client, collections, documents, vectors, embedder = _setup()
    dense_first = _record("workspace-a", name="dense-first", version="v1")
    lexical_match = _record("workspace-a", name="lexical-match", version="v2")
    for record in (dense_first, lexical_match):
        documents.commit(record)
    _upsert_chunk(
        client, collections.chunks, _payload(dense_first, text="generic study notes"), [1, 0, 0]
    )
    _upsert_chunk(
        client,
        collections.chunks,
        _payload(lexical_match, text="BM25 lexical retrieval evidence"),
        [0.9, 0.1, 0],
    )

    result = HybridRetriever(_retriever(documents, vectors, embedder), top_k=2).retrieve(
        RetrievalRequest(question="BM25 retrieval", workspace_id="workspace-a")
    )

    assert result.trace.strategy_id == "hybrid-dense-bm25"
    assert result.chunks[0].original_text == "BM25 lexical retrieval evidence"
    assert [chunk.source_id for chunk in result.chunks] == ["S1", "S2"]
