from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from src.ingestion.indexer import delete_document
from src.ingestion.models import DocumentRecord
from src.storage.document_store import DocumentStore
from src.storage.qdrant_client import QdrantCollections, ensure_collections
from src.storage.vector_store import VectorStore


def _record() -> DocumentRecord:
    return DocumentRecord(
        workspace_id="workspace-a",
        document_id="6e7c8135-9c2d-47c7-92cd-0b2a7458a217",
        version_id="version-a",
        file_name="lesson.pdf",
        normalized_name="lesson.pdf",
        title="lesson",
        file_type="pdf",
        content_hash="hash-a",
        chunk_count=1,
        extracted_chars=100,
        context_version="context-v1",
        created_at="2026-09-08T00:00:00+00:00",
    )


def test_delete_keeps_registry_record_when_vector_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = QdrantClient(location=":memory:")
    collections = QdrantCollections()
    ensure_collections(client, collections, embedding_dimension=3)
    document_store = DocumentStore(client, collections.documents)
    vector_store = VectorStore(client, collections.chunks)
    record = _record()
    document_store.commit(record)

    def fail_delete(_workspace_id: str, _document_id: str) -> None:
        raise RuntimeError("vector cleanup failed")

    monkeypatch.setattr(vector_store, "delete_document", fail_delete)

    try:
        delete_document(
            workspace_id=record.workspace_id,
            document_id=record.document_id,
            document_store=document_store,
            vector_store=vector_store,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("delete_document must surface vector cleanup failures")

    assert document_store.find_by_name(record.workspace_id, record.normalized_name) == record
