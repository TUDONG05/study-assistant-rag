from __future__ import annotations

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient, models

from src.ingestion.contextualizer import DeterministicContextualizer
from src.ingestion.indexer import DocumentIndexer, delete_document
from src.ingestion.models import ContextualChunk, UploadPayload
from src.storage.document_store import DocumentStore
from src.storage.qdrant_client import QdrantCollections, ensure_collections
from src.storage.vector_store import VectorStore
from tests.ingestion_fixtures import make_docx_bytes

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class _EmbeddingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: Sequence[str], *, title: str) -> list[list[float]]:
        del title
        self.calls += 1
        return [[float(index + 1), 0.5, 0.25] for index, _text in enumerate(texts)]


class _FailingAfterStageVectorStore(VectorStore):
    def stage(
        self,
        chunks: Sequence[ContextualChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        super().stage(chunks, vectors)
        raise RuntimeError("staging interrupted")


def _upload(text: str, *, file_name: str = "lesson.docx") -> UploadPayload:
    return UploadPayload(file_name, make_docx_bytes(first_text=text), DOCX_MIME)


def _pipeline(
    *,
    fail_after_stage: bool = False,
) -> tuple[
    QdrantClient,
    QdrantCollections,
    DocumentStore,
    VectorStore,
    _EmbeddingProvider,
    DocumentIndexer,
]:
    client = QdrantClient(location=":memory:")
    collections = QdrantCollections()
    ensure_collections(client, collections, embedding_dimension=3)
    documents = DocumentStore(client, collections.documents)
    vector_type = _FailingAfterStageVectorStore if fail_after_stage else VectorStore
    vectors = vector_type(client, collections.chunks, batch_size=2)
    embeddings = _EmbeddingProvider()
    indexer = DocumentIndexer(
        client=client,
        collections=collections,
        document_store=documents,
        vector_store=vectors,
        embedding_provider=embeddings,
        contextualizer=DeterministicContextualizer(max_prefix_chars=200),
        embedding_dimension=3,
        max_upload_bytes=2_000_000,
        max_zip_entries=1_000,
        max_zip_uncompressed_bytes=10_000_000,
        max_pages=20,
        max_extracted_chars=10_000,
        chunk_size_chars=80,
        chunk_overlap_chars=10,
    )
    return client, collections, documents, vectors, embeddings, indexer


def test_index_success_and_duplicate_short_circuit() -> None:
    _client, _collections, documents, vectors, embeddings, indexer = _pipeline()
    upload = _upload("Nội dung ổn định đủ dài để lập chỉ mục và kiểm tra trùng lặp.")

    first = indexer.index(upload, workspace_id="workspace-a")
    duplicate = indexer.index(upload, workspace_id="workspace-a")

    assert not first.deduplicated
    assert duplicate.deduplicated
    assert duplicate.document == first.document
    assert embeddings.calls == 1
    assert documents.list_active("workspace-a") == [first.document]
    assert vectors.count_version(
        "workspace-a", first.document.document_id, first.document.version_id
    ) == first.document.chunk_count


def test_replacement_activates_new_version_and_cleans_old_chunks() -> None:
    _client, _collections, documents, vectors, _embeddings, indexer = _pipeline()
    first = indexer.index(
        _upload("Phiên bản đầu tiên đủ dài để tạo dữ liệu chỉ mục."),
        workspace_id="workspace-a",
    )
    replacement = indexer.index(
        _upload("Phiên bản thứ hai có nội dung hoàn toàn khác để thay thế."),
        workspace_id="workspace-a",
    )

    assert replacement.document.version_id != first.document.version_id
    assert documents.list_active("workspace-a") == [replacement.document]
    assert vectors.count_version(
        "workspace-a", first.document.document_id, first.document.version_id
    ) == 0
    assert vectors.count_version(
        "workspace-a", replacement.document.document_id, replacement.document.version_id
    ) == replacement.document.chunk_count


def test_failed_staging_rolls_back_chunks_and_registry() -> None:
    client, collections, documents, _vectors, _embeddings, indexer = _pipeline(
        fail_after_stage=True
    )

    with pytest.raises(RuntimeError, match="staging interrupted"):
        indexer.index(
            _upload("Nội dung đủ dài để staging ghi vector trước khi mô phỏng lỗi."),
            workspace_id="workspace-a",
        )

    assert documents.list_active("workspace-a") == []
    assert client.count(collection_name=collections.chunks, exact=True).count == 0


def test_deduplication_and_records_are_workspace_scoped() -> None:
    _client, _collections, documents, _vectors, embeddings, indexer = _pipeline()
    upload = _upload("Một tài liệu được phép xuất hiện trong hai workspace độc lập.")

    first = indexer.index(upload, workspace_id="workspace-a")
    second = indexer.index(upload, workspace_id="workspace-b")

    assert not first.deduplicated and not second.deduplicated
    assert first.document.document_id != second.document.document_id
    assert documents.list_active("workspace-a") == [first.document]
    assert documents.list_active("workspace-b") == [second.document]
    assert embeddings.calls == 2


def test_delete_removes_vectors_and_registry() -> None:
    _client, _collections, documents, vectors, _embeddings, indexer = _pipeline()
    result = indexer.index(
        _upload("Nội dung đủ dài để kiểm tra xóa tài liệu thành công."),
        workspace_id="workspace-a",
    )

    delete_document(
        workspace_id="workspace-a",
        document_id=result.document.document_id,
        document_store=documents,
        vector_store=vectors,
    )

    assert documents.list_active("workspace-a") == []
    assert vectors.count_version(
        "workspace-a", result.document.document_id, result.document.version_id
    ) == 0


def test_existing_collection_dimension_mismatch_fails_early() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="wrong_chunks",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )

    with pytest.raises(ValueError, match="yêu cầu 3"):
        ensure_collections(
            client,
            QdrantCollections(chunks="wrong_chunks", documents="documents"),
            embedding_dimension=3,
        )


def test_existing_collection_distance_mismatch_fails_early() -> None:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="wrong_chunks",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT),
    )

    with pytest.raises(ValueError, match="Cosine"):
        ensure_collections(
            client,
            QdrantCollections(chunks="wrong_chunks", documents="documents"),
            embedding_dimension=3,
        )
