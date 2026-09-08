"""Transactional ingestion orchestration over non-transactional vector storage."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from qdrant_client import QdrantClient

from src.ingestion.chunking import chunk_document
from src.ingestion.contextualizer import CONTEXT_VERSION, Contextualizer
from src.ingestion.embeddings import EmbeddingProvider
from src.ingestion.models import (
    DocumentRecord,
    IndexResult,
    UploadPayload,
    make_document_id,
    make_version_id,
)
from src.ingestion.parsers import parse_document
from src.ingestion.validation import IngestionError, validate_upload
from src.storage.document_store import DocumentStore
from src.storage.qdrant_client import QdrantCollections, ensure_collections
from src.storage.vector_store import VectorStore

ProgressCallback = Callable[[str, float], None]


class DocumentIndexer:
    def __init__(
        self,
        *,
        client: QdrantClient,
        collections: QdrantCollections,
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        contextualizer: Contextualizer,
        embedding_dimension: int,
        max_upload_bytes: int,
        max_zip_entries: int,
        max_zip_uncompressed_bytes: int,
        max_pages: int,
        max_extracted_chars: int,
        chunk_size_chars: int,
        chunk_overlap_chars: int,
        context_version: str = CONTEXT_VERSION,
    ) -> None:
        self.client = client
        self.collections = collections
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.contextualizer = contextualizer
        self.embedding_dimension = embedding_dimension
        self.max_upload_bytes = max_upload_bytes
        self.max_zip_entries = max_zip_entries
        self.max_zip_uncompressed_bytes = max_zip_uncompressed_bytes
        self.max_pages = max_pages
        self.max_extracted_chars = max_extracted_chars
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.context_version = context_version

    def index(
        self,
        upload_payload: UploadPayload,
        *,
        workspace_id: str,
        on_progress: ProgressCallback | None = None,
    ) -> IndexResult:
        progress = on_progress or (lambda _message, _value: None)
        progress("Đang kiểm tra tệp…", 0.05)
        upload = validate_upload(
            upload_payload,
            max_upload_bytes=self.max_upload_bytes,
            max_zip_entries=self.max_zip_entries,
            max_zip_uncompressed_bytes=self.max_zip_uncompressed_bytes,
        )
        ensure_collections(
            self.client,
            self.collections,
            embedding_dimension=self.embedding_dimension,
        )
        duplicate = self.document_store.find_duplicate(
            workspace_id,
            upload.content_hash,
            self.context_version,
        )
        if duplicate:
            progress("Tài liệu đã được lập chỉ mục.", 1.0)
            return IndexResult(document=duplicate, deduplicated=True)

        document_id = make_document_id(workspace_id, upload.normalized_name)
        version_id = make_version_id(document_id, upload.content_hash, self.context_version)
        old_record = self.document_store.find_by_name(workspace_id, upload.normalized_name)
        staging_started = False
        committed = False

        try:
            progress("Đang trích xuất văn bản…", 0.15)
            parsed = parse_document(
                upload,
                max_pages=self.max_pages,
                max_extracted_chars=self.max_extracted_chars,
            )
            drafts = chunk_document(
                parsed,
                upload,
                workspace_id=workspace_id,
                document_id=document_id,
                version_id=version_id,
                max_chars=self.chunk_size_chars,
                overlap_chars=self.chunk_overlap_chars,
            )
            if not drafts:
                raise IngestionError(f"{upload.file_name}: không tạo được chunk hợp lệ.")

            progress("Đang bổ sung ngữ cảnh truy xuất…", 0.35)
            chunks = self.contextualizer.contextualize(drafts)
            progress("Đang tạo embedding…", 0.5)
            vectors = self.embedding_provider.embed_documents(
                [chunk.retrieval_text for chunk in chunks],
                title=upload.title,
            )
            if len(vectors) != len(chunks):
                raise IngestionError("Số embedding không khớp số chunk.")
            if any(len(vector) != self.embedding_dimension for vector in vectors):
                raise IngestionError("Embedding dimension không khớp cấu hình storage.")

            progress("Đang ghi vùng staging…", 0.75)
            staging_started = True
            self.vector_store.stage(chunks, vectors)
            stored_count = self.vector_store.count_version(
                workspace_id,
                document_id,
                version_id,
            )
            if stored_count != len(chunks):
                raise IngestionError(
                    f"Xác minh staging thất bại: cần {len(chunks)} chunk, nhận {stored_count}."
                )

            record = DocumentRecord.create(
                upload=upload,
                workspace_id=workspace_id,
                document_id=document_id,
                version_id=version_id,
                chunk_count=len(chunks),
                extracted_chars=parsed.extracted_chars,
                context_version=self.context_version,
            )
            progress("Đang kích hoạt phiên bản mới…", 0.9)
            self.document_store.commit(record)
            committed = True
            if old_record and old_record.version_id != version_id:
                try:
                    self.vector_store.delete_version(
                        workspace_id,
                        old_record.document_id,
                        old_record.version_id,
                    )
                except Exception:
                    progress("Phiên bản mới đã hoạt động nhưng chưa dọn được dữ liệu cũ.", 0.98)
            progress("Hoàn tất lập chỉ mục.", 1.0)
            return IndexResult(document=record, deduplicated=False)
        except Exception:
            if staging_started and not committed:
                with contextlib.suppress(Exception):
                    self.vector_store.delete_version(workspace_id, document_id, version_id)
            raise


def delete_document(
    *,
    workspace_id: str,
    document_id: str,
    document_store: DocumentStore,
    vector_store: VectorStore,
) -> None:
    # Keep the registry record active when vector cleanup fails so deletion can be retried safely.
    vector_store.delete_document(workspace_id, document_id)
    document_store.delete(workspace_id, document_id)
