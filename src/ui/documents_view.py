"""Upload, index and manage workspace-scoped study documents."""

from __future__ import annotations

import streamlit as st

from src.clients import get_gemini_client, get_qdrant_client
from src.config import AppSettings
from src.ingestion.contextualizer import (
    DeterministicContextualizer,
    GeminiContextualizer,
)
from src.ingestion.embeddings import GeminiEmbeddingProvider
from src.ingestion.indexer import DocumentIndexer, delete_document
from src.ingestion.models import UploadPayload
from src.ingestion.validation import IngestionError
from src.ingestion.versioning import make_pipeline_version
from src.storage import DocumentStore, QdrantCollections, VectorStore, ensure_collections
from src.ui.session_state import active_gemini_api_key, current_workspace_id


def render(settings: AppSettings) -> None:
    st.subheader("Tài liệu")
    st.write("Tải và quản lý tài liệu dùng làm nguồn kiến thức cho trợ lý.")
    workspace_id = current_workspace_id()

    try:
        client = get_qdrant_client(settings, workspace_id)
        collections = QdrantCollections(
            chunks=settings.chunks_collection,
            documents=settings.documents_collection,
        )
        ensure_collections(
            client,
            collections,
            embedding_dimension=settings.embedding_dimension,
        )
        document_store = DocumentStore(client, collections.documents)
        vector_store = VectorStore(client, collections.chunks)
    except Exception as exc:
        st.error(f"Không thể khởi tạo kho tài liệu: {exc}")
        return

    uploaded_files = st.file_uploader(
        "PDF, DOCX hoặc PPTX",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        help=f"Tối đa {settings.max_upload_mb} MB mỗi tệp.",
    )
    enrich_context = st.toggle(
        "Tăng cường contextual retrieval bằng Gemini",
        value=False,
        help=(
            "Tạo context riêng cho tối đa "
            f"{settings.max_llm_context_chunks} chunk; chậm và tốn thêm request. "
            "Khi tắt, hệ thống vẫn tạo context xác định từ metadata và chunk lân cận."
        ),
    )
    api_key = active_gemini_api_key()
    index_clicked = st.button(
        "Lập chỉ mục tài liệu",
        type="primary",
        disabled=not uploaded_files or not api_key,
    )
    if uploaded_files and not api_key:
        st.info("Nhập Gemini API key ở sidebar để tạo embedding.")

    if index_clicked and api_key:
        gemini_client = get_gemini_client(api_key, settings.request_timeout_ms)
        pipeline_version = make_pipeline_version(
            context_mode="gemini" if enrich_context else "deterministic",
            context_model=settings.chat_model if enrich_context else None,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            chunk_size_chars=settings.chunk_size_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
            context_prefix_chars=settings.context_prefix_chars,
            max_enriched_chunks=settings.max_llm_context_chunks if enrich_context else 0,
        )
        contextualizer = (
            GeminiContextualizer(
                gemini_client,
                model=settings.chat_model,
                max_prefix_chars=settings.context_prefix_chars,
                max_enriched_chunks=settings.max_llm_context_chunks,
                context_version=pipeline_version,
            )
            if enrich_context
            else DeterministicContextualizer(
                max_prefix_chars=settings.context_prefix_chars,
                context_version=pipeline_version,
            )
        )
        indexer = DocumentIndexer(
            client=client,
            collections=collections,
            document_store=document_store,
            vector_store=vector_store,
            embedding_provider=GeminiEmbeddingProvider(
                gemini_client,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                batch_size=settings.embedding_batch_size,
                max_retries=settings.embedding_max_retries,
            ),
            contextualizer=contextualizer,
            embedding_dimension=settings.embedding_dimension,
            max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
            max_zip_entries=settings.max_zip_entries,
            max_zip_uncompressed_bytes=settings.max_zip_uncompressed_mb * 1024 * 1024,
            max_pages=settings.max_document_pages,
            max_extracted_chars=settings.max_extracted_chars,
            chunk_size_chars=settings.chunk_size_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
            context_version=pipeline_version,
        )
        for uploaded_file in uploaded_files:
            _index_one(indexer, uploaded_file, workspace_id)

    st.divider()
    st.markdown("#### Đã lập chỉ mục")
    records = document_store.list_active(workspace_id)
    if not records:
        st.caption("Workspace này chưa có tài liệu.")
        return

    for record in records:
        info_column, action_column = st.columns([5, 1])
        with info_column:
            st.markdown(f"**{record.file_name}**")
            st.caption(
                f"{record.file_type.upper()} · {record.chunk_count} chunks · "
                f"{record.extracted_chars:,} ký tự · {record.context_version}"
            )
        with action_column:
            if st.button("Xóa", key=f"delete-{record.document_id}", use_container_width=True):
                try:
                    delete_document(
                        workspace_id=workspace_id,
                        document_id=record.document_id,
                        document_store=document_store,
                        vector_store=vector_store,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể xóa {record.file_name}: {exc}")


def _index_one(indexer: DocumentIndexer, uploaded_file: object, workspace_id: str) -> None:
    file_name = str(getattr(uploaded_file, "name", "document"))
    with st.status(f"Đang xử lý {file_name}", expanded=True) as status:
        progress_bar = st.progress(0.0)
        message = st.empty()

        def update_progress(label: str, value: float) -> None:
            message.write(label)
            progress_bar.progress(value)

        try:
            payload = UploadPayload(
                file_name=file_name,
                data=uploaded_file.getvalue(),  # type: ignore[attr-defined]
                mime_type=str(getattr(uploaded_file, "type", "")) or None,
            )
            result = indexer.index(
                payload,
                workspace_id=workspace_id,
                on_progress=update_progress,
            )
            if result.deduplicated:
                status.update(
                    label=f"{file_name}: đã tồn tại, không tạo embedding lại.", state="complete"
                )
            else:
                status.update(
                    label=f"{file_name}: đã lập chỉ mục {result.document.chunk_count} chunks.",
                    state="complete",
                )
        except IngestionError as exc:
            status.update(label=str(exc), state="error")
        except Exception:
            status.update(
                label=f"{file_name}: xử lý thất bại. Kiểm tra API key/kết nối rồi thử lại.",
                state="error",
            )
