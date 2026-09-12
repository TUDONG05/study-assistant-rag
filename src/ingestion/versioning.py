"""Stable ingestion-pipeline identity used by deduplication and replacement."""

from __future__ import annotations

import hashlib

PIPELINE_VERSION = "ingestion-v2"


def is_current_pipeline_version(value: str) -> bool:
    """Return whether an indexed record uses the current representation contract."""

    return value.startswith(f"{PIPELINE_VERSION}-")


def make_pipeline_version(
    *,
    context_mode: str,
    context_model: str | None,
    embedding_model: str,
    embedding_dimension: int,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    context_prefix_chars: int,
    max_enriched_chunks: int,
) -> str:
    """Fingerprint every setting that changes stored retrieval text or vectors."""

    # Chỉ đưa cấu hình ảnh hưởng truy xuất vào đây; cấu hình UI không được kích hoạt reindex.
    settings = (
        context_mode,
        context_model or "none",
        embedding_model,
        str(embedding_dimension),
        str(chunk_size_chars),
        str(chunk_overlap_chars),
        str(context_prefix_chars),
        str(max_enriched_chunks),
    )
    # Digest độ dài cố định giúp ID gọn nhưng vẫn đổi khi bất kỳ cấu hình nào thay đổi.
    digest = hashlib.sha256("\0".join(settings).encode()).hexdigest()[:16]
    return f"{PIPELINE_VERSION}-{digest}"
