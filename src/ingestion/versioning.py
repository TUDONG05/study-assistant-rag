"""Stable ingestion-pipeline identity used by deduplication and replacement."""

from __future__ import annotations

import hashlib

PIPELINE_VERSION = "ingestion-v1"


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
    digest = hashlib.sha256("\0".join(settings).encode()).hexdigest()[:16]
    return f"{PIPELINE_VERSION}-{digest}"
