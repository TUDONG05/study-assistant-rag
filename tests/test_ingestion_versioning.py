from __future__ import annotations

from src.ingestion.versioning import (
    PIPELINE_VERSION,
    is_current_pipeline_version,
    make_pipeline_version,
)


def _version(
    *,
    context_mode: str = "deterministic",
    context_model: str | None = None,
    embedding_model: str = "embedding-model",
    embedding_dimension: int = 768,
    chunk_size_chars: int = 1_600,
    chunk_overlap_chars: int = 240,
    context_prefix_chars: int = 360,
    max_enriched_chunks: int = 0,
) -> str:
    return make_pipeline_version(
        context_mode=context_mode,
        context_model=context_model,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        context_prefix_chars=context_prefix_chars,
        max_enriched_chunks=max_enriched_chunks,
    )


def test_pipeline_version_is_stable_for_same_settings() -> None:
    assert _version() == _version()
    assert _version().startswith(f"{PIPELINE_VERSION}-")


def test_pipeline_version_changes_with_indexed_representation() -> None:
    variants = [
        _version(context_mode="gemini"),
        _version(context_model="chat-model"),
        _version(embedding_model="new-embedding-model"),
        _version(embedding_dimension=1_536),
        _version(chunk_size_chars=800),
        _version(chunk_overlap_chars=120),
        _version(context_prefix_chars=180),
        _version(max_enriched_chunks=40),
    ]

    assert all(variant != _version() for variant in variants)


def test_only_current_pipeline_fingerprints_are_compatible() -> None:
    assert is_current_pipeline_version(_version())
    assert not is_current_pipeline_version("ingestion-v1-deadbeef")
    assert not is_current_pipeline_version("ingestion-v20-deadbeef")
