from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import pytest

from src.ingestion.embeddings import EmbeddingError, GeminiEmbeddingProvider


class _EmbeddingModels:
    def __init__(self, *, dimension: int, failures: int = 0) -> None:
        self.dimension = dimension
        self.failures = failures
        self.calls = 0

    def embed_content(
        self,
        *,
        contents: Sequence[object],
        **_kwargs: object,
    ) -> SimpleNamespace:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary failure")
        embeddings = [SimpleNamespace(values=[0.1] * self.dimension) for _ in contents]
        return SimpleNamespace(embeddings=embeddings)


def _provider(
    models: _EmbeddingModels,
    *,
    dimension: int = 3,
    batch_size: int = 2,
    max_retries: int = 2,
    sleep: Callable[[float], None] | None = None,
) -> GeminiEmbeddingProvider:
    return GeminiEmbeddingProvider(
        SimpleNamespace(models=models),
        model="embedding-model",
        dimension=dimension,
        batch_size=batch_size,
        max_retries=max_retries,
        sleep=sleep or (lambda _seconds: None),
    )


def test_embeddings_are_batched_and_order_count_is_preserved() -> None:
    models = _EmbeddingModels(dimension=3)

    vectors = _provider(models).embed_documents(["a", "b", "c", "d", "e"], title="Lesson")

    assert models.calls == 3
    assert len(vectors) == 5
    assert all(len(vector) == 3 for vector in vectors)


def test_transient_embedding_failure_retries_with_backoff() -> None:
    models = _EmbeddingModels(dimension=3, failures=1)
    sleeps: list[float] = []

    vectors = _provider(models, sleep=sleeps.append).embed_documents(["a"], title="Lesson")

    assert len(vectors) == 1
    assert models.calls == 2
    assert sleeps == [1.0]


def test_embedding_dimension_mismatch_is_rejected_without_retry() -> None:
    models = _EmbeddingModels(dimension=2)

    with pytest.raises(EmbeddingError, match="dimension 3"):
        _provider(models, dimension=3).embed_documents(["a"], title="Lesson")

    assert models.calls == 1
