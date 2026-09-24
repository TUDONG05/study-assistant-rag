from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from src.ingestion.embeddings import EmbeddingError, GeminiEmbeddingProvider


class _EmbeddingModels:
    def __init__(self, *, dimension: int, failures: int = 0) -> None:
        self.dimension = dimension
        self.failures = failures
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    def embed_content(
        self,
        *,
        contents: Sequence[object],
        **kwargs: object,
    ) -> SimpleNamespace:
        self.calls += 1
        self.requests.append({"contents": contents, **kwargs})
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
    first_content = models.requests[0]["contents"][0]  # type: ignore[index]
    assert first_content.parts[0].text == "title: Lesson | text: a"
    config: Any = models.requests[0]["config"]
    assert config.output_dimensionality == 3
    assert config.task_type is None
    assert config.title is None


def test_query_uses_question_answering_format() -> None:
    models = _EmbeddingModels(dimension=3)

    vector = _provider(models).embed_query("  Khái niệm RAG là gì?  ")

    assert len(vector) == 3
    content = models.requests[0]["contents"][0]  # type: ignore[index]
    assert content.parts[0].text == ("task: question answering | query: Khái niệm RAG là gì?")


def test_blank_query_is_rejected_without_api_call() -> None:
    models = _EmbeddingModels(dimension=3)

    with pytest.raises(EmbeddingError, match="không được để trống"):
        _provider(models).embed_query("   ")

    assert models.calls == 0


def test_transient_embedding_failure_retries_with_backoff() -> None:
    models = _EmbeddingModels(dimension=3, failures=1)
    sleeps: list[float] = []

    vectors = _provider(models, sleep=sleeps.append).embed_documents(["a"], title="Lesson")

    assert len(vectors) == 1
    assert models.calls == 2
    assert sleeps == [1.0]


def test_provider_retry_delay_overrides_short_local_backoff() -> None:
    models = _EmbeddingModels(dimension=3, failures=1)
    sleeps: list[float] = []

    original_embed = models.embed_content

    def quota_limited(*, contents: Sequence[object], **kwargs: object) -> SimpleNamespace:
        if models.calls == 0:
            models.calls += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED: retryDelay': '37.5s'")
        return original_embed(contents=contents, **kwargs)

    models.embed_content = quota_limited  # type: ignore[method-assign]

    vectors = _provider(models, sleep=sleeps.append).embed_documents(["a"], title="Lesson")

    assert len(vectors) == 1
    assert sleeps == [37.5]


def test_provider_retry_delay_is_capped() -> None:
    models = _EmbeddingModels(dimension=3, failures=1)
    sleeps: list[float] = []

    original_embed = models.embed_content

    def quota_limited(*, contents: Sequence[object], **kwargs: object) -> SimpleNamespace:
        if models.calls == 0:
            models.calls += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED: retryDelay: 999999999999999999s")
        return original_embed(contents=contents, **kwargs)

    models.embed_content = quota_limited  # type: ignore[method-assign]

    _provider(models, sleep=sleeps.append).embed_documents(["a"], title="Lesson")

    assert sleeps == [60.0]


def test_embedding_dimension_mismatch_is_rejected_without_retry() -> None:
    models = _EmbeddingModels(dimension=2)

    with pytest.raises(EmbeddingError, match="dimension 3"):
        _provider(models, dimension=3).embed_documents(["a"], title="Lesson")

    assert models.calls == 1
