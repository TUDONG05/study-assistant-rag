"""Embedding adapters with batching, retries and response validation."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from google.genai import types


class EmbeddingError(RuntimeError):
    """Raised when an embedding response is incomplete or malformed."""


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str], *, title: str) -> list[list[float]]: ...


class GeminiEmbeddingProvider:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        dimension: int,
        batch_size: int,
        max_retries: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.sleep = sleep

    def embed_documents(self, texts: Sequence[str], *, title: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, title=title))
        if len(vectors) != len(texts):
            raise EmbeddingError("Số embedding trả về không khớp số chunk.")
        return vectors

    def _embed_batch(self, texts: Sequence[str], *, title: str) -> list[list[float]]:
        contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        title=title,
                        output_dimensionality=self.dimension,
                    ),
                )
                embeddings = response.embeddings or []
                vectors = [list(embedding.values or []) for embedding in embeddings]
                if len(vectors) != len(texts):
                    raise EmbeddingError("Gemini trả về thiếu embedding.")
                if any(len(vector) != self.dimension for vector in vectors):
                    raise EmbeddingError(f"Embedding không đúng dimension {self.dimension}.")
                return vectors
            except EmbeddingError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self.sleep(float(2**attempt))
        raise EmbeddingError("Không thể tạo embedding sau nhiều lần thử.") from last_error
