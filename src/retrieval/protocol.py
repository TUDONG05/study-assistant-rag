"""Common retrieval interface for dense baseline and Advanced RAG strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.retrieval.models import RetrievalRequest, RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    strategy_id: str

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...
