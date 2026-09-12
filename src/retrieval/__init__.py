"""Stable retrieval contracts and dense baseline for grounded chat."""

from src.retrieval.models import (
    ConversationTurn,
    RetrievalError,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    RetrievedChunk,
)
from src.retrieval.protocol import Retriever
from src.retrieval.retriever import DenseRetriever

__all__ = [
    "ConversationTurn",
    "DenseRetriever",
    "RetrievedChunk",
    "RetrievalError",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalTrace",
    "Retriever",
]
