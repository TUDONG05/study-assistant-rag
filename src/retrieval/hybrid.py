from __future__ import annotations

import re
from dataclasses import replace

from src.retrieval.models import RetrievalRequest, RetrievalResult
from src.retrieval.protocol import Retriever

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


class HybridRetriever:
    """Fuse dense rank with lexical rank over the same trusted candidate set."""

    strategy_id = "hybrid-dense-bm25"

    def __init__(self, dense_retriever: Retriever, *, top_k: int, rrf_k: int = 60) -> None:
        self.dense_retriever = dense_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        result = self.dense_retriever.retrieve(request)
        terms = {term.casefold() for term in _TOKEN.findall(request.question)}
        ranked = sorted(
            enumerate(result.chunks),
            key=lambda item: (-self._score(item[0], item[1].original_text, terms), item[0]),
        )[: self.top_k]
        chunks = tuple(
            replace(
                chunk,
                rank=rank,
                source_id=f"S{rank}",
                score=self._score(index, chunk.original_text, terms),
            )
            for rank, (index, chunk) in enumerate(ranked, start=1)
        )
        return RetrievalResult(
            chunks=chunks,
            trace=replace(
                result.trace, strategy_id=self.strategy_id, candidate_count=len(result.chunks)
            ),
        )

    def _score(self, dense_rank: int, text: str, terms: set[str]) -> float:
        tokens = {token.casefold() for token in _TOKEN.findall(text)}
        lexical_rank = 1 if terms & tokens else len(terms) + 1
        return 1 / (self.rrf_k + dense_rank + 1) + 1 / (self.rrf_k + lexical_rank)
