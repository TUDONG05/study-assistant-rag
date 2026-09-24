"""Typed retrieval results derived from trusted vector payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class RetrievalError(RuntimeError):
    """Raised when retrieval cannot safely produce an evidence set."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """Bounded conversation context shared by retrieval and answer generation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Stable input contract for dense and future advanced retrievers."""

    question: str
    workspace_id: str
    selected_document_ids: tuple[str, ...] = ()
    history: tuple[ConversationTurn, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    trace_id: str
    strategy_id: str
    question: str
    selected_document_ids: tuple[str, ...]
    history_turns: int
    candidate_count: int
    effective_queries: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    source_id: str
    rank: int
    score: float
    chunk_id: str
    workspace_id: str
    document_id: str
    version_id: str
    file_name: str
    document_title: str
    source: str
    section: str | None
    page_number: int | None
    slide_number: int | None
    original_text: str

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        rank: int,
        score: float,
    ) -> RetrievedChunk:
        required = (
            "chunk_id",
            "workspace_id",
            "document_id",
            "version_id",
            "file_name",
            "document_title",
            "source",
            "original_text",
        )
        try:
            values = {field: _required_text(payload, field) for field in required}
            section = _optional_text(payload.get("section"))
            page_number = _optional_int(payload.get("page_number"), "page_number")
            slide_number = _optional_int(payload.get("slide_number"), "slide_number")
        except (KeyError, TypeError, ValueError) as exc:
            raise RetrievalError("Payload chunk trong kho vector không hợp lệ.") from exc
        return cls(
            source_id=f"S{rank}",
            rank=rank,
            score=float(score),
            section=section,
            page_number=page_number,
            slide_number=slide_number,
            **values,
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Evidence candidates plus provenance safe to persist in UI state."""

    chunks: tuple[RetrievedChunk, ...]
    trace: RetrievalTrace


def _required_text(payload: dict[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field)
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional text")
    return value.strip() or None


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(field)
    return int(value)
