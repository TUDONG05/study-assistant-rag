"""Public contracts for grounded answers and citation snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.retrieval.models import RetrievedChunk

CANONICAL_REFUSAL = (
    "Mình chưa tìm thấy đủ thông tin trong các tài liệu đã chọn để trả lời câu hỏi này."
)


class ChatError(RuntimeError):
    """Raised when answer generation fails for an operational reason."""


@dataclass(frozen=True, slots=True)
class Citation:
    source_id: str
    chunk_id: str
    document_id: str
    version_id: str
    file_name: str
    title: str
    location: str
    supporting_quote: str
    score: float

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> Citation:
        if chunk.page_number is not None:
            location = f"Trang {chunk.page_number}"
        elif chunk.slide_number is not None:
            location = f"Slide {chunk.slide_number}"
        elif chunk.section:
            location = chunk.section
        else:
            location = chunk.source
        return cls(
            source_id=chunk.source_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            file_name=chunk.file_name,
            title=chunk.document_title,
            location=location,
            supporting_quote=chunk.original_text,
            score=chunk.score,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    answer_markdown: str
    citations: tuple[Citation, ...]
    refused: bool
    refusal_reason: str | None = None

    @classmethod
    def refusal(cls, reason: str) -> GroundedAnswer:
        return cls(
            answer_markdown=CANONICAL_REFUSAL,
            citations=(),
            refused=True,
            refusal_reason=reason,
        )
