"""Typed contracts shared across the ingestion pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5


class DocumentKind(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"


class ContextSource(StrEnum):
    DETERMINISTIC = "deterministic"
    GEMINI = "gemini"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class UploadPayload:
    file_name: str
    data: bytes
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    file_name: str
    normalized_name: str
    title: str
    data: bytes
    kind: DocumentKind
    content_hash: str


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    block_index: int
    text: str
    source: str
    section: str | None = None
    page_number: int | None = None
    slide_number: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str
    kind: DocumentKind
    blocks: tuple[ParsedBlock, ...]
    extracted_chars: int


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    workspace_id: str
    document_id: str
    version_id: str
    content_hash: str
    chunk_id: str
    file_name: str
    document_title: str
    source: str
    section: str | None
    page_number: int | None
    slide_number: int | None
    block_index: int
    chunk_index: int
    original_text: str
    original_hash: str


@dataclass(frozen=True, slots=True)
class ContextualChunk:
    draft: ChunkDraft
    context_prefix: str
    retrieval_text: str
    context_source: ContextSource
    context_version: str

    def payload(self) -> dict[str, Any]:
        payload = asdict(self.draft)
        payload.update(
            context_prefix=self.context_prefix,
            retrieval_text=self.retrieval_text,
            context_source=self.context_source.value,
            context_version=self.context_version,
        )
        return payload


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    workspace_id: str
    document_id: str
    version_id: str
    file_name: str
    normalized_name: str
    title: str
    file_type: str
    content_hash: str
    chunk_count: int
    extracted_chars: int
    context_version: str
    created_at: str
    status: str = "active"

    @classmethod
    def create(
        cls,
        *,
        upload: ValidatedUpload,
        workspace_id: str,
        document_id: str,
        version_id: str,
        chunk_count: int,
        extracted_chars: int,
        context_version: str,
    ) -> DocumentRecord:
        return cls(
            workspace_id=workspace_id,
            document_id=document_id,
            version_id=version_id,
            file_name=upload.file_name,
            normalized_name=upload.normalized_name,
            title=upload.title,
            file_type=upload.kind.value,
            content_hash=upload.content_hash,
            chunk_count=chunk_count,
            extracted_chars=extracted_chars,
            context_version=context_version,
            created_at=datetime.now(UTC).isoformat(),
        )

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DocumentRecord:
        fields = {key: payload[key] for key in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class IndexResult:
    document: DocumentRecord
    deduplicated: bool


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def make_document_id(workspace_id: str, normalized_name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"study-assistant:{workspace_id}:{normalized_name}"))


def make_version_id(document_id: str, content_hash: str, context_version: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{document_id}:{content_hash}:{context_version}"))


def make_chunk_id(
    version_id: str,
    source: str,
    chunk_index: int,
    original_text: str,
) -> str:
    fingerprint = sha256_text(original_text)
    return str(uuid5(NAMESPACE_URL, f"{version_id}:{source}:{chunk_index}:{fingerprint}"))


def safe_title(file_name: str) -> str:
    return Path(file_name).stem.strip() or "Untitled document"
