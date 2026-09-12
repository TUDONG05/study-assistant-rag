"""Versioned contextual retrieval prefixes with safe Gemini fallback."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from google.genai import types

from src.ingestion.models import ChunkDraft, ContextSource, ContextualChunk

CONTEXT_VERSION = "context-v1"


class Contextualizer(Protocol):
    def contextualize(self, chunks: Sequence[ChunkDraft]) -> list[ContextualChunk]: ...


class DeterministicContextualizer:
    def __init__(self, *, max_prefix_chars: int, context_version: str = CONTEXT_VERSION) -> None:
        self.max_prefix_chars = max_prefix_chars
        self.context_version = context_version

    def contextualize(self, chunks: Sequence[ChunkDraft]) -> list[ContextualChunk]:
        result: list[ContextualChunk] = []
        for index, chunk in enumerate(chunks):
            previous = chunks[index - 1] if index > 0 else None
            following = chunks[index + 1] if index + 1 < len(chunks) else None
            prefix = self._prefix(chunk, previous, following)
            result.append(self._build(chunk, prefix, ContextSource.DETERMINISTIC))
        return result

    def _prefix(
        self,
        chunk: ChunkDraft,
        previous: ChunkDraft | None,
        following: ChunkDraft | None,
    ) -> str:
        parts = [f"Tài liệu: {chunk.document_title}", f"Vị trí: {chunk.source}"]
        if chunk.section:
            parts.append(f"Mục: {chunk.section}")
        if previous and previous.source == chunk.source:
            parts.append(f"Trước: {_snippet(previous.original_text)}")
        if following and following.source == chunk.source:
            parts.append(f"Sau: {_snippet(following.original_text)}")
        return " | ".join(parts)[: self.max_prefix_chars].rstrip()

    def _build(
        self,
        chunk: ChunkDraft,
        prefix: str,
        source: ContextSource,
    ) -> ContextualChunk:
        return ContextualChunk(
            draft=chunk,
            context_prefix=prefix,
            retrieval_text=f"[Ngữ cảnh truy xuất: {prefix}]\n\n{chunk.original_text}",
            context_source=source,
            context_version=self.context_version,
        )


class GeminiContextualizer(DeterministicContextualizer):
    """Enrich short prefixes; each failure falls back without losing ingestion."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_prefix_chars: int,
        max_enriched_chunks: int,
        context_version: str = CONTEXT_VERSION,
    ) -> None:
        super().__init__(max_prefix_chars=max_prefix_chars, context_version=context_version)
        self.client = client
        self.model = model
        self.max_enriched_chunks = max_enriched_chunks

    def contextualize(self, chunks: Sequence[ChunkDraft]) -> list[ContextualChunk]:
        deterministic = super().contextualize(chunks)
        result: list[ContextualChunk] = []
        for index, base in enumerate(deterministic):
            if index >= self.max_enriched_chunks:
                result.append(base)
                continue
            try:
                prefix = self._generate_prefix(base)
                result.append(self._build(base.draft, prefix, ContextSource.GEMINI))
            except Exception:
                result.append(self._build(base.draft, base.context_prefix, ContextSource.FALLBACK))
        return result

    def _generate_prefix(self, base: ContextualChunk) -> str:
        prompt = (
            "Viết một câu ngữ cảnh truy xuất ngắn, chỉ dùng dữ kiện được cung cấp. "
            "Không trả lời nội dung chunk, không thêm kiến thức bên ngoài.\n"
            f"Metadata: {base.context_prefix}\n"
            f"Chunk: {base.draft.original_text[:1800]}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {"context": {"type": "string"}},
                    "required": ["context"],
                },
            ),
        )
        payload = json.loads(response.text or "{}")
        context = str(payload.get("context", "")).strip()
        if not context:
            raise ValueError("Gemini returned empty context")
        return context[: self.max_prefix_chars].rstrip()


def _snippet(text: str, limit: int = 90) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1].rstrip()}…"
