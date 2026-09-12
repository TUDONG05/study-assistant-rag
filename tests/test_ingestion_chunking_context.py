from __future__ import annotations

from types import SimpleNamespace

from src.ingestion.chunking import chunk_document
from src.ingestion.contextualizer import (
    DeterministicContextualizer,
    GeminiContextualizer,
)
from src.ingestion.models import (
    ChunkDraft,
    ContextSource,
    DocumentKind,
    ParsedBlock,
    ParsedDocument,
    ValidatedUpload,
)


def _chunks(
    text: str,
    *,
    max_chars: int = 42,
    overlap_chars: int = 22,
) -> list[ChunkDraft]:
    upload = ValidatedUpload(
        file_name="lesson.pdf",
        normalized_name="lesson.pdf",
        title="Lesson",
        data=b"data",
        kind=DocumentKind.PDF,
        content_hash="content-hash",
    )
    parsed = ParsedDocument(
        title="Lesson",
        kind=DocumentKind.PDF,
        blocks=(
            ParsedBlock(0, text, "Trang 1", section="Mục A", page_number=1),
        ),
        extracted_chars=len(text),
    )
    return chunk_document(
        parsed,
        upload,
        workspace_id="workspace-a",
        document_id="document-a",
        version_id="version-a",
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def test_chunking_respects_boundaries_overlap_and_stable_ids() -> None:
    text = "Câu một có dữ liệu. Câu hai có dữ liệu. Câu ba có dữ liệu."

    first = _chunks(text)
    second = _chunks(text)

    assert len(first) == 2
    assert all(len(chunk.original_text) <= 42 for chunk in first)
    assert first[0].original_text.endswith("Câu hai có dữ liệu.")
    assert first[1].original_text.startswith("Câu hai có dữ liệu.")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_oversized_token_is_split_to_chunk_limit() -> None:
    chunks = _chunks("x" * 95, max_chars=30, overlap_chars=0)

    assert len(chunks) == 4
    assert all(len(chunk.original_text) <= 30 for chunk in chunks)


def test_overlap_is_dropped_when_next_unit_would_exceed_limit() -> None:
    text = f"{'a' * 19}. {'b' * 19}. {'c' * 29}."

    chunks = _chunks(text, max_chars=42, overlap_chars=22)

    assert all(len(chunk.original_text) <= 42 for chunk in chunks)


def test_deterministic_context_keeps_evidence_separate() -> None:
    draft = _chunks("Nội dung bằng chứng đủ dài để tạo một chunk duy nhất.", max_chars=100)[0]

    result = DeterministicContextualizer(max_prefix_chars=200).contextualize([draft])[0]

    assert result.context_source is ContextSource.DETERMINISTIC
    assert "Tài liệu: Lesson" in result.context_prefix
    assert "Mục: Mục A" in result.context_prefix
    assert result.draft.original_text == draft.original_text
    assert result.retrieval_text.endswith(draft.original_text)


class _GeminiModels:
    def __init__(self, response_text: str | None = None, error: Exception | None = None) -> None:
        self.response_text = response_text
        self.error = error

    def generate_content(self, **_kwargs: object) -> SimpleNamespace:
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.response_text)


def test_gemini_context_uses_response_and_falls_back_per_chunk() -> None:
    draft = _chunks("Nội dung đủ dài cho contextual retrieval.", max_chars=100)[0]
    success_client = SimpleNamespace(models=_GeminiModels('{"context": "Ngữ cảnh AI"}'))
    failure_client = SimpleNamespace(models=_GeminiModels(error=RuntimeError("unavailable")))

    success = GeminiContextualizer(
        success_client, model="model", max_prefix_chars=100, max_enriched_chunks=1
    ).contextualize([draft])[0]
    fallback = GeminiContextualizer(
        failure_client, model="model", max_prefix_chars=100, max_enriched_chunks=1
    ).contextualize([draft])[0]

    assert success.context_source is ContextSource.GEMINI
    assert success.context_prefix == "Ngữ cảnh AI"
    assert fallback.context_source is ContextSource.FALLBACK
    assert "Tài liệu: Lesson" in fallback.context_prefix
