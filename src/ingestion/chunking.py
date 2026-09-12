"""Boundary-aware chunking with deterministic identifiers."""

from __future__ import annotations

import re

from src.ingestion.models import (
    ChunkDraft,
    DocumentKind,
    ParsedBlock,
    ParsedDocument,
    ValidatedUpload,
    make_chunk_id,
    sha256_text,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n{2,}")
_DOCX_PARAGRAPH_SOURCE = re.compile(r"^Đoạn (?P<number>\d+)$")


def chunk_document(
    parsed: ParsedDocument,
    upload: ValidatedUpload,
    *,
    workspace_id: str,
    document_id: str,
    version_id: str,
    max_chars: int,
    overlap_chars: int,
) -> list[ChunkDraft]:
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Chunk size/overlap không hợp lệ.")

    chunks: list[ChunkDraft] = []
    chunk_index = 0
    blocks = (
        _merge_short_docx_paragraphs(parsed.blocks, max_chars)
        if parsed.kind is DocumentKind.DOCX
        else parsed.blocks
    )
    for block in blocks:
        # PDF, slide và bảng vẫn giữ nguyên ranh giới nguồn để trích dẫn chính xác.
        for text in _chunk_text(block.text, max_chars=max_chars, overlap_chars=overlap_chars):
            chunk_id = make_chunk_id(version_id, block.source, chunk_index, text)
            chunks.append(
                ChunkDraft(
                    workspace_id=workspace_id,
                    document_id=document_id,
                    version_id=version_id,
                    content_hash=upload.content_hash,
                    chunk_id=chunk_id,
                    file_name=upload.file_name,
                    document_title=parsed.title,
                    source=block.source,
                    section=block.section,
                    page_number=block.page_number,
                    slide_number=block.slide_number,
                    block_index=block.block_index,
                    chunk_index=chunk_index,
                    original_text=text,
                    original_hash=sha256_text(text),
                )
            )
            chunk_index += 1
    return chunks


def _merge_short_docx_paragraphs(
    blocks: tuple[ParsedBlock, ...], max_chars: int
) -> list[ParsedBlock]:
    """Gom các đoạn DOCX liền nhau trong cùng mục thành một nguồn trích dẫn."""

    merged: list[ParsedBlock] = []
    current: list[ParsedBlock] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length
        if not current:
            return
        first = current[0]
        last = current[-1]
        source = first.source
        if len(current) > 1:
            last_number = _DOCX_PARAGRAPH_SOURCE.fullmatch(last.source)
            if last_number:
                source = f"{first.source}–{last_number.group('number')}"
        merged.append(
            ParsedBlock(
                block_index=first.block_index,
                text="\n\n".join(block.text for block in current),
                source=source,
                section=first.section,
            )
        )
        current = []
        current_length = 0

    for block in blocks:
        is_paragraph = _DOCX_PARAGRAPH_SOURCE.fullmatch(block.source) is not None
        if not is_paragraph:
            # Bảng hoặc loại block khác luôn ngắt nhóm paragraph đang gom.
            flush()
            merged.append(block)
            continue

        added = len(block.text) + (2 if current else 0)
        changed_section = bool(current and block.section != current[0].section)
        if current and (changed_section or current_length + added > max_chars):
            flush()
            added = len(block.text)
        current.append(block)
        current_length += added

    flush()
    return merged


def _chunk_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    units: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        units.extend(_split_oversized(sentence, max_chars))

    # Ưu tiên chunk theo câu hoàn chỉnh; câu quá dài đã được tách ở trên.
    result: list[str] = []
    current: list[str] = []
    current_length = 0
    for unit in units:
        # Chỉ cần một khoảng trắng khi chunk hiện tại đã có nội dung.
        added = len(unit) + (1 if current else 0)
        if current and current_length + added > max_chars:
            result.append(" ".join(current))

            # Mang phần cuối sang chunk mới để giữ ngữ cảnh ở ranh giới khi truy xuất.
            current = _overlap_units(current, overlap_chars)
            current_length = len(" ".join(current))

            # Bỏ dần phần overlap cũ nhất đến khi còn đủ chỗ cho unit mới.
            while current and current_length + len(unit) + 1 > max_chars:
                current.pop(0)
                current_length = len(" ".join(current))
        current.append(unit)
        current_length += len(unit) + (1 if len(current) > 1 else 0)
    if current:
        candidate = " ".join(current)
        if not result or candidate != result[-1]:
            result.append(candidate)
    return result


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if len(word) > max_chars:
            # Một từ dài hơn cả giới hạn  .
            if current:
                parts.append(" ".join(current))
                current = []
                length = 0
            parts.extend(
                word[index : index + max_chars] for index in range(0, len(word), max_chars)
            )
            continue
        added = len(word) + (1 if current else 0)
        if current and length + added > max_chars:
            parts.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += added
    if current:
        parts.append(" ".join(current))
    return parts


def _overlap_units(units: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars == 0:
        return []
    overlap: list[str] = []
    length = 0
    for unit in reversed(units):
        # Duyệt ngược để giữ phần cuối, thay vì phần đầu, của chunk vừa hoàn tất.
        added = len(unit) + (1 if overlap else 0)
        if overlap and length + added > overlap_chars:
            break
        if not overlap and len(unit) > overlap_chars:
            break
        overlap.insert(0, unit)
        length += added
    return overlap
