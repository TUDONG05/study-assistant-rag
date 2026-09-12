from __future__ import annotations

import pytest

from src.ingestion.models import DocumentKind, UploadPayload, ValidatedUpload
from src.ingestion.parsers import parse_document
from src.ingestion.validation import IngestionError, validate_upload
from tests.ingestion_fixtures import make_docx_bytes, make_pdf_bytes, make_pptx_bytes

MIME_TYPES = {
    DocumentKind.PDF: "application/pdf",
    DocumentKind.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    DocumentKind.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _validated(file_name: str, data: bytes) -> ValidatedUpload:
    kind = DocumentKind(file_name.rsplit(".", maxsplit=1)[1])
    return validate_upload(
        UploadPayload(file_name=file_name, data=data, mime_type=MIME_TYPES[kind]),
        max_upload_bytes=2_000_000,
        max_zip_entries=1_000,
        max_zip_uncompressed_bytes=10_000_000,
    )


def test_docx_parser_preserves_table_order_and_section() -> None:
    upload = _validated("bai-hoc.docx", make_docx_bytes(include_table=True))

    parsed = parse_document(upload, max_pages=20, max_extracted_chars=10_000)

    assert [(block.source, block.section) for block in parsed.blocks] == [
        ("Đoạn 1", "Phần một"),
        ("Bảng 1", "Phần một"),
        ("Đoạn 2", "Phần hai"),
    ]


def test_pdf_parser_preserves_page_citation() -> None:
    parsed = parse_document(
        _validated("lesson.pdf", make_pdf_bytes()),
        max_pages=5,
        max_extracted_chars=10_000,
    )

    assert parsed.blocks[0].page_number == 1
    assert parsed.blocks[0].source == "Trang 1"
    assert "retrieval" in parsed.blocks[0].text


def test_pptx_parser_preserves_slide_title_and_number() -> None:
    parsed = parse_document(
        _validated("lesson.pptx", make_pptx_bytes()),
        max_pages=5,
        max_extracted_chars=10_000,
    )

    assert parsed.blocks[0].slide_number == 1
    assert parsed.blocks[0].source == "Slide 1"
    assert parsed.blocks[0].section == "Bài giảng RAG"
    assert "Dense | Vector" in parsed.blocks[0].text


def test_parser_rejects_documents_without_enough_text() -> None:
    upload = _validated("short.docx", make_docx_bytes(first_text="ngắn"))

    with pytest.raises(IngestionError, match="không tìm thấy đủ văn bản"):
        parse_document(upload, max_pages=5, max_extracted_chars=10_000)
