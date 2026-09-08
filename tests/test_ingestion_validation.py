from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from src.ingestion.models import DocumentKind, UploadPayload, ValidatedUpload
from src.ingestion.validation import IngestionError, validate_upload
from tests.ingestion_fixtures import make_docx_bytes, make_pdf_bytes


def _validate(upload: UploadPayload, **overrides: int) -> ValidatedUpload:
    limits = {
        "max_upload_bytes": 2_000_000,
        "max_zip_entries": 1_000,
        "max_zip_uncompressed_bytes": 10_000_000,
    }
    limits.update(overrides)
    return validate_upload(upload, **limits)


def test_valid_pdf_is_normalized_and_hashed() -> None:
    result = _validate(
        UploadPayload("  Lesson.PDF  ", make_pdf_bytes(), "application/pdf")
    )

    assert result.kind is DocumentKind.PDF
    assert result.normalized_name == "lesson.pdf"
    assert len(result.content_hash) == 64


@pytest.mark.parametrize(
    ("upload", "message"),
    [
        (UploadPayload("empty.pdf", b"", "application/pdf"), "tệp rỗng"),
        (UploadPayload("notes.txt", b"content", "text/plain"), "chỉ hỗ trợ"),
        (UploadPayload("fake.pdf", b"not a pdf", "application/pdf"), "không phải PDF"),
        (UploadPayload("file.pdf", b"%PDF-1.3", "text/plain"), "MIME type"),
    ],
)
def test_invalid_uploads_are_rejected(upload: UploadPayload, message: str) -> None:
    with pytest.raises(IngestionError, match=message):
        _validate(upload)


def test_upload_size_limit_is_enforced_before_parsing() -> None:
    with pytest.raises(IngestionError, match="vượt giới hạn"):
        _validate(
            UploadPayload("large.pdf", make_pdf_bytes(), "application/pdf"),
            max_upload_bytes=10,
        )


def test_openxml_path_traversal_is_rejected() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "types")
        archive.writestr("word/document.xml", "document")
        archive.writestr("../outside", "unsafe")

    with pytest.raises(IngestionError, match="đường dẫn không an toàn"):
        _validate(
            UploadPayload(
                "unsafe.docx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        )


def test_openxml_uncompressed_size_limit_is_enforced() -> None:
    with pytest.raises(IngestionError, match="dữ liệu giải nén"):
        _validate(
            UploadPayload(
                "large.docx",
                make_docx_bytes(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            max_zip_uncompressed_bytes=100,
        )
