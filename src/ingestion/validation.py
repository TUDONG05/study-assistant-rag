"""Defensive validation for untrusted document uploads."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from src.ingestion.models import (
    DocumentKind,
    UploadPayload,
    ValidatedUpload,
    safe_title,
    sha256_bytes,
)

SUPPORTED_MIME_TYPES = {
    DocumentKind.PDF: {"application/pdf"},
    DocumentKind.DOCX: {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    DocumentKind.PPTX: {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    },
}


class IngestionError(ValueError):
    """A safe, actionable error that can be shown in the UI."""


def validate_upload(
    upload: UploadPayload,
    *,
    max_upload_bytes: int,
    max_zip_entries: int,
    max_zip_uncompressed_bytes: int,
) -> ValidatedUpload:

    file_name = Path(upload.file_name).name.strip()
    if not file_name:
        raise IngestionError("Tệp không có tên hợp lệ.")
    if not upload.data:
        raise IngestionError(f"{file_name}: tệp rỗng.")
    if len(upload.data) > max_upload_bytes:
        limit_mb = max_upload_bytes // (1024 * 1024)
        raise IngestionError(f"{file_name}: vượt giới hạn {limit_mb} MB.")

    extension = Path(file_name).suffix.lower().lstrip(".")
    try:
        kind = DocumentKind(extension)
    except ValueError as exc:
        raise IngestionError(f"{file_name}: chỉ hỗ trợ PDF, DOCX và PPTX.") from exc

    if upload.mime_type and upload.mime_type not in {
        "application/octet-stream",
        *SUPPORTED_MIME_TYPES[kind],
    }:
        raise IngestionError(f"{file_name}: MIME type không khớp phần mở rộng.")

    if kind is DocumentKind.PDF:
        # Phần mở rộng và MIME không đáng tin hoàn toàn; kiểm tra cả chữ ký PDF.
        marker_index = upload.data[:1024].find(b"%PDF-")
        if marker_index < 0:
            raise IngestionError(f"{file_name}: nội dung không phải PDF hợp lệ.")
    else:
        # DOCX/PPTX là ZIP archive, cần chống archive bomb và path traversal.
        _validate_openxml(
            file_name,
            upload.data,
            kind,
            max_entries=max_zip_entries,
            max_uncompressed_bytes=max_zip_uncompressed_bytes,
        )

    normalized_name = re.sub(r"\s+", " ", file_name).casefold()
    return ValidatedUpload(
        file_name=file_name,
        normalized_name=normalized_name,
        title=safe_title(file_name),
        data=upload.data,
        kind=kind,
        content_hash=sha256_bytes(upload.data),
    )


def _validate_openxml(
    file_name: str,
    data: bytes,
    kind: DocumentKind,
    *,
    max_entries: int,
    max_uncompressed_bytes: int,
) -> None:
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise IngestionError(f"{file_name}: archive có quá nhiều thành phần.")

            total_size = 0
            names: set[str] = set()
            for info in entries:
                if info.flag_bits & 0x1:
                    raise IngestionError(f"{file_name}: không hỗ trợ archive được mã hóa.")
                path = PurePosixPath(info.filename)
                # Archive OpenXML không được phép thoát ra ngoài thư mục giải nén.
                if path.is_absolute() or ".." in path.parts:
                    raise IngestionError(f"{file_name}: archive chứa đường dẫn không an toàn.")
                total_size += info.file_size
                if total_size > max_uncompressed_bytes:
                    raise IngestionError(f"{file_name}: dữ liệu giải nén vượt giới hạn an toàn.")
                names.add(info.filename)
    except BadZipFile as exc:
        raise IngestionError(f"{file_name}: tệp Office bị hỏng hoặc sai định dạng.") from exc

    required = "word/document.xml" if kind is DocumentKind.DOCX else "ppt/presentation.xml"
    # ZIP thông thường không phải tệp Office nếu thiếu các thành phần OpenXML bắt buộc.
    if "[Content_Types].xml" not in names or required not in names:
        raise IngestionError(f"{file_name}: nội dung không khớp định dạng {kind.value.upper()}.")
