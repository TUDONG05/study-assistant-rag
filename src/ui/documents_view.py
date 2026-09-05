"""Documents workspace shell."""

import streamlit as st

from src.config import AppSettings


def render(settings: AppSettings) -> None:
    st.subheader("Tài liệu")
    st.write("Tải và quản lý tài liệu dùng làm nguồn kiến thức cho trợ lý.")

    st.file_uploader(
        "PDF, DOCX hoặc PPTX",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        disabled=True,
        help="Ingestion sẽ được kích hoạt ở Phase 3.",
    )
    st.info(
        f"Khung tải tệp đã sẵn sàng. Giới hạn dự kiến: {settings.max_upload_mb} MB mỗi tệp."
    )
