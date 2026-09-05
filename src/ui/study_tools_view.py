"""Sourced study tools workspace shell."""

import streamlit as st

from src.config import AppSettings


def render(settings: AppSettings) -> None:
    del settings
    st.subheader("Công cụ học tập")
    st.write("Nội dung sinh ra sẽ luôn đi kèm nguồn từ tài liệu đã chọn.")

    summary_tab, quiz_tab, mindmap_tab = st.tabs(["Tóm tắt", "Quiz", "Mindmap"])
    with summary_tab:
        st.button("Tạo bản tóm tắt", disabled=True, use_container_width=True)
    with quiz_tab:
        st.button("Tạo bộ câu hỏi", disabled=True, use_container_width=True)
    with mindmap_tab:
        st.button("Tạo sơ đồ tư duy", disabled=True, use_container_width=True)
