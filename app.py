"""Streamlit entry point for Study Assistant."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.config import AppSettings, ConfigurationError, load_settings
from src.ui import chat_view, documents_view, evaluation_view, study_tools_view
from src.ui.session_state import (
    ACTIVE_VIEW,
    GEMINI_API_KEY,
    REQUEST_COUNT,
    active_gemini_api_key,
    configure_workspace,
    current_workspace_id,
    initialize_session_state,
    reset_session_state,
)

ViewRenderer = Callable[[AppSettings], None]

VIEWS: dict[str, ViewRenderer] = {
    "Tài liệu": documents_view.render,
    "Chat": chat_view.render,
    "Đánh giá": evaluation_view.render,
    "Công cụ học tập": study_tools_view.render,
}


def _read_streamlit_secrets() -> Mapping[str, Any]:
    try:
        secrets = st.secrets.to_dict()
    except StreamlitSecretNotFoundError:
        return {}
    return cast(Mapping[str, Any], secrets)


def _render_sidebar(settings: AppSettings) -> str:
    with st.sidebar:
        st.title(settings.app_name)
        st.caption("Advanced RAG cho tài liệu học tập tiếng Việt")

        st.text_input(
            "Gemini API key",
            type="password",
            key=GEMINI_API_KEY,
            placeholder="Nhập key của bạn",
            help="Key BYOK chỉ tồn tại trong session và không được ghi log.",
        )
        if active_gemini_api_key():
            st.success("Đã nhận API key. Key sẽ được xác thực khi gọi tính năng AI.")
        else:
            st.info("Nhập API key để bật các tính năng AI.")

        selected_view = st.radio("Điều hướng", tuple(VIEWS), key=ACTIVE_VIEW)

        if st.button("Đặt lại phiên", use_container_width=True):
            reset_session_state()
            st.rerun()

        with st.expander("Diagnostics"):
            diagnostics = settings.public_diagnostics()
            diagnostics["Workspace"] = current_workspace_id()[:8]
            diagnostics["Requests used"] = int(st.session_state[REQUEST_COUNT])
            st.json(diagnostics)

    return str(selected_view)


def main() -> None:
    st.set_page_config(
        page_title="Study Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_session_state()

    try:
        settings = load_settings(_read_streamlit_secrets())
    except ConfigurationError as exc:
        st.error(f"Cấu hình không hợp lệ: {exc}")
        st.stop()

    configure_workspace(settings.storage_mode)
    selected_view = _render_sidebar(settings)
    st.title("Study Assistant")
    st.caption("Học từ tài liệu của bạn với câu trả lời có căn cứ và trích dẫn kiểm chứng.")
    VIEWS[selected_view](settings)


if __name__ == "__main__":
    main()
