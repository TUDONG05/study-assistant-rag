"""Grounded chat workspace shell."""

import streamlit as st

from src.config import AppSettings
from src.ui.session_state import MESSAGES


def render(settings: AppSettings) -> None:
    st.subheader("Chat có trích dẫn")
    st.caption(f"Model cấu hình: `{settings.chat_model}` · Retrieval: Advanced")

    messages = st.session_state[MESSAGES]
    if not messages:
        st.info("Hãy tải tài liệu trước khi đặt câu hỏi.")
    else:
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.chat_input("Đặt câu hỏi về tài liệu…", disabled=True)
