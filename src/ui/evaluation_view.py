"""Retrieval evaluation workspace shell."""

import streamlit as st

from src.config import AppSettings


def render(settings: AppSettings) -> None:
    st.subheader("Đánh giá RAG")
    st.write("So sánh dense baseline với pipeline Advanced trên cùng tập hold-out.")

    columns = st.columns(4)
    for column, label in zip(
        columns,
        ("Hit Rate@K", "Recall@K", "MRR", "Citation accuracy"),
        strict=True,
    ):
        column.metric(label, "—")

    st.caption(
        f"Chưa có benchmark · embedding `{settings.embedding_model}` / "
        f"{settings.embedding_dimension} chiều"
    )
