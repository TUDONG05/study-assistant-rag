"""Read-only dashboard for the latest reproducible evaluation report."""

from pathlib import Path

import streamlit as st

from src.config import AppSettings
from src.evaluation.reporting import load_report

_LATEST_REPORT = Path("evaluations/results/latest.json")


def render(settings: AppSettings) -> None:
    st.subheader("Đánh giá RAG")
    st.write("Hiển thị kết quả dense baseline trên tập hold-out đã được gán nhãn.")

    if not _LATEST_REPORT.exists():
        _render_empty(settings)
        return
    try:
        report = load_report(_LATEST_REPORT)
        metrics = report["metrics"]
        config = report["config"]
        top_k = int(config["top_k"])
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("Báo cáo benchmark gần nhất không hợp lệ. Hãy chạy lại evaluation CLI.")
        return

    columns = st.columns(4)
    for column, label, key in zip(
        columns,
        (f"Hit Rate@{top_k}", f"Recall@{top_k}", "MRR", "Citation precision"),
        ("hit_rate_at_k", "recall_at_k", "mrr", "citation_precision"),
        strict=True,
    ):
        column.metric(label, f"{float(metrics[key]):.1%}")

    st.caption(
        f"{int(report['evaluated_case_count'])} ca · split `{report.get('split', 'unknown')}` · "
        f"strategy `{config['strategy_id']}` · dataset `{report['dataset_version']}`"
    )
    st.markdown("#### Chất lượng câu trả lời")
    detail_columns = st.columns(4)
    details = (
        ("Answerability F1", "answerability_f1"),
        ("Citation coverage", "citation_coverage"),
        ("Fact coverage", "expected_fact_coverage"),
        ("Query drift", "query_drift_rate"),
    )
    for column, (label, key) in zip(detail_columns, details, strict=True):
        column.metric(label, f"{float(metrics[key]):.1%}")

    with st.expander("Cấu hình và vận hành"):
        st.json(
            {
                "model": config.get("answer_model"),
                "embedding": config.get("embedding_model"),
                "dimension": config.get("embedding_dimension"),
                "chunk_size/overlap": f"{config.get('chunk_size')}/{config.get('chunk_overlap')}",
                "score_threshold": config.get("score_threshold"),
                "retrieval_p95_ms": round(float(metrics["retrieval_latency_p95_ms"]), 2),
                "end_to_end_p95_ms": round(float(metrics["end_to_end_latency_p95_ms"]), 2),
                "model_calls": int(metrics["model_calls_total"]),
                "estimated_tokens": int(metrics["input_tokens_total"])
                + int(metrics["output_tokens_total"]),
                "estimated_cost_usd": round(float(metrics["estimated_cost_usd_total"]), 6),
            }
        )


def _render_empty(settings: AppSettings) -> None:
    columns = st.columns(4)
    for column, label in zip(
        columns,
        ("Hit Rate@K", "Recall@K", "MRR", "Citation precision"),
        strict=True,
    ):
        column.metric(label, "—")
    st.caption(
        f"Chưa có benchmark · embedding `{settings.embedding_model}` / "
        f"{settings.embedding_dimension} chiều"
    )
    st.code("uv run python -m scripts.evaluate_rag --split holdout", language="bash")
