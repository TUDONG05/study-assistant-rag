"""JSON and Markdown serialization for compact benchmark artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.models import EvaluationReport


def write_report(
    report: EvaluationReport,
    output_dir: str | Path,
    *,
    split: str,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    payload["generated_at"] = datetime.now(UTC).isoformat()
    payload["split"] = split
    json_path = directory / "latest.json"
    markdown_path = directory / "latest.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(report_to_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise ValueError("Báo cáo benchmark không hợp lệ.")
    return payload


def report_to_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    config = payload["config"]
    rows = (
        (f"Hit Rate@{config['top_k']}", metrics["hit_rate_at_k"]),
        (f"Recall@{config['top_k']}", metrics["recall_at_k"]),
        ("MRR", metrics["mrr"]),
        (f"nDCG@{config['top_k']}", metrics["ndcg_at_k"]),
        ("Answerability F1", metrics["answerability_f1"]),
        ("Citation precision", metrics["citation_precision"]),
        ("Citation recall", metrics["citation_recall"]),
        ("Citation coverage", metrics["citation_coverage"]),
        ("Expected-fact coverage", metrics["expected_fact_coverage"]),
    )
    table = "\n".join(f"| {name} | {float(value):.4f} |" for name, value in rows)
    return (
        f"# Benchmark {payload['dataset_id']}\n\n"
        f"- Dataset: `{payload['dataset_version']}`\n"
        f"- Corpus: `{payload['corpus_id']} / {payload['corpus_version']}`\n"
        f"- Split: `{payload['split']}`\n"
        f"- Strategy: `{config['strategy_id']}`\n"
        f"- Cases: {payload['evaluated_case_count']}\n"
        f"- Generated: `{payload['generated_at']}`\n\n"
        "| Chỉ số | Giá trị |\n|---|---:|\n"
        f"{table}\n\n"
        "Token được ước lượng theo số ký tự vì provider chưa trả usage qua contract hiện tại. "
        "Corpus chỉ có một tài liệu nên chưa đo được retrieval xuyên tài liệu.\n"
    )
