"""Run the versioned Vietnamese RAG benchmark against the local active corpus."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from src.chat import GroundedAnswerService
from src.config import AppSettings, StorageMode, load_settings
from src.evaluation import EvaluationConfig, load_dataset, run_evaluation, write_report
from src.ingestion.embeddings import GeminiEmbeddingProvider
from src.retrieval import DenseRetriever, HybridRetriever
from src.storage import DocumentStore, QdrantCollections, VectorStore, ensure_collections


def main() -> None:
    args = _arguments()
    settings = load_settings()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Thiếu GEMINI_API_KEY trong môi trường terminal.")

    dataset = load_dataset(args.dataset)
    client = _qdrant_client(settings)
    try:
        collections = QdrantCollections(
            chunks=settings.chunks_collection,
            documents=settings.documents_collection,
        )
        ensure_collections(client, collections, embedding_dimension=settings.embedding_dimension)
        document_store = DocumentStore(client, collections.documents)
        vector_store = VectorStore(client, collections.chunks)
        gemini = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=settings.request_timeout_ms),
        )
        dense_retriever = DenseRetriever(
            document_store=document_store,
            vector_store=vector_store,
            embedding_provider=GeminiEmbeddingProvider(
                gemini,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                batch_size=settings.embedding_batch_size,
                max_retries=settings.embedding_max_retries,
            ),
            top_k=settings.retrieval_top_k * (3 if args.strategy == "hybrid" else 1),
            score_threshold=settings.retrieval_score_threshold,
            max_question_chars=settings.max_question_chars,
        )
        retriever = (
            HybridRetriever(dense_retriever, top_k=settings.retrieval_top_k)
            if args.strategy == "hybrid"
            else dense_retriever
        )
        config = EvaluationConfig(
            strategy_id=retriever.strategy_id,
            top_k=settings.retrieval_top_k,
            answer_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            score_threshold=settings.retrieval_score_threshold,
            chunk_size=settings.chunk_size_chars,
            chunk_overlap=settings.chunk_overlap_chars,
            context_mode="deterministic",
            pipeline_version=dataset.context_version,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
        )
        report = run_evaluation(
            dataset,
            split=args.split,
            workspace_id=args.workspace_id,
            records=document_store.list_active(args.workspace_id),
            retriever=retriever,
            answer_provider=GroundedAnswerService(gemini, model=settings.chat_model),
            config=config,
        )
        json_path, markdown_path = write_report(report, args.output_dir, split=args.split)
    finally:
        client.close()

    metrics = report.metrics
    print(f"Đã đánh giá {report.evaluated_case_count} ca ({args.split}).")
    print(
        f"Hit Rate@{config.top_k}: {metrics.hit_rate_at_k:.3f} | "
        f"Recall@{config.top_k}: {metrics.recall_at_k:.3f} | MRR: {metrics.mrr:.3f}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá dense RAG trên corpus đã lập chỉ mục.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluations/datasets/lapzone-v1.json"),
    )
    parser.add_argument("--strategy", choices=("dense", "hybrid"), default="dense")
    parser.add_argument("--split", choices=("development", "holdout"), default="holdout")
    parser.add_argument("--workspace-id", default="local-default")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/results"))
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    return parser.parse_args()


def _qdrant_client(settings: AppSettings) -> QdrantClient:
    if settings.storage_mode is StorageMode.LOCAL:
        return QdrantClient(path=str(settings.qdrant_path))
    if settings.storage_mode is StorageMode.CLOUD:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=max(1, settings.request_timeout_ms // 1_000),
        )
    raise SystemExit(
        "Benchmark CLI không hỗ trợ STORAGE_MODE=demo vì dữ liệu chỉ tồn tại trong session."
    )


if __name__ == "__main__":
    main()
