"""Reliable document ingestion for Study Assistant."""

from src.ingestion.indexer import DocumentIndexer
from src.ingestion.models import IndexResult, UploadPayload

__all__ = ["DocumentIndexer", "IndexResult", "UploadPayload"]
