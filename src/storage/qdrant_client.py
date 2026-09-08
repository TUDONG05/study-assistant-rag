"""Qdrant collection lifecycle and workspace filter helpers."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models


@dataclass(frozen=True, slots=True)
class QdrantCollections:
    chunks: str = "study_chunks_v1"
    documents: str = "study_documents_v1"


def ensure_collections(
    client: QdrantClient,
    collections: QdrantCollections,
    *,
    embedding_dimension: int,
) -> None:
    if not client.collection_exists(collections.chunks):
        client.create_collection(
            collection_name=collections.chunks,
            vectors_config=models.VectorParams(
                size=embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        _create_keyword_indexes(
            client,
            collections.chunks,
            ("workspace_id", "document_id", "version_id", "content_hash"),
        )
    if not client.collection_exists(collections.documents):
        client.create_collection(
            collection_name=collections.documents,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        _create_keyword_indexes(
            client,
            collections.documents,
            ("workspace_id", "document_id", "normalized_name", "content_hash", "status"),
        )


def workspace_filter(workspace_id: str, **matches: str) -> models.Filter:
    conditions = [
        models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
    ]
    conditions.extend(
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in matches.items()
    )
    return models.Filter(must=conditions)


def _create_keyword_indexes(
    client: QdrantClient,
    collection_name: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
