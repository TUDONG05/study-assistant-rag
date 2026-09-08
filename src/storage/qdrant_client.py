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
    if client.collection_exists(collections.chunks):
        _validate_vector_size(client, collections.chunks, embedding_dimension)
    else:
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
    if client.collection_exists(collections.documents):
        _validate_vector_size(client, collections.documents, 1)
    else:
        client.create_collection(
            collection_name=collections.documents,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
        _create_keyword_indexes(
            client,
            collections.documents,
            ("workspace_id", "document_id", "normalized_name", "content_hash", "status"),
        )


def _validate_vector_size(client: QdrantClient, collection_name: str, expected: int) -> None:
    vectors = client.get_collection(collection_name).config.params.vectors
    actual = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)
    if actual != expected or distance is not models.Distance.COSINE:
        raise ValueError(
            f"Qdrant collection {collection_name!r} dùng vector dimension/distance "
            f"{actual}/{distance}; cấu hình hiện tại yêu cầu {expected}/Cosine. "
            "Đổi tên collection hoặc lập chỉ mục lại."
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
