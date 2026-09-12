"""Chunk staging, verification and workspace-scoped cleanup."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from src.ingestion.models import ContextualChunk
from src.storage.qdrant_client import workspace_filter


class VectorStore:
    def __init__(self, client: QdrantClient, collection_name: str, *, batch_size: int = 64) -> None:
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size

    def stage(self, chunks: Sequence[ContextualChunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Số chunk và vector không khớp.")
        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start : start + self.batch_size]
            batch_vectors = vectors[start : start + self.batch_size]
            points = [
                models.PointStruct(
                    id=chunk.draft.chunk_id,
                    vector=list(vector),
                    payload=chunk.payload(),
                )
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def count_version(self, workspace_id: str, document_id: str, version_id: str) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=workspace_filter(
                workspace_id,
                document_id=document_id,
                version_id=version_id,
            ),
            exact=True,
        )
        return int(result.count)

    def delete_version(self, workspace_id: str, document_id: str, version_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=workspace_filter(
                    workspace_id,
                    document_id=document_id,
                    version_id=version_id,
                )
            ),
            wait=True,
        )

    def delete_document(self, workspace_id: str, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=workspace_filter(workspace_id, document_id=document_id)
            ),
            wait=True,
        )

    def query(
        self,
        vector: Sequence[float],
        *,
        workspace_id: str,
        version_ids: Sequence[str],
        document_ids: Sequence[str] | None,
        limit: int,
        score_threshold: float,
    ) -> list[models.ScoredPoint]:
        """Search only committed versions from the current workspace."""

        if not version_ids:
            return []
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="workspace_id",
                match=models.MatchValue(value=workspace_id),
            ),
            models.FieldCondition(
                key="version_id",
                match=models.MatchAny(any=list(version_ids)),
            ),
        ]
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(document_ids)),
                )
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            query_filter=models.Filter(must=conditions),
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)
