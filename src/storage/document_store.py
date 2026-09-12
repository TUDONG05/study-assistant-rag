"""Active document registry; staging chunks are invisible until commit."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.ingestion.models import DocumentRecord
from src.storage.qdrant_client import workspace_filter


class DocumentStore:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    def find_by_name(self, workspace_id: str, normalized_name: str) -> DocumentRecord | None:
        return self._find_one(
            workspace_filter(workspace_id, normalized_name=normalized_name, status="active")
        )

    def find_duplicate(
        self,
        workspace_id: str,
        content_hash: str,
        context_version: str,
    ) -> DocumentRecord | None:
        return self._find_one(
            workspace_filter(
                workspace_id,
                content_hash=content_hash,
                context_version=context_version,
                status="active",
            )
        )

    def list_active(self, workspace_id: str) -> list[DocumentRecord]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=workspace_filter(workspace_id, status="active"),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        records = [DocumentRecord.from_payload(dict(point.payload or {})) for point in points]
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def commit(self, record: DocumentRecord) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=record.document_id,
                    vector=[0.0],
                    payload=record.payload(),
                )
            ],
            wait=True,
        )

    def delete(self, workspace_id: str, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=workspace_filter(workspace_id, document_id=document_id)
            ),
            wait=True,
        )

    def _find_one(self, query_filter: models.Filter) -> DocumentRecord | None:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        return DocumentRecord.from_payload(dict(points[0].payload or {}))
