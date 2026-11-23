from typing import Any, Dict, List

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


class QdrantService:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize the Async Qdrant client connection."""
        self.host = host
        self.port = port
        self.client = AsyncQdrantClient(host=self.host, port=self.port)

    async def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes required for fast filtering, atomic swaps, and rollbacks."""
        indexes: dict[str, PayloadSchemaType] = {
            "user_id": PayloadSchemaType.KEYWORD,
            "doc_id": PayloadSchemaType.KEYWORD,
            "generation": PayloadSchemaType.KEYWORD,
            "committed": PayloadSchemaType.BOOL,
        }

        for field_name, field_schema in indexes.items():
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )

    async def init_collection(
        self,
        collection_name: str = "rag_documents",
        dense_vector_size: int = 768,
    ) -> None:
        """Create collection with Named Vectors (Dense + Sparse) and payload indexes.

        - Dense: Used for semantic meaning (Google Embeddings).
        - Sparse: Used for exact keyword match (BM25 / FastEmbed).
        """
        collections = await self.client.get_collections()
        existing_names = [col.name for col in collections.collections]

        if collection_name not in existing_names:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_vector_size,
                        distance=Distance.COSINE,
                        on_disk=True,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
            )

        # Always verify payload indexes on startup (idempotent if they already exist)
        await self._ensure_payload_indexes(collection_name)

    async def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct],
    ) -> None:
        """Upload or update document points in Qdrant asynchronously."""
        await self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    async def hybrid_search(
        self,
        collection_name: str,
        user_id: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Perform Hybrid Search (Dense + Sparse) with RRF Fusion.

        Strictly filtered by owner user_id and excludes uncommitted in-flight chunks.
        """
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ],
            must_not=[
                FieldCondition(
                    key="committed",
                    match=MatchValue(value=False),
                )
            ],
        )

        formatted_sparse = SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"],
        )

        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    using="dense",
                    query=dense_vector,
                    filter=user_filter,
                    limit=20,
                ),
                Prefetch(
                    using="sparse",
                    query=formatted_sparse,
                    filter=user_filter,
                    limit=20,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )

        documents: List[Dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            documents.append(
                {
                    "id": point.id,
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                }
            )

        return documents