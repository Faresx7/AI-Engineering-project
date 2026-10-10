import os
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    models,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector
)


class QdrantService:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize the Async Qdrant client connection."""
        self.host = host
        self.port = port
        self.client = AsyncQdrantClient(host=self.host, port=self.port)

    async def init_collection(self,
                              collection_name: str = "rag_documents",
                              dense_vector_size: int = 768):

        """
        Create collection with Named Vectors (Dense + Sparse) if it doesn't exist.
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
                        memory = models.Memory.COLD # this will load data first time in the disk not ram
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )

    async def upsert_points(self,
        collection_name: str,
        points: List[PointStruct]):
        """Upload or update document points in Qdrant asynchronously."""
        await self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    async def hybrid_search(
        self,
        collection_name: str,
        user_id: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Perform Hybrid Search (Dense + Sparse) with RRF Fusion
        strictly filtered by the owner's user_id.
        """
        # Strict tenant isolation filter
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )

        # Convert dictionary input to Qdrant SparseVector model
        formatted_sparse = SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"]
        )

        # Execute combined search with RRF Fusion in Rust engine
        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    using="dense",
                    query=dense_vector,
                    filter=user_filter,
                    limit=20
                ),
                Prefetch(
                    using="sparse",
                    query=formatted_sparse,
                    filter=user_filter,
                    limit=20
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit
        )

        # Format and return extracted document texts
        documents = []
        for point in results.points:
            payload = point.payload or {}
            documents.append({
                "id": point.id,
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {})
            })

        return documents

