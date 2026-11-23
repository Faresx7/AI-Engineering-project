from src.services.qdrant_service import QdrantService
from src.rag.retrieval.rag_chain import RAGChain


class Container:
    def __init__(self):
        # Service instances declared as optional before startup
        self.qdrant_service: QdrantService | None = None
        self.rag_chain_instance: RAGChain | None = None

    async def initialize_services(self):
        """Initialize DB connections and wire dependent services."""
        # 1. Initialize DB Service
        self.qdrant_service = QdrantService(host="127.0.0.1", port=6333)
        await self.qdrant_service.init_collection(collection_name="rag_documents")

        # ! future update for RAGChain
        # 2. Inject QdrantService into RAGChain
        # self.rag_chain_instance = RAGChain(qdrant_service=self.qdrant_service)

        self.rag_chain_instance = RAGChain()

    async def close_services(self):
        """Close DB client connection cleanly on shutdown."""
        if self.qdrant_service and hasattr(self.qdrant_service.client, "close"):
            await self.qdrant_service.client.close()


# Global instance
container = Container()