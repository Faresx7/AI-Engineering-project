"""Application dependency injection container for managing shared service lifecycles.

Purpose and Overview:
    Centralizes initialization, wiring, and graceful shutdown of singleton backend
    services used across the FastAPI application, including Qdrant vector storage,
    the hybrid retrieval pipeline, and the RAG generation chain.

Key Components:
    - ``Container``: Lifecycle manager holding initialized service instances and
      wiring dependencies in topological order.
    - ``container``: Global singleton instance imported by ``main.py`` and API
      routers.

Raises:
    QdrantServiceError: Raised during startup if Qdrant connection or collection
        initialization fails.
    ValueError: Raised if invalid configuration values are passed to
        ``QdrantService`` or ``RetrievalPipeline``.
"""
from __future__ import annotations

import logging

from src.rag.clients import close_clients
from src.rag.constants import _DEFAULT_COLLECTION_NAME
from src.rag.retrieval.rag_chain import RAGChain
from src.rag.retrieval.retrieval_pipeline import RetrievalPipeline
from src.services.qdrant_service import QdrantService


# #############################################################################
# CONSTANTS & CONFIGURATION
# #############################################################################


logger = logging.getLogger(__name__)


# #############################################################################
# DEPENDENCY INJECTION CONTAINER
# #############################################################################

class Container:
    """Manages startup initialization, wiring, and shutdown of core RAG services."""

    def __init__(self) -> None:
        """Declare optional service slots prior to asynchronous startup initialization.

        Returns:
            None.
        """

        self.qdrant_service: QdrantService | None = None
        self.retrieval_pipeline: RetrievalPipeline | None = None
        self.rag_chain_instance: RAGChain | None = None

    async def initialize_services(self) -> None:
        """Initialize database connections and wire dependent RAG services in order.

        Returns:
            None.

        Raises:
            QdrantServiceError: If Qdrant collection or payload index setup fails.
            ValueError: If collection name or search limits are invalid.
        """

        # 1. Initialize Qdrant vector database service using settings.QDRANT_URL
        self.qdrant_service = QdrantService()
        await self.qdrant_service.init_collection(collection_name=_DEFAULT_COLLECTION_NAME)

        # 2. Initialize hybrid retrieval pipeline backed by QdrantService
        self.retrieval_pipeline = RetrievalPipeline(qdrant_service=self.qdrant_service,
                                                    collection_name=_DEFAULT_COLLECTION_NAME,
                                                   )

        # 3. Wire RetrievalPipeline into RAGChain (uncomment once RAGChain is ready)
        # self.rag_chain_instance = RAGChain(retrieval_pipeline=self.retrieval_pipeline)

        logger.info("CONTAINER_SERVICES_INITIALIZED")

    async def close_services(self) -> None:
        """Close database and client connections cleanly during application shutdown.

        Returns:
            None.

        Raises:
            QdrantServiceError: If closing the Qdrant client fails unexpectedly.
        """

        if self.qdrant_service is not None:
            await self.qdrant_service.close()
            self.qdrant_service = None

        self.retrieval_pipeline = None
        self.rag_chain_instance = None

        close_clients()
        logger.info("CONTAINER_SERVICES_CLOSED")


# #############################################################################
# GLOBAL CONTAINER INSTANCE
# #############################################################################

container = Container()