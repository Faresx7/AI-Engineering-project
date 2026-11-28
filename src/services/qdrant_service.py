"""Asynchronous Qdrant vector database service for RAG retrieval and ingestion.

Provides a hardened wrapper around ``AsyncQdrantClient`` featuring bounded
concurrency via an asyncio semaphore, per-request timeouts, exponential backoff
retries, idempotent collection and payload index initialization, and
tenant-isolated hybrid search (dense + sparse BM25 with RRF fusion).
"""
from __future__ import annotations

from qdrant_client.http.exceptions import UnexpectedResponse
from typing import Any, Callable, Coroutine, Dict, List
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (PayloadSchemaType,
                                  SparseVectorParams,
                                  FieldCondition,
                                  SparseVector,
                                  VectorParams,
                                  FusionQuery,
                                  PointStruct,
                                  MatchValue,
                                  Distance,
                                  Prefetch,
                                  Filter,
                                  Fusion,
                                 )
import asyncio
import logging
import random
import re
import time

from src.rag.constants import (_QDRANT_DEFAULT_TIMEOUT_SECONDS,
                               _NON_RETRYABLE_CODE_ERRORS,
                               _RETRY_BASE_DELAY_SECONDS,
                               _RETRY_MAX_DELAY_SECONDS,
                               _DEFAULT_COLLECTION_NAME,
                               _QDRANT_MAX_CONCURRENCY,
                               _DEFAULT_SEARCH_LIMIT,
                               _DENSE_EMBEDDING_DIM,
                               _QDRANT_MAX_RETRIES,
                               _MAX_SEARCH_LIMIT,
                              )
from src.core.config import settings

try:
    from celery.exceptions import SoftTimeLimitExceeded
except ImportError:
    SoftTimeLimitExceeded = ()

logger = logging.getLogger(__name__)

_ID_REGEX = re.compile(settings.ID_PATTERN)

_NON_RETRYABLE = (_NON_RETRYABLE_CODE_ERRORS
                  + ((SoftTimeLimitExceeded,)
                     if isinstance(SoftTimeLimitExceeded, type) else ())
                 )


class QdrantServiceError(RuntimeError):
    """Raised when a Qdrant operation fails after exhausting retries or is rejected."""


class QdrantService:
    """Async Qdrant client wrapper for collection setup and tenant-isolated hybrid search.

    Manages the lifecycle of a shared ``AsyncQdrantClient`` instance and enforces
    concurrency limits, timeouts, retries, and strict tenant isolation across all
    read and write operations.
    """

    def __init__(self,
                 url: str | None = None,
                 *,
                 host: str | None = None,
                 port: int | None = None,
                 api_key: str | None = None,
                 timeout: float = _QDRANT_DEFAULT_TIMEOUT_SECONDS,
                 max_concurrent_requests: int = _QDRANT_MAX_CONCURRENCY,
                ) -> None:
        """Initializes the Qdrant service client, timeout, and concurrency limiter.

        Args:
            url: Full Qdrant server URL. Overrides ``host``, ``port``, and default settings.
            host: Qdrant server hostname used when ``url`` is not provided.
            port: Qdrant server port used when ``url`` is not provided (defaults to 6333).
            api_key: Optional API key for authenticating with Qdrant. Falls back to settings.
            timeout: Request timeout in seconds; must be strictly positive.
            max_concurrent_requests: Maximum simultaneous in-flight Qdrant calls allowed.

        Returns:
            None.

        Raises:
            ValueError: If ``timeout`` is less than or equal to 0, or if
                ``max_concurrent_requests`` is less than 1.
        """
        
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}.")
        
        if max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be >= 1, got {max_concurrent_requests}."
            )

        if url:
            resolved_url = url
            
        elif host is not None or port is not None:
            resolved_url = f"http://{host or 'localhost'}:{port or 6333}"

        else:
            resolved_url = settings.QDRANT_URL

        resolved_key = (api_key
                        or (settings.QDRANT_API_KEY.get_secret_value()
                            if settings.QDRANT_API_KEY else None))

        self.url = resolved_url
        self.timeout = timeout
        self._closed = False
        
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        self.client = AsyncQdrantClient(url=self.url,
                                        api_key=resolved_key,
                                        timeout=int(timeout),
                                        )


    async def __aenter__(self) -> QdrantService:
        """Enters the asynchronous context manager and returns the active service.

        Args:
            None.

        Returns:
            QdrantService: The initialized ``self`` instance ready for operations.

        Raises:
            None.
        """
        return self


    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the asynchronous context manager and closes the underlying client.

        Args:
            exc_type: Exception class if an error occurred inside the ``async with`` block.
            exc_val: Exception instance raised inside the block, if any.
            exc_tb: Traceback object associated with the exception, if any.

        Returns:
            None. Always returns ``None`` so any exception raised inside the block propagates.

        Raises:
            QdrantServiceError: If closing the underlying Qdrant client fails.
        """
        await self.close()


    async def close(self) -> None:
        """Closes the underlying Qdrant client safely and idempotently.

        Marks the service as closed immediately to reject any subsequent calls,
        then closes the underlying network connection pool. Calling this method
        multiple times is a safe no-op after the first call.

        Args:
            None.

        Returns:
            None.

        Raises:
            QdrantServiceError: If ``self.client.close()`` encounters an unexpected error
                while shutting down connections.
        """
        if self._closed:
            return
        
        self._closed = True

        try:
            await self.client.close()
            logger.info("QDRANT_CLIENT_CLOSED | url=%s", self.url)

        except Exception as exc:
            logger.error(
                        "QDRANT_CLIENT_CLOSE_FAILED | url=%s err_type=%s err=%r",
                        self.url,
                        type(exc).__name__,
                        str(exc),
                        )
            
            raise QdrantServiceError("Failed to close Qdrant client cleanly.") from exc


    async def _execute(self,
                       coro_factory: Callable[[], Coroutine[Any, Any, Any]],
                       *,
                       what: str,
                       attempts: int = _QDRANT_MAX_RETRIES,
                       base_delay: float = _RETRY_BASE_DELAY_SECONDS,
                      ) -> Any:
        
        """Executes a Qdrant coroutine under semaphore concurrency control, timeout, and retries.

        Creates a fresh coroutine on each attempt via ``coro_factory``, enforces a
        10-second timeout per attempt, and retries transient failures using jittered
        exponential backoff.

        Args:
            coro_factory: Zero-argument callable returning a fresh Qdrant coroutine per attempt.
            what: Human-readable operation description used in logs and error messages.
            attempts: Maximum number of execution attempts before raising an error.
            base_delay: Initial backoff delay in seconds before the first retry.

        Returns:
            Any: The resolved return value from the awaited Qdrant coroutine.

        Raises:
            QdrantServiceError: If the service is already closed, if Qdrant rejects the
                request with a non-429 4xx HTTP error, or if all retry attempts fail.
            Exception: Any non-retryable programming error (``_NON_RETRYABLE_CODE_ERRORS``)
                or Celery ``SoftTimeLimitExceeded`` is re-raised immediately without retrying.
        """
        if self._closed:
            raise QdrantServiceError(f"{what} attempted on a closed QdrantService.")

        last_exc: BaseException | None = None

        for attempt in range(1, attempts + 1):
            try:
                async with self._semaphore:
                    result = await asyncio.wait_for(coro_factory(), timeout=10.0)
                    
                    return result

            except _NON_RETRYABLE:
                raise

            except UnexpectedResponse as exc:
                status = getattr(exc, "status_code", None)
                
                if status is not None and 400 <= status < 500 and status != 429:
                    logger.warning(
                        "QDRANT_REJECTED | op=%s status=%d err=%r action=raising_no_retry",
                        what, status, str(exc),
                    )
                
                    raise QdrantServiceError(
                        f"{what} rejected by Qdrant (HTTP {status}): {exc}"
                    ) from exc
                
                last_exc = exc

            except Exception as exc:
                last_exc = exc

            if attempt >= attempts:
                break

            delay = min(_RETRY_MAX_DELAY_SECONDS,
                        base_delay * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2),
                        )
            
            logger.warning(
                "QDRANT_RETRY | op=%s attempt=%d/%d err_type=%s delay_s=%.2f",
                what, attempt, attempts,
                type(last_exc).__name__, delay,
            )

            await asyncio.sleep(delay)

        logger.error(
            "QDRANT_FAILED | op=%s attempts=%d err_type=%s err=%r",
            what, attempts, type(last_exc).__name__, str(last_exc),
        )

        raise QdrantServiceError(
            f"{what} failed after {attempts} attempts."
        ) from last_exc


    async def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Idempotently creates required payload indexes on a Qdrant collection.

        Fetches the current collection schema and creates keyword/boolean indexes
        for ``user_id``, ``doc_id``, ``generation``, and ``committed`` only if they
        do not already exist.

        Args:
            collection_name: Name of the target Qdrant collection.

        Returns:
            None.

        Raises:
            QdrantServiceError: If fetching collection metadata or creating any missing
                payload index fails after retries or is rejected by Qdrant.
        """
        
        indexes: dict[str, PayloadSchemaType] = {"user_id": PayloadSchemaType.KEYWORD,
                                                 "doc_id": PayloadSchemaType.KEYWORD,
                                                 "generation": PayloadSchemaType.KEYWORD,
                                                 "committed": PayloadSchemaType.BOOL,
                                                }

        collection_info = await self._execute(
            lambda: self.client.get_collection(collection_name=collection_name),
            what="get collection info",
        )

        existing = set(collection_info.payload_schema.keys()) if collection_info.payload_schema else set()

        for field_name, field_schema in indexes.items():
            if field_name in existing:
                continue

            await self._execute(
                lambda fn=field_name, fs=field_schema:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=fn,
                        field_schema=fs,
                        wait=True,
                    ),
                what=f"create payload index '{field_name}'",
                            )


    async def init_collection(self,
                              collection_name: str = _DEFAULT_COLLECTION_NAME,
                              dense_vector_size: int = _DENSE_EMBEDDING_DIM,
                             ) -> None:
        """Creates the hybrid collection if missing and ensures all payload indexes exist.

        Safely handles concurrent startup races where another worker creates the
        collection simultaneously (HTTP 409 conflict).

        Args:
            collection_name: Name of the Qdrant collection to initialize.
            dense_vector_size: Dimensionality of the dense embedding vectors.

        Returns:
            None.

        Raises:
            ValueError: If ``collection_name`` is empty/whitespace or ``dense_vector_size <= 0``.
            QdrantServiceError: If checking collection existence, creating the collection
                (for non-409 errors), or creating payload indexes fails.
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")
        if dense_vector_size <= 0:
            raise ValueError(
                f"dense_vector_size must be positive, got {dense_vector_size}."
            )

        exists = await self._execute(
            lambda: self.client.collection_exists(collection_name=collection_name),
            what=f"check collection '{collection_name}'",
        )

        if not exists:
            try:
                await self._execute(
                    lambda: self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={"dense": VectorParams(
                            size=dense_vector_size,
                            distance=Distance.COSINE,
                            on_disk=True,
                        )},
                        sparse_vectors_config={"sparse": SparseVectorParams()},
                    ),
                    what=f"create collection '{collection_name}'",
                    attempts=2,
                )
                logger.info(
                    "QDRANT_COLLECTION_CREATED | collection=%s dim=%d",
                    collection_name, dense_vector_size,
                )
            except QdrantServiceError as exc:
                cause = exc.__cause__
                status = getattr(cause, "status_code", None)
                if status == 409 or "already exists" in str(cause or exc).lower():
                    logger.info(
                        "QDRANT_COLLECTION_EXISTS | collection=%s", collection_name
                    )
                else:
                    raise

        await self._ensure_payload_indexes(collection_name)
        logger.info(
            "QDRANT_COLLECTION_READY | collection=%s", collection_name
        )


    async def upsert_points(self,
                            collection_name: str,
                            points: List[PointStruct],
                           ) -> None:
        """Uploads or updates a batch of document points in Qdrant with retries.

        Waits until Qdrant persists and indexes the batch (``wait=True``) and logs
        the operation latency. Skips network calls if ``points`` is empty.

        Args:
            collection_name: Name of the target Qdrant collection.
            points: List of ``PointStruct`` objects containing vectors and payloads.

        Returns:
            None.

        Raises:
            ValueError: If ``collection_name`` is empty or whitespace.
            QdrantServiceError: If Qdrant rejects the batch or fails after exhausting retries.
        """
        
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")

        if not points:
            logger.debug("QDRANT_UPSERT_SKIPPED | collection=%s reason=empty_points", collection_name)
            return

        started_at = time.perf_counter()

        await self._execute(
            lambda: self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            ),
            what=f"upsert {len(points)} points",
        )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "QDRANT_UPSERT_COMPLETED | collection=%s count=%d latency_ms=%.1f",
            collection_name,
            len(points),
            elapsed_ms,
        )


    async def hybrid_search(self,
                            collection_name: str,
                            user_id: str,
                            dense_vector: List[float],
                            sparse_vector: Dict[str, Any],
                            limit: int = _DEFAULT_SEARCH_LIMIT,
                            doc_id: str | None = None,
                           ) -> List[Dict[str, Any]]:
        """Runs a tenant-isolated hybrid search combining dense and sparse vectors via RRF.

        Filters strictly by ``user_id`` and ``committed=True`` (and optionally ``doc_id``).
        Automatically falls back to a dense-only search if ``sparse_vector`` contains no
        active indices.

        Args:
            collection_name: Name of the Qdrant collection to query.
            user_id: Tenant identifier used to isolate search results; must match ``_ID_REGEX``.
            dense_vector: Query semantic embedding vector of length ``_DENSE_EMBEDDING_DIM``.
            sparse_vector: Dictionary containing ``indices`` and ``values`` lists of equal length.
            limit: Maximum number of matching chunks to return (must be between 1 and 10).
            doc_id: Optional document identifier to restrict search to a single document.

        Returns:
            List[Dict[str, Any]]: Ordered list of matched chunk dictionaries, each containing
                ``id``, ``text``, ``doc_id``, ``chunk_index``, ``score``, and ``metadata``.

        Raises:
            ValueError: If ``collection_name`` is empty, ``user_id`` or ``doc_id`` has an
                invalid format, ``limit`` is out of bounds, ``dense_vector`` has the wrong
                dimension, or ``sparse_vector`` ``indices`` and ``values`` lengths mismatch.
            QdrantServiceError: If Qdrant rejects the search query or fails after retries.
        """

        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")
        
        if not user_id or not _ID_REGEX.fullmatch(user_id):
            raise ValueError("Invalid user_id format.")
        
        if doc_id is not None and (not doc_id or not _ID_REGEX.fullmatch(doc_id)):
            raise ValueError("Invalid doc_id format.")
        
        if not (1 <= limit <= _MAX_SEARCH_LIMIT):
            raise ValueError(f"limit must be in [1, {_MAX_SEARCH_LIMIT}], got {limit}.")
        
        if len(dense_vector) != _DENSE_EMBEDDING_DIM:
            raise ValueError(
                f"dense_vector must have dimension {_DENSE_EMBEDDING_DIM}, got {len(dense_vector)}."
            )

        indices = sparse_vector.get("indices") or []
        values = sparse_vector.get("values") or []

        if len(indices) != len(values):
            raise ValueError("sparse_vector 'indices' and 'values' must have equal length.")

        must_conditions: list[Any] = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="committed", match=MatchValue(value=True)),
                                    ]
        if doc_id is not None:
            must_conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                                    )

        user_filter = Filter(must=must_conditions)
        prefetch_limit = max(20, limit * 3)

        started_at = time.perf_counter()

        if indices:
            formatted_sparse = SparseVector(
                indices=list(indices),
                values=list(values),
            )

            results = await self._execute(
                lambda: self.client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        Prefetch(
                            using="dense",
                            query=dense_vector,
                            filter=user_filter,
                            limit=prefetch_limit,
                        ),
                        Prefetch(
                            using="sparse",
                            query=formatted_sparse,
                            filter=user_filter,
                            limit=prefetch_limit,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
                what="Qdrant hybrid search",
            )
        
        else:
            results = await self._execute(
                lambda: self.client.query_points(
                    collection_name=collection_name,
                    using="dense",
                    query=dense_vector,
                    query_filter=user_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
                what="Qdrant dense search",
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        points = getattr(results, "points", []) or []

        documents: List[Dict[str, Any]] = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")
            
            if not text:
                continue
            
            raw_chunk_idx = payload.get("chunk_index")
            
            documents.append({
                "id": point.id,
                "text": text,
                "doc_id": payload.get("doc_id") or "",
                "filename": payload.get("filename") or "unknown",
                "chunk_index": int(raw_chunk_idx) if raw_chunk_idx is not None else -1,
                "score": float(point.score) if point.score is not None else 0.0,
                "metadata": payload.get("metadata") or {},
            })

        logger.info(
            "QDRANT_SEARCH_OK | collection=%s user_id=%s hits=%d latency_ms=%.1f",
            collection_name, user_id, len(documents), elapsed_ms,
        )

        return documents