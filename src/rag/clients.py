"""Lazy, thread-safe client singletons and embedding helpers for RAG ingestion and retrieval.

Purpose:
    Centralizes process-level client initialization (Qdrant, Google GenAI, and local
    FastEmbed BM25) using double-checked locking so heavy models and network pools
    are instantiated once per worker/app process. Also provides synchronous batch
    document embedding for Celery ingestion and asynchronous, concurrency-bounded
    query embedding for FastAPI retrieval.

Raises:
    IngestionError: Raised when required configuration values are missing, client
        initialization fails, input validation fails, or embedding API calls
        return malformed data or exhaust their retry budget.
"""
from __future__ import annotations

from typing import Any, Optional, cast
from threading import Lock
import asyncio
import logging
import random
import time

from google.genai import types as genai_types
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from google import genai

from src.core.config import settings
from src.core.validation import IngestionError
from src.rag.constants import (
    _DENSE_EMBEDDING_DIM,
    _DENSE_MODEL_NAME,
    _EMBED_MAX_RETRIES,
    _GEMINI_MAX_EMBED_BATCH,
    _GEMINI_QUERY_TIMEOUT_SECONDS,
    _GEMINI_TIMEOUT_MS,
    _MAX_CONCURRENT_QUERY_EMBEDS,
    _RETRY_BASE_DELAY_SECONDS,
    _SPARSE_MODEL_NAME,
    _SPARSE_NUM_THREADS,
)

from src.rag.ingestion.pipeline_utils import _retry

logger = logging.getLogger(__name__)


# ##############################################################################
# Process-Level Locks, Singleton Caches, and Loop-Bound Semaphores
# ##############################################################################
_qdrant_lock: Lock = Lock()
_genai_lock: Lock = Lock()
_sparse_lock: Lock = Lock()

_qdrant_client: Optional[QdrantClient] = None
_genai_client: Optional[genai.Client] = None
_sparse_model: Optional[SparseTextEmbedding] = None

_query_embed_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


# ##############################################################################
# Concurrency & Client Initialization Helpers
# ##############################################################################
def _get_query_embed_semaphore() -> asyncio.Semaphore:
    """Returns a concurrency semaphore bound to the currently running event loop.

    Creates the semaphore lazily on first access within the active asyncio event
    loop to prevent cross-loop binding issues during startup or test reloads.

    Args:
        None.

    Returns:
        asyncio.Semaphore: A semaphore capped at ``_MAX_CONCURRENT_QUERY_EMBEDS``
            permits for the active event loop.

    Raises:
        RuntimeError: If called when no asyncio event loop is running in the
            current thread.
    """
    loop = asyncio.get_running_loop()
    semaphore = _query_embed_semaphores.get(loop)

    if semaphore is None:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_QUERY_EMBEDS)
        _query_embed_semaphores[loop] = semaphore

    return semaphore


def _get_qdrant() -> QdrantClient:
    """Returns the process-level sync Qdrant client, creating it once under a lock.

    Uses double-checked locking so concurrent worker threads share a single
    Qdrant connection pool without redundant initialization.

    Args:
        None.

    Returns:
        QdrantClient: The initialized synchronous Qdrant client instance.

    Raises:
        IngestionError: If ``QDRANT_URL`` is missing/blank in settings or if
            ``QdrantClient`` fails to initialize.
    """
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        qdrant_url = (settings.QDRANT_URL or "").strip()
        if not qdrant_url:
            raise IngestionError("QDRANT_URL is missing in configuration.")

        api_key = (
            settings.QDRANT_API_KEY.get_secret_value()
            if settings.QDRANT_API_KEY
            else None
        )

        try:
            _qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=30)
            logger.info("QDRANT_SYNC_CLIENT_INITIALIZED | url=%s", qdrant_url)

        except Exception as exc:
            raise IngestionError("Failed to initialize Qdrant client.") from exc

    return _qdrant_client


def _get_genai_client() -> genai.Client:
    """Returns the process-level Gemini client, creating it once under a lock.

    Uses double-checked locking and configures a hard HTTP timeout via
    ``HttpOptions`` so stalled API requests do not hang workers indefinitely.

    Args:
        None.

    Returns:
        genai.Client: The initialized Google GenAI client instance.

    Raises:
        IngestionError: If ``GEMINI_API_KEY`` is missing/empty in settings or if
            ``genai.Client`` initialization fails.
    """
    global _genai_client

    if _genai_client is not None:
        return _genai_client

    with _genai_lock:
        if _genai_client is not None:
            return _genai_client

        api_key = (
            settings.GEMINI_API_KEY.get_secret_value().strip()
            if settings.GEMINI_API_KEY else "")

        if not api_key:
            raise IngestionError("GEMINI_API_KEY is missing or empty in configuration.")

        try:
            _genai_client = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(timeout=int(_GEMINI_TIMEOUT_MS)),
            )
            logger.info("GENAI_CLIENT_INITIALIZED")
        
        except Exception as exc:
            raise IngestionError("Failed to initialize Gemini client.") from exc

    return _genai_client


def _get_sparse_model() -> SparseTextEmbedding:
    """Returns the process-level BM25 sparse model with bounded CPU threads.

    Loads the local FastEmbed ONNX model once under a lock while capping ONNX
    runtime threads so CPU-bound sparse encoding does not starve FastAPI or Qdrant.

    Args:
        None.

    Returns:
        SparseTextEmbedding: The loaded local BM25 sparse embedding model.

    Raises:
        IngestionError: If loading the sparse embedding model fails.
    """
    global _sparse_model

    if _sparse_model is not None:
        return _sparse_model

    with _sparse_lock:
        if _sparse_model is not None:
            return _sparse_model

        try:
            _sparse_model = SparseTextEmbedding(model_name=_SPARSE_MODEL_NAME,
                                                threads=_SPARSE_NUM_THREADS,)

        except Exception as exc:
            raise IngestionError("Failed to load sparse embedding model.") from exc

    return _sparse_model


def close_clients() -> None:
    """Safely closes cached client connections and resets singletons on shutdown.

    Releases network sockets held by Qdrant and GenAI clients and clears the
    cached BM25 model reference so fresh instances are created on next access
    (useful during worker shutdown, hot-reload, or test teardowns).

    Args:
        None.

    Returns:
        None.

    Raises:
        None: Any exception raised while closing a client is caught and logged
            as a warning so shutdown cleanup continues uninterrupted.
    """
    global _qdrant_client, _genai_client, _sparse_model

    with _qdrant_lock:
        if _qdrant_client is not None:
            
            try:
                _qdrant_client.close()
            
            except Exception as exc:
                logger.warning("QDRANT_SYNC_CLOSE_FAILED | err=%r", str(exc))
            
            finally:
                _qdrant_client = None

    with _genai_lock:
        if _genai_client is not None:
            
            try:
                _genai_client.close()
            
            except Exception as exc:
                logger.warning("GENAI_CLIENT_CLOSE_FAILED | err=%r", str(exc))
            
            finally:
                _genai_client = None

    # SparseTextEmbedding holds no network socket to close; dropping the
    # singleton reference ensures a clean instance on reload or test fixtures.
    with _sparse_lock:
        _sparse_model = None


# ##############################################################################
# Embedding Execution Helpers (Sync Ingestion & Async Query Retrieval)
# ##############################################################################
def _embed_dense_batch(texts: list[str]) -> list[list[float]]:
    """Synchronously embeds document chunks for Celery ingestion with retries.

    Splits large inputs into sub-batches capped at ``_GEMINI_MAX_EMBED_BATCH``
    items, calls Gemini ``embed_content`` with ``task_type="RETRIEVAL_DOCUMENT"``,
    and validates vector counts and dimensions so malformed responses trigger a
    retry instead of failing downstream in Qdrant.

    Args:
        texts: Ordered list of document chunk strings to embed.

    Returns:
        list[list[float]]: Dense embedding vectors aligned one-to-one with
            ``texts``, where each vector has length ``_DENSE_EMBEDDING_DIM``.
            Returns an empty list immediately if ``texts`` is empty.

    Raises:
        IngestionError: If client initialization fails, the API returns a
            mismatched vector count or invalid dimension, or all retry attempts
            are exhausted.
    """
    if not texts:
        return []

    client = _get_genai_client()
    started_at = time.perf_counter()
    all_vectors: list[list[float]] = []

    for offset in range(0, len(texts), _GEMINI_MAX_EMBED_BATCH):
        chunk = texts[offset:offset + _GEMINI_MAX_EMBED_BATCH]

        def _call(batch: list[str] = chunk) -> list[list[float]]:
            response = client.models.embed_content(
                model=_DENSE_MODEL_NAME,
                contents=cast(Any, batch),
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=_DENSE_EMBEDDING_DIM,
                ),
            )

            embeddings = response.embeddings or []
            
            if len(embeddings) != len(batch):
                raise IngestionError(
                    f"Gemini embeddings API returned {len(embeddings)} vectors "
                    f"for {len(batch)} input texts."
                )

            vectors: list[list[float]] = []
            for idx, emb in enumerate(embeddings):
                if emb.values is None or len(emb.values) != _DENSE_EMBEDDING_DIM:
                    got_dim = len(emb.values) if emb.values is not None else 0
                   
                    raise IngestionError(
                        f"Gemini embedding at index {idx} has invalid dimension {got_dim} "
                        f"(expected {_DENSE_EMBEDDING_DIM}).")
                
                vectors.append(list(emb.values))

            return vectors

        all_vectors.extend(
            _retry(
                _call,
                attempts=_EMBED_MAX_RETRIES,
                base_delay=_RETRY_BASE_DELAY_SECONDS,
                what="Gemini embed_content",
            )
        )

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    logger.info(
        "GEMINI_EMBED_BATCH_OK | count=%d latency_ms=%.1f",
        len(all_vectors),
        elapsed_ms,
    )
   
    return all_vectors


async def _embed_query_async(query: str) -> list[float]:
    """Asynchronously embeds a single user search query for FastAPI retrieval.

    Uses ``task_type="RETRIEVAL_QUERY"`` and non-blocking ``client.aio`` guarded
    by a per-loop concurrency semaphore and an outer timeout. Deterministic
    ``IngestionError`` failures (empty vector or dimension mismatch) fail fast
    without retrying, whereas transient network or rate-limit errors retry with
    jittered exponential backoff.

    Args:
        query: Raw user search query string.

    Returns:
        list[float]: A single dense query embedding vector of length
            ``_DENSE_EMBEDDING_DIM``.

    Raises:
        IngestionError: If ``query`` is empty/whitespace, the API returns an
            empty or wrong-sized vector, or transient errors persist after
            ``_EMBED_MAX_RETRIES`` attempts.
    """
    if not query or not query.strip():
        raise IngestionError("Query text must be non-empty.")

    client = _get_genai_client()
    semaphore = _get_query_embed_semaphore()
    started_at = time.perf_counter()
    last_exc: BaseException | None = None

    for attempt in range(1, _EMBED_MAX_RETRIES + 1):
        try:
            async with semaphore:
                response = await asyncio.wait_for(
                    client.aio.models.embed_content(
                        model=_DENSE_MODEL_NAME,
                        contents=query,
                        config=genai_types.EmbedContentConfig(
                            task_type="RETRIEVAL_QUERY",
                            output_dimensionality=_DENSE_EMBEDDING_DIM,
                        ),
                    ),
                    timeout=_GEMINI_QUERY_TIMEOUT_SECONDS,
                )

            embeddings = response.embeddings or []
           
            if not embeddings or embeddings[0].values is None:
                raise IngestionError("Gemini returned an empty embedding for query.")

            vector = list(embeddings[0].values)
           
            if len(vector) != _DENSE_EMBEDDING_DIM:
                raise IngestionError(
                    f"Query embedding dimension mismatch: got {len(vector)}, "
                    f"expected {_DENSE_EMBEDDING_DIM}."
                )

            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            logger.debug("GEMINI_QUERY_EMBED_OK | latency_ms=%.1f", elapsed_ms)
           
            return vector

        except IngestionError:
            logger.warning(
                "GEMINI_QUERY_EMBED_NON_RETRYABLE | action=raising_no_retry",
            )
           
            raise

        except Exception as exc:
            last_exc = exc
           
            if attempt >= _EMBED_MAX_RETRIES:
                break

            delay = _RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            logger.warning(
                "GEMINI_QUERY_EMBED_RETRY | attempt=%d/%d err_type=%s delay_s=%.2f",
                attempt,
                _EMBED_MAX_RETRIES,
                type(exc).__name__,
                delay,
            )

            await asyncio.sleep(delay)

    raise IngestionError(
        f"Query embedding failed after {_EMBED_MAX_RETRIES} attempts."
        ) from last_exc