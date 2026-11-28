"""Asynchronous hybrid retrieval pipeline for multi-tenant RAG queries.

Purpose and Overview:
    Coordinates live user query retrieval by generating dense (Gemini) and
    sparse (local FastEmbed BM25) query representations in parallel, executing
    tenant-isolated hybrid search via ``QdrantService``, deduplicating results,
    and formatting clean context blocks for downstream LLM generation.

Key Components:
    - ``RetrievalError``: Domain exception raised on validation, embedding, or
      Qdrant search failures.
    - ``RetrievedChunk``: Immutable data model representing a deduplicated
      search hit and its metadata.
    - ``_safe_int``, ``_safe_float``, ``_sanitize_filename``: Defensive helpers
      for normalizing payload fields and preventing prompt format breaks.
    - ``_encode_sparse_query``: Synchronous BM25 query encoder offloaded to a
      worker thread so ONNX CPU inference never blocks the event loop.
    - ``RetrievalPipeline``: Stateless async orchestrator exposing ``retrieve``
      and ``format_context``.

Raises:
    RetrievalError: Raised when tenant or query validation fails, query
        embedding generation fails, or Qdrant hybrid search encounters an error.
    ValueError: Raised during ``RetrievalPipeline`` initialization if
        ``collection_name`` is blank or ``default_top_k`` is out of bounds.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from src.core.config import settings
from src.core.validation import IngestionError
from src.rag.clients import _embed_query_async, _get_sparse_model
from src.rag.constants import (_DEFAULT_COLLECTION_NAME,
                               _DEFAULT_SEARCH_LIMIT,
                               _MAX_SEARCH_LIMIT,
                              )
from src.services.qdrant_service import QdrantService, QdrantServiceError


# #############################################################################
# CONSTANTS & CONFIGURATION
# #############################################################################


logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_ID_REGEX = re.compile(settings.ID_PATTERN)

_MAX_QUERY_CHARS: int = 2000
_MAX_FILENAME_LEN: int = 100


# #############################################################################
# DOMAIN EXCEPTIONS & DATA MODELS
# #############################################################################


class RetrievalError(Exception):
    """Raised when live query retrieval fails at validation, embedding, or search."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Immutable representation of a single retrieved document chunk."""

    text: str
    score: float
    doc_id: str
    filename: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


# #############################################################################
# SAFE CONVERSION & SANITIZATION HELPERS
# #############################################################################


def _safe_int(value: Any, default: int = -1) -> int:
    """Convert a value to an integer safely, rejecting boolean values.

    Args:
        value (Any): Raw value to convert to an integer.
        default (int): Fallback integer returned if conversion fails or ``value``
            is a boolean.

    Returns:
        int: Parsed integer value, or ``default`` when ``value`` is a boolean
            or cannot be converted.
    """

    if isinstance(value, bool):
        return default

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float safely, rejecting booleans and NaN/Inf.

    Args:
        value (Any): Raw value to convert to a floating-point number.
        default (float): Fallback float returned on conversion error, boolean
            input, or non-finite values.

    Returns:
        float: Finite floating-point value, or ``default`` when conversion fails
            or produces a non-finite number.
    """

    if isinstance(value, bool):
        return default

    try:
        parsed = float(value)

        return parsed if math.isfinite(parsed) else default

    except (TypeError, ValueError):
        return default


def _sanitize_filename(name: Any) -> str:
    """Sanitize a filename for safe inclusion in LLM prompt context blocks.

    Normalizes whitespace, strips closing square brackets that could break
    ``[Source N | name]`` headers, truncates overly long names, and falls back
    to ``"unknown"`` when empty.

    Args:
        name (Any): Raw filename metadata value from a retrieved payload.

    Returns:
        str: Cleaned, length-bounded filename string safe for prompt formatting.
    """

    cleaned = _WHITESPACE_RE.sub(" ", str(name or "")).replace("]", "").strip()

    if not cleaned:
        return "unknown"

    if len(cleaned) > _MAX_FILENAME_LEN:
        cleaned = cleaned[:_MAX_FILENAME_LEN]

    return cleaned


# #############################################################################
# SPARSE QUERY ENCODING HELPER
# #############################################################################


def _encode_sparse_query(query: str) -> dict[str, Any]:
    """Synchronously compute BM25 sparse indices and values for a single query.

    Designed to run inside ``asyncio.to_thread`` so ONNX CPU inference never
    blocks the FastAPI event loop. Returns a dictionary matching the exact
    ``sparse_vector`` contract expected by ``QdrantService.hybrid_search``.

    Args:
        query (str): Normalized non-empty search query string.

    Returns:
        dict[str, Any]: Dictionary containing ``indices`` (list[int]) and
            ``values`` (list[float]). Returns empty lists when no vocabulary
            tokens match.

    Raises:
        IngestionError: If loading the sparse embedding model fails.
    """

    model = _get_sparse_model()
    sparse_result = next(iter(model.query_embed(query)), None)

    if sparse_result is None:
        return {"indices": [], "values": []}

    return {
        "indices": sparse_result.indices.tolist(),
        "values": sparse_result.values.tolist(),
    }


# #############################################################################
# RETRIEVAL PIPELINE SERVICE
# #############################################################################


class RetrievalPipeline:
    """Stateless, async hybrid retrieval orchestrator backed by QdrantService."""

    def __init__(self,
                 qdrant_service: QdrantService,
                 collection_name: str = _DEFAULT_COLLECTION_NAME,
                 default_top_k: int = _DEFAULT_SEARCH_LIMIT,
                ) -> None:
        """Initialize lightweight retrieval configuration without loading models.

        Heavy embedding clients and models are resolved lazily via
        ``src.rag.clients`` singletons, while Qdrant operations delegate to the
        injected ``QdrantService`` instance.

        Args:
            qdrant_service (QdrantService): Initialized ``QdrantService`` instance
                managing async Qdrant connections.
            collection_name (str): Target Qdrant collection storing tenant chunks.
            default_top_k (int): Default number of top fused chunks to return.

        Raises:
            ValueError: If ``collection_name`` is blank or ``default_top_k`` is
                outside ``[1, _MAX_SEARCH_LIMIT]``.
        """

        cleaned_collection = (collection_name or "").strip()

        if not cleaned_collection:
            raise ValueError("collection_name must be a non-empty string.")

        if not (1 <= default_top_k <= _MAX_SEARCH_LIMIT):
            raise ValueError(f"default_top_k must be between 1 and {_MAX_SEARCH_LIMIT}.")

        self._qdrant_service = qdrant_service
        self._collection_name = cleaned_collection
        self._default_top_k = default_top_k

    async def retrieve(self,
                       query: str,
                       user_id: str,
                       *,
                       top_k: Optional[int] = None,
                       doc_id: Optional[str] = None,
                      ) -> list[RetrievedChunk]:
        """Retrieve the most relevant tenant-scoped chunks using hybrid search.

        Validates tenant identity and query text, computes dense (Gemini) and
        sparse (BM25) query representations concurrently, executes filtered
        hybrid search in Qdrant, and deduplicates overlapping hits.

        Args:
            query (str): Raw user question from chat.
            user_id (str): Tenant/owner identifier used to strictly isolate Qdrant
                search.
            top_k (Optional[int]): Optional override for number of chunks to return.
            doc_id (Optional[str]): Optional document ID to restrict search to a
                single file.

        Returns:
            list[RetrievedChunk]: Ranked, deduplicated chunks ready for prompt
                assembly. Returns an empty list if ``query`` has no searchable
                alphanumeric characters or no matching chunks exist.

        Raises:
            RetrievalError: If ``user_id`` or ``doc_id`` is invalid, ``top_k`` is
                out of bounds, embedding generation fails, or Qdrant search fails.
        """

        clean_user_id = (user_id or "").strip()

        if not clean_user_id or not _ID_REGEX.fullmatch(clean_user_id):
            raise RetrievalError("Valid user_id is required for tenant-isolated retrieval.")

        clean_doc_id = (doc_id or "").strip() or None

        if clean_doc_id is not None and not _ID_REGEX.fullmatch(clean_doc_id):
            raise RetrievalError("Invalid doc_id format.")

        effective_top_k = self._default_top_k if top_k is None else top_k

        if not (1 <= effective_top_k <= _MAX_SEARCH_LIMIT):
            raise RetrievalError(
                f"top_k must be between 1 and {_MAX_SEARCH_LIMIT}, got {effective_top_k}."
            )

        clean_query = _WHITESPACE_RE.sub(" ", (query or "")).strip()

        if not clean_query or not any(ch.isalnum() for ch in clean_query):
            return []

        if len(clean_query) > _MAX_QUERY_CHARS:
            original_len = len(clean_query)
            clean_query = clean_query[:_MAX_QUERY_CHARS]
            logger.debug("RETRIEVAL_QUERY_TRUNCATED | original_len=%d max=%d",
                         original_len,
                         _MAX_QUERY_CHARS,
                        )

        started_at = time.perf_counter()

        dense_task = asyncio.create_task(_embed_query_async(clean_query))
        sparse_task = asyncio.create_task(asyncio.to_thread(_encode_sparse_query, clean_query))

        try:
            dense_vector, sparse_vector = await asyncio.gather(dense_task, sparse_task)

        except IngestionError as exc:
            await self._cancel_pending(dense_task, sparse_task)
            logger.error("RETRIEVAL_EMBED_FAILED | user_id=%s err=%r",
                         clean_user_id,
                         str(exc),
                        )

            raise RetrievalError("Failed to generate query embeddings.") from exc

        except Exception as exc:
            await self._cancel_pending(dense_task, sparse_task)
            logger.error("RETRIEVAL_EMBED_UNEXPECTED | user_id=%s err_type=%s err=%r",
                         clean_user_id,
                         type(exc).__name__,
                         str(exc),
                        )

            raise RetrievalError("Unexpected error during query embedding.") from exc

        try:
            raw_hits = await self._qdrant_service.hybrid_search(collection_name=self._collection_name,
                                                                user_id=clean_user_id,
                                                                dense_vector=dense_vector,
                                                                sparse_vector=sparse_vector,
                                                                limit=effective_top_k,
                                                                doc_id=clean_doc_id,
                                                               )

        except QdrantServiceError as exc:
            logger.error("RETRIEVAL_SEARCH_FAILED | user_id=%s err=%r",
                         clean_user_id,
                         str(exc),
                        )

            raise RetrievalError("Hybrid search failed in Qdrant.") from exc

        except Exception as exc:
            logger.error("RETRIEVAL_SEARCH_UNEXPECTED | user_id=%s err_type=%s err=%r",
                         clean_user_id,
                         type(exc).__name__,
                         str(exc),
                        )

            raise RetrievalError("Unexpected error during hybrid search.") from exc

        chunks = self._normalize_and_deduplicate(raw_hits, effective_top_k)

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info("RETRIEVAL_OK | user_id=%s hits=%d latency_ms=%.1f",
                    clean_user_id,
                    len(chunks),
                    elapsed_ms,
                   )

        return chunks

    @staticmethod
    async def _cancel_pending(*tasks: asyncio.Future[Any]) -> None:
        """Cancel any still-running tasks and await their cleanup cleanly.

        Args:
            *tasks (asyncio.Future[Any]): Asyncio tasks or futures to cancel if
                not yet completed.
        """

        for task in tasks:

            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _normalize_and_deduplicate(raw_hits: Sequence[dict[str, Any]],
                                   limit: int,
                                  ) -> list[RetrievedChunk]:
        """Convert raw Qdrant hit dictionaries into deduplicated ``RetrievedChunk`` items.

        Preserves nested ``metadata`` dictionaries returned by ``QdrantService``
        as well as any extra top-level payload keys while guarding against
        malformed numeric fields.

        Args:
            raw_hits (Sequence[dict[str, Any]]): List of hit dictionaries returned
                by ``hybrid_search``.
            limit (int): Maximum number of unique chunks to keep.

        Returns:
            list[RetrievedChunk]: Ordered unique chunks with non-empty text.
        """

        results: list[RetrievedChunk] = []
        seen_keys: set[tuple[str, int] | str] = set()

        reserved_keys = {
            "id",
            "score",
            "payload",
            "text",
            "page_content",
            "doc_id",
            "chunk_index",
            "filename",
            "user_id",
            "metadata",
        }

        for hit in raw_hits:

            if not isinstance(hit, dict):
                continue

            nested_payload = hit.get("payload")
            payload: dict[str, Any] = nested_payload if isinstance(nested_payload, dict) else hit

            score = _safe_float(hit.get("score", 0.0))
            text = str(payload.get("text") or payload.get("page_content") or "").strip()

            if not text:
                continue

            doc_id = str(payload.get("doc_id") or "")
            chunk_index = _safe_int(payload.get("chunk_index"), -1)

            nested_meta = payload.get("metadata")
            base_meta: dict[str, Any] = dict(nested_meta) if isinstance(nested_meta, dict) else {}

            raw_filename = payload.get("filename") or base_meta.get("filename")
            filename = _sanitize_filename(raw_filename)

            dedup_key: tuple[str, int] | str = (
                (doc_id, chunk_index) if doc_id and chunk_index >= 0 else text
            )

            if dedup_key in seen_keys:
                continue

            seen_keys.add(dedup_key)

            extra_meta = {
                **base_meta,
                **{k: v for k, v in payload.items() if k not in reserved_keys},
            }

            results.append(RetrievedChunk(text=text,
                                          score=score,
                                          doc_id=doc_id,
                                          filename=filename,
                                          chunk_index=chunk_index,
                                          metadata=extra_meta,
                                         ))

            if len(results) >= limit:
                break

        return results

    @staticmethod
    def format_context(chunks: Sequence[RetrievedChunk]) -> str:
        """Format retrieved chunks into a clean context string for LLM prompts.

        Args:
            chunks (Sequence[RetrievedChunk]): Ordered sequence of retrieved
                document chunks.

        Returns:
            str: Formatted context string separated by dividers, or an empty
                string when ``chunks`` is empty.
        """

        if not chunks:
            return ""

        blocks: list[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            clean_text = (chunk.text or "").strip()

            if not clean_text:
                continue

            safe_filename = _sanitize_filename(chunk.filename)
            blocks.append(f"[Source {idx} | {safe_filename}]\n{clean_text}")

        return "\n\n---\n\n".join(blocks)