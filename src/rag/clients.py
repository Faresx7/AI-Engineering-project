"""Lazy, thread-safe client singletons and embedding services for RAG ingestion."""
from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Optional, cast

from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types as genai_types
from qdrant_client import QdrantClient

from src.core.config import settings
from src.core.validation import IngestionError
from src.rag.ingestion.budget import _retry
from src.rag.constants import (
    _DENSE_EMBEDDING_DIM,
    _DENSE_MODEL_NAME,
    _EMBED_MAX_RETRIES,
    _RETRY_BASE_DELAY_SECONDS,
    _SPARSE_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Lazy, thread-safe per-process singletons
# ══════════════════════════════════════════════════════════════════════════════
_qdrant_lock: Lock = Lock()
_genai_lock: Lock = Lock()
_sparse_lock: Lock = Lock()
_qdrant_client: Optional[QdrantClient] = None
_genai_client: Optional[genai.Client] = None
_sparse_model: Optional[SparseTextEmbedding] = None


def _get_qdrant() -> QdrantClient:
    """Returns the process-level Qdrant client, creating it once under a lock."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        qdrant_url = getattr(settings, "QDRANT_URL", None)
        if not qdrant_url:
            raise IngestionError("QDRANT_URL is missing in configuration.")

        api_key: Optional[str] = None
        qdrant_key = getattr(settings, "QDRANT_API_KEY", None)
        if qdrant_key is not None:
            api_key = (
                qdrant_key.get_secret_value()
                if hasattr(qdrant_key, "get_secret_value")
                else str(qdrant_key)
            )

        try:
            _qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=30)
        except Exception as exc:
            raise IngestionError("Failed to initialize Qdrant client.") from exc

    return _qdrant_client


def _get_genai_client() -> genai.Client:
    """Returns the process-level Gemini client, creating it once under a lock."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client

    with _genai_lock:
        if _genai_client is not None:
            return _genai_client

        gemini_key = getattr(settings, "GEMINI_API_KEY", None)
        api_key: Optional[str] = None
        if gemini_key is not None:
            api_key = (
                gemini_key.get_secret_value()
                if hasattr(gemini_key, "get_secret_value")
                else str(gemini_key)
            )
        if not api_key:
            raise IngestionError("GEMINI_API_KEY is missing or empty in configuration.")

        try:
            _genai_client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise IngestionError("Failed to initialize Gemini client.") from exc

    return _genai_client


def _get_sparse_model() -> SparseTextEmbedding:
    """Returns the process-level local BM25 sparse model, creating it once under a lock."""
    global _sparse_model
    if _sparse_model is not None:
        return _sparse_model

    with _sparse_lock:
        if _sparse_model is None:
            try:
                _sparse_model = SparseTextEmbedding(model_name=_SPARSE_MODEL_NAME)
            except Exception as exc:
                raise IngestionError("Failed to load sparse embedding model.") from exc

    return _sparse_model


def _embed_dense_batch(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts via the online Gemini embeddings API.

    Bounded retries (network blips / transient 429s); fails fast and raises
    IngestionError once the retry budget is exhausted, and never silently
    returns a mismatched or partial result.
    """
    client = _get_genai_client()

    def _call() -> list[list[float]]:
        response = client.models.embed_content(
            model=_DENSE_MODEL_NAME,
            contents=cast(Any, texts),
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=_DENSE_EMBEDDING_DIM,
            ),
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise IngestionError(
                f"Gemini embeddings API returned {len(embeddings)} vectors "
                f"for {len(texts)} input texts."
            )
        return [list(e.values) if e.values is not None else [] for e in embeddings]

    return _retry(
        _call,
        attempts=_EMBED_MAX_RETRIES,
        base_delay=_RETRY_BASE_DELAY_SECONDS,
        what="Gemini embed_content",
    )
