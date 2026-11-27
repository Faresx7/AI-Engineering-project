"""Constants, safety guardrails, and Celery policy limits for the RAG pipeline.

What This File Does:
    Centralizes all model identifiers, file-parsing safety caps, retry counts,
    Qdrant collection defaults, and Celery worker timeout/lock thresholds so
    no magic numbers are hardcoded across pipeline modules.

Sections:
    1. Model Identifiers: Dense and sparse embedding model names and dimensions.
    2. Parsing & Chunking Limits: Size caps for PDFs, text files, and chunks.
    3. Retry Policy: Bounded retry attempts and backoff delays for external APIs.
    4. Qdrant & Orphan Sweep Defaults: Collection name and memory caps.
    5. Celery Worker & Beat Policy: Timeouts, locks, non-retryable errors, and purge timings.
    6. File Extension Sets: Immutable extension groups for MarkItDown and zip checks.
"""
from __future__ import annotations

from src.core.config import settings


# ##############################################################################
# Model identifiers
# ##############################################################################

_DENSE_MODEL_NAME: str = getattr(settings, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
_DENSE_EMBEDDING_DIM: int = int(getattr(settings, "GEMINI_EMBEDDING_DIM", 768))
_SPARSE_MODEL_NAME: str = "Qdrant/bm25"


# ##############################################################################
# Qdrant client & hybrid search operational constants
# ##############################################################################
_QDRANT_DEFAULT_TIMEOUT_SECONDS: float = 15.0
_QDRANT_MAX_CONCURRENCY: int = 32
_RETRY_MAX_DELAY_SECONDS: float = 30.0

_DEFAULT_SEARCH_LIMIT: int = 5
_MAX_SEARCH_LIMIT: int = 10

# ##############################################################################
# Parsing & chunking safety limits
# ##############################################################################

_BATCH_SIZE: int = 64
_MAX_PDF_PAGES: int = 200
_MAX_CHUNKS_PER_DOC: int = 20_000
_MAX_RECORD_CHARS: int = 4_000
_MAX_TEXT_BYTES: int = 10 * 1024 * 1024

# ##############################################################################
# Embedding & Client Runtime Limits (clients.py)
# ##############################################################################
# Cap ONNX threads so FastEmbed does not starve FastAPI/Qdrant on small VPS CPUs
_SPARSE_NUM_THREADS: int = 2

# Gemini embed_content API safe batch limit per request (hard API cap is 100)
_GEMINI_MAX_EMBED_BATCH: int = 95

# HTTP client timeout in milliseconds for google-genai HttpOptions
_GEMINI_TIMEOUT_MS: int = 20_000

# Outer asyncio timeout (seconds) for live user query embeddings (fail fast)
_GEMINI_QUERY_TIMEOUT_SECONDS: float = 15.0

# Max concurrent live query embeddings from FastAPI to prevent 429 rate-limit spikes
_MAX_CONCURRENT_QUERY_EMBEDS: int = 20

# ##############################################################################
# Retry policy (bounded, fail-fast after a few attempts)
# ##############################################################################

_EMBED_MAX_RETRIES: int = 3
_QDRANT_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY_SECONDS: float = 2.0


# ##############################################################################
# Qdrant & orphan generation sweep defaults
# ##############################################################################

_DEFAULT_COLLECTION_NAME: str = getattr(settings, "QDRANT_COLLECTION_NAME", "rag_documents")
_DEFAULT_PURGE_SCROLL_BATCH: int = 512

# Cap distinct (user_id, doc_id) pairs held in RAM per sweep to prevent OOM
_DEFAULT_PURGE_MAX_DOCS: int = 100_000


# ##############################################################################
# Celery worker & Beat maintenance policy
# ##############################################################################

# Code bugs that will never succeed on retry and should fail immediately
_NON_RETRYABLE_CODE_ERRORS: tuple[type[Exception], ...] = (NotImplementedError,
                                                           AttributeError,
                                                           TypeError,
                                                           ValueError,
                                                           KeyError,
                                                          )

# Ingestion task execution limits and Redis per-document lock TTL
_INGEST_SOFT_TIME_LIMIT: int = settings.INGESTION_SOFT_TIME_LIMIT_SECONDS
_INGEST_HARD_TIME_LIMIT: int = settings.INGESTION_HARD_TIME_LIMIT_SECONDS
_LOCK_TTL_SECONDS: int = _INGEST_HARD_TIME_LIMIT + 45

# Guarantee orphan grace period is always safely above the hard task kill limit
_DEFAULT_PURGE_MIN_AGE_SECONDS: float = max(3600.0, float(_INGEST_HARD_TIME_LIMIT + 600))


_PURGE_FILE_SOFT_TIME_LIMIT: int = settings.PURGE_FILE_SOFT_TIME_LIMIT
_PURGE_FILE_HARD_TIME_LIMIT: int = settings.PURGE_FILE_HARD_TIME_LIMIT

# Periodic orphan-generation purge timeouts and Beat schedule timings
_PURGE_GEN_SOFT_TIME_LIMIT: int = settings.PURGE_GEN_SOFT_TIME_LIMIT
_PURGE_GEN_HARD_TIME_LIMIT: int = settings.PURGE_GEN_HARD_TIME_LIMIT

_PURGE_GEN_SCHEDULE_MINUTES: int = 30
_PURGE_GEN_TASK_EXPIRES_SECONDS: int = max(300, (_PURGE_GEN_SCHEDULE_MINUTES * 60) - 300)


# ##############################################################################
# Immutable file extension sets
# ##############################################################################

# Extensions routed to the MarkItDown converter
_MARKITDOWN_EXTENSIONS: frozenset[str] = frozenset({".docx",
                                                    ".xlsx",
                                                    ".pptx",
                                                    ".html",
                                                    ".json",
                                                    ".htm",
                                                    ".xls",
                                                    ".csv",
                                                    ".txt",
                                                    ".msg",
                                                    ".md",
                                                   })

# Archive-based Office extensions requiring zip-bomb expansion checks
_ZIP_EXTENSIONS: frozenset[str] = frozenset({".docx",
                                             ".xlsx",
                                             ".pptx",
                                            })