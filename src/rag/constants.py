"""Constants and safety limits for the RAG ingestion pipeline."""
from __future__ import annotations

from docx.oxml.ns import qn

from src.core.config import settings

# ── Model identifiers ─────────────────────────────────────────────────────────
_DENSE_MODEL_NAME: str = getattr(settings, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
_DENSE_EMBEDDING_DIM: int = int(getattr(settings, "GEMINI_EMBEDDING_DIM", 768))
_SPARSE_MODEL_NAME: str = "Qdrant/bm25"

# ── Safety / performance limits ───────────────────────────────────────────────
_BATCH_SIZE: int = 64
_MAX_PDF_PAGES: int = 200
_MAX_TABLE_ROWS: int = 50_000
_MAX_CHUNKS_PER_DOC: int = 20_000
_MAX_RECORD_CHARS: int = 4_000
_MAX_CONSECUTIVE_EMPTY_ROWS: int = 50
_MAX_CELL_CHARS: int = 1_000
_MAX_CELLS_PER_ROW: int = 500
_MAX_TEXT_BYTES: int = 10 * 1024 * 1024
_MAX_LINE_CHARS: int = 50_000

# ── Retry policy: bounded, fail-fast after a few attempts ────────────────────
_EMBED_MAX_RETRIES: int = 3
_QDRANT_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY_SECONDS: float = 1.0

# ── Orphan generation sweep defaults ──────────────────────────────────────────
_DEFAULT_PURGE_MIN_AGE_SECONDS: float = 3600.0
_DEFAULT_PURGE_SCROLL_BATCH: int = 512
# Maximum number of distinct (user_id, doc_id) pairs to hold in memory during
# the purge sweep. If the collection has millions of documents this prevents
# the ``seen`` dict from growing without bound and eating all available RAM.
_DEFAULT_PURGE_MAX_DOCS: int = 100_000

# XML tag constants (namespace-safe; `str.endswith` on raw tags is NOT safe)
_WD_PARA_TAG: str = qn("w:p")
_WD_TABLE_TAG: str = qn("w:tbl")
