"""Synchronous ingestion pipeline. Runs ONLY inside the Celery worker.

Everything here is deliberately synchronous (no async/await): it is CPU- and
IO-heavy work (parsing files, running ML models) that must never run on the
FastAPI event loop. The worker is a separate process, so even a 10-second PDF
parse cannot slow down your Instagram/WhatsApp webhooks.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional, cast, Sequence
from dataclasses import dataclass
from pathlib import Path
import threading
import logging
import json
import uuid
import csv
import io
from contextlib import contextmanager

from google import genai
from qdrant_client import QdrantClient
from src.core.config import settings

from threading import Lock


import chardet
import docx
import pypdf
import openpyxl
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    Range,
    SparseVector,
)

from src.core.config import settings
from src.core.validation import (
    IngestionError,
    check_zip_expansion,
    sniff_matches_extension,
)
from src.core.config import settings


logger = logging.getLogger(__name__)


DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"



# ── Safety limits: every loop over untrusted input needs an upper bound ──────────
BATCH_SIZE = 64            # chunks embedded + sent to Qdrant per round trip
MAX_PDF_PAGES = 500
MAX_TABLE_ROWS = 50_000    # data rows per Excel/CSV/JSON-list file
MAX_CHUNKS_PER_DOC = 20_000
MAX_RECORD_CHARS = 4_000   # a single row/record longer than this gets split anyway
MAX_CONSECUTIVE_EMPTY_ROWS = 50

ProgressCallback = Callable[[int, int], None]


# ══════════════════════════════════════════════════════════════════════════════
# Lazy, per-process singletons (models and the Qdrant client are expensive)
# ══════════════════════════════════════════════════════════════════════════════
#making initial variables stores model for lazy loading
_genai_client: Optional[genai.Client] = None
_qdrant: Optional[QdrantClient] = None


def _get_genai_client() -> genai.Client:
    """Lazy initializer for Google GenAI Client (for Dense Embeddings)."""
    global _genai_client

    if _genai_client is None:
        gemini_api_key = settings.GEMINI_API_KEY.get_secret_value() if settings.GEMINI_API_KEY else None

        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing or empty in configuration.")

        try:
            _genai_client = genai.Client(api_key=gemini_api_key)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize GenAI Client: {exc}") from exc

    return _genai_client


def _get_qdrant() -> QdrantClient:
    """Lazy initializer for Qdrant Client."""
    global _qdrant

    if _qdrant is None:
        if not getattr(settings, "QDRANT_URL", None):
            raise ValueError("QDRANT_URL is missing in configuration.")

        api_key = (
            settings.QDRANT_API_KEY.get_secret_value()
            if settings.QDRANT_API_KEY
            else None
        )

    try:
        _qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
            timeout=30
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Qdrant Client: {exc}") from exc
    
    return _qdrant


# def get_dense_embeddings(texts: list[str]) -> list[list[float]]:
#     """Fetch dense embeddings directly from Google API without local GPU/RAM models."""
#     client = _get_genai_client()
#     response = client.models.embed_content(
#         model="text-embedding-004",
#         contents=cast(Any, texts),
#     )
#     embeddings = response.embeddings or []
#     return [cast(list[float], embedding.values) for embedding in embeddings]

#  ══════════════════════════════════════════════════════════════════════════════
#  Counter Class
#  ══════════════════════════════════════════════════════════════════════════════
class _RowBudget:
    """Counts rows across a whole file (all sheets) and refuses to go past the limit."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._left = limit
        self._lock = Lock()

    def spend(self) -> None:
        with self._lock:
            if self._left <= 0:
                raise IngestionError(
                    f"The file has more than {self._limit:,} rows. Please split it into smaller files."
                )
            self._left -= 1


MAX_TABLE_ROWS = 50000
MAX_CONSECUTIVE_EMPTY_ROWS = 30


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_text(path: Path, max_bytes: int = 10 * 1024 * 1024) -> str:
    """Read text file safely with size limit and fast fallback encoding detection."""
    file_size = path.stat().st_size
    if file_size > max_bytes:
        raise IngestionError(f"Text file size exceeds maximum limit of {max_bytes // (1024 * 1024)} MB.")

    raw = path.read_bytes()

    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass

    sample = raw[:32768]
    detected = chardet.detect(sample)
    encoding = detected.get("encoding") or "latin-1"
    logger.warning("File %s is not standard UTF-8. Using detected encoding: %s", path.name, encoding)

    try:
        return raw.decode(encoding)
    except Exception as exc:
        raise IngestionError("Unable to decode text file with supported encodings.") from exc


def _is_row_empty(row: tuple) -> bool:
    """Check whether a row contains only None values or empty whitespace strings."""
    if not row:
        return True
    return all(
        cell is None or (isinstance(cell, str) and not cell.strip())
        for cell in row
    )


def _is_potential_header(cells: list) -> bool:
    """Smart header check: Skips title banners spanning across columns while preserving 1-column tables."""
    non_empty = [c for c in cells if c is not None and str(c).strip() != ""]
    if not non_empty:
        return False

    # If row spans multiple columns in the sheet but only 1 cell is filled -> Title Banner
    if len(cells) > 1 and len(non_empty) == 1:
        return False

    return True


def _trim_trailing_nones(cells: list) -> list:
    """Remove trailing None/empty cells from the right side of a row."""
    while cells and (cells[-1] is None or str(cells[-1]).strip() == ""):
        cells.pop()
    return cells


def _table_documents(
    rows: Iterable[Sequence[Any]],
    filename: str,
    extra_meta: dict,
    budget: _RowBudget,
    max_cell_chars: int = 1000,
) -> Iterator[Document]:
    """Converts table rows into Documents with smart Header/Title detection and duplicate handling."""
    headers: Optional[list[str]] = None

    for row_idx, row in enumerate(rows, start=1):
        cells = list(row)

        if _is_row_empty(tuple(cells)):
            continue

        # 1. Smart Header Resolution
        if headers is None:
            # Check potential header on UNTRIMMED cells to detect title banners correctly
            if not _is_potential_header(cells):
                logger.debug("Skipping title/banner row at index %d in file %s", row_idx, filename)
                continue

            trimmed_cells = _trim_trailing_nones(cells.copy())
            if not trimmed_cells:
                continue

            raw_headers = []
            used_headers: set[str] = set()
            last_valid_header = "col"

            for col_idx, raw_cell in enumerate(trimmed_cells):
                raw_str = str(raw_cell).strip() if raw_cell is not None else ""

                if not raw_str:
                    clean_val = f"{last_valid_header}_sub_{col_idx}"
                else:
                    clean_val = raw_str
                    last_valid_header = clean_val

                candidate_val = clean_val
                counter = 1
                while candidate_val in used_headers:
                    candidate_val = f"{clean_val}_{counter}"
                    counter += 1

                used_headers.add(candidate_val)
                raw_headers.append(candidate_val)

            headers = raw_headers
            continue

        # 2. Extract Data Rows
        pairs = []
        for idx, val in enumerate(cells):
            if val is None:
                continue

            val_str = str(val).strip()
            if not val_str:
                continue

            if len(val_str) > max_cell_chars:
                val_str = val_str[:max_cell_chars] + "... [truncated]"

            header_name = headers[idx] if idx < len(headers) else f"extra_col_{idx}"
            pairs.append((header_name, val_str))

        if not pairs:
            continue

        budget.spend()

        yield Document(
            page_content="\n".join(f"{key}: {value}" for key, value in pairs),
            metadata={
                "source": filename,
                **extra_meta,
                "row": row_idx,
                "pre_chunked": True,
            },
        )


def _stream_rows(sheet, sheet_name: str) -> Iterator[tuple]:
    """Yields non-empty rows from a worksheet while respecting consecutive empty row limits."""
    consecutive_empty_count = 0
    has_started_data = False

    for row in sheet.iter_rows(values_only=True):
        if _is_row_empty(row):
            # Skip top padding empty rows without triggering MAX_CONSECUTIVE_EMPTY_ROWS
            if not has_started_data:
                continue

            consecutive_empty_count += 1
            if consecutive_empty_count >= MAX_CONSECUTIVE_EMPTY_ROWS:
                logger.info(
                    "Reached max consecutive empty rows (%d) in sheet '%s'. Stopping sheet.",
                    MAX_CONSECUTIVE_EMPTY_ROWS,
                    sheet_name,
                )
                break
            continue

        has_started_data = True
        consecutive_empty_count = 0
        yield row


@contextmanager
def _open_workbook(path: Path):
    """Safely opens and closes openpyxl workbook without masking downstream runtime errors."""
    workbook = None
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise IngestionError("Could not open the Excel file (corrupt or unsupported).") from exc

    try:
        yield workbook
    finally:
        if workbook is not None:
            try:
                workbook.close()
            except Exception as close_exc:
                logger.warning("Failed to close workbook cleanly: %s", close_exc)


# ══════════════════════════════════════════════════════════════════════════════
# file loaders
# ══════════════════════════════════════════════════════════════════════════════

def _excel_loader(path: Path, filename: str) -> Iterator[Document]:
    """Production-ready Excel loader with hidden sheet skipping and streaming safety."""
    check_zip_expansion(path)

    with _open_workbook(path) as workbook:
        budget = _RowBudget(MAX_TABLE_ROWS)

        for sheet in workbook.worksheets:
            if sheet.sheet_state != "visible":
                logger.info("Skipping non-visible sheet '%s' (state: %s)", sheet.title, sheet.sheet_state)
                continue

            yield from _table_documents(
                rows=_stream_rows(sheet, sheet.title),
                filename=filename,
                extra_meta={"sheet": sheet.title},
                budget=budget,
            )



def _csv_loader(path: Path, filename: str) -> Iterator[Document]:
    """Streams CSV rows directly from disk with automatic delimiter detection."""
    budget = _RowBudget(MAX_TABLE_ROWS)

    try:
        # 1. Detect CSV dialect (delimiter) safely from the first 8KB sample
        with open(path, mode="r", encoding="utf-8-sig", errors="replace") as sample_file:
            
            sample_bytes = sample_file.read(8192)
            try:
                dialect = csv.Sniffer().sniff(sample_bytes)
            
            except csv.Error:
                dialect = csv.excel  # Fallback to standard comma separator

        # 2. Open true file stream from disk not from RAM (Memory remains O(1))
        file_stream = open(path, mode="r", encoding="utf-8-sig", errors="replace")
    
    except Exception as exc:
        raise IngestionError(f"Could not open CSV file {filename}: {exc}") from exc

    try:
        reader = csv.reader(file_stream, dialect=dialect)
        yield from _table_documents(
            rows=reader,
            filename=filename,
            extra_meta={},
            budget=budget,
        )
   
    except csv.Error as exc:
        raise IngestionError(f"Invalid CSV syntax in file {filename}: {exc}") from exc
   
    except IngestionError:
        raise
   
    except Exception as exc:
        logger.exception("Unexpected error while parsing CSV file %s", filename)
        raise IngestionError(f"Failed to process CSV file {filename}") from exc
   
    finally:
        file_stream.close()


def _json_loader(
                    path: Path, 
                    filename: str, 
                    max_item_chars: int = 2000
                ) -> Iterator[Document]:
    """Safely streams JSON elements as Document objects with payload size guards."""
    
    raw_text = _read_text(path)
    try:
        data = json.loads(raw_text)
    
    except (json.JSONDecodeError, RecursionError) as exc:
        raise IngestionError("Invalid JSON file syntax.") from exc

    budget = _RowBudget(MAX_TABLE_ROWS)

    # Case 1: JSON is a list of records/objects
    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if isinstance(item, dict):
                lines = []
                for key, value in item.items():
                    # Format nested dicts/lists cleanly as JSON strings
                    val_str = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value).strip()
                    if val_str:
                        lines.append(f"{key}: {val_str}")
                content = "\n".join(lines)
            else:
                content = str(item).strip()

            if not content:
                continue

            # Protect Embedding API payload size
            if len(content) > max_item_chars:
                content = content[:max_item_chars] + "... [truncated]"

            budget.spend()

            yield Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "row": idx,
                    "pre_chunked": True,
                },
            )
        return

    # Case 2: Single JSON Object or scalar
    content = json.dumps(data, ensure_ascii=False, indent=2)
    if len(content) > max_item_chars:
        content = content[:max_item_chars] + "... [truncated]"

    budget.spend()

    yield Document(
        page_content=content,
        metadata={"source": filename, "pre_chunked": True},
    )   


def _pdf_loader(path: Path, filename: str) -> list[Document]:
    
    try:
        reader = pypdf.PdfReader(str(path))
    
        if reader.is_encrypted:
            raise IngestionError("Password-protected PDFs are not supported.")
    
        if len(reader.pages) > MAX_PDF_PAGES:
            raise IngestionError(
                f"The PDF has more than {MAX_PDF_PAGES} pages. Please split it."
            )
    
        docs = []
    
        for page_idx, page in enumerate(reader.pages, start=1):
    
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_idx},
                    )
                )
        
        return docs
    
    except IngestionError:
        raise
    
    except Exception as exc:
        raise IngestionError("Could not read the PDF (corrupt or unsupported).") from exc


def _docx_loader(path: Path, filename: str) -> list[Document]:
    
    check_zip_expansion(path)
    
    try:
        document = docx.Document(str(path))
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    
    except Exception as exc:
        raise IngestionError("Could not read the Word document (corrupt or unsupported).") from exc
    
    if not text.strip():
        return []
    
    return [Document(page_content=text, metadata={"source": filename})]


def _text_loader(path: Path, filename: str) -> list[Document]:
    text = _read_text(path)
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": filename})]


_LOADERS: dict[str, Callable[[Path, str], list[Document]]] = {
    ".xlsx": _excel_loader,
    ".csv": _csv_loader,
    ".json": _json_loader,
    ".pdf": _pdf_loader,
    ".docx": _docx_loader,
    ".txt": _text_loader,
    ".md": _text_loader,
}


# ══════════════════════════════════════════════════════════════════════════════
# The pipeline
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    chunks: int


class IngestionPipeline:
    def __init__(
        self,
        collection_name: str = "rag_documents",
        chunk_overlap: int = 130,
        chunk_size: int = 800,
    ) -> None:
        self.collection_name = collection_name
        self.chunk_overlap = chunk_overlap
        self.chunk_size = chunk_size
        # Built once here instead of on every call.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── chunking ───────────────────────────────────────────────────────────
    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        """Split documents into chunks while preserving their original order.

        Table rows / JSON records (flagged `pre_chunked`) stay whole unless they are
        absurdly long. Everything else goes through the text splitter. Deciding by
        this flag (not by file extension) is what fixes the old bug where a CSV or a
        big JSON object became one single giant chunk.
        """
        chunks: list[Document] = []
        for doc in docs:
            if doc.metadata.get("pre_chunked") and len(doc.page_content) <= MAX_RECORD_CHARS:
                chunks.append(doc)
            else:
                chunks.extend(self._splitter.split_documents([doc]))
        return [c for c in chunks if c.page_content.strip()]

    # ── main entry point ───────────────────────────────────────────────────
    def ingest_file(
        self,
        path: Path,
        original_name: str,
        user_id: str,
        doc_id: Optional[str] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> IngestionResult:
        path = Path(path)
        ext = path.suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise IngestionError(f"Unsupported file type: {ext or '(none)'}")
        # Defense in depth: the API already checked this, but the worker must not
        # trust whatever arrives through the queue.
        if not sniff_matches_extension(path, ext):
            raise IngestionError("The file content does not match its extension.")

        docs = _LOADERS[ext](path, original_name)
        chunks = self._chunk_documents(docs) if docs else []
        if not chunks:
            raise IngestionError(
                "No readable text was found in the file (scanned PDFs need OCR)."
            )
        if len(chunks) > MAX_CHUNKS_PER_DOC:
            raise IngestionError(
                f"The file produced more than {MAX_CHUNKS_PER_DOC:,} chunks. Please split it."
            )

        # user_id is part of the seed so two users uploading "prices.pdf" get different ids.
        target_doc_id = doc_id or f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{user_id}_{original_name}')}"

        dense_model, sparse_model = _get_models()
        client = _get_qdrant()
        total = len(chunks)
        stored = 0

        # Embed + upsert one batch at a time: memory stays flat no matter how big the
        # document is, and if something fails we lose one batch, not the whole file.
        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [c.page_content for c in batch]
            # ONE call per batch (the models are built for lists), not one per chunk.
            dense_vecs = list(dense_model.embed(texts, batch_size=BATCH_SIZE))
            sparse_vecs = list(sparse_model.embed(texts, batch_size=BATCH_SIZE))

            points: list[PointStruct] = []
            for offset, (chunk, dense, sparse) in enumerate(
                zip(batch, dense_vecs, sparse_vecs)
            ):
                index = start + offset
                chunk_id = f"{target_doc_id}_chunk_{index}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}_{chunk_id}"))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense.tolist(),
                            "sparse": SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist(),
                            ),
                        },
                        payload={
                            "user_id": user_id,
                            "doc_id": target_doc_id,
                            "chunk_id": chunk_id,
                            "chunk_index": index,
                            "text": chunk.page_content,
                            "metadata": {
                                **{k: v for k, v in chunk.metadata.items() if k != "pre_chunked"},
                                "filename": original_name,
                            },
                        },
                    )
                )

            client.upsert(collection_name=self.collection_name, points=points, wait=True)
            stored += len(points)
            if progress_cb:
                progress_cb(stored, total)

        # Re-upload clean-up. Same ids are simply overwritten above; but if the new
        # version is SHORTER, the tail chunks of the old version would stay in Qdrant
        # and the bot would answer from outdated text. We delete them only AFTER the
        # new version is fully stored, so the document is never "missing" in between.
        client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="doc_id", match=MatchValue(value=target_doc_id)),
                        FieldCondition(key="chunk_index", range=Range(gte=total)),
                    ]
                )
            ),
            wait=True,
        )

        logger.info(
            "Ingestion complete | doc_id=%s | chunks=%d | user=%s", target_doc_id, stored, user_id
        )
        return IngestionResult(doc_id=target_doc_id, chunks=stored)
