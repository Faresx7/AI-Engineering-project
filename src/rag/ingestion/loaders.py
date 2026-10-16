"""Streaming document loaders and format parsers for RAG ingestion.

Every loader here is a generator yielding LangChain ``Document`` objects lazily.
No file or dataset is ever materialized fully in memory.
"""
from __future__ import annotations

import csv
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import chardet
import docx
import fitz
import openpyxl
from docx.document import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from langchain_core.documents import Document

from src.core.validation import IngestionError, check_zip_expansion
from src.rag.ingestion.budget import _RowBudget
from src.rag.constants import (
    _MAX_CELL_CHARS,
    _MAX_CELLS_PER_ROW,
    _MAX_CONSECUTIVE_EMPTY_ROWS,
    _MAX_LINE_CHARS,
    _MAX_PDF_PAGES,
    _MAX_RECORD_CHARS,
    _MAX_TABLE_ROWS,
    _MAX_TEXT_BYTES,
    _WD_PARA_TAG,
    _WD_TABLE_TAG,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _read_text_bytes(path: Path, max_bytes: int = _MAX_TEXT_BYTES) -> str:
    """Reads a text file into a string with size guard and encoding auto-detection."""
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IngestionError(f"Cannot access file '{path.name}'.") from exc

    if not resolved.is_file():
        raise IngestionError(f"'{path.name}' is not a regular file.")

    try:
        file_size = resolved.stat().st_size
    except OSError as exc:
        raise IngestionError(f"Cannot stat file '{path.name}'.") from exc

    if file_size > max_bytes:
        raise IngestionError(
            f"File '{path.name}' exceeds the {max_bytes // (1024 * 1024)} MB read limit."
        )

    raw = resolved.read_bytes()

    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass

    detected = chardet.detect(raw[:32_768])
    encoding: str = "latin-1"
    if isinstance(detected, dict) and detected.get("encoding"):
        encoding = str(detected["encoding"])
    logger.warning("File '%s' is not UTF-8; detected encoding '%s'.", path.name, encoding)
    try:
        return raw.decode(encoding)
    except Exception as exc:
        raise IngestionError(
            f"Could not decode '{path.name}' with any supported encoding."
        ) from exc


def _is_row_empty(row: Sequence[Any]) -> bool:
    """Returns True when every cell in the row is None or blank whitespace."""
    return not row or all(
        cell is None or (isinstance(cell, str) and not cell.strip())
        for cell in row
    )


def _is_potential_header(cells: list[Any]) -> bool:
    """Returns False for single-filled-cell title banners in multi-column rows."""
    non_empty = [c for c in cells if c is not None and str(c).strip()]
    if not non_empty:
        return False
    if len(cells) > 1 and len(non_empty) == 1:
        return False
    return True


def _trim_trailing_nones(cells: list[Any]) -> list[Any]:
    """Removes trailing None/blank cells from the right end of a row (non-destructive copy)."""
    result = list(cells)
    while result and (result[-1] is None or str(result[-1]).strip() == ""):
        result.pop()
    return result


def _sanitize_cell(value: Any, max_chars: int = _MAX_CELL_CHARS) -> str:
    """Converts a cell value to a clean, length-bounded string."""
    text = str(value).strip() if value is not None else ""
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Core table → Document converter
# ══════════════════════════════════════════════════════════════════════════════

def _table_documents(
    rows: Iterable[Sequence[Any]],
    filename: str,
    extra_meta: dict[str, Any],
    budget: _RowBudget,
) -> Iterator[Document]:
    """Converts an iterable of table rows into Documents with header detection.

    Stops producing Documents when the row budget is exhausted, but never
    raises an exception — so already-stored batches are preserved.
    """
    headers: Optional[list[str]] = None

    for row_idx, row in enumerate(rows, start=1):
        cells = list(row)

        if _is_row_empty(cells):
            continue

        if headers is None:
            if not _is_potential_header(cells):
                logger.debug("Skipping title/banner row %d in '%s'.", row_idx, filename)
                continue

            trimmed = _trim_trailing_nones(cells)
            if not trimmed:
                continue

            # Cap header columns to prevent unbounded memory on wide sheets.
            if len(trimmed) > _MAX_CELLS_PER_ROW:
                logger.warning(
                    "Header row in '%s' has %d columns; capping at %d.",
                    filename, len(trimmed), _MAX_CELLS_PER_ROW,
                )
                trimmed = trimmed[:_MAX_CELLS_PER_ROW]

            raw_headers: list[str] = []
            used: set[str] = set()
            last_valid = "col"

            for col_idx, raw_cell in enumerate(trimmed):
                raw_str = str(raw_cell).strip() if raw_cell is not None else ""
                base = raw_str if raw_str else f"{last_valid}_sub_{col_idx}"
                if raw_str:
                    last_valid = raw_str

                candidate = base
                counter = 1
                while candidate in used:
                    candidate = f"{base}_{counter}"
                    counter += 1

                used.add(candidate)
                raw_headers.append(candidate)

            headers = raw_headers
            continue

        pairs: list[tuple[str, str]] = []
        for idx, val in enumerate(cells):
            if idx >= _MAX_CELLS_PER_ROW:
                break
            if val is None:
                continue
            val_str = _sanitize_cell(val)
            if not val_str:
                continue
            key = headers[idx] if idx < len(headers) else f"extra_col_{idx}"
            pairs.append((key, val_str))

        if not pairs:
            continue

        if not budget.spend():
            # Budget exhausted — stop producing rows gracefully.
            return

        yield Document(
            page_content="\n".join(f"{k}: {v}" for k, v in pairs),
            metadata={
                "source": filename,
                **extra_meta,
                "row": row_idx,
                "pre_chunked": True,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# Context managers for external resources
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def _open_workbook(path: Path) -> Iterator[openpyxl.Workbook]:
    workbook: Optional[openpyxl.Workbook] = None
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("Could not open the Excel file (corrupt or unsupported).") from exc

    try:
        yield workbook
    finally:
        if workbook is not None:
            try:
                workbook.close()
            except Exception as close_exc:
                logger.warning("Failed to close workbook for '%s': %s", path.name, close_exc)


@contextmanager
def _open_pdf(path: Path) -> Iterator[fitz.Document]:
    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(str(path))
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("Could not open the PDF file (corrupt or unsupported).") from exc

    try:
        yield doc
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception as close_exc:
                logger.warning("Failed to close PDF for '%s': %s", path.name, close_exc)


@contextmanager
def _open_docx(path: Path) -> Iterator[DocxDocument]:
    doc: Optional[DocxDocument] = None
    try:
        doc = docx.Document(str(path))
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("Could not open the Word document (corrupt or unsupported).") from exc

    try:
        yield doc
    finally:
        # Let Python's GC handle cleanup — no need to mutate the XML tree.
        del doc


# ══════════════════════════════════════════════════════════════════════════════
# Excel helpers
# ══════════════════════════════════════════════════════════════════════════════

def _stream_sheet_rows(sheet: Any, sheet_name: str) -> Iterator[tuple]:
    consecutive_empty = 0
    data_started = False

    for row in sheet.iter_rows(values_only=True):
        if _is_row_empty(row):
            if not data_started:
                continue
            consecutive_empty += 1
            if consecutive_empty >= _MAX_CONSECUTIVE_EMPTY_ROWS:
                logger.info(
                    "Stopping sheet '%s' after %d consecutive empty rows.",
                    sheet_name, _MAX_CONSECUTIVE_EMPTY_ROWS,
                )
                break
            continue

        data_started = True
        consecutive_empty = 0
        yield row


# ══════════════════════════════════════════════════════════════════════════════
# DOCX table helpers
# ══════════════════════════════════════════════════════════════════════════════

def _docx_row_cells(row: Any) -> list[str]:
    seen_ids: set[int] = set()
    cells: list[str] = []

    for i, cell in enumerate(row.cells):
        if i >= _MAX_CELLS_PER_ROW:
            logger.warning("DOCX table row truncated at %d cells.", _MAX_CELLS_PER_ROW)
            break
        cell_id = id(cell._tc)
        if cell_id in seen_ids:
            cells.append("")
        else:
            seen_ids.add(cell_id)
            cells.append(cell.text.strip().replace("\n", " ").replace("|", "\\|"))
    return cells


def _format_docx_table(table: Any, budget: _RowBudget) -> str:
    rows_data: list[list[str]] = []

    for row in table.rows:
        row_cells = _docx_row_cells(row)
        if not any(row_cells):
            continue
        if not budget.spend():
            logger.warning("Row budget exhausted while formatting DOCX table. Truncating.")
            break
        rows_data.append(row_cells)

    if not rows_data or not rows_data[0]:
        return ""

    col_count = len(rows_data[0])
    lines = [
        "| " + " | ".join(rows_data[0]) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in rows_data[1:]:
        padded = row + [""] * max(0, col_count - len(row))
        lines.append("| " + " | ".join(padded[:col_count]) + " |")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# File loaders — every one is a true generator, nothing is materialized in RAM
# ══════════════════════════════════════════════════════════════════════════════

def _excel_loader(path: Path, filename: str) -> Iterator[Document]:
    check_zip_expansion(path)

    with _open_workbook(path) as workbook:
        budget = _RowBudget(_MAX_TABLE_ROWS)

        try:
            for sheet in workbook.worksheets:
                if sheet.sheet_state != "visible":
                    logger.info(
                        "Skipping non-visible sheet '%s' (state: %s).",
                        sheet.title, sheet.sheet_state,
                    )
                    continue

                yield from _table_documents(
                    rows=_stream_sheet_rows(sheet, sheet.title),
                    filename=filename,
                    extra_meta={"sheet": sheet.title},
                    budget=budget,
                )

        except IngestionError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error reading Excel file '%s'.", filename)
            raise IngestionError("Could not read the contents of the Excel file.") from exc


def _csv_loader(path: Path, filename: str) -> Iterator[Document]:
    budget = _RowBudget(_MAX_TABLE_ROWS)

    try:
        with open(path, mode="r", encoding="utf-8-sig", errors="replace") as probe:
            sample = probe.read(8_192)
        try:
            dialect: Any = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
    except Exception as exc:
        raise IngestionError(f"Could not open CSV file '{filename}'.") from exc

    file_stream = None
    try:
        file_stream = open(path, mode="r", encoding="utf-8-sig", errors="replace")
        reader = csv.reader(file_stream, dialect=dialect)

        yield from _table_documents(
            rows=reader, filename=filename, extra_meta={}, budget=budget,
        )

    except csv.Error as exc:
        raise IngestionError(f"Invalid CSV syntax in '{filename}'.") from exc
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error reading CSV file '%s'.", filename)
        raise IngestionError(f"Could not read the contents of CSV file '{filename}'.") from exc
    finally:
        if file_stream is not None:
            try:
                file_stream.close()
            except Exception:
                pass


def _json_loader(path: Path, filename: str) -> Iterator[Document]:
    raw_text = _read_text_bytes(path)

    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise IngestionError("Invalid or malformed JSON file.") from exc

    # Free the raw string immediately — ``json.loads`` already made its own parsed copy.
    del raw_text

    budget = _RowBudget(_MAX_TABLE_ROWS)

    if isinstance(data, list):
        items = data
        del data

        for idx, item in enumerate(items, start=1):
            if isinstance(item, dict):
                lines = []
                for key, value in item.items():
                    val_str = (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else str(value).strip()
                    )
                    if val_str:
                        lines.append(f"{key}: {val_str}")
                content = "\n".join(lines)
            else:
                content = str(item).strip()

            if not content:
                continue

            if len(content) > _MAX_RECORD_CHARS:
                content = content[:_MAX_RECORD_CHARS] + "... [truncated]"

            if not budget.spend():
                return

            yield Document(
                page_content=content,
                metadata={"source": filename, "row": idx, "pre_chunked": True},
            )

        del items
        return

    content = json.dumps(data, ensure_ascii=False, indent=2)
    del data

    if len(content) > _MAX_RECORD_CHARS:
        content = content[:_MAX_RECORD_CHARS] + "... [truncated]"

    if not budget.spend():
        return

    yield Document(
        page_content=content,
        metadata={"source": filename, "pre_chunked": True},
    )


def _pdf_loader(path: Path, filename: str) -> Iterator[Document]:
    check_zip_expansion(path)

    with _open_pdf(path) as doc:
        if doc.is_encrypted:
            raise IngestionError("Password-protected PDFs are not supported.")

        total_pages = len(doc)
        if total_pages > _MAX_PDF_PAGES:
            raise IngestionError(
                f"PDF exceeds the {_MAX_PDF_PAGES}-page limit (found {total_pages} pages)."
            )

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            try:
                page = doc[page_idx]
                raw_text = page.get_text("text")
                text = raw_text if isinstance(raw_text, str) else ""
            except Exception as page_exc:
                logger.warning(
                    "Failed to extract text from page %d of '%s': %s",
                    page_num, filename, page_exc,
                )
                continue

            clean_text = text.strip()
            if clean_text:
                yield Document(
                    page_content=clean_text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "total_pages": total_pages,
                        "pre_chunked": False,
                    },
                )


def _docx_loader(path: Path, filename: str) -> Iterator[Document]:
    """Streams Word document paragraphs and tables cleanly.

    Paragraphs are accumulated and chunked (pre_chunked=False), while tables are
    yielded as standalone pre-formatted Markdown blocks (pre_chunked=True) so
    the text splitter never slices Markdown table rows.
    """
    check_zip_expansion(path)

    with _open_docx(path) as doc:
        budget = _RowBudget(_MAX_TABLE_ROWS)
        block_idx = 1
        accumulated: list[str] = []
        current_len = 0
        max_chunk_chars = 2_000

        try:
            for element in doc.element.body:
                # 1. Handle Paragraphs
                if element.tag == _WD_PARA_TAG:
                    para = DocxParagraph(element, doc)
                    text = para.text.strip()
                    if text:
                        if len(text) > _MAX_RECORD_CHARS:
                            logger.warning(
                                "Paragraph in '%s' truncated from %d to %d chars.",
                                filename, len(text), _MAX_RECORD_CHARS,
                            )
                            text = text[:_MAX_RECORD_CHARS]

                        accumulated.append(text)
                        current_len += len(text)

                        # Flush text block if accumulated size exceeds limit
                        if current_len >= max_chunk_chars:
                            yield Document(
                                page_content="\n\n".join(accumulated),
                                metadata={
                                    "source": filename,
                                    "block": block_idx,
                                    "block_id": f"{filename}::block::{block_idx}",
                                    "pre_chunked": False,
                                },
                            )
                            block_idx += 1
                            accumulated = []
                            current_len = 0

                # 2. Handle Tables
                elif element.tag == _WD_TABLE_TAG:
                    # Flush any pending paragraph text before processing the table
                    if accumulated:
                        yield Document(
                            page_content="\n\n".join(accumulated),
                            metadata={
                                "source": filename,
                                "block": block_idx,
                                "block_id": f"{filename}::block::{block_idx}",
                                "pre_chunked": False,
                            },
                        )
                        block_idx += 1
                        accumulated = []
                        current_len = 0

                    # Yield table as a standalone pre_chunked Document
                    tbl = DocxTable(element, doc)
                    md = _format_docx_table(tbl, budget)
                    if md:
                        yield Document(
                            page_content=md,
                            metadata={
                                "source": filename,
                                "block": block_idx,
                                "block_id": f"{filename}::block::{block_idx}",
                                "pre_chunked": True,  # Ensures text splitter leaves table intact
                            },
                        )
                        block_idx += 1

            # Flush any remaining paragraph text at the end of the document
            if accumulated:
                yield Document(
                    page_content="\n\n".join(accumulated),
                    metadata={
                        "source": filename,
                        "block": block_idx,
                        "block_id": f"{filename}::block::{block_idx}",
                        "pre_chunked": False,
                    },
                )

        except IngestionError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error reading DOCX file '%s'.", filename)
            raise IngestionError("Could not read the contents of the Word document.") from exc


def _text_loader(path: Path, filename: str) -> Iterator[Document]:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IngestionError(f"Cannot access file '{filename}'.") from exc

    if not resolved.is_file():
        raise IngestionError(f"'{filename}' is not a regular file.")

    try:
        file_size = resolved.stat().st_size
    except OSError as exc:
        raise IngestionError(f"Cannot stat file '{filename}'.") from exc

    if file_size > _MAX_TEXT_BYTES:
        raise IngestionError(
            f"Text file '{filename}' exceeds the {_MAX_TEXT_BYTES // (1024 * 1024)} MB limit."
        )

    accumulated: list[str] = []
    current_len = 0
    block_idx = 1
    max_chunk_chars = 2_000

    try:
        with open(resolved, mode="r", encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                if len(line) > _MAX_LINE_CHARS:
                    logger.warning(
                        "Line in '%s' exceeds %d chars; truncating.", filename, _MAX_LINE_CHARS,
                    )
                    line = line[:_MAX_LINE_CHARS]

                clean = line.strip()
                if not clean:
                    continue

                accumulated.append(clean)
                current_len += len(clean)

                if current_len >= max_chunk_chars:
                    yield Document(
                        page_content="\n\n".join(accumulated),
                        metadata={
                            "source": filename,
                            "block": block_idx,
                            "block_id": f"{filename}::block::{block_idx}",
                            "pre_chunked": False,
                        },
                    )
                    block_idx += 1
                    accumulated = []
                    current_len = 0

            if accumulated:
                yield Document(
                    page_content="\n\n".join(accumulated),
                    metadata={
                        "source": filename,
                        "block": block_idx,
                        "block_id": f"{filename}::block::{block_idx}",
                        "pre_chunked": False,
                    },
                )

    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error reading text file '%s'.", filename)
        raise IngestionError(f"Could not read the contents of text file '{filename}'.") from exc


_LOADERS: dict[str, Callable[[Path, str], Iterator[Document]]] = {
    ".xlsx": _excel_loader,
    ".csv":  _csv_loader,
    ".json": _json_loader,
    ".pdf":  _pdf_loader,
    ".docx": _docx_loader,
    ".txt":  _text_loader,
    ".md":   _text_loader,
}
