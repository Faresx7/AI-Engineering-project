"""Fault-tolerant streaming document loaders for RAG ingestion.

Routes PDF files to PyMuPDF for page-by-page extraction and delegates all other
supported document formats to Microsoft MarkItDown with bounded text slicing.
"""
from __future__ import annotations

from typing import Callable, Generator, Iterator
from langchain_core.documents import Document
from contextlib import closing
from pathlib import Path
from markitdown import MarkItDown
import logging
import pymupdf

from src.core.validation import IngestionError, check_zip_expansion
from src.core.config import settings

from src.rag.constants import (_MAX_PDF_PAGES,
                               _MAX_TEXT_BYTES,
                               _MARKITDOWN_EXTENSIONS,
                               _ZIP_EXTENSIONS,)



logger = logging.getLogger(__name__)

_MD = MarkItDown()



def _validate_file(path: Path, filename: str, ext: str) -> int:
    """Verify file existence, size limits, and archive expansion safety.

    Args:
        path: Local filesystem path to the stored upload.
        filename: Original user-facing filename used for logging and error messages.
        ext: Normalized lowercase file extension (including leading dot).

    Returns:
        int: File size in bytes.

    Raises:
        IngestionError: If the file is missing, unreadable, exceeds size limits,
            or fails zip expansion validation.
    """
    if not path.is_file():
        logger.warning("LOADER_VALIDATION_FILE_FAILED | file not found. \nfile=%s reason=not_a_file action=rejecting", filename)
        raise IngestionError(f"File '{filename}' not found.")

    try:
        file_size = path.stat().st_size
    
    except OSError as exc:
        logger.warning(
            "LOADER_VALIDATION_FILE_FAILED | Can't access file.\nfile=%s reason=stat_error err_type=%s err=%r action=rejecting",
            filename,
            type(exc).__name__,
            exc,
        )
        raise IngestionError(f"Cannot access file '{filename}'.") from exc

    if file_size > _MAX_TEXT_BYTES:
        logger.warning(
            "LOADER_VALIDATION_FILE_FAILED | file is too large.\nfile=%s reason=size_exceeded size=%d limit=%d action=rejecting",
            filename,
            file_size,
            _MAX_TEXT_BYTES,
        )
        raise IngestionError(
            f"File '{filename}' exceeds the {_MAX_TEXT_BYTES // (1024 * 1024)} MB limit."
        )

    # Guard against zip bombs in compressed Office XML formats.
    if ext in _ZIP_EXTENSIONS:
        try:
            check_zip_expansion(path)
        except IngestionError:
            logger.warning(
                "LOADER_VALIDATION_FILE_FAILED | Zip expansion ratio exceeded.\nfile=%s ext=%s reason=zip_expansion_exceeded action=rejecting",
                filename,
                ext,
            )
            raise

    logger.debug(
        "LOADER_VALIDATION_PASSED | File passed validation checks.\nfile=%s ext=%s size=%d action=proceeding",
        filename,
        ext,
        file_size,
    )
    return file_size


def _iter_text_slices(text: str, slice_size: int = settings._STREAM_SLICE_CHARS) -> Iterator[str]:
    """Split a large string into bounded slices along natural text boundaries.

    Searches backward from each slice boundary for paragraph breaks, newlines,
    or spaces so words are not split mid-token.

    Args:
        text: Full extracted document text to partition.
        slice_size: Target maximum number of characters per slice.

    Yields:
        str: Non-empty trimmed text segments up to ``slice_size`` characters.
    """
    total_len = len(text)
    start = 0

    while start < total_len:
        end = min(start + slice_size, total_len)
        if end < total_len:
            # Look backward in the second half of the slice for a clean split point.
            split_at = text.rfind("\n\n", start + slice_size // 2, end)
            if split_at == -1:
                split_at = text.rfind("\n", start + slice_size // 2, end)
            if split_at == -1:
                split_at = text.rfind(" ", start + slice_size // 2, end)
            if split_at != -1:
                end = split_at

        piece = text[start:end].strip()
        if piece:
            yield piece

        start = end


def _pdf_loader(path: Path, filename: str) -> Generator[Document, None, None]:
    """Extract text page-by-page from a PDF using PyMuPDF.

    Skips empty or corrupted pages individually while preserving page metadata,
    failing only if the document is invalid, encrypted, oversized, or yields no text.

    Args:
        path: Local filesystem path to the PDF file.
        filename: Original user-facing filename stored in document metadata.

    Yields:
        Document: One LangChain Document per readable PDF page.

    Raises:
        IngestionError: If validation fails, the PDF is encrypted/corrupt/scanned,
            or page count exceeds ``_MAX_PDF_PAGES``.
    """
    _validate_file(path, filename, ".pdf")
    logger.info("LOADER_PDF_STARTED | Starting PDF extraction.\nfile=%s loader=pdf action=opening_document", filename)

    try:
        with pymupdf.open(str(path)) as doc:
            if not doc.is_pdf:
                logger.warning("LOADER_PDF_INVALID | File is not a valid PDF.\nfile=%s reason=not_a_pdf action=rejecting", filename)
                raise IngestionError(f"File '{filename}' is not a valid PDF.")

            if doc.is_encrypted:
                logger.warning("LOADER_PDF_ENCRYPTED | PDF is password-protected.\nfile=%s action=rejecting", filename)
                raise IngestionError("Password-protected PDFs are not supported.")

            total_pages = len(doc)
            if total_pages > _MAX_PDF_PAGES:
                logger.warning(
                    "LOADER_PDF_PAGE_LIMIT | PDF exceeds maximum page limit.\nfile=%s pages=%d limit=%d action=rejecting",
                    filename,
                    total_pages,
                    _MAX_PDF_PAGES,
                )
            
                raise IngestionError(
                    f"PDF exceeds the {_MAX_PDF_PAGES}-page limit (found {total_pages} pages)."
                )

            if total_pages == 0:
                logger.warning("LOADER_PDF_EMPTY | PDF document has zero pages.\nfile=%s pages=0 action=rejecting", filename)
                
                raise IngestionError(f"PDF '{filename}' has 0 pages.")

            logger.info("LOADER_PDF_SCANNING | Extracting text from PDF pages.\nfile=%s total_pages=%d action=extracting_pages", filename, total_pages)

            pages_extracted = 0
            pages_failed = 0

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                
                try:
                    raw_text = doc[page_idx].get_text("text")
                    clean_text = raw_text.strip() if isinstance(raw_text, str) else ""
                
                except Exception as page_exc:
                    # Isolate single-page extraction failures so valid pages still ingest.
                    pages_failed += 1
                    logger.warning(
                        "LOADER_PDF_PAGE_FAILED | Failed to extract text from corrupted page.\nfile=%s page=%d/%d err_type=%s err=%r action=skipping_corrupt_page",
                        filename,
                        page_num,
                        total_pages,
                        type(page_exc).__name__,
                        page_exc,
                    )
                    
                    continue

                if not clean_text:
                    logger.debug(
                        "LOADER_PDF_PAGE_EMPTY | Page contains no extractable text.\nfile=%s page=%d/%d action=skipping_empty_page",
                        filename,
                        page_num,
                        total_pages,
                    )
                    
                    continue

                pages_extracted += 1
                
                yield Document(
                    page_content=clean_text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "total_pages": total_pages,
                        "pre_chunked": False,
                    },
                )

            if pages_extracted == 0:
                if pages_failed == total_pages:
                    logger.error(
                        "LOADER_PDF_ALL_PAGES_FAILED | Failed to extract text from any PDF page.\nfile=%s pages=%d failed_pages=%d action=rejecting",
                        filename,
                        total_pages,
                        pages_failed,
                    )
                    
                    raise IngestionError(f"Could not extract text from any page of '{filename}'.")

                logger.warning(
                    "LOADER_PDF_NO_TEXT | PDF has no extractable text and is likely scanned.\nfile=%s pages=%d failed_pages=%d action=rejecting_scanned",
                    filename,
                    total_pages,
                    pages_failed,
                )
                
                raise IngestionError(
                    f"PDF '{filename}' contains no extractable text (likely scanned)."
                )

            if pages_failed > 0:
                logger.warning(
                    "LOADER_PDF_PARTIAL | PDF extracted with some corrupted pages skipped.\nfile=%s total_pages=%d extracted=%d failed=%d action=completed_with_skips",
                    filename,
                    total_pages,
                    pages_extracted,
                    pages_failed,
                )

            logger.info(
                "LOADER_PDF_COMPLETED | PDF text extraction finished successfully.\nfile=%s total_pages=%d extracted=%d failed=%d action=done",
                filename,
                total_pages,
                pages_extracted,
                pages_failed,
            )

    
    except IngestionError:
        raise
    
    except Exception as exc:
        logger.exception(
            "LOADER_PDF_READ_FAILED | Unexpected error while reading PDF file.\nfile=%s err_type=%s err=%r action=raising",
            filename,
            type(exc).__name__,
            exc,
        )
        
        raise IngestionError(f"Could not read PDF file '{filename}'.") from exc


def _markitdown_loader(path: Path, filename: str) -> Generator[Document, None, None]:
    """Convert non-PDF documents via MarkItDown and yield bounded text slices.

    Args:
        path: Local filesystem path to the document.
        filename: Original user-facing filename used for extension routing and metadata.

    Yields:
        Document: Sequential LangChain Documents containing bounded Markdown/text slices.

    Raises:
        IngestionError: If file validation fails, MarkItDown raises an error,
            or the converted document contains no readable text.
    """
    ext = Path(filename).suffix.lower()
    _validate_file(path, filename, ext)
    logger.info("LOADER_MARKITDOWN_STARTED | Starting MarkItDown conversion.\nfile=%s loader=markitdown ext=%s action=converting", filename, ext)

    try:
        result = _MD.convert(str(path))
    
    except Exception as exc:
        logger.exception(
            "LOADER_MARKITDOWN_FAILED | MarkItDown failed to convert file.\nfile=%s ext=%s err_type=%s err=%r action=raising",
            filename,
            ext,
            type(exc).__name__,
            exc,
        )
    
        raise IngestionError(f"Could not extract text from '{filename}'.") from exc

    text = (result.text_content or "").strip()
    # Release the converter result object before yielding slices,
    # to avoid memory doubling.
    del result

    if not text:
        logger.warning(
            "LOADER_MARKITDOWN_EMPTY | No readable text extracted from file.\nfile=%s ext=%s action=rejecting",
            filename,
            ext,
        )
    
        raise IngestionError(f"No readable text found in '{filename}'.")

    total_chars = len(text)
    logger.info(
        "LOADER_MARKITDOWN_STREAMING | Streaming extracted text in slices.\nfile=%s ext=%s total_chars=%d slice_size=%d action=yielding_slices",
        filename,
        ext,
        total_chars,
        settings._STREAM_SLICE_CHARS,
    )

    slices_yielded = 0
    for block_idx, slice_text in enumerate(_iter_text_slices(text), start=1):
        
        slices_yielded += 1
        
        yield Document(
            page_content=slice_text,
            metadata={
                "source": filename,
                "block": block_idx,
                "block_id": f"{filename}::block::{block_idx}",
                "pre_chunked": False,
            },
        )

    logger.info(
        "LOADER_MARKITDOWN_COMPLETED | MarkItDown extraction and streaming finished.\nfile=%s ext=%s slices=%d total_chars=%d action=done",
        filename,
        ext,
        slices_yielded,
        total_chars,
    )


_LOADERS: dict[str, Callable[[Path, str], Generator[Document, None, None]]] = {
    ".pdf": _pdf_loader,
    **{ext: _markitdown_loader for ext in _MARKITDOWN_EXTENSIONS},
}


def stream_documents(path: Path, filename: str) -> Iterator[Document]:
    """Route a file to its format-specific loader and stream extracted Documents.

    Wraps the underlying generator in ``contextlib.closing`` to guarantee immediate
    file handle cleanup if downstream consumption aborts early.

    Args:
        path: Local filesystem path to the file to ingest.
        filename: Original filename used to determine the file extension.

    Yields:
        Document: Extracted document pages or slices ready for chunking.

    Raises:
        IngestionError: If the file extension is unsupported or loader extraction fails.
    """
    ext = Path(filename).suffix.lower()
    logger.info("LOADER_DISPATCH_STARTED | Routing file to appropriate loader.\nfile=%s ext=%s action=routing", filename, ext)

    loader = _LOADERS.get(ext)
    if loader is None:
        logger.warning("LOADER_UNSUPPORTED_EXT | File extension is not supported.\nfile=%s ext=%s action=rejecting", filename, ext)
       
        raise IngestionError(f"Unsupported file extension '{ext}'.")

    with closing(loader(path, filename)) as doc_stream:
        yield from doc_stream