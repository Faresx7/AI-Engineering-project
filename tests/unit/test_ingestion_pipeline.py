"""
tests/unit/test_ingestion_pipeline.py
========================================
Unit tests for src.rag.ingestion.loaders and src.rag.ingestion.ingestion_pipeline.IngestionPipeline

REAL-LIFE CONDITIONS SIMULATED:
  1.  _json_loader: list of dicts → one Document per dict with key-values
  2.  _json_loader: single dict object → one Document with JSON dump
  3.  _json_loader: list containing primitives → str() fallback
  4.  _json_loader: source metadata is properly set
  5.  _json_loader: Unicode / Arabic text preserved
  6.  _excel_loader: multi-sheet workbook with data rows
  7.  _excel_loader: sheet with only headers → skipped
  8.  _excel_loader: cells with None values are excluded from content
  9.  _excel_loader: metadata includes sheet name and row number
  10. _stream_chunks: JSON/CSV/XLSX (pre_chunked) sources bypass text splitter
  11. _stream_chunks: TXT/PDF sources are split into chunks
  12. _stream_chunks: CSV source bypasses splitter
  13. _stream_chunks: empty docs list returns empty chunks
  14. file validation: unsupported file extension raises IngestionError
  15. file validation: magic byte mismatch (sniff) raises IngestionError
  16. file validation: empty readable text raises IngestionError
  17. preview_chunks: no chunks prints warning without crash
  18. preview_chunks: limit=None prints all chunks
  19. preview_chunks: limit=N prints first N chunks
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.validation import IngestionError
from src.rag.ingestion.loaders import _json_loader, _excel_loader
from src.rag.ingestion.ingestion_pipeline import IngestionPipeline


# ──────────────────────────────────────────────────────────────────────────────
# _json_loader
# ──────────────────────────────────────────────────────────────────────────────

class TestJsonLoader:

    def test_list_of_dicts_creates_one_doc_per_item(self, tmp_path):
        """
        REAL-LIFE: FAQ knowledge base stored as a JSON array of Q&A objects.
        Each object must become exactly one Document.
        """
        data = [
            {"question": "What is the price?", "answer": "$99"},
            {"question": "Do you ship?", "answer": "Yes"},
        ]
        json_file = tmp_path / "faq.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        docs = list(_json_loader(json_file, "faq.json"))
        assert len(docs) == 2
        assert "question: What is the price?" in docs[0].page_content
        assert "answer: $99" in docs[0].page_content

    def test_single_dict_creates_one_doc_with_json_dump(self, tmp_path):
        """
        REAL-LIFE: Product catalogue stored as a single top-level JSON object.
        Must be serialised as a single Document.
        """
        data = {"product": "iPhone 15", "price": "$999"}
        json_file = tmp_path / "product.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        docs = list(_json_loader(json_file, "product.json"))
        assert len(docs) == 1
        assert "iPhone 15" in docs[0].page_content

    def test_list_of_primitives_uses_str_fallback(self, tmp_path):
        """
        EDGE: List of strings / numbers, not dicts.
        Must not crash — each element becomes str().
        """
        data = ["item_one", "item_two", 42]
        json_file = tmp_path / "primitives.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        docs = list(_json_loader(json_file, "primitives.json"))
        assert len(docs) == 3
        assert docs[2].page_content == "42"

    def test_source_metadata_is_set(self, tmp_path):
        """Source metadata must point to the file path for traceability."""
        data = [{"k": "v"}]
        json_file = tmp_path / "meta_test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        docs = list(_json_loader(json_file, "meta_test.json"))
        assert docs[0].metadata["source"] == "meta_test.json"

    def test_unicode_arabic_content_preserved(self, tmp_path):
        """
        REAL-LIFE: Arabic product descriptions must survive JSON round-trip.
        """
        data = [{"اسم": "هاتف", "السعر": "500 ريال"}]
        json_file = tmp_path / "arabic.json"
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        docs = list(_json_loader(json_file, "arabic.json"))
        assert "هاتف" in docs[0].page_content


# ──────────────────────────────────────────────────────────────────────────────
# _excel_loader
# ──────────────────────────────────────────────────────────────────────────────

class TestExcelLoader:

    def _make_xlsx(self, tmp_path: Path, rows: list[list]) -> Path:
        """Helper: writes a simple xlsx file using openpyxl."""
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        for row in rows:
            ws.append(row)
        file_path = tmp_path / "test.xlsx"
        wb.save(file_path)
        return file_path

    def test_basic_rows_produce_documents(self, tmp_path):
        """
        REAL-LIFE: Product inventory in Excel.
        Each data row (after header) must become one Document.
        """
        rows = [
            ["Name", "Price", "Stock"],
            ["Phone A", "200", "50"],
            ["Phone B", "300", "20"],
        ]
        xlsx = self._make_xlsx(tmp_path, rows)
        docs = list(_excel_loader(xlsx, "test.xlsx"))

        assert len(docs) == 2
        assert "Name: Phone A" in docs[0].page_content
        assert "Price: 200" in docs[0].page_content

    def test_header_only_sheet_is_skipped(self, tmp_path):
        """
        EDGE: Sheet has headers but no data rows.
        Must not produce any Documents (avoids embedding empty vectors).
        """
        rows = [["ID", "Name"]]  # single row = headers only
        xlsx = self._make_xlsx(tmp_path, rows)
        docs = list(_excel_loader(xlsx, "test.xlsx"))
        assert docs == []

    def test_none_cells_are_excluded(self, tmp_path):
        """
        REAL-LIFE: Sparse Excel sheets with blank cells.
        None values must be skipped in the Document content.
        """
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Name", "Price", "Notes"])
        ws.append(["Widget", None, None])     # only Name is filled
        xlsx_path = tmp_path / "sparse.xlsx"
        wb.save(xlsx_path)

        docs = list(_excel_loader(xlsx_path, "sparse.xlsx"))
        assert len(docs) == 1
        assert "Notes" not in docs[0].page_content
        assert "Name: Widget" in docs[0].page_content

    def test_metadata_includes_sheet_and_row(self, tmp_path):
        """Metadata must expose sheet name and row number for debugging."""
        rows = [["Product"], ["Gadget"]]
        xlsx = self._make_xlsx(tmp_path, rows)
        docs = list(_excel_loader(xlsx, "test.xlsx"))
        assert docs[0].metadata["sheet"] is not None
        assert docs[0].metadata["row"] == 2   # first data row = row 2


# ──────────────────────────────────────────────────────────────────────────────
# _stream_chunks()
# ──────────────────────────────────────────────────────────────────────────────

class TestChunkDocuments:

    def test_json_docs_bypass_splitter(self):
        """
        REAL-LIFE: JSON/CSV/XLSX records must not be fragmented by the
        text splitter because chunking would break their key-value structure.
        """
        pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
        doc = Document(
            page_content="name: iPhone\nprice: 999",
            metadata={"source": "data.json", "pre_chunked": True},
        )
        chunks = list(pipeline._stream_chunks(iter([doc])))

        # The original document should pass through intact
        assert len(chunks) == 1
        assert chunks[0].page_content == doc.page_content

    def test_text_docs_are_split(self):
        """
        REAL-LIFE: A product manual with thousands of characters
        must be chunked so each piece fits within the embedding model's context window.
        """
        pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
        long_text = "word " * 500         # 2500 characters
        doc = Document(
            page_content=long_text,
            metadata={"source": "manual.txt", "pre_chunked": False},
        )
        chunks = list(pipeline._stream_chunks(iter([doc])))

        # Text longer than chunk_size must produce multiple chunks
        assert len(chunks) > 1

    def test_csv_doc_bypasses_splitter(self):
        """CSV source must also bypass the text splitter when pre_chunked."""
        pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
        doc = Document(
            page_content="col_a: val1\ncol_b: val2",
            metadata={"source": "data.csv", "pre_chunked": True},
        )
        chunks = list(pipeline._stream_chunks(iter([doc])))
        assert len(chunks) == 1

    def test_empty_docs_list_returns_empty_chunks(self):
        """BOUNDARY: No documents to chunk → empty list returned."""
        pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
        chunks = list(pipeline._stream_chunks(iter([])))
        assert chunks == []


# ──────────────────────────────────────────────────────────────────────────────
# File validation and loading tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFileValidation:

    def test_unsupported_extension_raises_ingestion_error(self, tmp_path):
        """
        REAL-LIFE: Operator uploads unsupported script or binary file.
        Must raise IngestionError with descriptive message.
        """
        bad_file = tmp_path / "script.py"
        bad_file.write_text("print('hello')", encoding="utf-8")
        pipeline = IngestionPipeline()
        with pytest.raises(IngestionError, match="Unsupported file type"):
            pipeline.ingest_file(bad_file, "script.py", user_id="u1")

    def test_extension_content_mismatch_raises_ingestion_error(self, tmp_path):
        """
        SECURITY / CHAOS: File renamed to .pdf but contains plain text.
        Sniffer must detect mismatch and reject before model processing.
        """
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("This is definitely not a PDF file", encoding="utf-8")
        pipeline = IngestionPipeline()
        with pytest.raises(IngestionError, match="does not match its declared extension"):
            pipeline.ingest_file(fake_pdf, "fake.pdf", user_id="u1")

    @patch("src.rag.ingestion.ingestion_pipeline._get_qdrant")
    @patch("src.rag.ingestion.ingestion_pipeline._get_sparse_model")
    def test_empty_readable_text_raises_ingestion_error(
        self, mock_sparse, mock_qdrant, tmp_path
    ):
        """
        BOUNDARY: Empty file or only whitespace text.
        Must raise IngestionError stating no readable text was found.
        """
        mock_qdrant.return_value = MagicMock()
        mock_sparse.return_value = MagicMock()

        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_text("    \n\n  \t  ", encoding="utf-8")
        pipeline = IngestionPipeline()
        with pytest.raises(IngestionError, match="No readable text was found"):
            pipeline.ingest_file(empty_txt, "empty.txt", user_id="u1")


# ──────────────────────────────────────────────────────────────────────────────
# Streaming & Chunk Overlap Validation
# ──────────────────────────────────────────────────────────────────────────────

class TestStreamChunksProperties:

    def test_empty_content_documents_are_skipped(self):
        """Documents with empty string or whitespace only are filtered out."""
        pipeline = IngestionPipeline()
        docs = [
            Document(page_content="", metadata={"source": "a.txt"}),
            Document(page_content="   \n\t  ", metadata={"source": "b.txt"}),
            Document(page_content="Valid content here", metadata={"source": "c.txt"}),
        ]
        chunks = list(pipeline._stream_chunks(iter(docs)))
        assert len(chunks) == 1
        assert chunks[0].page_content == "Valid content here"

    def test_custom_chunk_size_and_overlap_applied(self):
        """Verify custom chunk_size and chunk_overlap configure text splitter."""
        pipeline = IngestionPipeline(chunk_size=50, chunk_overlap=10)
        assert pipeline.chunk_size == 50
        assert pipeline.chunk_overlap == 10
        assert pipeline._splitter._chunk_size == 50
        assert pipeline._splitter._chunk_overlap == 10

    def test_pre_chunked_oversized_record_is_split(self):
        """Pre-chunked record that exceeds _MAX_RECORD_CHARS is passed to splitter."""
        from src.rag.constants import _MAX_RECORD_CHARS
        pipeline = IngestionPipeline(chunk_size=100, chunk_overlap=20)
        oversized = "x" * (_MAX_RECORD_CHARS + 500)
        doc = Document(page_content=oversized, metadata={"source": "large.json", "pre_chunked": True})
        chunks = list(pipeline._stream_chunks(iter([doc])))
        # Since it exceeds _MAX_RECORD_CHARS, it shouldn't stay whole
        assert len(chunks) > 1
