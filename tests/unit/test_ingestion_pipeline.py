"""
tests/unit/test_ingestion_pipeline.py
========================================
Unit tests for src.rag.ingestion_pipeline.IngestionPipeline

REAL-LIFE CONDITIONS SIMULATED:
  1.  _json_loader: list of dicts → one Document per dict
  2.  _json_loader: single dict object → one Document with JSON dump
  3.  _json_loader: list containing primitives → str() fallback
  4.  _excel_loader: multi-sheet workbook with data rows
  5.  _excel_loader: sheet with only one row (headers only) → skipped
  6.  _excel_loader: cells with None values are excluded from content
  7.  _chunk_documents: JSON/CSV/XLSX sources bypass text splitter
  8.  _chunk_documents: TXT/PDF sources are split into chunks
  9.  _load_documents: directory does not exist → FileNotFoundError
  10. _load_documents: directory exists but no supported files → FileNotFoundError
  11. preview_chunks: no chunks → prints warning without crash
  12. preview_chunks: limit=-1 prints all chunks
  13. preview_chunks: limit=N prints first N chunks
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
from langchain_core.documents import Document

# We import only the static methods / pure functions to avoid triggering
# heavy ML model downloads (HuggingFaceEmbeddings, Chroma) in __init__.
from src.rag.ingestion_pipeline import IngestionPipeline


# ──────────────────────────────────────────────────────────────────────────────
# _json_loader (static method)
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

        docs = IngestionPipeline._json_loader(json_file)
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

        docs = IngestionPipeline._json_loader(json_file)
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

        docs = IngestionPipeline._json_loader(json_file)
        assert len(docs) == 3
        assert docs[2].page_content == "42"

    def test_source_metadata_is_set(self, tmp_path):
        """Source metadata must point to the file path for traceability."""
        data = [{"k": "v"}]
        json_file = tmp_path / "meta_test.json"
        json_file.write_text(json.dumps(data), encoding="utf-8")

        docs = IngestionPipeline._json_loader(json_file)
        assert str(json_file) in docs[0].metadata["source"]

    def test_unicode_arabic_content_preserved(self, tmp_path):
        """
        REAL-LIFE: Arabic product descriptions must survive JSON round-trip.
        """
        data = [{"اسم": "هاتف", "السعر": "500 ريال"}]
        json_file = tmp_path / "arabic.json"
        json_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        docs = IngestionPipeline._json_loader(json_file)
        assert "هاتف" in docs[0].page_content


# ──────────────────────────────────────────────────────────────────────────────
# _excel_loader (static method)
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
        docs = IngestionPipeline._excel_loader(xlsx)

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
        docs = IngestionPipeline._excel_loader(xlsx)
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

        docs = IngestionPipeline._excel_loader(xlsx_path)
        assert len(docs) == 1
        assert "Notes" not in docs[0].page_content
        assert "Name: Widget" in docs[0].page_content

    def test_metadata_includes_sheet_and_row(self, tmp_path):
        """Metadata must expose sheet name and row number for debugging."""
        rows = [["Product"], ["Gadget"]]
        xlsx = self._make_xlsx(tmp_path, rows)
        docs = IngestionPipeline._excel_loader(xlsx)
        assert docs[0].metadata["sheet"] is not None
        assert docs[0].metadata["row"] == 2   # first data row = row 2


# ──────────────────────────────────────────────────────────────────────────────
# _chunk_documents()
# ──────────────────────────────────────────────────────────────────────────────

class TestChunkDocuments:

    def _make_pipeline(self) -> IngestionPipeline:
        """
        Create an IngestionPipeline instance without touching the filesystem
        or loading any ML models.
        """
        with patch.object(IngestionPipeline, "__init__", lambda self, *a, **kw: None):
            pipeline = IngestionPipeline.__new__(IngestionPipeline)
            pipeline.chunk_size = 100
            pipeline.chunk_overlap = 20
            pipeline.chunked_files = []
            pipeline.chunks = []
            return pipeline

    def test_json_docs_bypass_splitter(self):
        """
        REAL-LIFE: JSON/CSV/XLSX records must not be fragmented by the
        text splitter because chunking would break their key-value structure.
        """
        pipeline = self._make_pipeline()
        doc = Document(page_content="name: iPhone\nprice: 999", metadata={"source": "data.json"})
        chunks = pipeline._chunk_documents([doc])

        # The original document should pass through intact
        assert len(chunks) == 1
        assert chunks[0].page_content == doc.page_content

    def test_text_docs_are_split(self):
        """
        REAL-LIFE: A PDF product manual with thousands of characters
        must be chunked so each piece fits within the embedding model's
        512-token context window.
        """
        pipeline = self._make_pipeline()
        long_text = "word " * 500         # 2500 characters
        doc = Document(page_content=long_text, metadata={"source": "manual.pdf"})
        chunks = pipeline._chunk_documents([doc])

        # Text longer than chunk_size must produce multiple chunks
        assert len(chunks) > 1

    def test_csv_doc_bypasses_splitter(self):
        """CSV source must also bypass the text splitter."""
        pipeline = self._make_pipeline()
        doc = Document(page_content="col_a: val1\ncol_b: val2", metadata={"source": "data.csv"})
        chunks = pipeline._chunk_documents([doc])
        assert len(chunks) == 1

    def test_empty_docs_list_returns_empty_chunks(self):
        """BOUNDARY: No documents to chunk → empty list returned."""
        pipeline = self._make_pipeline()
        chunks = pipeline._chunk_documents([])
        assert chunks == []


# ──────────────────────────────────────────────────────────────────────────────
# _load_documents()
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadDocuments:

    def test_nonexistent_directory_raises_file_not_found(self, tmp_path):
        """
        REAL-LIFE: Operator forgets to create/mount the docs volume.
        Must raise FileNotFoundError with a descriptive message.
        """
        missing_dir = tmp_path / "does_not_exist"
        pipeline = IngestionPipeline(docs_dir=missing_dir,
                                     db_dir=tmp_path / "db",
                                     bm25_dir=tmp_path / "bm25.pkl")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            pipeline._load_documents()

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        """
        REAL-LIFE: The docs folder was created but no files were uploaded yet.
        Must raise FileNotFoundError (not silently return empty list).
        """
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        pipeline = IngestionPipeline(docs_dir=empty_dir,
                                     db_dir=tmp_path / "db",
                                     bm25_dir=tmp_path / "bm25.pkl")
        with pytest.raises(FileNotFoundError, match="No supported files"):
            pipeline._load_documents()

    def test_unsupported_files_only_raises_file_not_found(self, tmp_path):
        """
        EDGE: Directory contains only .py or .log files (unsupported).
        Must raise FileNotFoundError, not silently return empty list.
        """
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "script.py").write_text("print('hello')")
        (docs_dir / "app.log").write_text("INFO startup")

        pipeline = IngestionPipeline(docs_dir=docs_dir,
                                     db_dir=tmp_path / "db",
                                     bm25_dir=tmp_path / "bm25.pkl")
        with pytest.raises(FileNotFoundError):
            pipeline._load_documents()

    def test_allowed_extensions_filter_applies(self, tmp_path):
        """
        REAL-LIFE: Operator restricts ingestion to PDF files only.
        TXT/JSON files in the same directory must be ignored.
        """
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        # Create a minimal valid JSON file
        (docs_dir / "data.json").write_text('[{"k":"v"}]')
        # A .txt file that should be excluded when filtering for .pdf only
        (docs_dir / "note.txt").write_text("some text")

        pipeline = IngestionPipeline(docs_dir=docs_dir,
                                     db_dir=tmp_path / "db",
                                     bm25_dir=tmp_path / "bm25.pkl")

        # Filter to .json only — txt must be excluded
        docs = pipeline._load_documents(allowed_extensions=[".json"])
        sources = [d.metadata.get("source", "") for d in docs]
        assert all(".json" in s for s in sources)
        assert not any(".txt" in s for s in sources)


# ──────────────────────────────────────────────────────────────────────────────
# preview_chunks()
# ──────────────────────────────────────────────────────────────────────────────

class TestPreviewChunks:

    def _pipeline(self) -> IngestionPipeline:
        with patch.object(IngestionPipeline, "__init__", lambda self, *a, **kw: None):
            p = IngestionPipeline.__new__(IngestionPipeline)
            p.chunks = []
            return p

    def test_no_chunks_prints_warning(self, capsys):
        """Empty chunks must print a warning, not raise."""
        pipeline = self._pipeline()
        pipeline.preview_chunks(chunks=[])
        captured = capsys.readouterr()
        assert "No chunks" in captured.out

    def test_limit_none_prints_all(self, capsys):
        """limit=None (or -1) must print every chunk."""
        pipeline = self._pipeline()
        chunks = [
            Document(page_content=f"chunk {i}", metadata={"source": "test.txt"})
            for i in range(5)
        ]
        pipeline.preview_chunks(chunks=chunks, limit=None)
        captured = capsys.readouterr()
        for i in range(5):
            assert f"chunk {i}" in captured.out

    def test_limit_n_prints_first_n(self, capsys):
        """limit=2 must display only the first 2 chunks."""
        pipeline = self._pipeline()
        chunks = [
            Document(page_content=f"chunk {i}", metadata={"source": "test.txt"})
            for i in range(5)
        ]
        pipeline.preview_chunks(chunks=chunks, limit=2)
        captured = capsys.readouterr()
        assert "chunk 0" in captured.out
        assert "chunk 1" in captured.out
        assert "chunk 4" not in captured.out
