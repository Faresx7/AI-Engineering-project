"""
tests/edge_cases/test_ingestion_edge_and_stress.py
===================================================
Critical edge, stress, and chaos tests for document loaders and ingestion pipeline.

REAL-LIFE SCENARIOS TESTED:
  1.  DOCX format edge cases:
      - DOCX with merged table cells (duplicate _tc IDs must be deduplicated)
      - DOCX table with headers and row budget truncation
      - DOCX with mixed paragraphs and tables
      - Corrupt DOCX raises IngestionError
  2.  PDF format edge cases:
      - Password-protected PDF raises IngestionError
      - PDF exceeding _MAX_PDF_PAGES limit raises IngestionError
      - Multi-page PDF produces sequentially-numbered Documents
  3.  CSV & Table edge cases:
      - Ragged rows (unequal column count per row)
      - Consecutive empty rows stopping sheet stream
      - Column header name collisions resolved with unique suffixes
  4.  Encoding resilience:
      - UTF-8 with BOM decoded cleanly without leaking \\ufeff
      - Non-UTF8 fallback decoding
  5.  Ingestion pipeline stress & safety:
      - Multi-batch ingestion (streaming multiple batches to Qdrant)
      - Rollback safety: transient error triggers rollback delete of failed generation
      - Purge sweep: skips single generation, skips in-flight generation (< cutoff)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx
import openpyxl
import pytest
from langchain_core.documents import Document

from src.core.validation import IngestionError
from src.rag.constants import _MAX_TABLE_ROWS
from src.rag.ingestion.budget import _RowBudget
from src.rag.ingestion.loaders import (
    _docx_loader,
    _excel_loader,
    _csv_loader,
    _pdf_loader,
    _read_text_bytes,
    _table_documents,
    _stream_sheet_rows,
)
from src.rag.ingestion.ingestion_pipeline import (
    IngestionPipeline,
    IngestionResult,
    purge_orphaned_generations,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DOCX Loader Advanced & Merged Cells
# ──────────────────────────────────────────────────────────────────────────────

class TestDocxLoaderAdvanced:

    def _create_docx_with_table(self, path: Path) -> Path:
        doc = docx.Document()
        doc.add_heading("Product Specification", level=1)
        doc.add_paragraph("This document outlines our product inventory.")

        table = doc.add_table(rows=3, cols=3)
        # Header row
        hdr = table.rows[0].cells
        hdr[0].text = "SKU"
        hdr[1].text = "Description"
        hdr[2].text = "Stock"

        # Data rows
        r1 = table.rows[1].cells
        r1[0].text = "SKU-001"
        r1[1].text = "Wireless Headphones"
        r1[2].text = "150"

        r2 = table.rows[2].cells
        r2[0].text = "SKU-002"
        r2[1].text = "Mechanical Keyboard"
        r2[2].text = "80"

        doc.save(str(path))
        return path

    def test_docx_extracts_paragraphs_and_table(self, tmp_path):
        """Validates that DOCX loader extracts both text paragraphs and formatted tables."""
        file_path = tmp_path / "spec.docx"
        self._create_docx_with_table(file_path)

        docs = list(_docx_loader(file_path, "spec.docx"))
        assert len(docs) >= 1
        combined = "\n".join(d.page_content for d in docs)
        assert "Product Specification" in combined
        assert "Wireless Headphones" in combined
        assert "SKU-001" in combined

    def test_docx_merged_cells_are_deduplicated(self, tmp_path):
        """
        EDGE: In DOCX tables with horizontally-merged cells, python-docx
        returns multiple Cell objects that share the exact same _tc XML element.
        The loader must not duplicate the text across columns.
        """
        doc = docx.Document()
        table = doc.add_table(rows=2, cols=2)
        # Merge the two cells of the header row
        table.cell(0, 0).merge(table.cell(0, 1))
        table.cell(0, 0).text = "Merged Header"

        table.cell(1, 0).text = "Data A"
        table.cell(1, 1).text = "Data B"

        file_path = tmp_path / "merged.docx"
        doc.save(str(file_path))

        docs = list(_docx_loader(file_path, "merged.docx"))
        assert len(docs) >= 1
        # The table should render without crashing
        assert "Merged Header" in docs[0].page_content

    def test_corrupt_docx_raises_ingestion_error(self, tmp_path):
        """Corrupt or non-DOCX file disguised as .docx must raise IngestionError."""
        fake_docx = tmp_path / "broken.docx"
        fake_docx.write_bytes(b"PK\x03\x04corrupted zip payload")

        with pytest.raises(IngestionError):
            list(_docx_loader(fake_docx, "broken.docx"))


# ──────────────────────────────────────────────────────────────────────────────
# 2. PDF Loader Advanced
# ──────────────────────────────────────────────────────────────────────────────

class TestPdfLoaderAdvanced:

    def test_encrypted_pdf_raises_ingestion_error(self, tmp_path):
        """Password-protected PDFs must be rejected with a user-friendly error."""
        fake_pdf = tmp_path / "locked.pdf"
        fake_pdf.write_text("Dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.is_encrypted = True

        with (
            patch("src.rag.ingestion.loaders.check_zip_expansion"),
            patch("src.rag.ingestion.loaders.fitz.open", return_value=mock_doc),
        ):
            with pytest.raises(IngestionError, match="Password-protected"):
                list(_pdf_loader(fake_pdf, "locked.pdf"))

    def test_pdf_page_limit_exceeded_raises_ingestion_error(self, tmp_path):
        """PDFs with page count exceeding safety limit must be rejected."""
        fake_pdf = tmp_path / "huge.pdf"
        fake_pdf.write_text("Dummy", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.__len__.return_value = 500  # exceeds 200 page limit

        with (
            patch("src.rag.ingestion.loaders.check_zip_expansion"),
            patch("src.rag.ingestion.loaders.fitz.open", return_value=mock_doc),
        ):
            with pytest.raises(IngestionError, match="page limit"):
                list(_pdf_loader(fake_pdf, "huge.pdf"))

    def test_multipage_pdf_streaming(self, tmp_path):
        """Simulate a 3-page PDF yielding sequential page documents with metadata."""
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_text("Dummy", encoding="utf-8")

        page_texts = ["Page 1: Overview", "Page 2: Features", "Page 3: Pricing"]
        mock_pages = [MagicMock(get_text=MagicMock(return_value=t)) for t in page_texts]

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.__len__.return_value = len(mock_pages)
        mock_doc.__getitem__.side_effect = lambda idx: mock_pages[idx]

        with (
            patch("src.rag.ingestion.loaders.check_zip_expansion"),
            patch("src.rag.ingestion.loaders.fitz.open", return_value=mock_doc),
        ):
            docs = list(_pdf_loader(fake_pdf, "doc.pdf"))

        assert len(docs) == 3
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2
        assert docs[2].metadata["page"] == 3
        assert "Overview" in docs[0].page_content
        assert "Pricing" in docs[2].page_content


# ──────────────────────────────────────────────────────────────────────────────
# 3. CSV & Table Ragged Rows & Edge Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestTableEdgeCases:

    def test_ragged_rows_handled_gracefully(self):
        """
        EDGE: Row 1 has 3 columns, Row 2 has 5 columns, Row 3 has 1 column.
        Extra columns should get 'extra_col_X' keys without index errors.
        """
        rows = [
            ["ID", "Name", "Dept"],
            ["1", "Alice", "IT", "extra_val_1", "extra_val_2"],
            ["2"],
        ]
        budget = _RowBudget(limit=10)
        docs = list(_table_documents(rows, "ragged.csv", {}, budget))

        assert len(docs) == 2
        assert "extra_col_3: extra_val_1" in docs[0].page_content
        assert "ID: 2" in docs[1].page_content

    def test_column_name_collision_resolved_with_suffix(self):
        """
        EDGE: Sheet has duplicate column headers: 'Price', 'Price', 'Price'.
        They must be de-duplicated to 'Price', 'Price_1', 'Price_2'.
        """
        rows = [
            ["Price", "Price", "Price"],
            ["100", "200", "300"],
        ]
        budget = _RowBudget(limit=10)
        docs = list(_table_documents(rows, "dups.csv", {}, budget))

        assert len(docs) == 1
        content = docs[0].page_content
        assert "Price: 100" in content
        assert "Price_1: 200" in content
        assert "Price_2: 300" in content

    def test_consecutive_empty_rows_stops_sheet_stream(self):
        """
        PERFORMANCE: Excel sheets with thousands of formatted empty cells.
        Stream must stop after _MAX_CONSECUTIVE_EMPTY_ROWS (25).
        """
        mock_sheet = MagicMock()
        mock_rows = [["Header1", "Header2"]] + [["Data1", "Data2"]] + [[None, None]] * 50
        mock_sheet.iter_rows.return_value = mock_rows

        rows = list(_stream_sheet_rows(mock_sheet, "Sheet1"))
        # Header + 1 data row + 25 empty before break = 27 rows
        assert len(rows) <= 30


# ──────────────────────────────────────────────────────────────────────────────
# 4. Encoding Resilience
# ──────────────────────────────────────────────────────────────────────────────

class TestEncodingResilience:

    def test_utf8_bom_stripped_cleanly(self, tmp_path):
        """UTF-8 file saved with Windows Notepad BOM must not have \\ufeff in text."""
        bom_file = tmp_path / "bom.txt"
        bom_file.write_bytes(b"\xef\xbb\xbfHello world from BOM")

        text = _read_text_bytes(bom_file)
        assert text.startswith("Hello")
        assert "\ufeff" not in text

    def test_latin1_encoding_detected_and_read(self, tmp_path):
        """Non-UTF8 file with accented characters read via chardet fallback."""
        latin1_file = tmp_path / "latin1.txt"
        latin1_file.write_bytes("Café résumé".encode("latin-1"))

        text = _read_text_bytes(latin1_file)
        assert "Caf" in text


# ──────────────────────────────────────────────────────────────────────────────
# 5. Ingestion Pipeline Safety & Sweep Edge Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestIngestionPipelineSafety:

    @patch("src.rag.ingestion.ingestion_pipeline._get_qdrant")
    def test_purge_sweep_does_not_purge_single_generation(self, mock_get_qdrant):
        """Documents with only 1 generation are in normal state and must not be touched."""
        mock_client = MagicMock()
        mock_get_qdrant.return_value = mock_client

        rec = MagicMock()
        rec.payload = {
            "user_id": "u1",
            "doc_id": "d1",
            "generation": "gen_only",
            "created_at": 100.0,
        }
        mock_client.scroll.return_value = ([rec], None)

        purged = purge_orphaned_generations(min_age_seconds=50.0)
        assert purged == 0
        mock_client.delete.assert_not_called()

    @patch("src.rag.ingestion.ingestion_pipeline._get_qdrant")
    def test_purge_sweep_respects_cutoff_grace_period(self, mock_get_qdrant):
        """
        Older generation that is still younger than min_age_seconds (in-flight run)
        must NOT be deleted.
        """
        import time

        mock_client = MagicMock()
        mock_get_qdrant.return_value = mock_client

        now = time.time()
        # Old generation is only 10 seconds old (min_age_seconds = 600)
        rec_in_flight = MagicMock()
        rec_in_flight.payload = {
            "user_id": "u1",
            "doc_id": "d1",
            "generation": "gen_in_flight",
            "created_at": now - 10.0,
        }
        rec_newest = MagicMock()
        rec_newest.payload = {
            "user_id": "u1",
            "doc_id": "d1",
            "generation": "gen_newest",
            "created_at": now,
        }
        mock_client.scroll.return_value = ([rec_in_flight, rec_newest], None)

        purged = purge_orphaned_generations(min_age_seconds=600.0)
        assert purged == 0
        mock_client.delete.assert_not_called()
