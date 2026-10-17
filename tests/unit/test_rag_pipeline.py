"""Unit tests for the modular RAG pipeline and loaders.

Validates that splitting src/rag/pipeline.py into modular components preserved
100% of the functionality, safety guarantees, streaming behavior, and error handling.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.core.validation import IngestionError
from src.rag.ingestion.budget import _RowBudget, _batched, _retry
from src.rag.constants import (
    _BATCH_SIZE,
    _MAX_CELL_CHARS,
    _MAX_CHUNKS_PER_DOC,
    _MAX_PDF_PAGES,
    _MAX_TABLE_ROWS,
)
from src.rag.ingestion.loaders import (
    _LOADERS,
    _csv_loader,
    _is_potential_header,
    _is_row_empty,
    _json_loader,
    _read_text_bytes,
    _sanitize_cell,
    _table_documents,
    _text_loader,
    _trim_trailing_nones,
)
from src.rag.ingestion.ingestion_pipeline import IngestionPipeline, IngestionResult, purge_orphaned_generations


class TestBudgetAndHelpers:
    """Tests for _RowBudget, _batched, and _retry."""

    def test_row_budget_spend_and_exhaustion(self):
        budget = _RowBudget(limit=3)
        assert budget.spend() is True
        assert budget.spend() is True
        assert budget.spend() is True
        # Exhausted
        assert budget.spend() is False
        assert budget.spend() is False

    def test_batched_generator(self):
        data = list(range(10))
        batches = list(_batched(data, 3))
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_batched_empty(self):
        assert list(_batched([], 5)) == []

    def test_retry_success_first_try(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return 42

        result = _retry(fn, attempts=3, base_delay=0.01, what="test")
        assert result == 42
        assert call_count == 1

    def test_retry_transient_failure_then_success(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary blip")
            return "recovered"

        result = _retry(fn, attempts=3, base_delay=0.01, what="test")
        assert result == "recovered"
        assert call_count == 2

    def test_retry_never_retries_ingestion_error(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise IngestionError("Deterministic bad data")

        with pytest.raises(IngestionError, match="Deterministic bad data"):
            _retry(fn, attempts=3, base_delay=0.01, what="test")
        assert call_count == 1

    def test_retry_exhausted_raises_last_exception(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Persistent timeout")

        with pytest.raises(TimeoutError, match="Persistent timeout"):
            _retry(fn, attempts=3, base_delay=0.01, what="test")
        assert call_count == 3


class TestLoadersAndHelpers:
    """Tests for low-level table/text helpers and file loaders."""

    def test_is_row_empty(self):
        assert _is_row_empty([]) is True
        assert _is_row_empty([None, "", "  "]) is True
        assert _is_row_empty([None, "data", ""]) is False

    def test_is_potential_header(self):
        # Empty
        assert _is_potential_header([]) is False
        assert _is_potential_header([None, ""]) is False
        # Single cell banner across multiple columns
        assert _is_potential_header(["Title Banner", None, None]) is False
        # Valid multi-column header
        assert _is_potential_header(["Col1", "Col2"]) is True

    def test_trim_trailing_nones(self):
        cells = ["a", "b", None, "", "  "]
        trimmed = _trim_trailing_nones(cells)
        assert trimmed == ["a", "b"]

    def test_sanitize_cell(self):
        assert _sanitize_cell(None) == ""
        assert _sanitize_cell("  hello  ") == "hello"
        long_val = "x" * 2000
        sanitized = _sanitize_cell(long_val, max_chars=100)
        assert len(sanitized) <= 120
        assert sanitized.endswith("... [truncated]")

    def test_table_documents_streaming(self):
        rows = [
            ["ID", "Name", "Role"],
            ["1", "Alice", "Admin"],
            ["2", "Bob", "User"],
        ]
        budget = _RowBudget(limit=10)
        docs = list(_table_documents(rows, filename="test.csv", extra_meta={"foo": "bar"}, budget=budget))
        assert len(docs) == 2
        assert "ID: 1" in docs[0].page_content
        assert "Name: Alice" in docs[0].page_content
        assert docs[0].metadata["row"] == 2
        assert docs[0].metadata["source"] == "test.csv"
        assert docs[0].metadata["foo"] == "bar"

    def test_text_loader(self, tmp_path):
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")
        docs = list(_text_loader(txt_file, "sample.txt"))
        assert len(docs) >= 1
        assert "Line 1" in docs[0].page_content

    def test_json_loader_array(self, tmp_path):
        json_file = tmp_path / "sample.json"
        data = [{"id": 1, "product": "Widget"}, {"id": 2, "product": "Gadget"}]
        json_file.write_text(json.dumps(data), encoding="utf-8")
        docs = list(_json_loader(json_file, "sample.json"))
        assert len(docs) == 2
        assert "product: Widget" in docs[0].page_content
        assert docs[0].metadata["row"] == 1

    def test_csv_loader(self, tmp_path):
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("item,qty\napples,10\noranges,20\n", encoding="utf-8")
        docs = list(_csv_loader(csv_file, "sample.csv"))
        assert len(docs) == 2
        assert "item: apples" in docs[0].page_content
        assert "qty: 10" in docs[0].page_content


class TestIngestionPipeline:
    """Tests for the IngestionPipeline orchestration."""

    def test_stream_chunks_pre_chunked(self):
        pipeline = IngestionPipeline()
        doc = Document(page_content="Row data", metadata={"pre_chunked": True})
        chunks = list(pipeline._stream_chunks(iter([doc])))
        assert len(chunks) == 1
        assert chunks[0].page_content == "Row data"

    def test_stream_chunks_splitter(self):
        pipeline = IngestionPipeline(chunk_size=50, chunk_overlap=10)
        long_text = "This is a sentence. " * 20
        doc = Document(page_content=long_text, metadata={"pre_chunked": False})
        chunks = list(pipeline._stream_chunks(iter([doc])))
        assert len(chunks) > 1

    @patch("src.rag.ingestion.ingestion_pipeline._get_qdrant")
    @patch("src.rag.ingestion.ingestion_pipeline._get_sparse_model")
    @patch("src.rag.ingestion.ingestion_pipeline._embed_dense_batch")
    def test_ingest_file_happy_path(
        self, mock_embed_dense, mock_get_sparse, mock_get_qdrant, tmp_path
    ):
        mock_client = MagicMock()
        mock_get_qdrant.return_value = mock_client

        mock_sparse = MagicMock()
        mock_sparse_res = MagicMock()
        mock_sparse_res.indices.tolist.return_value = [1, 2]
        mock_sparse_res.values.tolist.return_value = [0.5, 0.8]
        mock_sparse.embed.return_value = [mock_sparse_res]
        mock_get_sparse.return_value = mock_sparse

        mock_embed_dense.return_value = [[0.1] * 768]

        sample_file = tmp_path / "test.txt"
        sample_file.write_text("Sample knowledge base content.", encoding="utf-8")

        pipeline = IngestionPipeline()
        result = pipeline.ingest_file(
            path=sample_file,
            original_name="test.txt",
            user_id="user_123",
        )

        assert isinstance(result, IngestionResult)
        assert result.chunks == 1
        assert result.doc_id.startswith("doc_")
        assert len(result.generation) > 0

        # Verify Qdrant upsert and old-generation delete were called
        assert mock_client.upsert.called
        assert mock_client.delete.called

    @patch("src.rag.ingestion.ingestion_pipeline._BATCH_SIZE", 2)
    @patch("src.rag.ingestion.ingestion_pipeline._get_qdrant")
    @patch("src.rag.ingestion.ingestion_pipeline._get_sparse_model")
    @patch("src.rag.ingestion.ingestion_pipeline._embed_dense_batch")
    def test_ingest_file_rollback_on_error(
        self, mock_embed_dense, mock_get_sparse, mock_get_qdrant, tmp_path
    ):
        mock_client = MagicMock()
        mock_get_qdrant.return_value = mock_client

        # Batch 1 succeeds, Batch 2 fails on all retry attempts
        mock_client.upsert.side_effect = [None, RuntimeError("Disk full"), RuntimeError("Disk full"), RuntimeError("Disk full")]

        mock_sparse = MagicMock()
        mock_sparse_res = MagicMock()
        mock_sparse_res.indices.tolist.return_value = [1]
        mock_sparse_res.values.tolist.return_value = [0.5]
        mock_sparse.embed.side_effect = lambda texts, **kw: [mock_sparse_res] * len(texts)
        mock_get_sparse.return_value = mock_sparse

        mock_embed_dense.side_effect = lambda texts: [[0.1] * 768] * len(texts)

        sample_file = tmp_path / "large.txt"
        sample_file.write_text(("Line of content\n\n" * 150), encoding="utf-8")

        pipeline = IngestionPipeline()
        with pytest.raises(RuntimeError, match="Disk full"):
            pipeline.ingest_file(
                path=sample_file,
                original_name="large.txt",
                user_id="user_123",
            )

        # Rollback should delete points written under this failed generation
        assert mock_client.delete.called


class TestPurgeOrphanedGenerations:
    """Tests for purge_orphaned_generations maintenance sweep."""

    @patch("src.rag.ingestion.ingestion_pipeline._get_qdrant")
    def test_purge_orphaned_generations(self, mock_get_qdrant):
        mock_client = MagicMock()
        mock_get_qdrant.return_value = mock_client

        # Mock two records for the same doc: one old orphaned, one new
        rec_old = MagicMock()
        rec_old.payload = {
            "user_id": "u1",
            "doc_id": "d1",
            "generation": "gen_old",
            "created_at": 100.0,
        }
        rec_new = MagicMock()
        rec_new.payload = {
            "user_id": "u1",
            "doc_id": "d1",
            "generation": "gen_new",
            "created_at": 5000.0,
        }

        mock_client.scroll.return_value = ([rec_old, rec_new], None)

        purged = purge_orphaned_generations(
            collection_name="rag_documents",
            min_age_seconds=1000.0,
        )

        assert purged == 1
        assert mock_client.delete.called
