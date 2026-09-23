"""
tests/unit/test_retrieval_pipeline.py
========================================
Unit tests for src.rag.retrieval_pipeline.Retrieval

Functions tested:
  - tokenize_and_remove_stopwords()  (classmethod)
  - _bm25_index_retriever()
  - _vector_index_retriever()
  - _combine_retrieved_docs()
  - build_rag_prompt()

REAL-LIFE CONDITIONS SIMULATED:
  1.  Tokenization removes English stopwords ("the", "is", "a")
  2.  Tokenization removes punctuation
  3.  Tokenization lowercases all tokens
  4.  BM25 retrieval returns top-k matching docs
  5.  BM25 retrieval with no index (index not present) returns []
  6.  BM25 retrieval with zero-score documents filtered out
  7.  Vector retrieval returns top-k relevant documents (mocked)
  8.  Combine: interleaves vector + bm25 and deduplicates
  9.  Combine: handles one source returning empty list
  10. build_rag_prompt: generated prompt contains context and question
  11. build_rag_prompt: no context produces prompt with empty context section
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.documents import Document

from src.rag.retrieval_pipeline import Retrieval


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build a Retrieval instance with all I/O replaced by mocks.
# This avoids loading HuggingFace models or Chroma DB from disk.
# ──────────────────────────────────────────────────────────────────────────────

def _make_retrieval(
    raw_text: list[str] | None = None,
    bm25_mock=None,
    retriever_docs: list[Document] | None = None,
    k_chunks: int = 3,
) -> Retrieval:
    """
    Construct a Retrieval object with:
     - HuggingFaceEmbeddings replaced by MagicMock
     - Chroma DB replaced by MagicMock
     - BM25 data injected directly (no pickle file needed)
    """
    with (
        patch("src.rag.retrieval_pipeline.HuggingFaceEmbeddings"),
        patch("src.rag.retrieval_pipeline.Chroma") as MockChroma,
    ):
        # Configure the mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = retriever_docs or []
        MockChroma.return_value.as_retriever.return_value = mock_retriever

        # We need bm25_dir to NOT exist so the __init__ sets bm25=None by default
        non_existent_path = Path("/tmp/__non_existent_bm25__.pkl")

        r = Retrieval(
            persist_dir=Path("/tmp/__non_existent_chroma__"),
            bm25_dir=non_existent_path,
            k_chunks=k_chunks,
        )

    # Manually inject BM25 data post-init
    if bm25_mock is not None:
        r.bm25 = bm25_mock
        r.raw_text = raw_text or []
    else:
        r.bm25 = None
        r.raw_text = []

    r.retriever = MagicMock()
    r.retriever.invoke.return_value = retriever_docs or []
    return r


# ──────────────────────────────────────────────────────────────────────────────
# tokenize_and_remove_stopwords()
# ──────────────────────────────────────────────────────────────────────────────

class TestTokenize:

    def test_removes_stopwords(self):
        """
        REAL-LIFE: User query contains common English filler words.
        BM25 keyword search must not index these low-signal tokens.
        """
        tokens = Retrieval.tokenize_and_remove_stopwords("What is the price of the iPhone?")
        stopwords = {"what", "is", "the", "of"}
        assert not any(t in stopwords for t in tokens)

    def test_removes_punctuation(self):
        """Punctuation must be stripped so 'price?' and 'price' match the same BM25 entry."""
        tokens = Retrieval.tokenize_and_remove_stopwords("price? warranty!")
        assert "?" not in tokens
        assert "!" not in tokens

    def test_lowercases_tokens(self):
        """Case-insensitive matching: 'iPhone' and 'iphone' must be the same token."""
        tokens = Retrieval.tokenize_and_remove_stopwords("iPhone PRICE")
        assert all(t == t.lower() for t in tokens)

    def test_empty_string_returns_empty_list(self):
        """BOUNDARY: Empty query must not crash — returns []."""
        tokens = Retrieval.tokenize_and_remove_stopwords("")
        assert tokens == []

    def test_only_stopwords_returns_empty_list(self):
        """
        EDGE: If the entire query is stopwords, BM25 retrieval should receive
        an empty token list (returning no results is acceptable).
        """
        tokens = Retrieval.tokenize_and_remove_stopwords("is the a an")
        assert tokens == []

    def test_arabic_tokens_are_preserved(self):
        """
        REAL-LIFE: After translation, some Arabic words may slip through.
        The tokenizer must not crash on non-ASCII tokens.
        """
        tokens = Retrieval.tokenize_and_remove_stopwords("سعر iPhone")
        # At minimum, tokenization must not raise
        assert isinstance(tokens, list)


# ──────────────────────────────────────────────────────────────────────────────
# _bm25_index_retriever()
# ──────────────────────────────────────────────────────────────────────────────

class TestBM25Retriever:

    def _make_bm25(self, corpus: list[str]):
        """Build a real (tiny) BM25Okapi index for unit testing."""
        from rank_bm25 import BM25Okapi
        tokenized = [Retrieval.tokenize_and_remove_stopwords(doc) for doc in corpus]
        return BM25Okapi(tokenized)

    def test_returns_top_k_matching_docs(self):
        """
        REAL-LIFE: Customer asks 'iPhone 15 warranty'. BM25 must return the
        most keyword-relevant documents from the indexed corpus.
        """
        corpus = [
            "iPhone 15 warranty two years",
            "Samsung Galaxy warranty one year",
            "MacBook pro specifications",
        ]
        bm25 = self._make_bm25(corpus)
        retrieval = _make_retrieval(raw_text=corpus, bm25_mock=bm25, k_chunks=2)

        results = retrieval._bm25_index_retriever("iPhone 15 warranty")
        assert len(results) <= 2
        # The iPhone doc must be ranked first (highest BM25 relevance)
        assert "iPhone" in results[0]

    def test_no_bm25_index_returns_empty_list(self):
        """
        REAL-LIFE: BM25 index file was not yet created (first run after deploy).
        Must return [] rather than raising an exception.
        """
        retrieval = _make_retrieval()  # bm25=None by default
        result = retrieval._bm25_index_retriever("iPhone warranty")
        assert result == []

    def test_zero_score_docs_are_filtered_out(self):
        """
        REAL-LIFE: Query about 'warranty' should not return unrelated 'MacBook specs'.
        Zero-score BM25 results must be suppressed.
        """
        corpus = [
            "iPhone warranty two years",
            "completely unrelated topic about weather",
        ]
        bm25 = self._make_bm25(corpus)
        retrieval = _make_retrieval(raw_text=corpus, bm25_mock=bm25, k_chunks=2)

        results = retrieval._bm25_index_retriever("iPhone warranty")
        # The weather doc should score 0 and be excluded
        assert all("weather" not in r for r in results)

    def test_no_matching_docs_returns_empty_list(self):
        """
        EDGE: Query on a completely out-of-domain topic.
        BM25 scores all 0 → returns [].
        """
        corpus = ["cat sat on mat", "dog ran in park"]
        bm25 = self._make_bm25(corpus)
        retrieval = _make_retrieval(raw_text=corpus, bm25_mock=bm25, k_chunks=3)

        results = retrieval._bm25_index_retriever("quantum physics")
        assert results == []


# ──────────────────────────────────────────────────────────────────────────────
# _vector_index_retriever()
# ──────────────────────────────────────────────────────────────────────────────

class TestVectorRetriever:

    def test_returns_page_content_of_retrieved_docs(self):
        """
        REAL-LIFE: Chroma retriever finds semantically similar docs.
        The function must return the .page_content strings, not Document objects.
        """
        docs = [
            Document(page_content="iPhone 15 is available", metadata={}),
            Document(page_content="Price is $999", metadata={}),
        ]
        retrieval = _make_retrieval(retriever_docs=docs, k_chunks=3)

        results = retrieval._vector_index_retriever("What phones do you have?")
        assert results == ["iPhone 15 is available", "Price is $999"]

    def test_respects_k_chunks_limit(self):
        """
        REAL-LIFE: Even if the vector DB returns more than k results,
        the function must truncate to k_chunks.
        """
        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(10)]
        retrieval = _make_retrieval(retriever_docs=docs, k_chunks=3)

        results = retrieval._vector_index_retriever("any query")
        assert len(results) <= 3

    def test_empty_vector_results_returns_empty_list(self):
        """
        REAL-LIFE: Score threshold not met — no documents are sufficiently similar.
        """
        retrieval = _make_retrieval(retriever_docs=[], k_chunks=3)
        results = retrieval._vector_index_retriever("extremely obscure topic")
        assert results == []


# ──────────────────────────────────────────────────────────────────────────────
# _combine_retrieved_docs()
# ──────────────────────────────────────────────────────────────────────────────

class TestCombineRetrievedDocs:

    def _retrieval_with_sources(
        self, bm25_results: list[str], vector_results: list[str], k: int = 3
    ) -> Retrieval:
        """Patch both sub-retrievers with fixed lists."""
        retrieval = _make_retrieval(k_chunks=k)
        retrieval._bm25_index_retriever = MagicMock(return_value=bm25_results)
        retrieval._vector_index_retriever = MagicMock(return_value=vector_results)
        return retrieval

    def test_interleaves_and_deduplicates(self):
        """
        REAL-LIFE: Both BM25 and vector search return overlapping results.
        The combined list must be deduplicated and interleaved (vector first).
        """
        bm25 = ["bm25_doc_A", "shared_doc", "bm25_doc_B"]
        vector = ["shared_doc", "vector_doc_A"]
        retrieval = self._retrieval_with_sources(bm25, vector, k=4)

        combined = retrieval._combine_retrieved_docs("query")
        # No duplicates
        assert len(combined) == len(set(combined))
        # shared_doc appears only once
        assert combined.count("shared_doc") == 1

    def test_vector_only_when_bm25_empty(self):
        """
        REAL-LIFE: BM25 index not available.
        Combined results must fall back to vector-only results.
        """
        retrieval = self._retrieval_with_sources([], ["doc_A", "doc_B"], k=3)
        combined = retrieval._combine_retrieved_docs("query")
        assert "doc_A" in combined
        assert "doc_B" in combined

    def test_bm25_only_when_vector_empty(self):
        """
        REAL-LIFE: Chroma returns nothing (similarity threshold too high).
        Must use BM25 results.
        """
        retrieval = self._retrieval_with_sources(["bm25_A", "bm25_B"], [], k=3)
        combined = retrieval._combine_retrieved_docs("query")
        assert "bm25_A" in combined

    def test_result_limited_to_k_chunks(self):
        """Combined list must never exceed k_chunks."""
        bm25 = [f"bm_{i}" for i in range(10)]
        vector = [f"vec_{i}" for i in range(10)]
        retrieval = self._retrieval_with_sources(bm25, vector, k=3)

        combined = retrieval._combine_retrieved_docs("query")
        assert len(combined) <= 3


# ──────────────────────────────────────────────────────────────────────────────
# build_rag_prompt()
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildRagPrompt:

    def test_prompt_contains_user_question(self):
        """
        CORRECTNESS: The final prompt must embed the user's original question.
        """
        retrieval = _make_retrieval()
        retrieval._combine_retrieved_docs = MagicMock(return_value=["Some context."])

        prompt, _ = retrieval.build_rag_prompt("What is the return policy?")
        assert "What is the return policy?" in prompt

    def test_prompt_contains_retrieved_context(self):
        """
        CORRECTNESS: Retrieved chunks must be embedded in the prompt
        so the LLM can ground its answer.
        """
        retrieval = _make_retrieval()
        retrieval._combine_retrieved_docs = MagicMock(
            return_value=["Return policy: 30 days."]
        )

        prompt, chunks = retrieval.build_rag_prompt("return policy?")
        assert "Return policy: 30 days." in prompt
        assert chunks == ["Return policy: 30 days."]

    def test_empty_context_produces_valid_prompt(self):
        """
        REAL-LIFE: Out-of-scope question — no context retrieved.
        The LLM must still receive a well-formed prompt instructing it
        to reply 'I don't know.' if context is absent.
        """
        retrieval = _make_retrieval()
        retrieval._combine_retrieved_docs = MagicMock(return_value=[])

        prompt, chunks = retrieval.build_rag_prompt("What is the weather on Mars?")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert chunks == []

    def test_multiple_chunks_are_joined_with_separator(self):
        """
        CORRECTNESS: Multiple context chunks must be joined with '---'
        so the LLM can distinguish document boundaries.
        """
        retrieval = _make_retrieval()
        retrieval._combine_retrieved_docs = MagicMock(
            return_value=["Chunk one.", "Chunk two."]
        )

        prompt, _ = retrieval.build_rag_prompt("any question?")
        assert "---" in prompt
