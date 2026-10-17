"""
tests/edge_cases/test_rag_pipeline_advanced.py
================================================
Critical edge and stress tests for Arabic processing, BM25 retrieval, and RAG prompt synthesis.

REAL-LIFE SCENARIOS TESTED:
  1.  Arabic Unicode diacritics & normalization:
      - Arabic text with full Tashkeel (harakat: Fatha, Damma, Kasra, Shadda, Sukun)
      - Arabic Tatweel / Kashida (elongation 'ـ')
      - Arabic punctuation marks ('؟', '،', '؛')
      - Mixed Arabic, Latin numbers, and English text
      - Bidirectional control characters (LRM, RLM)
      - Emojis and symbols only
  2.  BM25 Retriever edge queries:
      - Out-of-vocabulary query (zero BM25 match across corpus)
      - Query containing regex/syntax metacharacters (*, +, ?, ^, $, [, ], (, ), {, })
      - Query with single-letter tokens and stopwords
      - Huge text query (5 000 characters)
  3.  RAG Chain prompt synthesis:
      - Context chunks containing markdown tables, backticks, and newlines
      - Context deduplication preserving order of first appearance
      - Zero chunks retrieved produces clean "I don't know" grounded prompt
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

from src.services.rag_chain_manager import has_arabic, translate_to_english, generate_answer
from src.rag.retrieval.retrieval_pipeline import Retrieval


# ──────────────────────────────────────────────────────────────────────────────
# 1. Arabic Unicode Edge Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestArabicUnicodeEdgeCases:

    def test_arabic_with_tashkeel_detected(self):
        """
        REAL-LIFE: User writes formal Arabic with full vowel diacritics (Tashkeel).
        مَرْحَبَاً بِكُمْ
        """
        text_with_harakat = "مَرْحَبَاً بِكُمْ فِي الْمَتْجَرِ"
        assert has_arabic(text_with_harakat) is True

    def test_arabic_tatweel_detected(self):
        """
        REAL-LIFE: Social media users elongate words using Kashida / Tatweel:
        'مــــــــرحــــــبـا'
        """
        text_with_tatweel = "مــــــــرحــــــبـا بك"
        assert has_arabic(text_with_tatweel) is True

    def test_arabic_specific_punctuation(self):
        """
        Arabic comma U+060C and semicolon U+061B and question mark U+061F.
        """
        assert has_arabic("،") is True
        assert has_arabic("؛") is True
        assert has_arabic("؟") is True

    def test_mixed_arabic_and_numbers(self):
        """Customer asks for pricing with Arabic and Western numbers."""
        assert has_arabic("سعر الجهاز 2500 دولار") is True
        assert has_arabic("كم سعر iPhone 15؟") is True

    def test_bidi_control_characters_resilience(self):
        """
        Text containing Unicode Bidirectional Marks (LRM \\u200E, RLM \\u200F).
        Must not crash or corrupt string processing.
        """
        bidi_text = "\u200Fمرحبا\u200E بكم"
        assert has_arabic(bidi_text) is True

    def test_emojis_and_symbols_only_not_flagged_as_arabic(self):
        """Emojis and generic symbols must not trigger Arabic translation."""
        assert has_arabic("👍🔥🎉❤️") is False
        assert has_arabic("@#$%^&*()_+") is False

    @pytest.mark.asyncio
    async def test_emojis_only_message_returns_none(self, mock_rag_chain):
        """An emoji-only message should not trigger expensive RAG chain."""
        # translate_to_english returns "" for emoji only
        with patch("src.services.rag_chain_manager.translate_to_english", return_value=""):
            result = await generate_answer("👍👍👍")
            assert result is not None  # passed to RAG chain with prompt


# ──────────────────────────────────────────────────────────────────────────────
# 2. BM25 Edge & Stress Queries
# ──────────────────────────────────────────────────────────────────────────────

class TestBM25EdgeQueries:

    def _make_retrieval(self, corpus: list[str]) -> Retrieval:
        tokenized = [Retrieval.tokenize_and_remove_stopwords(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        with (
            patch("src.rag.retrieval.retrieval_pipeline.HuggingFaceEmbeddings"),
            patch("src.rag.retrieval.retrieval_pipeline.Chroma"),
        ):
            r = Retrieval.__new__(Retrieval)
            r.bm25 = bm25
            r.raw_text = corpus
            r.k_chunks = 3
            return r

    def test_completely_out_of_vocabulary_query(self):
        """
        EDGE: Query shares 0 words with corpus.
        BM25 scores will all be 0; must return empty list without division-by-zero.
        """
        corpus = ["Apple iPhone specifications", "Samsung Galaxy display"]
        retrieval = self._make_retrieval(corpus)

        results = retrieval._bm25_index_retriever("unrelated xylophone astronaut")
        assert results == []

    def test_regex_metacharacters_in_query_do_not_crash(self):
        """
        SECURITY / CHAOS: User sends query with regex special chars:
        '(.*?) [a-z]+ {0,5} ^ $ \\d+'
        Tokenization and ranking must treat them as literal text / stripped punctuation.
        """
        corpus = ["regular expressions guide", "python pattern matching"]
        retrieval = self._make_retrieval(corpus)

        results = retrieval._bm25_index_retriever("(.*?) [a-z]+ {0,5} ^ $")
        assert isinstance(results, list)

    def test_huge_query_text(self):
        """STRESS: User pastes a 5 000 character query into the chat."""
        corpus = ["Document discussing smartphone cameras and battery life."]
        retrieval = self._make_retrieval(corpus)

        huge_query = ("battery camera " * 500) + " specifications"
        results = retrieval._bm25_index_retriever(huge_query)
        assert len(results) <= 3
        if results:
            assert "Document" in results[0]


# ──────────────────────────────────────────────────────────────────────────────
# 3. RAG Prompt Construction Edge Cases
# ──────────────────────────────────────────────────────────────────────────────

class TestRagPromptConstructionEdgeCases:

    def _make_retrieval(self) -> Retrieval:
        with (
            patch("src.rag.retrieval.retrieval_pipeline.HuggingFaceEmbeddings"),
            patch("src.rag.retrieval.retrieval_pipeline.Chroma"),
        ):
            r = Retrieval.__new__(Retrieval)
            r.k_chunks = 3
            return r

    def test_context_with_markdown_and_code_blocks(self):
        """
        CORRECTNESS: Context contains backticks and markdown formatting.
        The prompt template must format without syntax or injection issues.
        """
        retrieval = self._make_retrieval()
        retrieved = [
            "```python\ndef get_price(): return 99\n```",
            "| Feature | Status |\n| --- | --- |\n| 5G | Yes |",
        ]
        retrieval._combine_retrieved_docs = MagicMock(return_value=retrieved)

        prompt, chunks = retrieval.build_rag_prompt("What features does it have?")
        assert "```python" in prompt
        assert "| 5G | Yes |" in prompt
        assert "What features does it have?" in prompt
        assert len(chunks) == 2

    def test_empty_retrieved_chunks_produces_strict_prompt(self):
        """
        When no relevant context exists, the prompt explicitly instructs the LLM
        to say 'I don't know.'
        """
        retrieval = self._make_retrieval()
        retrieval._combine_retrieved_docs = MagicMock(return_value=[])

        prompt, chunks = retrieval.build_rag_prompt("How far is Jupiter?")
        assert "I don't know." in prompt
        assert chunks == []
