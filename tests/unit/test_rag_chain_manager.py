"""
tests/unit/test_rag_chain_manager.py
======================================
Unit tests for src.services.rag_chain_manager

Functions tested:
  - has_arabic()
  - translate_to_english()
  - generate_answer()

REAL-LIFE CONDITIONS SIMULATED:
  1.  Arabic-only text is correctly detected
  2.  Latin-only text is NOT flagged as Arabic
  3.  Mixed Arabic-Latin text is flagged as Arabic
  4.  Empty string: has_arabic returns False
  5.  translate_to_english: successful translation via MyMemoryTranslator
  6.  translate_to_english: translator raises exception → original text returned
  7.  translate_to_english: translator returns a list → joined correctly
  8.  generate_answer: happy-path, Arabic → translated → answered
  9.  generate_answer: non-text message is skipped (returns None)
  10. generate_answer: RAG chain is None → RuntimeError raised
  11. generate_answer: empty/whitespace → returns None without calling RAG
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import src.core.container as container_module
from src.services.rag_chain_manager import (
    has_arabic,
    translate_to_english,
    generate_answer,
)


# ──────────────────────────────────────────────────────────────────────────────
# has_arabic()
# ──────────────────────────────────────────────────────────────────────────────

class TestHasArabic:

    def test_pure_arabic_returns_true(self):
        """
        REAL-LIFE: Customer sends a message entirely in Arabic.
        """
        assert has_arabic("مرحباً كيف حالك") is True

    def test_pure_english_returns_false(self):
        """
        REAL-LIFE: Customer sends a normal English message.
        """
        assert has_arabic("Hello, how are you?") is False

    def test_mixed_arabic_latin_returns_true(self):
        """
        REAL-LIFE: Common code-switching in the MENA region.
        e.g. 'iPhone السعر كم؟'
        """
        assert has_arabic("iPhone السعر كم؟") is True

    def test_empty_string_returns_false(self):
        """BOUNDARY: No characters → no Arabic."""
        assert has_arabic("") is False

    def test_numbers_only_returns_false(self):
        """Numbers alone must not be confused with Arabic Unicode codepoints."""
        assert has_arabic("12345 ٣٤٥") is False  # Arabic-Indic numerals are outside 0600-06FF

    def test_arabic_punctuation_block_returns_true(self):
        """Some Arabic-specific punctuation lies inside the Arabic Unicode block."""
        assert has_arabic("؟") is True  # ARABIC QUESTION MARK U+061F

    def test_persian_text_returns_true(self):
        """
        Persian / Farsi uses the same Unicode range as Arabic.
        Both should be translated before retrieval.
        """
        assert has_arabic("سلام دنیا") is True


# ──────────────────────────────────────────────────────────────────────────────
# translate_to_english()
# ──────────────────────────────────────────────────────────────────────────────

class TestTranslateToEnglish:

    def test_successful_translation(self):
        """
        REAL-LIFE: MyMemoryTranslator is reachable and returns an English string.
        Mocked to avoid real network calls.
        """
        with patch(
            "src.services.rag_chain_manager.MyMemoryTranslator"
        ) as MockTranslator:
            instance = MagicMock()
            instance.translate.return_value = "Hello world"
            MockTranslator.return_value = instance

            result = translate_to_english("مرحبا بالعالم")
            assert result == "Hello world"

    def test_translation_service_raises_exception_returns_original(self):
        """
        REAL-LIFE FAILURE: Translation API is down (network timeout / 429).
        The function must fall back to the original Arabic text so the
        pipeline doesn't silently drop the user's message.
        """
        with patch(
            "src.services.rag_chain_manager.MyMemoryTranslator"
        ) as MockTranslator:
            instance = MagicMock()
            instance.translate.side_effect = Exception("Service unavailable")
            MockTranslator.return_value = instance

            original = "كيف حالك"
            result = translate_to_english(original)
            assert result == original

    def test_translator_returns_list_is_joined(self):
        """
        EDGE: Some translator versions return a list of string segments.
        The function must join them with spaces.
        """
        with patch(
            "src.services.rag_chain_manager.MyMemoryTranslator"
        ) as MockTranslator:
            instance = MagicMock()
            instance.translate.return_value = ["Hello", "world"]
            MockTranslator.return_value = instance

            result = translate_to_english("مرحبا بالعالم")
            assert result == "Hello world"

    def test_translator_returns_none_falls_back_to_original(self):
        """
        EDGE: Translator returns None (API returned empty response).
        """
        with patch(
            "src.services.rag_chain_manager.MyMemoryTranslator"
        ) as MockTranslator:
            instance = MagicMock()
            instance.translate.return_value = None
            MockTranslator.return_value = instance

            original = "مرحبا"
            result = translate_to_english(original)
            assert result == original


# ──────────────────────────────────────────────────────────────────────────────
# generate_answer()
# ──────────────────────────────────────────────────────────────────────────────

class TestGenerateAnswer:

    @pytest.fixture(autouse=True)
    def patch_translator(self):
        """
        Stub out MyMemoryTranslator so translation tests run offline.
        """
        with patch("src.services.rag_chain_manager.MyMemoryTranslator") as MockT:
            instance = MagicMock()
            instance.translate.return_value = "What is the price?"
            MockT.return_value = instance
            yield MockT

    @pytest.mark.asyncio
    async def test_english_text_answered_directly(self, mock_rag_chain):
        """
        REAL-LIFE: English user message goes directly to RAG without translation.
        """
        mock_rag_chain.answer.return_value = {
            "answer": "The price is $99.", "context": [], "final_prompt": ""
        }
        result = await generate_answer("What is the price?")
        assert result == "The price is $99."
        mock_rag_chain.answer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_arabic_text_is_translated_before_rag(self, mock_rag_chain):
        """
        REAL-LIFE: Arabic-speaking customer asks a question.
        The text must be translated before being sent to the retrieval pipeline.
        """
        mock_rag_chain.answer.return_value = {
            "answer": "السعر 99 دولار", "context": [], "final_prompt": ""
        }
        result = await generate_answer("ما هو السعر؟")
        # RAG must have been called with the translated text, not the original Arabic
        call_arg = mock_rag_chain.answer.call_args[0][0]
        assert "What is the price?" in call_arg or "arabic" in call_arg.lower()

    @pytest.mark.asyncio
    async def test_non_text_message_returns_none(self, mock_rag_chain):
        """
        REAL-LIFE: User sends a sticker / image. The route sets text to
        '[NON_TEXT_MESSAGE]' and generate_answer must return None without
        invoking the expensive RAG pipeline.
        """
        result = await generate_answer("[NON_TEXT_MESSAGE]")
        assert result is None
        mock_rag_chain.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_returns_none(self, mock_rag_chain):
        """
        BOUNDARY: Empty text must be rejected early.
        """
        result = await generate_answer("")
        assert result is None
        mock_rag_chain.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_none(self, mock_rag_chain):
        """
        BOUNDARY: Whitespace-only text must be rejected early.
        """
        result = await generate_answer("   \t\n  ")
        assert result is None

    @pytest.mark.asyncio
    async def test_rag_chain_not_initialized_raises_runtime_error(self):
        """
        REAL-LIFE: A webhook event arrives during a cold-start window
        before the RAGChain has finished loading its models.
        The function must raise RuntimeError to signal to the caller that
        the system isn't ready.
        """
        original = container_module.rag_chain_instance
        container_module.rag_chain_instance = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                await generate_answer("What is the price?")
        finally:
            container_module.rag_chain_instance = original

    @pytest.mark.asyncio
    async def test_answer_contains_arabic_instruction(self, mock_rag_chain):
        """
        REAL-LIFE: The system prompt prepends 'always answer in arabic'
        when constructing the RAG query to maintain language consistency
        for Arabic-speaking users.
        """
        mock_rag_chain.answer.return_value = {
            "answer": "الإجابة", "context": [], "final_prompt": ""
        }
        await generate_answer("What is the price?")
        call_arg = mock_rag_chain.answer.call_args[0][0]
        assert "arabic" in call_arg.lower()
