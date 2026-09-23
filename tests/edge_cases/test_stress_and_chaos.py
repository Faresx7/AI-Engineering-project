"""
tests/edge_cases/test_stress_and_chaos.py
==========================================
Stress, chaos, and adversarial edge-case tests.

REAL-LIFE CONDITIONS SIMULATED:
  1.  Memory storm: 50 000 concurrent duplicate MIDs hitting the cache
  2.  Extreme payload fields: null, empty arrays, missing keys at all levels
  3.  Unicode bomb / emoji floods in message text
  4.  Extremely long message text (10 000 chars)
  5.  Deeply nested / unexpected JSON structure accepted gracefully
  6.  All three routes simultaneously receive their maximum batch size
  7.  Mixed valid + invalid entries in one payload (partial batch)
  8.  BM25 retriever called with only stopwords (empty token list)
  9.  RAGChain.answer() called with whitespace / control characters
  10. Retry function with retries=0 → returns None immediately
  11. Cache eviction under high throughput (LRU stress)
  12. Semaphore: 50 concurrent HTTP posts (5x the semaphore limit)
  13. Security: Signature with correct prefix but wrong hex digest
  14. Security: sha256 prefix repeated ('sha256=sha256=...')
  15. HMAC with empty secret
"""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from tests.conftest import make_valid_signature, make_instagram_payload, make_whatsapp_payload

APP_SECRET = "test_app_secret_xyz"
WA_APP_SECRET = "wa_app_secret_fake"
MS_APP_SECRET = "ms_app_secret_fake"


# ──────────────────────────────────────────────────────────────────────────────
# 1. MessageCache under memory storm
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheMemoryStorm:

    def test_50k_unique_inserts_respects_max_size(self):
        """
        REAL-LIFE STRESS: High-traffic day with 50 000 unique messages.
        Cache must never grow beyond its declared max_size.
        """
        from src.core.cache import MessageCache
        cache = MessageCache(max_size=5000)
        for i in range(50_000):
            cache.set_if_absent(f"wamid_{i:010d}", f"msg_{i}")
        assert len(cache._cache) <= 5000

    def test_50k_same_mid_only_one_accepted(self):
        """
        REAL-LIFE DDOS: Attacker re-sends the same MID 50 000 times hoping
        to trigger repeated RAG calls.  Only one must succeed.
        """
        from src.core.cache import MessageCache
        cache = MessageCache(max_size=10_000)
        results = [cache.set_if_absent("same_mid", "text") for _ in range(50_000)]
        assert results.count(True) == 1
        assert results.count(False) == 49_999


# ──────────────────────────────────────────────────────────────────────────────
# 2. Payload field extremes
# ──────────────────────────────────────────────────────────────────────────────

class TestPayloadExtremes:

    def _post(self, client, payload: dict, secret: str = APP_SECRET, route: str = "/webhook/instagram"):
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, secret)
        return client.post(
            route,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(body)),
            },
        )

    def test_null_text_field_does_not_crash(self, test_client):
        """
        EDGE: Meta sends message with text=null instead of omitting the key.
        """
        payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": "p1",
                    "messaging": [
                        {
                            "sender": {"id": "u1"},
                            "recipient": {"id": "b1"},
                            "message": {"mid": "null_text_mid", "text": None},
                        }
                    ],
                }
            ],
        }
        resp = self._post(test_client, payload)
        assert resp.status_code == 200

    def test_empty_entry_list(self, test_client):
        """EDGE: Ping with empty entries."""
        payload = {"object": "instagram", "entry": []}
        resp = self._post(test_client, payload)
        assert resp.status_code == 200

    def test_missing_sender_id_does_not_crash(self, test_client):
        """
        EDGE: 'sender' key present but 'id' is missing.
        The bot can't reply, but must not crash.
        """
        payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": "p1",
                    "messaging": [
                        {
                            "sender": {},                      # no 'id'
                            "recipient": {"id": "b1"},
                            "message": {"mid": "no_sender_mid", "text": "Hello"},
                        }
                    ],
                }
            ],
        }
        resp = self._post(test_client, payload)
        assert resp.status_code == 200

    def test_missing_message_key_does_not_crash(self, test_client):
        """
        EDGE: Event has no 'message' key (e.g., postback-only event).
        """
        payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": "p1",
                    "messaging": [{"sender": {"id": "u1"}, "recipient": {"id": "b1"}}],
                }
            ],
        }
        resp = self._post(test_client, payload)
        assert resp.status_code == 200

    def test_unicode_bomb_emoji_in_text(self, test_client):
        """
        REAL-LIFE: User sends a flood of emojis.
        No encoding crash, no JSON parse failure.
        """
        emoji_text = "🔥" * 2000 + "💥" * 500
        payload = make_instagram_payload(mid="emoji_mid_001", text=emoji_text)
        resp = self._post(test_client, payload)
        assert resp.status_code == 200

    def test_very_long_text_message(self, test_client):
        """
        REAL-LIFE: User pastes a 10 000-char text block (copy-paste from doc).
        Must be accepted and processed without OOM or timeout.
        """
        long_text = "A" * 10_000
        payload = make_instagram_payload(mid="long_text_mid", text=long_text)
        resp = self._post(test_client, payload)
        assert resp.status_code == 200

    def test_mixed_valid_and_null_events_in_batch(self, test_client):
        """
        REAL-LIFE: Partially malformed batch — some entries valid, some have
        empty messaging lists.  Valid ones should still be processed.
        """
        payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": "p1",
                    "messaging": [
                        {
                            "sender": {"id": "u1"},
                            "recipient": {"id": "b1"},
                            "message": {"mid": "partial_valid_mid", "text": "Good message"},
                        }
                    ],
                },
                {"id": "p2", "messaging": []},    # empty — no events
            ],
        }
        resp = self._post(test_client, payload)
        assert resp.status_code == 200

    def test_deeply_nested_referral_object(self, test_client):
        """
        EDGE: Referral object contains unexpected nested keys.
        Code that calls .get() defensively must not crash.
        """
        payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": "p1",
                    "messaging": [
                        {
                            "sender": {"id": "u_deep"},
                            "recipient": {"id": "b1"},
                            "message": {
                                "mid": "deep_referral_mid",
                                "text": "nested",
                                "referral": {
                                    "ad_id": "ad_1",
                                    "extra": {"nested": {"deeper": "value"}},
                                },
                            },
                        }
                    ],
                }
            ],
        }
        resp = self._post(test_client, payload)
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# 3. Security adversarial tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAdversarialSecurity:

    def test_correct_prefix_wrong_hex_rejected(self, test_client):
        """
        ATTACK: Header has correct 'sha256=' prefix but wrong hex value.
        Must return 403, not silently process.
        """
        body = json.dumps(make_instagram_payload()).encode()
        fake_sig = "sha256=" + "a" * 64   # valid format, wrong digest
        resp = test_client.post(
            "/webhook/instagram",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": fake_sig,
                "Content-Length": str(len(body)),
            },
        )
        assert resp.status_code == 403

    def test_repeated_prefix_rejected(self, test_client):
        """
        ATTACK: 'sha256=sha256=<hex>' — double prefix injection.
        Must fail because compare_digest will detect the mismatch.
        """
        body = json.dumps(make_instagram_payload()).encode()
        real_digest = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
        doubled_sig = f"sha256=sha256={real_digest}"
        resp = test_client.post(
            "/webhook/instagram",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": doubled_sig,
                "Content-Length": str(len(body)),
            },
        )
        assert resp.status_code == 403

    def test_empty_secret_signature_rejected(self):
        """
        CONFIGURATION ERROR: APP_SECRET accidentally set to empty string.
        A payload signed with "" must still be rejected by the real secret.
        """
        from src.core.security import verify_signature
        body = b'{"entry":[]}'
        sig_with_empty_secret = "sha256=" + hmac.new(b"", body, hashlib.sha256).hexdigest()
        # Should fail when compared against the real non-empty secret
        result = verify_signature(body, sig_with_empty_secret, "real_secret")
        assert result is False

    def test_sha1_signature_rejected(self, test_client):
        """
        ATTACK: Attacker provides 'sha1=...' hoping for a fallback check.
        The route only accepts sha256= prefix.
        """
        body = json.dumps(make_instagram_payload()).encode()
        sha1_sig = "sha1=" + hmac.new(APP_SECRET.encode(), body, hashlib.sha1).hexdigest()
        resp = test_client.post(
            "/webhook/instagram",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sha1_sig,
                "Content-Length": str(len(body)),
            },
        )
        assert resp.status_code == 403


# ──────────────────────────────────────────────────────────────────────────────
# 4. HTTP client stress
# ──────────────────────────────────────────────────────────────────────────────

class TestHTTPClientStress:

    @pytest.mark.asyncio
    async def test_50_concurrent_sends_complete_without_deadlock(self, mock_http_client):
        """
        REAL-LIFE STRESS: 50 concurrent coroutines all attempt to send a reply
        simultaneously.  The semaphore (limit=10) must queue them without deadlock
        and all must eventually resolve.
        """
        import src.core.http_client as hc

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ok": True}
        mock_http_client.post.return_value = mock_resp

        tasks = [
            hc.send_message_with_retry(
                "https://example.com/send",
                {"Authorization": "Bearer fake"},
                {"recipient": {"id": str(i)}, "message": {"text": "hi"}},
                retries=1,
            )
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        assert all(r == {"ok": True} for r in results)

    @pytest.mark.asyncio
    async def test_retries_0_returns_none_immediately(self, mock_http_client):
        """
        BOUNDARY: retries=0 means no attempts.
        The loop range(1, 1) is empty → returns None immediately.
        """
        import src.core.http_client as hc
        result = await hc.send_message_with_retry("https://x.com", {}, {}, retries=0)
        assert result is None
        mock_http_client.post.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────────────
# 5. RAG chain edge inputs
# ──────────────────────────────────────────────────────────────────────────────

class TestRAGChainEdgeInputs:

    @pytest.mark.asyncio
    async def test_whitespace_only_skipped_by_rag_chain(self, mock_rag_chain):
        """
        REAL-LIFE: User accidentally sends a blank message.
        RAGChain.answer() must not be called with empty/whitespace prompt.
        """
        from src.rag.rag_chain import RAGChain
        import src.rag.retrieval_pipeline as retriever_mod
        import src.rag.cloud_model as model_mod

        with (
            patch.object(retriever_mod, "Retrieval"),
            patch.object(model_mod, "AIModel"),
        ):
            chain = RAGChain.__new__(RAGChain)
            chain.retrieval = MagicMock()
            chain.model = AsyncMock()

            result = await chain.answer("   ")
            assert result["answer"] == "Please provide a valid Question."
            chain.model.generate_response.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_string_returns_validation_message(self, mock_rag_chain):
        """BOUNDARY: Empty string → immediate validation response."""
        from src.rag.rag_chain import RAGChain
        import src.rag.retrieval_pipeline as retriever_mod
        import src.rag.cloud_model as model_mod

        with (
            patch.object(retriever_mod, "Retrieval"),
            patch.object(model_mod, "AIModel"),
        ):
            chain = RAGChain.__new__(RAGChain)
            chain.retrieval = MagicMock()
            chain.model = AsyncMock()

            result = await chain.answer("")
            assert "valid" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_bm25_with_stopwords_only_returns_empty(self):
        """
        REAL-LIFE: User sends 'the is a' — after tokenization all tokens
        are stopwords. BM25 must return [] gracefully (no ZeroDivisionError).
        """
        from rank_bm25 import BM25Okapi
        from src.rag.retrieval_pipeline import Retrieval

        corpus = ["iPhone 15 price", "Samsung warranty"]
        tokenized = [Retrieval.tokenize_and_remove_stopwords(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        # Manually construct a minimal Retrieval with mocked dependencies
        with (
            patch("src.rag.retrieval_pipeline.HuggingFaceEmbeddings"),
            patch("src.rag.retrieval_pipeline.Chroma"),
        ):
            retrieval = Retrieval.__new__(Retrieval)
            retrieval.bm25 = bm25
            retrieval.raw_text = corpus
            retrieval.k_chunks = 3

        # Stopwords-only query produces empty token list
        results = retrieval._bm25_index_retriever("the is a")
        assert results == []


# ──────────────────────────────────────────────────────────────────────────────
# 6. All three routes simultaneous stress
# ──────────────────────────────────────────────────────────────────────────────

class TestAllRoutesSimultaneousLoad:

    def test_all_three_routes_accept_max_batch(self, test_client):
        """
        REAL-LIFE STRESS: All three social channels fire their maximum batch
        at the same time (simulating a campaign launch).
        All routes must respond with EVENT_RECEIVED.
        """
        # Instagram batch (5 entries)
        ig_payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": f"ig_p_{i}",
                    "messaging": [
                        {
                            "sender": {"id": f"ig_u_{i}"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": f"ig_mid_{i}", "text": f"IG msg {i}"},
                        }
                    ],
                }
                for i in range(5)
            ],
        }
        ig_body = json.dumps(ig_payload).encode()
        ig_sig = make_valid_signature(ig_body, APP_SECRET)
        ig_resp = test_client.post(
            "/webhook/instagram",
            content=ig_body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": ig_sig,
                "Content-Length": str(len(ig_body)),
            },
        )
        assert ig_resp.status_code == 200

        # WhatsApp batch (5 messages)
        wa_payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_biz",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": [
                                    {
                                        "id": f"wamid.batch_{i}",
                                        "from": f"+201234{i:06d}",
                                        "type": "text",
                                        "text": {"body": f"WA msg {i}"},
                                    }
                                    for i in range(5)
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        wa_body = json.dumps(wa_payload).encode()
        wa_sig = make_valid_signature(wa_body, "wa_app_secret_fake")
        wa_resp = test_client.post(
            "/webhook/whatsapp",
            content=wa_body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": wa_sig,
                "Content-Length": str(len(wa_body)),
            },
        )
        assert wa_resp.status_code == 200

        # Messenger batch (3 entries)
        ms_payload = {
            "object": "page",
            "entry": [
                {
                    "id": f"ms_p_{i}",
                    "messaging": [
                        {
                            "sender": {"id": f"ms_u_{i}"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": f"ms_mid_{i}", "text": f"MS msg {i}"},
                        }
                    ],
                }
                for i in range(3)
            ],
        }
        ms_body = json.dumps(ms_payload).encode()
        ms_sig = make_valid_signature(ms_body, "ms_app_secret_fake")
        ms_resp = test_client.post(
            "/webhook/messenger",
            content=ms_body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": ms_sig,
                "Content-Length": str(len(ms_body)),
            },
        )
        assert ms_resp.status_code == 200
