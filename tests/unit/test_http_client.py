"""
tests/unit/test_http_client.py
================================
Unit tests for src.core.http_client

Functions tested:
  - send_message_with_retry()
  - get_message_by_mid()

REAL-LIFE CONDITIONS SIMULATED:
  1.  Successful HTTP 200 on first attempt
  2.  HTTP 500 on all attempts → returns None (server down)
  3.  Network timeout on every attempt → returns None
  4.  HTTP 500 twice, then 200 on third → retry recovery
  5.  http_client not initialised (lifespan not started)
  6.  Semaphore: 12 concurrent sends (exceeds concurrency=10 cap)
  7.  get_message_by_mid: HTTP 200 with valid JSON
  8.  get_message_by_mid: HTTP 404 returns empty dict
  9.  get_message_by_mid: Network exception returns empty dict
  10. get_message_by_mid: http_client is None
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx


# ──────────────────────────────────────────────────────────────────────────────
# Import under test
# ──────────────────────────────────────────────────────────────────────────────
import src.core.http_client as hc


URL = "https://graph.facebook.com/v26.0/me/messages"
HEADERS = {"Authorization": "Bearer FAKE_TOKEN"}
JSON_DATA = {"recipient": {"id": "123"}, "message": {"text": "hi"}}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a mock httpx.Response
# ──────────────────────────────────────────────────────────────────────────────

def _mock_response(status_code: int, json_body: dict) -> AsyncMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# send_message_with_retry()
# ──────────────────────────────────────────────────────────────────────────────

class TestSendMessageWithRetry:

    @pytest.fixture(autouse=True)
    def patch_sleep(self):
        """
        PERFORMANCE: Replace asyncio.sleep with a no-op so retry-delay tests
        complete in milliseconds rather than real seconds.
        """
        with patch("src.core.http_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            yield mock_sleep

    @pytest.mark.asyncio
    async def test_returns_json_on_first_200(self, mock_http_client):
        """
        REAL-LIFE GREEN PATH: API responds immediately with 200.
        """
        mock_http_client.post.return_value = _mock_response(200, {"message_id": "abc"})
        result = await hc.send_message_with_retry(URL, HEADERS, JSON_DATA, retries=3)
        assert result == {"message_id": "abc"}
        mock_http_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_all_retries_fail_with_500(self, mock_http_client):
        """
        REAL-LIFE: Graph API is degraded and returns 500 on every attempt.
        After exhausting retries the function must silently return None
        (so the calling coroutine can decide how to handle the failure).
        """
        mock_http_client.post.return_value = _mock_response(500, {"error": "server error"})
        result = await hc.send_message_with_retry(URL, HEADERS, JSON_DATA, retries=3)
        assert result is None
        assert mock_http_client.post.await_count == 3   # exactly 3 attempts

    @pytest.mark.asyncio
    async def test_returns_none_on_network_timeout(self, mock_http_client):
        """
        REAL-LIFE: Network timeout on every attempt (e.g., Meta API unreachable).
        The exception must be caught and None returned — never raised to caller.
        """
        mock_http_client.post.side_effect = httpx.TimeoutException("timed out")
        result = await hc.send_message_with_retry(URL, HEADERS, JSON_DATA, retries=3)
        assert result is None

    @pytest.mark.asyncio
    async def test_retry_recovery_on_third_attempt(self, mock_http_client):
        """
        REAL-LIFE: Transient server error recovers on the third attempt.
        Common with Meta APIs during brief overload windows.
        """
        fail_resp = _mock_response(500, {"error": "overloaded"})
        ok_resp = _mock_response(200, {"message_id": "recovered"})
        mock_http_client.post.side_effect = [fail_resp, fail_resp, ok_resp]

        result = await hc.send_message_with_retry(URL, HEADERS, JSON_DATA, retries=3)
        assert result == {"message_id": "recovered"}
        assert mock_http_client.post.await_count == 3

    @pytest.mark.asyncio
    async def test_returns_none_when_http_client_is_none(self):
        """
        REAL-LIFE: A message arrives before FastAPI lifespan has finished
        initialising the shared http_client.
        """
        original = hc.http_client
        hc.http_client = None
        try:
            result = await hc.send_message_with_retry(URL, HEADERS, JSON_DATA)
            assert result is None
        finally:
            hc.http_client = original

    @pytest.mark.asyncio
    async def test_sleep_called_between_retries(self, mock_http_client, patch_sleep):
        """
        CORRECTNESS: asyncio.sleep must be called between retry attempts
        to implement the backoff strategy (not called after the last attempt).
        """
        mock_http_client.post.return_value = _mock_response(500, {})
        await hc.send_message_with_retry(URL, HEADERS, JSON_DATA, retries=3)
        # 3 attempts → sleep is called after attempt 1 and 2 (2 times)
        assert patch_sleep.await_count == 2

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mock_http_client):
        """
        REAL-LIFE STRESS: 12 concurrent callers compete for the semaphore
        (capacity=10).  All must eventually complete without deadlock.
        """
        mock_http_client.post.return_value = _mock_response(200, {"ok": True})

        tasks = [
            hc.send_message_with_retry(URL, HEADERS, JSON_DATA, retries=1)
            for _ in range(12)
        ]
        results = await asyncio.gather(*tasks)
        assert all(r == {"ok": True} for r in results)


# ──────────────────────────────────────────────────────────────────────────────
# get_message_by_mid()
# ──────────────────────────────────────────────────────────────────────────────

class TestGetMessageByMid:

    MID_URL = "https://graph.facebook.com/v26.0/mid_abc123"
    HEADER = {"Authorization": "Bearer FAKE_TOKEN"}
    PARAMS = {"fields": "message"}

    @pytest.mark.asyncio
    async def test_returns_json_on_200(self, mock_http_client):
        """
        REAL-LIFE GREEN PATH: Graph API returns the quoted message body.
        """
        mock_http_client.get.return_value = _mock_response(200, {"message": "Original text"})
        result = await hc.get_message_by_mid(self.MID_URL, self.HEADER, self.PARAMS)
        assert result == {"message": "Original text"}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_404(self, mock_http_client):
        """
        REAL-LIFE: Message was deleted by the user after being sent.
        Graph API returns 404; the caller receives {} not an exception.
        """
        mock_http_client.get.return_value = _mock_response(404, {"error": "not found"})
        result = await hc.get_message_by_mid(self.MID_URL, self.HEADER, self.PARAMS)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_network_exception(self, mock_http_client):
        """
        REAL-LIFE: DNS failure or connection reset while fetching reply context.
        Must return {} so the webhook handler can still reply without the context.
        """
        mock_http_client.get.side_effect = ConnectionError("DNS failure")
        result = await hc.get_message_by_mid(self.MID_URL, self.HEADER, self.PARAMS)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_none_when_http_client_is_none(self):
        """
        REAL-LIFE: get_message_by_mid called during cold-start race
        before the http_client lifespan completes.
        """
        original = hc.http_client
        hc.http_client = None
        try:
            result = await hc.get_message_by_mid(self.MID_URL, self.HEADER, self.PARAMS)
            assert result is None
        finally:
            hc.http_client = original

    @pytest.mark.asyncio
    async def test_passes_correct_params_to_get(self, mock_http_client):
        """
        CORRECTNESS: Verify the function forwards the params dict to
        httpx.get (token must NOT appear as a query param — security fix).
        """
        mock_http_client.get.return_value = _mock_response(200, {"message": "hi"})
        await hc.get_message_by_mid(self.MID_URL, self.HEADER, self.PARAMS)

        call_kwargs = mock_http_client.get.call_args.kwargs
        assert call_kwargs.get("params") == self.PARAMS
        # Token must be in headers, NOT in params
        assert "Authorization" in call_kwargs.get("headers", {})
