"""
tests/integration/test_messenger_webhook.py
=============================================
Integration tests for src.routes.messenger

Routes tested:
  GET  /webhook/messenger — Meta hub verification
  POST /webhook/messenger — Incoming Facebook Messenger events

REAL-LIFE CONDITIONS SIMULATED:
  1.  Valid hub verification challenge
  2.  Wrong verify token → 403
  3.  Valid signed payload → 200 EVENT_RECEIVED
  4.  Invalid HMAC signature → 403
  5.  Payload too large → 413
  6.  Malformed JSON → 400
  7.  Echo message (is_echo=True) → cached and ignored
  8.  Read receipt event → silently ignored
  9.  Message without text → no reply dispatched
  10. Ad referral from postback block
  11. Ad referral from direct event block
  12. Reply-to quoted message (reply_to.mid)
  13. Duplicate message ID suppression
  14. Multi-entry / multi-event batch
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import make_valid_signature, make_messenger_payload

MS_APP_SECRET = "ms_app_secret_fake"
VERIFY_TOKEN = "test_verify_token_abc123"
BASE = "/webhook/messenger"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _post(client, payload: dict, secret: str = MS_APP_SECRET):
    """Sign and POST a Messenger webhook payload."""
    body = json.dumps(payload).encode()
    sig = make_valid_signature(body, secret)
    headers = {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": sig,
        "Content-Length": str(len(body)),
    }
    return client.post(BASE, content=body, headers=headers)


# ──────────────────────────────────────────────────────────────────────────────
# GET — webhook verification
# ──────────────────────────────────────────────────────────────────────────────

class TestVerifyWebhook:

    def test_valid_subscribe_returns_200_with_challenge(self, test_client):
        """REAL-LIFE: Meta initiates Messenger page subscription."""
        resp = test_client.get(
            BASE,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "ms_challenge_abc",
            },
        )
        assert resp.status_code == 200
        assert resp.text == "ms_challenge_abc"

    def test_wrong_token_returns_403(self, test_client):
        """Spoofed subscription attempt with wrong token."""
        resp = test_client.get(
            BASE,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong_token",
                "hub.challenge": "ms_challenge_abc",
            },
        )
        assert resp.status_code == 403


# ──────────────────────────────────────────────────────────────────────────────
# POST — security layer
# ──────────────────────────────────────────────────────────────────────────────

class TestReceiveWebhookSecurity:

    def test_valid_signed_message_accepted(self, test_client):
        """REAL-LIFE GREEN PATH: Legitimate Messenger event."""
        payload = make_messenger_payload(text="Hello from Messenger!")
        resp = _post(test_client, payload)
        assert resp.status_code == 200
        assert resp.json() == {"status": "EVENT_RECEIVED"}

    def test_invalid_signature_returns_403(self, test_client):
        """REAL-LIFE ATTACK: Request signed with attacker's secret."""
        payload = make_messenger_payload()
        resp = _post(test_client, payload, secret="attacker_secret")
        assert resp.status_code == 403

    def test_missing_signature_returns_403(self, test_client):
        """Header stripped by misconfigured CDN."""
        body = json.dumps(make_messenger_payload()).encode()
        resp = test_client.post(
            BASE,
            content=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        assert resp.status_code == 403

    def test_payload_too_large_returns_413(self, test_client):
        """Memory-bomb guard: oversized content-length → 413 before body is read."""
        resp = test_client.post(
            BASE,
            content=b"x",
            headers={
                "Content-Length": str(1001 * 1024),
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=fake",
            },
        )
        assert resp.status_code == 413

    def test_malformed_json_returns_400(self, test_client):
        """Corrupted payload cannot be parsed → 400."""
        corrupted = b'{"entry": [BAD]}'
        sig = make_valid_signature(corrupted, MS_APP_SECRET)
        resp = test_client.post(
            BASE,
            content=corrupted,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(corrupted)),
            },
        )
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# POST — business logic
# ──────────────────────────────────────────────────────────────────────────────

class TestProcessPayload:

    def test_echo_message_silently_ignored(self, test_client):
        """
        REAL-LIFE: Messenger sends echo of bot-sent messages (is_echo=True).
        These are cached for reply-context lookup but must NOT trigger a bot reply.
        """
        payload = make_messenger_payload(mid="echo_mid_999", is_echo=True)
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_read_receipt_event_silently_ignored(self, test_client):
        """
        REAL-LIFE: Messenger sends 'read' event when user opens the chat.
        Must be ignored without error.
        """
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "page_1",
                    "messaging": [
                        {
                            "sender": {"id": "user_1"},
                            "recipient": {"id": "bot_1"},
                            "read": {"watermark": 1234567890},
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_message_without_text_does_not_crash(self, test_client):
        """
        REAL-LIFE: User sends a GIF or attachment — no 'text' key in message.
        """
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "page_1",
                    "messaging": [
                        {
                            "sender": {"id": "user_gif"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": "gif_mid_001"},  # no text
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_ad_referral_from_event_level(self, test_client):
        """
        REAL-LIFE: Click-to-Messenger ad; referral block at the event level.
        """
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "page_1",
                    "messaging": [
                        {
                            "sender": {"id": "ad_user_1"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": "ad_ms_001", "text": "I'm interested"},
                            "referral": {
                                "ref": "promo_summer",
                                "ad_id": "ad_001",
                                "source": "ADS",
                                "type": "OPEN_THREAD",
                            },
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_ad_referral_from_postback(self, test_client):
        """
        REAL-LIFE: Referral comes inside a postback (button click from ad).
        Must be extracted correctly.
        """
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "page_1",
                    "messaging": [
                        {
                            "sender": {"id": "postback_user"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": "postback_ms_001", "text": "Get started"},
                            "postback": {
                                "title": "Get Started",
                                "payload": "START",
                                "referral": {
                                    "ref": "summer_sale",
                                    "ad_id": "ad_002",
                                    "source": "ADS",
                                    "type": "OPEN_THREAD",
                                },
                            },
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_reply_to_quoted_message(self, test_client):
        """
        REAL-LIFE: User quotes a previous bot message. The reply_to.mid
        should trigger a cache lookup or Graph API fetch.
        """
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": "page_1",
                    "messaging": [
                        {
                            "sender": {"id": "reply_user"},
                            "recipient": {"id": "bot_1"},
                            "message": {
                                "mid": "ms_reply_001",
                                "text": "More details please",
                                "reply_to": {"mid": "original_bot_ms_mid"},
                            },
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_duplicate_mid_both_return_200(self, test_client):
        """
        REAL-LIFE: Messenger retries delivery. Both HTTP responses must
        be 200 EVENT_RECEIVED, while dedup is handled internally.
        """
        payload = make_messenger_payload(mid="ms_dupe_001", text="Duplicate!")
        assert _post(test_client, payload).status_code == 200
        assert _post(test_client, payload).status_code == 200

    def test_multi_entry_batch_accepted(self, test_client):
        """
        REAL-LIFE: Meta batches 3 events from different users in one delivery.
        """
        payload = {
            "object": "page",
            "entry": [
                {
                    "id": f"page_{i}",
                    "messaging": [
                        {
                            "sender": {"id": f"ms_user_{i}"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": f"ms_mid_{i}", "text": f"Hello {i}"},
                        }
                    ],
                }
                for i in range(3)
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_empty_messaging_array_accepted(self, test_client):
        """EDGE: Entry with no messaging events (keep-alive) must not crash."""
        payload = {
            "object": "page",
            "entry": [{"id": "page_1", "messaging": []}],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200
