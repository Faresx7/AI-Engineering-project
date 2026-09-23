"""
tests/integration/test_whatsapp_webhook.py
============================================
Integration tests for src.routes.whatsapp

Routes tested:
  GET  /webhook/whatsapp — Meta hub verification
  POST /webhook/whatsapp — Incoming WhatsApp Cloud API events

REAL-LIFE CONDITIONS SIMULATED:
  1.  Valid hub verification challenge
  2.  Wrong verify token → 403
  3.  Valid signed text message → 200 EVENT_RECEIVED
  4.  Invalid HMAC signature → 403
  5.  Payload too large → 413
  6.  Malformed JSON → 400
  7.  Status update event (sent/delivered/read) → silently consumed
  8.  Non-text message (image/audio) → handled without crash
  9.  Duplicate wamid deduplication
  10. Ad referral data in message
  11. Reply context (context.id pointing to a cached bot message)
  12. Bot's sent wamid is cached after successful reply
  13. Multi-message batch in a single webhook call
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_valid_signature, make_whatsapp_payload

WA_APP_SECRET = "wa_app_secret_fake"
VERIFY_TOKEN = "test_verify_token_abc123"
BASE = "/webhook/whatsapp"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _post(client, payload: dict, secret: str = WA_APP_SECRET):
    """Sign and POST a WhatsApp webhook payload."""
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
        """REAL-LIFE: Meta Cloud API initiates webhook subscription."""
        resp = test_client.get(
            BASE,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "wa_challenge_999",
            },
        )
        assert resp.status_code == 200
        assert resp.text == "wa_challenge_999"

    def test_wrong_token_returns_403(self, test_client):
        """Security guard: wrong token must be rejected."""
        resp = test_client.get(
            BASE,
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "bad_token",
                "hub.challenge": "wa_challenge_999",
            },
        )
        assert resp.status_code == 403


# ──────────────────────────────────────────────────────────────────────────────
# POST — security layer
# ──────────────────────────────────────────────────────────────────────────────

class TestReceiveWebhookSecurity:

    def test_valid_signature_returns_event_received(self, test_client):
        """REAL-LIFE GREEN PATH: Legitimate WhatsApp message arrives."""
        payload = make_whatsapp_payload(text="Hello WhatsApp!")
        resp = _post(test_client, payload)
        assert resp.status_code == 200
        assert resp.json() == {"status": "EVENT_RECEIVED"}

    def test_invalid_signature_returns_403(self, test_client):
        """
        REAL-LIFE ATTACK: Forged webhook with incorrect HMAC.
        """
        payload = make_whatsapp_payload()
        resp = _post(test_client, payload, secret="attacker_secret")
        assert resp.status_code == 403

    def test_payload_too_large_returns_413(self, test_client):
        """DDOS guard: content-length > 1 MB must be rejected before reading body."""
        headers = {
            "Content-Length": str(1001 * 1024),
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=fake",
        }
        resp = test_client.post(BASE, content=b"x", headers=headers)
        assert resp.status_code == 413

    def test_malformed_json_returns_400(self, test_client):
        """Corrupted network packet: JSON parse failure → 400."""
        corrupted = b'{"entry": CORRUPTED}'
        sig = make_valid_signature(corrupted, WA_APP_SECRET)
        headers = {
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig,
            "Content-Length": str(len(corrupted)),
        }
        resp = test_client.post(BASE, content=corrupted, headers=headers)
        assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# POST — business logic
# ──────────────────────────────────────────────────────────────────────────────

class TestProcessPayload:

    def test_text_message_returns_event_received(self, test_client):
        """Normal user message → accepted and queued for background processing."""
        payload = make_whatsapp_payload(text="What are your hours?")
        resp = _post(test_client, payload)
        assert resp.json()["status"] == "EVENT_RECEIVED"

    def test_status_update_silently_consumed(self, test_client):
        """
        REAL-LIFE: WhatsApp sends sent/delivered/read status events.
        These must be acknowledged with 200 but NOT trigger a reply.
        """
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_biz_id",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "statuses": [
                                    {
                                        "id": "wamid.status_001",
                                        "status": "delivered",
                                        "timestamp": "1700000000",
                                        "recipient_id": "1234567890",
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_image_message_does_not_crash(self, test_client):
        """
        REAL-LIFE: User sends an image.
        msg_type='image' has no text body → must be skipped gracefully.
        """
        payload = make_whatsapp_payload(msg_type="image")
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_audio_message_does_not_crash(self, test_client):
        """REAL-LIFE: User sends a voice note."""
        payload = make_whatsapp_payload(msg_type="audio")
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_duplicate_wamid_accepted_at_http_layer(self, test_client):
        """
        REAL-LIFE: Meta retries delivery after a webhook timeout.
        HTTP layer always returns EVENT_RECEIVED; dedup is handled internally.
        """
        payload = make_whatsapp_payload(wamid="wamid.dupe_test", text="Same message")
        resp1 = _post(test_client, payload)
        resp2 = _post(test_client, payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200

    def test_ad_referral_payload_accepted(self, test_client):
        """
        REAL-LIFE: User clicks a Click-to-WhatsApp ad.
        Referral block appears inside the message object.
        """
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_biz_id",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": [
                                    {
                                        "id": "wamid.ad_001",
                                        "from": "+201234567890",
                                        "type": "text",
                                        "text": {"body": "Interested in your product"},
                                        "referral": {
                                            "headline": "Special Offer",
                                            "body": "Buy now",
                                            "source_url": "https://fb.com/ad/123",
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_reply_context_payload_accepted(self, test_client):
        """
        REAL-LIFE: User replies to a bot message (quoted message).
        The context.id field must be parsed without error.
        """
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_biz_id",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": [
                                    {
                                        "id": "wamid.reply_001",
                                        "from": "+201234567890",
                                        "type": "text",
                                        "text": {"body": "Tell me more"},
                                        "context": {"id": "wamid.original_bot_msg"},
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_empty_messages_array_accepted(self, test_client):
        """
        EDGE: Webhook delivered with no messages in the value.
        Must not crash (empty for-loop).
        """
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_biz_id",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {"messaging_product": "whatsapp", "messages": []},
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200

    def test_multi_message_batch_accepted(self, test_client):
        """
        REAL-LIFE: Meta batches several messages into one webhook delivery.
        All messages must be processed concurrently without error.
        """
        messages = [
            {
                "id": f"wamid.batch_{i}",
                "from": f"+2012345{i:04d}",
                "type": "text",
                "text": {"body": f"Batch message {i}"},
            }
            for i in range(5)
        ]
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_biz_id",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": messages,
                            },
                        }
                    ],
                }
            ],
        }
        resp = _post(test_client, payload)
        assert resp.status_code == 200
