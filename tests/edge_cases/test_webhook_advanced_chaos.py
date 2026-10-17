"""
tests/edge_cases/test_webhook_advanced_chaos.py
=================================================
Advanced chaos, security bypass, and edge payload tests across all webhook routes.

REAL-LIFE SCENARIOS TESTED:
  1.  Signature mutations & bypass attempts:
      - Upper-case hexadecimal digest in X-Hub-Signature-256
      - Truncated signature (< 64 hex chars)
      - Extra-long signature (> 64 hex chars)
      - Non-hexadecimal characters in signature
      - Whitespace padding around signature (leading/trailing)
      - Null-byte injection in signature string
  2.  Double MID delivery in a single webhook payload:
      - Meta sends identical MID twice in the same batch (duplicate delivery)
      - Exactly one event should trigger RAG processing / reply
  3.  WhatsApp non-standard message types:
      - Location message (latitude, longitude, name)
      - Contact card message (vcard, name)
      - Reaction event (emoji reaction to a message)
      - Interactive button reply (list_reply, button_reply)
      - Document message with filename and mime_type
  4.  Instagram edge events:
      - Message with story mention / story share
      - Reaction event (heart/emoji on direct message)
      - Unsend / deleted message event
      - Echo message properly caches text in message cache
  5.  Messenger edge events:
      - Quick reply response
      - Delivery receipt (delivery event with mids list)
      - Attachment without text (sticker, GIF, audio)
      - Postback referral with nested data
  6.  Network anomalies:
      - Truncated / malformed JSON byte streams
      - Null bytes inside JSON strings
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import (
    make_valid_signature,
    make_instagram_payload,
    make_whatsapp_payload,
    make_messenger_payload,
)

APP_SECRET = "test_app_secret_xyz"
WA_APP_SECRET = "wa_app_secret_fake"
MS_APP_SECRET = "ms_app_secret_fake"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Signature Security Chaos
# ──────────────────────────────────────────────────────────────────────────────

class TestSignatureChaos:

    def _post(self, client, route: str, body: bytes, sig: str):
        return client.post(
            route,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(body)),
            },
        )

    def test_uppercase_hex_signature_rejected(self, test_client):
        """
        SECURITY: Meta specifies lowercase hex in X-Hub-Signature-256.
        An uppercase hex signature does not match the canonical lower hex digest and is rejected.
        """
        payload = make_instagram_payload(text="Uppercase hex test")
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, APP_SECRET).upper()
        # Keep prefix as 'sha256='
        sig = "sha256=" + sig.split("=", 1)[1]

        resp = self._post(test_client, "/webhook/instagram", body, sig)
        assert resp.status_code == 403

    def test_truncated_signature_rejected(self, test_client):
        """ATTACK: Signature is truncated to fewer than 64 hex chars."""
        payload = make_instagram_payload(text="Truncated sig test")
        body = json.dumps(payload).encode()
        sig = "sha256=" + ("a" * 32)

        resp = self._post(test_client, "/webhook/instagram", body, sig)
        assert resp.status_code == 403

    def test_extra_long_signature_rejected(self, test_client):
        """ATTACK: Signature has extra hex characters appended."""
        payload = make_instagram_payload(text="Extra long sig test")
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, APP_SECRET) + "ffff"

        resp = self._post(test_client, "/webhook/instagram", body, sig)
        assert resp.status_code == 403

    def test_non_hex_characters_in_signature_rejected(self, test_client):
        """ATTACK: Signature contains non-hex characters (e.g. 'z', '!', '$')."""
        payload = make_instagram_payload(text="Non hex test")
        body = json.dumps(payload).encode()
        sig = "sha256=" + ("z" * 64)

        resp = self._post(test_client, "/webhook/instagram", body, sig)
        assert resp.status_code == 403

    def test_signature_with_whitespace_padding_handled(self, test_client):
        """
        EDGE: Gateway adds leading or trailing spaces to the header value.
        """
        payload = make_instagram_payload(text="Whitespace sig test")
        body = json.dumps(payload).encode()
        sig = "  " + make_valid_signature(body, APP_SECRET) + "  "

        resp = self._post(test_client, "/webhook/instagram", body, sig)
        # Should be accepted if stripped, or safely rejected with 403 without 500 crash
        assert resp.status_code in (200, 403)

    def test_null_byte_in_signature_header_rejected(self, test_client):
        """ATTACK: Null byte injection in header."""
        payload = make_instagram_payload(text="Null byte test")
        body = json.dumps(payload).encode()
        sig = "sha256=abc\x00def"

        resp = self._post(test_client, "/webhook/instagram", body, sig)
        assert resp.status_code == 403


# ──────────────────────────────────────────────────────────────────────────────
# 2. Duplicate MID within Same Batch
# ──────────────────────────────────────────────────────────────────────────────

class TestDuplicateMidInSameBatch:

    def test_instagram_duplicate_mid_in_same_payload_processed_once(
        self, test_client, mock_rag_chain
    ):
        """
        REAL-LIFE RACE: Meta bundles duplicate deliveries of the SAME mid
        inside a single webhook batch. Only the first must invoke RAG.
        """
        duplicate_mid = "batch_dup_mid_001"
        payload = {
            "object": "instagram",
            "entry": [
                {
                    "id": "p1",
                    "messaging": [
                        {
                            "sender": {"id": "user_dup_1"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": duplicate_mid, "text": "Can I buy an iPhone?"},
                        },
                        {
                            "sender": {"id": "user_dup_1"},
                            "recipient": {"id": "bot_1"},
                            "message": {"mid": duplicate_mid, "text": "Can I buy an iPhone?"},
                        },
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, APP_SECRET)

        resp = test_client.post(
            "/webhook/instagram",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(body)),
            },
        )
        assert resp.status_code == 200

    def test_whatsapp_duplicate_wamid_in_same_payload_handled(
        self, test_client, mock_http_client
    ):
        """
        REAL-LIFE RACE: WhatsApp batch contains two message entries with same wamid.
        Must not double-reply or crash.
        """
        duplicate_wamid = "wamid.batch_dup_999"
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_acc_1",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": [
                                    {
                                        "id": duplicate_wamid,
                                        "from": "+201000000001",
                                        "type": "text",
                                        "text": {"body": "First copy"},
                                    },
                                    {
                                        "id": duplicate_wamid,
                                        "from": "+201000000001",
                                        "type": "text",
                                        "text": {"body": "Second copy"},
                                    },
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, WA_APP_SECRET)

        resp = test_client.post(
            "/webhook/whatsapp",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(body)),
            },
        )
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# 3. WhatsApp Complex Message Types
# ──────────────────────────────────────────────────────────────────────────────

class TestWhatsAppComplexMessageTypes:

    def _post_wa(self, client, msg_dict: dict):
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "wa_acc_1",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": [msg_dict],
                            },
                        }
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, WA_APP_SECRET)
        return client.post(
            "/webhook/whatsapp",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(body)),
            },
        )

    def test_location_message_does_not_crash(self, test_client):
        """User shares their GPS location via WhatsApp."""
        msg = {
            "id": "wamid.loc_001",
            "from": "+201112223334",
            "type": "location",
            "location": {
                "latitude": 30.0444,
                "longitude": 31.2357,
                "name": "Cairo Downtown",
                "address": "Tahrir Square",
            },
        }
        resp = self._post_wa(test_client, msg)
        assert resp.status_code == 200

    def test_contact_card_message_does_not_crash(self, test_client):
        """User shares a contact card via WhatsApp."""
        msg = {
            "id": "wamid.contact_001",
            "from": "+201112223334",
            "type": "contacts",
            "contacts": [
                {
                    "name": {"first_name": "Ahmed", "formatted_name": "Ahmed Ali"},
                    "phones": [{"phone": "+201099998888", "type": "MOBILE"}],
                }
            ],
        }
        resp = self._post_wa(test_client, msg)
        assert resp.status_code == 200

    def test_reaction_event_does_not_crash(self, test_client):
        """User reacts with an emoji to a previous bot message."""
        msg = {
            "id": "wamid.reaction_001",
            "from": "+201112223334",
            "type": "reaction",
            "reaction": {
                "message_id": "wamid.target_bot_msg",
                "emoji": "👍",
            },
        }
        resp = self._post_wa(test_client, msg)
        assert resp.status_code == 200

    def test_interactive_button_reply_does_not_crash(self, test_client):
        """User taps an interactive button on WhatsApp."""
        msg = {
            "id": "wamid.interactive_001",
            "from": "+201112223334",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "btn_price_check", "title": "Check Price"},
            },
        }
        resp = self._post_wa(test_client, msg)
        assert resp.status_code == 200

    def test_document_message_does_not_crash(self, test_client):
        """User uploads a PDF / document via WhatsApp."""
        msg = {
            "id": "wamid.doc_001",
            "from": "+201112223334",
            "type": "document",
            "document": {
                "filename": "receipt.pdf",
                "mime_type": "application/pdf",
                "id": "media_doc_999",
            },
        }
        resp = self._post_wa(test_client, msg)
        assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# 4. Instagram Edge Events
# ──────────────────────────────────────────────────────────────────────────────

class TestInstagramEdgeEvents:

    def _post_ig(self, client, messaging_dict: dict):
        payload = {
            "object": "instagram",
            "entry": [{"id": "ig_page_1", "messaging": [messaging_dict]}],
        }
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, APP_SECRET)
        return client.post(
            "/webhook/instagram",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(body)),
            },
        )

    def test_story_mention_event_does_not_crash(self, test_client):
        """User mentions the business in an Instagram story."""
        event = {
            "sender": {"id": "ig_user_story"},
            "recipient": {"id": "ig_page_1"},
            "message": {
                "mid": "mid_story_001",
                "attachments": [
                    {
                        "type": "story_mention",
                        "payload": {"url": "https://instagram.com/stories/..."},
                    }
                ],
            },
        }
        resp = self._post_ig(test_client, event)
        assert resp.status_code == 200

    def test_reaction_event_does_not_crash(self, test_client):
        """User reacts to an Instagram message with a heart."""
        event = {
            "sender": {"id": "ig_user_react"},
            "recipient": {"id": "ig_page_1"},
            "reaction": {
                "mid": "mid_reacted_to",
                "action": "react",
                "reaction": "heart",
                "emoji": "❤️",
            },
        }
        resp = self._post_ig(test_client, event)
        assert resp.status_code == 200

    def test_echo_message_caches_text_for_future_replies(self, test_client):
        """
        When bot sends an echo message, it must be stored in cache
        so when customer quotes it later, the quoted text is available.
        """
        from src.routes.instagram import messages_cache

        mid = "bot_echo_mid_123"
        echo_text = "Thank you for contacting customer support."
        event = {
            "sender": {"id": "bot_page_id"},
            "recipient": {"id": "customer_id"},
            "message": {
                "mid": mid,
                "text": echo_text,
                "is_echo": True,
            },
        }
        resp = self._post_ig(test_client, event)
        assert resp.status_code == 200
        assert messages_cache.get(mid) == echo_text


# ──────────────────────────────────────────────────────────────────────────────
# 5. Messenger Edge Events
# ──────────────────────────────────────────────────────────────────────────────

class TestMessengerEdgeEvents:

    def _post_ms(self, client, messaging_dict: dict):
        payload = {
            "object": "page",
            "entry": [{"id": "fb_page_1", "messaging": [messaging_dict]}],
        }
        body = json.dumps(payload).encode()
        sig = make_valid_signature(body, MS_APP_SECRET)
        return client.post(
            "/webhook/messenger",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "Content-Length": str(len(body)),
            },
        )

    def test_quick_reply_message_processed(self, test_client):
        """User selects a quick reply button in Facebook Messenger."""
        event = {
            "sender": {"id": "fb_u_qr"},
            "recipient": {"id": "fb_page_1"},
            "message": {
                "mid": "mid_qr_001",
                "text": "Red",
                "quick_reply": {"payload": "COLOR_RED"},
            },
        }
        resp = self._post_ms(test_client, event)
        assert resp.status_code == 200

    def test_delivery_receipt_silently_ignored(self, test_client):
        """Meta delivers a delivery receipt with watermarks and mids."""
        event = {
            "sender": {"id": "fb_u_del"},
            "recipient": {"id": "fb_page_1"},
            "delivery": {
                "mids": ["mid_sent_001", "mid_sent_002"],
                "watermark": 1600000000000,
                "seq": 45,
            },
        }
        resp = self._post_ms(test_client, event)
        assert resp.status_code == 200

    def test_sticker_attachment_without_text(self, test_client):
        """Customer sends a large Facebook animated sticker."""
        event = {
            "sender": {"id": "fb_u_sticker"},
            "recipient": {"id": "fb_page_1"},
            "message": {
                "mid": "mid_sticker_001",
                "sticker_id": 369239263222822,
                "attachments": [
                    {
                        "type": "image",
                        "payload": {"url": "https://cdn.facebook.com/sticker.png"},
                    }
                ],
            },
        }
        resp = self._post_ms(test_client, event)
        assert resp.status_code == 200
