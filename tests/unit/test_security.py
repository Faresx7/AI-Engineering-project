"""
tests/unit/test_security.py
============================
Unit tests for src.core.security

Functions tested:
  - verify_signature()     — HMAC-SHA256 webhook payload signing
  - verify_webhooks()      — Meta hub challenge-response handshake

REAL-LIFE CONDITIONS SIMULATED:
  1. Legitimate Meta webhook (correct signature)
  2. Tampered payload (body modified after signing — replay / MITM attack)
  3. Missing X-Hub-Signature-256 header (header dropped by proxy)
  4. Malformed header prefix (attacker sends "md5=..." instead of "sha256=")
  5. Wrong secret used for signing (cross-account payload confusion)
  6. Empty body with valid signature (edge: ping from Meta)
  7. Webhook verification: valid subscribe handshake
  8. Webhook verification: wrong token (spoofed hub request)
  9. Webhook verification: wrong mode (mode != "subscribe")
  10. Webhook verification: missing query params
"""

import hashlib
import hmac

import pytest
from unittest.mock import MagicMock

# ──────────────────────────────────────────────────────────────────────────────
# Import under test
# ──────────────────────────────────────────────────────────────────────────────
from src.core.security import verify_signature, verify_webhooks


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SECRET = "super_secret_app_key"
BODY = b'{"object":"instagram","entry":[]}'


def _sign(body: bytes, secret: str) -> str:
    """Produce a valid sha256= header value using the given secret."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _make_request(mode="subscribe", token="test_token", challenge="challenge_xyz"):
    """
    Builds a minimal mock of FastAPI Request with query_params.
    Avoids a real ASGI app; pure unit isolation.
    """
    mock_req = MagicMock()
    mock_req.query_params = {
        "hub.mode": mode,
        "hub.verify_token": token,
        "hub.challenge": challenge,
    }
    return mock_req


# ──────────────────────────────────────────────────────────────────────────────
# verify_signature()
# ──────────────────────────────────────────────────────────────────────────────

class TestVerifySignature:

    def test_valid_signature_returns_true(self):
        """
        REAL-LIFE: Meta sends a legitimately signed payload.
        The HMAC must match and the function must return True.
        """
        sig = _sign(BODY, SECRET)
        assert verify_signature(BODY, sig, SECRET) is True

    def test_tampered_body_returns_false(self):
        """
        REAL-LIFE ATTACK: Man-in-the-middle modifies the body content
        after Meta signed it.  Signature must no longer match.
        """
        sig = _sign(BODY, SECRET)
        tampered = BODY + b" malicious_extra_data"
        assert verify_signature(tampered, sig, SECRET) is False

    def test_wrong_secret_returns_false(self):
        """
        REAL-LIFE: Payload signed with a different app's secret
        (cross-app confusion or misconfiguration).
        """
        sig = _sign(BODY, "wrong_secret")
        assert verify_signature(BODY, sig, SECRET) is False

    def test_missing_signature_header_returns_false(self):
        """
        REAL-LIFE: Load-balancer or proxy strips the X-Hub-Signature-256
        header.  Function must return False without raising.
        """
        assert verify_signature(BODY, "", SECRET) is False

    def test_none_signature_header_returns_false(self):
        """
        REAL-LIFE: FastAPI returns empty string when header absent.
        Guard against None being passed by caller.
        """
        assert verify_signature(BODY, None, SECRET) is False  # type: ignore

    def test_wrong_prefix_md5_returns_false(self):
        """
        REAL-LIFE ATTACK: Attacker sends 'md5=...' instead of 'sha256=...'
        hoping the code blindly strips the prefix.
        """
        raw_digest = hmac.new(SECRET.encode(), BODY, hashlib.sha256).hexdigest()
        fake_sig = f"md5={raw_digest}"
        assert verify_signature(BODY, fake_sig, SECRET) is False

    def test_empty_body_with_valid_signature_returns_true(self):
        """
        REAL-LIFE: Meta occasionally sends empty-body ping events.
        An empty body can still be validly signed.
        """
        empty_body = b""
        sig = _sign(empty_body, SECRET)
        assert verify_signature(empty_body, sig, SECRET) is True

    def test_high_entropy_body_with_valid_signature(self):
        """
        REAL-LIFE: Large payload (~10 KB) with nested JSON.
        HMAC computation must remain correct regardless of body size.
        """
        large_body = b'{"data": "' + b"x" * 10_000 + b'"}'
        sig = _sign(large_body, SECRET)
        assert verify_signature(large_body, sig, SECRET) is True

    def test_signature_comparison_is_constant_time(self):
        """
        SECURITY: The function must use hmac.compare_digest, not ==,
        to prevent timing-based side-channel attacks.
        Verified by confirming the source uses compare_digest.
        (Static logic check — no runtime timing measurement needed.)
        """
        import inspect
        import src.core.security as sec_module
        source = inspect.getsource(sec_module.verify_signature)
        assert "compare_digest" in source, (
            "verify_signature must use hmac.compare_digest for timing-safe comparison"
        )


# ──────────────────────────────────────────────────────────────────────────────
# verify_webhooks()
# ──────────────────────────────────────────────────────────────────────────────

class TestVerifyWebhooks:

    VERIFY_TOKEN = "test_verify_token_abc123"

    def test_valid_subscribe_returns_200_with_challenge(self):
        """
        REAL-LIFE: Meta initiates a webhook subscription verification.
        The response must echo the hub.challenge with HTTP 200.
        """
        req = _make_request(mode="subscribe", token=self.VERIFY_TOKEN, challenge="ch_99")
        response = verify_webhooks(req, self.VERIFY_TOKEN)
        assert response.status_code == 200
        assert response.body == b"ch_99"

    def test_wrong_verify_token_returns_403(self):
        """
        REAL-LIFE ATTACK: A third party tries to verify a webhook they don't own
        by guessing the token.  Must be rejected with 403.
        """
        req = _make_request(mode="subscribe", token="WRONG_TOKEN", challenge="ch_99")
        response = verify_webhooks(req, self.VERIFY_TOKEN)
        assert response.status_code == 403

    def test_wrong_mode_returns_403(self):
        """
        REAL-LIFE: Malformed request with mode != 'subscribe'.
        """
        req = _make_request(mode="unsubscribe", token=self.VERIFY_TOKEN, challenge="ch_99")
        response = verify_webhooks(req, self.VERIFY_TOKEN)
        assert response.status_code == 403

    def test_missing_mode_returns_403(self):
        """
        REAL-LIFE: Bot / scraper sends a GET without hub.mode param.
        """
        mock_req = MagicMock()
        mock_req.query_params = {
            "hub.verify_token": self.VERIFY_TOKEN,
            "hub.challenge": "ch_99",
        }
        response = verify_webhooks(mock_req, self.VERIFY_TOKEN)
        assert response.status_code == 403

    def test_missing_token_returns_403(self):
        """
        REAL-LIFE: Request is missing hub.verify_token entirely.
        """
        mock_req = MagicMock()
        mock_req.query_params = {
            "hub.mode": "subscribe",
            "hub.challenge": "ch_99",
        }
        response = verify_webhooks(mock_req, self.VERIFY_TOKEN)
        assert response.status_code == 403

    def test_challenge_is_returned_as_plain_text(self):
        """
        REAL-LIFE: Meta requires the challenge body as text/plain.
        Verify the media_type header is set correctly.
        """
        req = _make_request(mode="subscribe", token=self.VERIFY_TOKEN, challenge="unique_challenge")
        response = verify_webhooks(req, self.VERIFY_TOKEN)
        assert "text/plain" in response.media_type

    def test_empty_challenge_is_reflected(self):
        """
        BOUNDARY: If Meta sends an empty challenge string, the empty body
        must still be echoed (so Meta doesn't time out).
        """
        req = _make_request(mode="subscribe", token=self.VERIFY_TOKEN, challenge="")
        response = verify_webhooks(req, self.VERIFY_TOKEN)
        assert response.status_code == 200
        assert response.body == b""
