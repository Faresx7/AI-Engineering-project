import hmac
import hashlib

def verify_signature(raw_body: bytes, signature_header: str, APP_SECRET: str) -> bool:
    """Verify that the provided payload was signed with the Meta App Secret.

    The function expects a raw request body and the value of the X-Hub-Signature-256
    header, which should be formatted as ``sha256=<hex-digest>``. It recomputes the
    HMAC-SHA256 digest over ``raw_body`` using ``APP_SECRET`` and compares it against
    the supplied signature using a constant-time comparison.

    Args:
        raw_body: The raw request payload bytes to verify.
        signature_header: The signature header value from the incoming request.
        APP_SECRET: The Meta App Secret used to generate the HMAC.

    Returns:
        True if the signature is valid, otherwise False.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        APP_SECRET.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature_header)
