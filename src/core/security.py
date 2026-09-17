from fastapi import Request, Response
import hashlib
import hmac

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


def verify_webhooks(request: Request,verify_token: str):
        """Verify Meta webhook subscription requests.

        This function validates the hub.mode and hub.verify_token query parameters
        that Meta sends during webhook verification. If the mode is "subscribe"
        and the token matches the expected verify token, it responds with the
        provided challenge string to complete the verification handshake.

        Args:
            request: The incoming FastAPI request containing the query parameters.
            verify_token: The expected verification token configured for the app.

        Returns:
            A FastAPI Response containing the challenge string with status code 200
            when verification succeeds, or a 403 response when it fails.
        """
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
    
        if mode == "subscribe" and token == verify_token:
            print("\n[SUCCESS] Webhook verified successfully by Meta!")
            return Response(content=challenge, media_type="text/plain", status_code=200)
    
        print("\n[ERROR] Verification failed.")
        return Response(content="Verification failed", status_code=403)
    
