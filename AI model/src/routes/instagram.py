from fastapi import APIRouter, BackgroundTasks, Request, Response
from dotenv import load_dotenv
import os


import src.core.http_client as hc
import src.core.cache as cache
import src.core.security as sec

# ─────────────────────────────────────────────
# Setup & Config
# ─────────────────────────────────────────────
load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "").strip()
PAGE_ACCESS_TOKEN = os.getenv("INSTAGRAM_TOKEN", "").strip()
APP_SECRET = os.getenv("APP_SECRET", "").strip()

if not VERIFY_TOKEN:
    raise ValueError("VERIFY_TOKEN not set in .env")
if not PAGE_ACCESS_TOKEN:
    raise ValueError("INSTAGRAM_TOKEN not set in .env")
if not APP_SECRET:
    raise ValueError("APP_SECRET not set in .env")

# ─────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────
messages_cache = cache.MessageCache(max_size=7000)

# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────
router = APIRouter(
    prefix="/webhook/instagram",
    tags=["Instagram"],
)


# ─────────────────────────────────────────────
# HTTPX Async Helpers
# ─────────────────────────────────────────────
async def send_auto_reply(recipient_id: str, text_message: str, retries: int = 3):
    """Send automated reply via Meta Graph API with retry logic and backoff."""
    reply_url = "https://graph.instagram.com/v26.0/me/messages"
    headers = {"Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"}
    json_data = {
        "recipient": {"id": recipient_id},
        "message": {"text": text_message},
    }

    await hc.send_with_retry(reply_url, headers, json_data, retries=retries)


async def get_message_by_mid(message_id: str) -> dict:
    """Fetch message details using its mid via Meta Graph API."""
    if hc.http_client is None:
        print("[ERROR] http_client not initialized yet!")
        return {}

    url = f"https://graph.instagram.com/v26.0/{message_id}"
    params = {
        "fields": "message",
        "access_token": PAGE_ACCESS_TOKEN,
    }

    return await hc.get_message_by_mid(url, params) or {}


# ─────────────────────────────────────────────
# Core Webhook Processing Logic
# ─────────────────────────────────────────────
async def process_webhook_payload(payload: dict):
    """Process incoming webhook events asynchronously in background."""
    try:
        entries = payload.get("entry", [])
        for entry in entries:
            messaging_events = entry.get("messaging", [])

            for event in messaging_events:
                # 1. Ignore read/delivery events
                if "read" in event or "delivery" in event:
                    continue

                message_data = event.get("message")
                if not message_data or message_data.get("is_echo"):
                    print("[ECHO]")
                    continue

                mid = message_data.get("mid")
                text = message_data.get("text")

                # 2. Duplicate delivery protection
                if mid and messages_cache.has(mid):
                    print(f"[DUPLICATE] Skipping already-processed message {mid}")
                    continue

                # Cache incoming message regardless of text presence
                if mid:
                    messages_cache.add(mid, text or "[NON_TEXT_MESSAGE]")

                # 3. Extract Ad Referral Data (Click to Instagram Direct Ads)
                referral = message_data.get("referral") or event.get("referral", {})
                if referral:
                    ad_id = referral.get("ad_id")
                    headline = referral.get("headline", "N/A")
                    body = referral.get("body", "N/A")
                    image_url = referral.get("image_url", "N/A")

                    print("\n[AD REFERRAL DETECTED]")
                    print(f" - Ad ID: {ad_id}")
                    print(f" - Headline: {headline}")
                    print(f" - Body: {body}")
                    print(f" - Image URL: {image_url}\n")

                # 4. Extract Reply Context
                reply_to = message_data.get("reply_to") or {}
                story_url = reply_to.get("story", {}).get("url")
                replied_mid = reply_to.get("mid")

                if story_url:
                    print(f"[REPLY FROM STORY] {story_url}")
                elif replied_mid:
                    original_text = messages_cache.get(replied_mid)

                    if not original_text:
                        original_msg = await get_message_by_mid(replied_mid)
                        original_text = original_msg.get("message", "N/A")

                    print(f"[REPLY TO] {original_text}")

                # 5. Dispatch Auto Reply
                sender_id = event.get("sender", {}).get("id")
                if text and sender_id:
                    print(f"[NEW MESSAGE] {text}")
                    await send_auto_reply(sender_id, text)

    except Exception as err:
        print(f"[EXCEPTIONAL ERROR]: {err}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@router.get("")
async def verify_webhook(request: Request):
    """Webhook verification endpoint for Meta."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("\n[SUCCESS] Webhook verified successfully by Meta!")
        return Response(content=challenge, media_type="text/plain", status_code=200)

    print("\n[ERROR] Verification failed.")
    return Response(content="Verification failed", status_code=403)


@router.post("")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive and validate incoming webhook payload."""
    raw_body = await request.body()

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not sec.verify_signature(
        raw_body=raw_body, signature_header=signature, APP_SECRET=APP_SECRET
    ):
        print("[SECURITY] Invalid signature — request rejected.")
        return Response(content="Invalid signature", status_code=403)

    try:
        payload = await request.json()
    except Exception:
        print("[ERROR] Failed to parse JSON payload.")
        return Response(content="Bad request", status_code=400)

    print("\n================ [NEW WEBHOOK EVENT] ================")
    background_tasks.add_task(process_webhook_payload, payload)
    return {"status": "EVENT_RECEIVED"}