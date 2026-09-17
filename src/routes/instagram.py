from fastapi import APIRouter, BackgroundTasks, Request, Response
from starlette.requests import ClientDisconnect
import asyncio
import json

from src.core.config import settings
import src.core.http_client as hc
import src.core.security as sec
import src.core.cache as cache
import src.services.rag_chain_manager as srg

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
    reply_url = f"https://graph.instagram.com/{settings.FB_GRAPH_API_VERSION}/me/messages"
    headers = {"Authorization": f"Bearer {settings.INSTAGRAM_TOKEN.get_secret_value()}"}
    rag = srg.answer(text_message)
    json_data = {
        "recipient": {"id": recipient_id},
        "message": {"text": rag},
    }

    await hc.send_with_retry(reply_url, headers, json_data, retries=retries)


async def get_message_by_mid(message_id: str) -> dict | None:
    """Fetch message details using its mid via Meta Graph API."""
    
    if hc.http_client is None:
        print("[ERROR] http_client not initialized yet!")
        return None

    url = f"https://graph.facebook.com/{settings.FB_GRAPH_API_VERSION}/{message_id}"
    header = {"Authorization": f"Bearer {settings.INSTAGRAM_TOKEN.get_secret_value()}"}
    params = {"fields": "message"}

    return await hc.get_message_by_mid(url, header, params) or None


# ─────────────────────────────────────────────
# Core Webhook Processing Logic
# ─────────────────────────────────────────────
async def _handle_single_event(event: dict):
    
    if "read" in event or "delivery" in event:
        return

    message_data = event.get("message")
    if not message_data :
        return

    mid = message_data.get("mid")
    text = message_data.get("text") or "[NON_TEXT_MESSAGE]"

    if message_data.get("is_echo"):
        if mid:
            messages_cache.set_if_absent(mid, text)
        return

    # ! fixed race condition
    # Duplicate delivery protection
    if mid and not messages_cache.set_if_absent(mid, text):
        print(f"[DUPLICATE] Skipping already-processed message {mid}")
        return

    # Extract Ad Referral Data (Click to Instagram Direct Ads)
    referral = message_data.get("referral") or event.get("referral", {})
    if referral:
        ad_id = referral.get("ad_id")
        headline = referral.get("headline", None)
        body = referral.get("body", None)
        image_url = referral.get("image_url", None)

        print("\n[AD REFERRAL DETECTED]")
        print(f" - Ad ID: {ad_id}")
        print(f" - Headline: {headline}")
        print(f" - Body: {body}")
        print(f" - Image URL: {image_url}\n")

    # Extract Reply Context
    reply_to = message_data.get("reply_to") or {}
    story_url = reply_to.get("story", {}).get("url")
    replied_mid = reply_to.get("mid")
    original_text = ""

    if story_url:
        print(f"[REPLY FROM STORY] {story_url}")
    elif replied_mid:
        original_text = messages_cache.get(replied_mid)

        if not original_text:
            original_msg = await get_message_by_mid(replied_mid)
            if original_msg:
                original_text = original_msg.get("message", None)

        print(f"[REPLY TO] {original_text}")

    # Dispatch Auto Reply
    sender_id = event.get("sender", {}).get("id")
    if text and sender_id:
        print(f"[NEW MESSAGE] {text}")
        await send_auto_reply(sender_id, text + f"\n{original_text}")



async def process_webhook_payload(payload: dict):
    """Process incoming webhook events asynchronously in background."""
    try:
        tasks = []
        entries = payload.get("entry", [])

        for entry in entries:
            messaging_events = entry.get("messaging", [])

            for event in messaging_events:
                tasks.append(_handle_single_event(event))


        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions = True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"[ERROR] Task failed silently {result}")

    except asyncio.CancelledError:
        raise
    except Exception as err:
        print(f"[EXCEPTIONAL ERROR]: {err}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@router.get("")
async def verify_webhook(request: Request):
    """Webhook verification endpoint for Meta."""
    return sec.verify_webhooks(request, settings.VERIFY_TOKEN)


@router.post("")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive and validate incoming webhook payload."""
    content_length = int(request.headers.get('content-length', 0))
    if content_length > 1000 * 1024:
        return Response(content="[PAYLOAD IS TOO LARGE]", status_code=413)
    
    try:
        raw_body = await request.body()

    except ClientDisconnect:
        print("[ERROR] Client disconnected before body was fully received.")
        return Response(content="Client disconnected", status_code=400)

    signature = request.headers.get("X-Hub-Signature-256", "")
    
    if not sec.verify_signature(
        raw_body=raw_body,
        signature_header=signature,
        APP_SECRET=settings.APP_SECRET.get_secret_value()
                                ):
        
        print("[SECURITY] Invalid signature — request rejected.")
        return Response(content="Invalid signature", status_code=403)

    try:
        payload = json.loads(raw_body)

    except Exception:
        print("[ERROR] Failed to parse JSON payload.")
        return Response(content="Bad request", status_code=400)

    print("\n================ [NEW INSTAGRAM EVENT] ================")
    background_tasks.add_task(process_webhook_payload, payload)
    return {"status": "EVENT_RECEIVED"}