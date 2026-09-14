from fastapi import BackgroundTasks,APIRouter, Request, Response
from starlette.requests import ClientDisconnect
import json

from src.core.config import settings
import src.core.http_client as hc
import src.core.security as sec
import src.core.cache as cache


# * DONE['cache','verify_webhook','router','send_reply'.'separate loading tokens','safer raw_body']


# ─────────────────────────────────────────────
# router
# ─────────────────────────────────────────────
router = APIRouter(prefix="/webhook/messenger",
                   tags=['messenger'])

# ─────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────
messages_cache = cache.MessageCache(max_size=7000)

# ─────────────────────────────────────────────
# HTTPX Async Helpers
# ─────────────────────────────────────────────
async def send_auto_reply(recipient_id: str, text_message: str):
    """Send automated reply via Facebook Messenger Graph API with retry logic and backoff."""
    reply_url = "https://graph.facebook.com/v26.0/me/messages"
    headers = {"Authorization": f"Bearer {settings.MESSENGER_TOKEN.get_secret_value()}"}
    json_data = {
        "recipient": {"id": recipient_id},
        "message": {"text": text_message},
    }
    await hc.send_with_retry(reply_url,headers,json_data)


async def get_message_by_mid(message_id: str) -> dict:
    """Fetch message details using its mid via Facebook Graph API."""
    url = f"https://graph.facebook.com/v26.0/{message_id}"
    params = {
        "fields": "message",
        "access_token": settings.MESSENGER_TOKEN.get_secret_value(),
    }

    return await hc.get_message_by_mid(url, params) or {}
                        

# ─────────────────────────────────────────────
# Core Webhook Processing Logic
# ─────────────────────────────────────────────
async def process_webhook_payload(payload: dict):
    """Process incoming Facebook Messenger webhook events asynchronously in background."""
    try:
        entries = payload.get("entry", [])
        for entry in entries:
            messaging_events = entry.get("messaging", [])

            for event in messaging_events:
                # 1. Silently ignore read receipts
                if "read" in event:
                    continue

                message_data = event.get("message")
                if not message_data or message_data.get("is_echo"):
                    continue

                mid = message_data.get("mid")
                text = message_data.get("text")
                if mid and messages_cache.get(mid):
                    print(f"[DUPLICATE] Skipping already-processed message {mid}")
                    continue


                # Cache incoming message text
                if mid:
                    messages_cache.add(mid, text or "[NON_TEXT_MESSAGE]")
                # 2. Extract Ad Referral Data (Click-to-Messenger Ads)
                referral = (
                            event.get("referral")
                            or event.get("postback", {}).get("referral")
                            or message_data.get("referral")
                        )
                if referral:
                    ref_code = referral.get("ref", "N/A")
                    ad_id = referral.get("ad_id", "N/A")
                    source = referral.get("source", "N/A")
                    type_str = referral.get("type", "N/A")

                    print(f"\n[MESSENGER AD REFERRAL DETECTED]")
                    print(f" - Ref Code: {ref_code}")
                    print(f" - Ad ID: {ad_id}")
                    print(f" - Source: {source}")
                    print(f" - Type: {type_str}\n")

                # 3. Extract Reply Context (Quoted message)
                reply_to = message_data.get("reply_to") or {}
                replied_mid = reply_to.get("mid")

                if replied_mid:
                    original_text = messages_cache.get(replied_mid)

                    if not original_text:
                        original_msg = await get_message_by_mid(replied_mid)
                        original_text = original_msg.get("message", "N/A")

                    print(f"[REPLY TO] {original_text}")

                # 4. Dispatch Auto Reply
                sender_id = event.get("sender", {}).get("id")
                if text and sender_id:
                    print(f"[NEW MESSENGER MESSAGE] From {sender_id}: {text}")
                    await send_auto_reply(sender_id, text)

    except Exception as err:
        print(f"[EXCEPTIONAL ERROR]: {err}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@router.get("")
async def verify_webhook(request: Request):
    """Webhook verification endpoint for Facebook Messenger."""
    return sec.verify_webhooks(request, settings.VERIFY_TOKEN)


@router.post("")
async def receive_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    """Receive and validate incoming Facebook Messenger webhook payload."""
    try:
        raw_body = await request.body()
    
    except ClientDisconnect:
        print("[ERROR] Client disconnected before body was fully received.")
        return Response(content="Client disconnected", status_code=400)

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not sec.verify_signature(raw_body, signature,settings.MESSENGER_APP_SECRET.get_secret_value()):
        print("[SECURITY] Invalid signature — request rejected.")
        return Response(content="Invalid signature", status_code=403)

    try:
        payload = json.loads(raw_body)
    
    except Exception:
        print("[ERROR] Failed to parse JSON payload.")
        return Response(content="Bad request", status_code=400)

    print("\n================ [NEW MESSENGER EVENT] ================")
    background_tasks.add_task(process_webhook_payload, payload)
    return {"status": "EVENT_RECEIVED"}