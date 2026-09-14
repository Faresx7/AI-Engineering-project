from fastapi import BackgroundTasks, APIRouter, Request, Response
from starlette.requests import ClientDisconnect
import json

from src.core.config import settings
import src.core.http_client as hc
import src.core.security as sec
import src.core.cache as cache

# ─────────────────────────────────────────────
# Router 
# ─────────────────────────────────────────────
router = APIRouter(prefix="/webhook/whatsapp",
                   tags=['whatsapp'])

# ─────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────
messages_cache = cache.MessageCache(max_size=15000)

# ─────────────────────────────────────────────
# HTTPX Async Helpers
# ─────────────────────────────────────────────
async def send_auto_reply(
    recipient_phone: str, text_message: str):
    """Send automated reply via WhatsApp Cloud API with retry logic and backoff."""
    reply_url = (
        f"https://graph.facebook.com/v26.0/{settings.PHONE_NUMBER_ID.get_secret_value()}/messages"
    )
    headers = {"Authorization": f"Bearer {settings.WHATSAPP_TOKEN.get_secret_value()}"}
    json_data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone,
        "type": "text",
        "text": {"preview_url": False, "body": text_message},
    }

    return await hc.send_with_retry(reply_url, headers, json_data)


# ─────────────────────────────────────────────
# Core Webhook Processing Logic
# ─────────────────────────────────────────────
async def process_webhook_payload(payload: dict):
    """Process incoming WhatsApp webhook events asynchronously in background."""
    try:
        entries = payload.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])

            for change in changes:
                value = change.get("value", {})

                # 1. Silently skip status updates (sent, delivered, read receipts)
                if "statuses" in value:
                    status = value.get("statuses", [{}])[0].get("status")
                    print(f"[STATUS UPDATE] {status}")
                    continue

                messages = value.get("messages", [])
                for msg in messages:
                    msg_id = msg.get("id")  # WhatsApp Message ID (wamid)
                    sender_phone = msg.get("from")
                    msg_type = msg.get("type")

                    # duplicated message check
                    if msg_id and messages_cache.get(msg_id):
                        print(f"[DUPLICATE] Skipping already-processed message {msg_id}")
                        continue

                    # Extract text content
                    text_body = ""
                    if msg_type == "text":
                        text_body = msg.get("text", {}).get("body", "")

                    # Cache incoming message ID regardless of type to prevent duplicate processing
                    if msg_id:
                        messages_cache.add(msg_id, text_body or f"[{msg_type.upper()}_MESSAGE]")

                    # 2. Extract Ad Referral Data (Click to WhatsApp Ads)
                    referral = msg.get("referral") or value.get("referral", {})
                    if referral:
                        headline = referral.get("headline", "N/A")
                        body = referral.get("body", "N/A")
                        source_url = referral.get("source_url", "N/A")

                        print(f"\n[AD REFERRAL DETECTED]")
                        print(f" - Headline: {headline}")
                        print(f" - Body: {body}")
                        print(f" - Source URL: {source_url}\n")

                    # 3. Extract Reply Context (Quoted message)
                    context = msg.get("context", {})
                    replied_wamid = context.get("id")

                    if replied_wamid:
                        original_text = messages_cache.get(replied_wamid)
                        print(f"[REPLY TO] {original_text}")

                    # 4. Dispatch Auto Reply
                    if text_body and sender_phone:
                        print(f"[NEW MESSAGE] From {sender_phone}: {text_body}")
                        await send_auto_reply(sender_phone, text_body)

    except Exception as err:
        print(f"[EXCEPTIONAL ERROR]: {err}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@router.get("")
async def verify_webhook(request: Request):
    """Webhook verification endpoint for WhatsApp Meta."""
    return sec.verify_webhooks(request, settings.VERIFY_TOKEN)


@router.post("")
async def receive_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    """Receive and validate incoming webhook payload."""
    try:
        raw_body = await request.body()
        
    except ClientDisconnect:
        print("[ERROR] Client disconnected before body was fully received.")
        return Response(content="Client disconnected", status_code=400)

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not sec.verify_signature(raw_body, signature, settings.WHATSAPP_APP_SECRET.get_secret_value()):
        print("[SECURITY] Invalid signature — request rejected.")
        return Response(content="Invalid signature", status_code=403)

    try:
        payload = json.loads(raw_body)
    except Exception:
        print("[ERROR] Failed to parse JSON payload.")
        return Response(content="Bad request", status_code=400)

    print("\n================ [NEW WHATSAPP EVENT] ================")
    background_tasks.add_task(process_webhook_payload, payload)
    return {"status": "EVENT_RECEIVED"}
