import asyncio
import json
from fastapi import APIRouter, BackgroundTasks, Request, Response
from starlette.requests import ClientDisconnect

import src.core.cache as cache
from src.core.config import settings
import src.core.http_client as hc
import src.core.security as sec

# ─────────────────────────────────────────────
# Router & Cache
# ─────────────────────────────────────────────
router = APIRouter(prefix="/webhook/whatsapp", tags=["whatsapp"])
messages_cache = cache.MessageCache(max_size=15000)


# ─────────────────────────────────────────────
# HTTPX Async Helpers
# ─────────────────────────────────────────────
async def send_auto_reply(recipient_phone: str, text_message: str):
    """Send automated reply via WhatsApp Cloud API and cache outgoing message ID."""
    reply_url = (
        f"https://graph.facebook.com/{settings.FB_GRAPH_API_VERSION}/"
        f"{settings.PHONE_NUMBER_ID.get_secret_value()}/messages"
    )
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_TOKEN.get_secret_value()}"
    }
    json_data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone,
        "type": "text",
        "text": {"preview_url": False, "body": text_message},
    }

    response = await hc.send_with_retry(reply_url, headers, json_data)

    # 💡 تخزين الـ wamid الخاص برسات البوت فور إرسالها لضمان وجودها عند الرد عليها
    if response and isinstance(response, dict):
        messages = response.get("messages", [])
        if messages:
            sent_wamid = messages[0].get("id")
            if sent_wamid:
                messages_cache.set_if_absent(sent_wamid, text_message)

    return response


# ─────────────────────────────────────────────
# Core Webhook Processing Logic
# ─────────────────────────────────────────────
async def _handle_single_event(value: dict, msg: dict):
    msg_id = msg.get("id")
    sender_phone = msg.get("from")
    msg_type = msg.get("type", "unknown")

    text_body = ""
    if msg_type == "text":
        text_body = msg.get("text", {}).get("body", "")

    cache_value = text_body or f"[{msg_type.upper()}_MESSAGE]"
    if msg_id and not messages_cache.set_if_absent(msg_id, cache_value):
        print(f"[DUPLICATE] Skipping already-processed message {msg_id}")
        return

    # Extract Ad Referral Data
    referral = msg.get("referral") or (value.get("referral") or {})
    if referral:
        headline = referral.get("headline")
        body = referral.get("body")
        source_url = referral.get("source_url")

        print("\n[AD REFERRAL DETECTED]")
        print(f" - Headline: {headline}")
        print(f" - Body: {body}")
        print(f" - Source URL: {source_url}\n")

    # Extract Reply Context (Quoted message)
    context = msg.get("context") or {}
    replied_wamid = context.get("id")
    original_text = ""

    if replied_wamid:
        original_text = messages_cache.get(replied_wamid) or ""
        print(f"[REPLY TO] {original_text}")

    # Dispatch Auto Reply with context
    if text_body and sender_phone:
        print(f"[NEW MESSAGE] From {sender_phone}: {text_body}")
        full_reply = (
            f"{text_body}\n\n[Quoted]: {original_text}"
            if original_text
            else text_body
        )
        await send_auto_reply(sender_phone, full_reply)


async def process_webhook_payload(payload: dict):
    """Process incoming WhatsApp webhook events asynchronously in background."""
    try:
        tasks = []
        entries = payload.get("entry", [])

        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})

                if "statuses" in value:
                    status = value.get("statuses", [{}])[0].get("status")
                    print(f"[STATUS UPDATE] {status}")
                    continue

                messages = value.get("messages", [])
                for msg in messages:
                    tasks.append(_handle_single_event(value, msg))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"[ERROR] Task failed silently: {result}")

    except asyncio.CancelledError:
        raise
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
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receive and validate incoming webhook payload."""
    content_length = int(request.headers.get("content-length", 0))
    if content_length > 1000 * 1024:
        return Response(content="[PAYLOAD IS TOO LARGE]", status_code=413)

    try:
        raw_body = await request.body()
    except ClientDisconnect:
        print("[ERROR] Client disconnected before body was fully received.")
        return Response(content="Client disconnected", status_code=400)

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not sec.verify_signature(
        raw_body,
        signature,
        settings.WHATSAPP_APP_SECRET.get_secret_value(),
    ):
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