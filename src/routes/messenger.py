from fastapi import BackgroundTasks,APIRouter, Request, Response
from starlette.requests import ClientDisconnect
import asyncio 
import json

from src.core.config import settings
import src.core.http_client as hc
import src.core.security as sec
import src.core.cache as cache

# ! use shared cache between all workers


# * DONE[content_size, api version separation, token leakage in params variable, race condition,
# * used asyncio.gather,]

# ─────────────────────────────────────────────
# Router
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
    reply_url = f"https://graph.facebook.com/{settings.FB_GRAPH_API_VERSION}/me/messages"
    headers = {"Authorization": f"Bearer {settings.MESSENGER_TOKEN.get_secret_value()}"}
    json_data = {
        "recipient": {"id": recipient_id},
        "message": {"text": text_message},
    }
    
    await hc.send_with_retry(reply_url,headers,json_data)


async def get_message_by_mid(message_id: str):
    """Fetch message details using its mid via Facebook Graph API."""
    url = f"https://graph.facebook.com/{settings.FB_GRAPH_API_VERSION}/{message_id}"
    # ! its not optimal to send token in params, it be be sent in header
    
    # * token sent to public in params have been fixed
    header = {"Authorization": f"Bearer {settings.MESSENGER_TOKEN.get_secret_value()}"}
    params = {"fields": "message"}

    return await hc.get_message_by_mid(url, header, params) or None
                        

# ─────────────────────────────────────────────
# Core Webhook Processing Logic
# ─────────────────────────────────────────────
async def _handle_single_event(event: dict):
    
    if "read" in event:
        return



    message_data = event.get("message")
    if not message_data:
        return

    mid = message_data.get("mid")
    text = message_data.get("text")

    # Cache outgoing bot replies (echoes) before ignoring them
    if message_data.get("is_echo"):
        if mid:
            messages_cache.set_if_absent(mid, text or "[NON_TEXT_MESSAGE]")
        return

    # ! fixed race condition
    if mid and not messages_cache.set_if_absent(mid, text or "[NON TEXT MESSAGE]"):
        print(f"[DUPLICATE] Skipping already-processed message {mid}")
        return 
    
    # Extract Ad Referral Data (Click-to-Messenger Ads)
    referral = (
        event.get("referral")
        or event.get("postback", {}).get("referral")
        or message_data.get("referral")
    )

    if referral:
        ref_code = referral.get("ref", None)
        ad_id = referral.get("ad_id", None)
        source = referral.get("source", None)
        type_str = referral.get("type", None)

        print("\n[MESSENGER AD REFERRAL DETECTED]")
        print(f" - Ref Code: {ref_code}")
        print(f" - Ad ID: {ad_id}")
        print(f" - Source: {source}")
        print(f" - Type: {type_str}\n")

    # Extract Reply Context (Quoted message)
    original_text = ""
    reply_to = message_data.get("reply_to") or {}
    replied_mid = reply_to.get("mid")

    if replied_mid:
        original_text = messages_cache.get(replied_mid)

        if not original_text:
            original_msg = await get_message_by_mid(replied_mid)
            if original_msg:
                original_text = original_msg.get("message", None)

        print(f"[REPLY TO] {original_text}")

    # Dispatch Auto Reply
    sender_id = event.get("sender", {}).get("id")
    if text and sender_id:

        # ! user personal data leakage
        print(f"[NEW MESSENGER MESSAGE] From {sender_id}: {text}")
        
        # for testing
        await send_auto_reply(sender_id, text + (f"\n{original_text}" or ""))


async def process_webhook_payload(payload: dict):
    """Process incoming Facebook Messenger webhook events asynchronously in background."""
    try:
        tasks = []
        entries = payload.get("entry", [])

        # Collect all event tasks from all entries
        for entry in entries:
            messaging_events = entry.get("messaging", [])
            
            for event in messaging_events:
                tasks.append(_handle_single_event(event))

        # SRun all tasks concurrently OUTSIDE the loops
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
    """Webhook verification endpoint for Facebook Messenger."""
    return sec.verify_webhooks(request, settings.VERIFY_TOKEN)


@router.post("")
async def receive_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    """Receive and validate incoming Facebook Messenger webhook payload."""
    content_length = int(request.headers.get('content-length', 0))
    if content_length > 1000 * 1024:
        return Response(content="[PAYLOAD IS TOO LARGE]", status_code=413)
        
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