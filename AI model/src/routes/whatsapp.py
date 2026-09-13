from fastapi import BackgroundTasks, FastAPI, Request, Response
from contextlib import asynccontextmanager
from collections import OrderedDict
from dotenv import load_dotenv
import asyncio
import uvicorn
import httpx
import os

import core.security as sec
import core.cache as cache

# ─────────────────────────────────────────────
# Setup & Config
# ─────────────────────────────────────────────
load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "").strip()
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "").strip()
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "").strip()
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "").strip()

if not VERIFY_TOKEN:
    raise ValueError("VERIFY_TOKEN not set in .env")
if not WHATSAPP_TOKEN or WHATSAPP_TOKEN == "":
    raise ValueError("WHATSAPP_TOKEN not set in .env")
if not PHONE_NUMBER_ID:
    raise ValueError("PHONE_NUMBER_ID not set in .env")
if not WHATSAPP_APP_SECRET:
    raise ValueError("WHATSAPP_APP_SECRET not set in .env")


# ─────────────────────────────────────────────
# Lifespan 
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.send_semaphore = asyncio.Semaphore(10)
    app.state.client = httpx.AsyncClient(timeout=10)
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


# ─────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────
messages_cache = cache.MessageCache(max_size=7000)

# ─────────────────────────────────────────────
# HTTPX Async Helpers
# ─────────────────────────────────────────────
#! DRY
async def send_auto_reply(
    recipient_phone: str, text_message: str, retries: int = 3
):
    """Send automated reply via WhatsApp Cloud API with retry logic and backoff."""
    reply_url = (
        f"https://graph.facebook.com/v26.0/{PHONE_NUMBER_ID}/messages"
    )
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    json_data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone,
        "type": "text",
        "text": {"preview_url": False, "body": text_message},
    }

    client = app.state.client

    semaphore = app.state.send_semaphore
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(
                    reply_url, headers=headers, json=json_data
                    )

                if response.status_code == 200:
                    print(
                        f"[REPLY SUCCESS] Status: 200 | Response: {response.json()}"
                    )
                    break

                print(
                    f"[REPLY ERROR] Attempt {attempt}/{retries} | Status: {response.status_code} | Body: {response.json()}"
                )

            except Exception as err:
                print(
                    f"[REPLY FAILED] Attempt {attempt}/{retries} | Error: {err}")

            if attempt < retries:
                await asyncio.sleep(1)


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
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Webhook verification endpoint for WhatsApp Meta."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    print(
        f"\n[VERIFY TRY] Token from Meta: '{token}' | Expected: '{VERIFY_TOKEN}'"
    )
    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("\n[SUCCESS] Webhook verified successfully by Meta!")
        return Response(
            content=challenge, media_type="text/plain", status_code=200
        )

    print("\n[ERROR] Verification failed.")
    return Response(content="Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    """Receive and validate incoming webhook payload."""
    raw_body = await request.body()

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not sec.verify_signature(raw_body, signature, WHATSAPP_APP_SECRET):
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


if __name__ == "__main__":
    uvicorn.run("whatsapp:app", reload=True, host="127.0.0.1", port=8000)