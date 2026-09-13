import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import hashlib
import hmac
import os
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, Response
import httpx
import uvicorn



#* add duplicate message checker 

# ─────────────────────────────────────────────
# Setup & Config
# ─────────────────────────────────────────────
load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "").strip()
PAGE_ACCESS_TOKEN = os.getenv("MESSENGER_TOKEN", "").strip()
APP_SECRET = os.getenv("MESSENGER_APP_SECRET", "").strip()

if not VERIFY_TOKEN:
    raise ValueError("VERIFY_TOKEN not set in .env")
if not PAGE_ACCESS_TOKEN:
    raise ValueError("MESSENGER_TOKEN not set in .env")
if not APP_SECRET:
    raise ValueError("MESSENGER_APP_SECRET not set in .env")

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
# LRU Cache Setup
# ─────────────────────────────────────────────
MAX_CACHE = 1200
messages_cache = OrderedDict()

#! DRY
def cache_message(mid: str, text: str):
    """Store message in LRU cache and evict oldest if limit is exceeded."""
    messages_cache[mid] = text
    if len(messages_cache) > MAX_CACHE:
        messages_cache.popitem(last=False)


# ─────────────────────────────────────────────
# Signature verification
# ─────────────────────────────────────────────
#! DRY
def verify_signature(raw_body: bytes, signature_header: str) -> bool:
    """Verify payload authenticity using Meta App Secret HMAC-SHA256."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        APP_SECRET.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature_header)


# ─────────────────────────────────────────────
# HTTPX Async Helpers
# ─────────────────────────────────────────────
#! DRY
async def send_auto_reply(
    recipient_id: str, text_message: str, retries: int = 3
):
    """Send automated reply via Facebook Messenger Graph API with retry logic and backoff."""
    reply_url = "https://graph.facebook.com/v26.0/me/messages"
    headers = {"Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"}
    json_data = {
        "recipient": {"id": recipient_id},
        "message": {"text": text_message},
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
                    break  # Stop retry loop immediately on success

                print(
                    f"[REPLY ERROR] Attempt {attempt}/{retries} | Status: {response.status_code} | Body: {response.json()}"
                )

            except Exception as err:
                print(
                    f"[REPLY FAILED] Attempt {attempt}/{retries} | Error: {err}"                )

            if attempt < retries:
                await asyncio.sleep(1)


async def get_message_by_mid(message_id: str) -> dict:
    """Fetch message details using its mid via Facebook Graph API."""
    url = f"https://graph.facebook.com/v26.0/{message_id}"
    params = {
        "fields": "message",
        "access_token": PAGE_ACCESS_TOKEN,
    }

    try:
        client = app.state.client
        response = await client.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(
                    f"[FETCH ERROR] Status: {response.status_code} | Body: {response.json()}"
            )
            return {}

    except Exception as err:
        print(f"[REQUEST FAILED]: {err}")
        return {}


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
                if mid and mid in messages_cache:
                    print(f"[DUPLICATE] Skipping already-processed message {mid}")
                    continue


                # Cache incoming message text
                if mid:
                    cache_message(mid, text or "[NON_TEXT_MESSAGE]")
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
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Webhook verification endpoint for Facebook Messenger."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("\n[SUCCESS] Messenger Webhook verified successfully by Meta!")
        return Response(
            content=challenge, media_type="text/plain", status_code=200
        )

    print("\n[ERROR] Verification failed.")
    return Response(content="Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    """Receive and validate incoming Facebook Messenger webhook payload."""
    raw_body = await request.body()

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(raw_body, signature):
        print("[SECURITY] Invalid signature — request rejected.")
        return Response(content="Invalid signature", status_code=403)

    try:
        payload = await request.json()
    except Exception:
        print("[ERROR] Failed to parse JSON payload.")
        return Response(content="Bad request", status_code=400)

    print("\n================ [NEW MESSENGER EVENT] ================")
    background_tasks.add_task(process_webhook_payload, payload)
    return {"status": "EVENT_RECEIVED"}


if __name__ == "__main__":
    uvicorn.run("fast_m:app", reload=True, host="127.0.0.1", port=8000)