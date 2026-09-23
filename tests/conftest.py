"""
conftest.py — Global pytest fixtures shared across all test modules.

REAL-LIFE SIMULATION:
  This file bootstraps a fully isolated test environment that mirrors
  production without touching any live external services.

  - All secret tokens are stubbed with syntactically-valid fake values.
  - httpx.AsyncClient is replaced with a controllable AsyncMock.
  - The RAGChain container is pre-wired with a MagicMock to avoid
    loading GPU models or Chroma DB during CI runs.
  - A lightweight FastAPI TestClient (httpx-based) replaces uvicorn.
"""

import asyncio
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ──────────────────────────────────────────────────────────────────────────────
# Make the project root importable in every test worker
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Inject fake environment variables BEFORE any src.* import so that
# pydantic-settings (Settings) never tries to read a real .env file.
# ──────────────────────────────────────────────────────────────────────────────
FAKE_ENV = {
    "VERIFY_TOKEN": "test_verify_token_abc123",
    "APP_SECRET": "test_app_secret_xyz",
    "INSTAGRAM_TOKEN": "IGTOKEN_fake_1234567890",
    "WHATSAPP_TOKEN": "WATOKEN_fake_1234567890",
    "WHATSAPP_APP_SECRET": "wa_app_secret_fake",
    "PHONE_NUMBER_ID": "123456789",
    "MESSENGER_TOKEN": "MSTOKEN_fake_1234567890",
    "MESSENGER_APP_SECRET": "ms_app_secret_fake",
    "FB_GRAPH_API_VERSION": "v26.0",
    "GEMINI_API_KEY": "fake_gemini_key",
}

for key, value in FAKE_ENV.items():
    os.environ.setdefault(key, value)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_valid_signature(body: bytes, secret: str) -> str:
    """
    Re-creates the exact HMAC-SHA256 signature that Meta's servers send.
    Used to produce green-path authenticated webhook requests in tests.
    """
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def make_instagram_payload(
    sender_id: str = "user_001",
    mid: str = "mid_test_001",
    text: str = "Hello bot",
    is_echo: bool = False,
) -> dict:
    """Factory: canonical Instagram webhook payload."""
    msg: dict = {"mid": mid, "text": text}
    if is_echo:
        msg["is_echo"] = True
    return {
        "object": "instagram",
        "entry": [
            {
                "id": "page_123",
                "messaging": [
                    {"sender": {"id": sender_id}, "recipient": {"id": "bot_456"}, "message": msg}
                ],
            }
        ],
    }


def make_whatsapp_payload(
    sender_phone: str = "+201234567890",
    wamid: str = "wamid.test001",
    text: str = "Hello WA",
    msg_type: str = "text",
) -> dict:
    """Factory: canonical WhatsApp Cloud API webhook payload."""
    msg: dict = {"id": wamid, "from": sender_phone, "type": msg_type}
    if msg_type == "text":
        msg["text"] = {"body": text}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "wa_business_id",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [msg],
                        },
                    }
                ],
            }
        ],
    }


def make_messenger_payload(
    sender_id: str = "fb_user_001",
    mid: str = "mid_fb_001",
    text: str = "Hello Messenger",
    is_echo: bool = False,
) -> dict:
    """Factory: canonical Facebook Messenger webhook payload."""
    msg: dict = {"mid": mid, "text": text}
    if is_echo:
        msg["is_echo"] = True
    return {
        "object": "page",
        "entry": [
            {
                "id": "page_fb_123",
                "messaging": [
                    {"sender": {"id": sender_id}, "recipient": {"id": "bot_789"}, "message": msg}
                ],
            }
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pytest Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Provide a single asyncio event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def app_settings():
    """
    Return the loaded Settings object (populated from FAKE_ENV above).
    Scope=session because Settings is immutable after construction.
    """
    from src.core.config import settings
    return settings


@pytest.fixture()
def mock_http_client():
    """
    Replace the global httpx.AsyncClient with a controllable AsyncMock.

    REAL-LIFE SIMULATION:
        Simulates the shared HTTP client that is normally initialised during
        FastAPI lifespan. Tests configure .post.return_value or
        .get.return_value to simulate any API response without a real network.
    """
    import src.core.http_client as hc_module

    mock_client = AsyncMock()
    mock_client.post = AsyncMock()
    mock_client.get = AsyncMock()

    original = hc_module.http_client
    hc_module.http_client = mock_client
    yield mock_client
    hc_module.http_client = original


@pytest.fixture()
def mock_rag_chain():
    """
    Inject a MagicMock RAGChain into the DI container.

    REAL-LIFE SIMULATION:
        Prevents the test from loading a 1.5 GB Qwen/Gemini model or
        connecting to a Chroma DB while exercising all code paths that
        call container.rag_chain_instance.answer().
    """
    import src.core.container as container_module

    mock_chain = AsyncMock()
    mock_chain.answer = AsyncMock(
        return_value={
            "answer": "Mocked RAG answer",
            "context": ["context chunk 1"],
            "final_prompt": "mocked prompt",
        }
    )
    original = container_module.rag_chain_instance
    container_module.rag_chain_instance = mock_chain
    yield mock_chain
    container_module.rag_chain_instance = original


@pytest.fixture()
def test_client(mock_http_client, mock_rag_chain):
    """
    Build a FastAPI TestClient that routes through the real application
    router tree but with all external dependencies mocked.

    REAL-LIFE SIMULATION:
        Mirrors a production HTTP request flow at the ASGI layer without
        spinning up uvicorn or connecting to Meta Graph API.
    """
    from fastapi import FastAPI
    from src.routes import instagram, messenger, whatsapp

    test_app = FastAPI()
    test_app.include_router(instagram.router)
    test_app.include_router(messenger.router)
    test_app.include_router(whatsapp.router)

    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client
