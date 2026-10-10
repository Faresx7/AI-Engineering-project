from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
import jwt
import redis.asyncio as aioredis
from slowapi import Limiter

from src.core.config import settings
from src.services.upload_service import submit_upload

logger = logging.getLogger(__name__)

# !!!!!!!!!!!!!!!!!!!!!!!!!
# ! clean code and improve naming
# ! check if the task_id matches user_id
# ! IDOR FIX
# ! create doc_id
# !!!!!!!!!!!!!!!!!!!!!!!!!

# Async Redis Client with expanded connection pool for concurrent SSE connections
redis_client = aioredis.from_url(
    settings.CELERY_BROKER_URL,
    decode_responses=True,
    max_connections=500,
)



def get_real_ip(request: Request) -> str:
    """Extract real client IP address behind reverse proxies (Nginx/Cloudflare/ALB)."""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    # if founded
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"



limiter = Limiter(
    key_func=get_real_ip, 
    storage_uri=settings.CELERY_BROKER_URL # to ensue all workers in the same limit
)

# Configure Rate Limiter with real IP resolution
limiter = Limiter(key_func=get_real_ip)
router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SECRET_KEY = settings.JWT_SECRET_KEY.get_secret_value()

ALGORITHM = "HS256"




def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Reject callers that don't know the shared secret using constant-time comparison."""
    expected = settings.INGEST_API_KEY.get_secret_value().encode()
    if not hmac.compare_digest(x_api_key.encode(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )



def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Validate JWT token and enforce strict ID_PATTERN regex on the extracted user_id."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing sub claim.",
            )

        if not re.match(settings.ID_PATTERN, user_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user_id format in token payload.",
            )

        return user_id

    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
        )



async def _fetch_state_async(task_id: str) -> tuple[str, dict | None]:
    """Pure async fetch of Celery task metadata directly from Redis without thread pool overhead."""
    raw_meta = await redis_client.get(f"celery-task-meta-{task_id}")
    if not raw_meta:
        return "PENDING", None

    try:
        meta = json.loads(raw_meta)
        state = meta.get("status", "PENDING")
        result = meta.get("result")
        info = result if isinstance(result, dict) else None
    
        return state, info
    
    except Exception:
        logger.exception(f"Failed to parse Celery metadata from Redis for task {task_id}")
        return "PENDING", None


# -----------------------------------------
# Upload Route
# -----------------------------------------


@router.post("/upload", status_code=202, dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    doc_id: Optional[str] = Form(None, pattern=settings.ID_PATTERN)
):
    """Accept a file upload, stream it to disk, and queue it asynchronously for ingestion."""
    task_id = await submit_upload(file, user_id, doc_id)
    return {"task_id": task_id, "status": "queued"}


# -----------------------------------------
# Basic Polling Route
# -----------------------------------------


@router.get("/status/{task_id}", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def ingestion_status(request: Request, task_id: str):
    """Poll task status by UUID (Standard REST polling endpoint)."""
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown task.")

    state, info = await _fetch_state_async(task_id)

    if state == "PENDING":
        return {"status": "queued"}
    
    if state == "STARTED":
        return {"status": "running"}
    
    if state == "PROGRESS" and isinstance(info, dict):
        return {"status": "running", "done": info.get("done", 0), "total": info.get("total", 0)}
    
    if state == "RETRY":
        return {"status": "retrying"}
    
    if state in ["SUCCESS", "completed", "done"] and isinstance(info, dict):
        return info

    return {"status": "failed", "error": "Processing failed. Please try again later."}


# -----------------------------------------
# SSE - Server-Sent Event Route
# -----------------------------------------


async def _sse_status_generator(request: Request, task_id: str) -> AsyncGenerator[str, None]:
    channel_name = f"task_status:{task_id}"
    pubsub = redis_client.pubsub()

    # Maximum stream duration limit (15 minutes) to prevent zombie connections
    max_stream_duration = 900.0
    start_time = time.monotonic()

    # Terminal states to determine task completion
    terminal_states = {"SUCCESS", "FAILURE", "REVOKED", "done", "failed", "completed"}
    success_states = {"SUCCESS", "done", "completed"}
    failed_states = {"FAILURE", "REVOKED", "failed"}

    try:
        # 1. Subscribe to channel first so Redis buffers messages sent during initial status fetch
        await pubsub.subscribe(channel_name)

        # 2. Asynchronously fetch initial task status directly from Redis
        state, info = await _fetch_state_async(task_id)

        if state in success_states:
            initial_status = "completed"

        elif state in failed_states:
            initial_status = "failed"
        
        elif state in {"STARTED", "PROGRESS"}:
            initial_status = "running"
        
        else:
            initial_status = "queued"


        if isinstance(info, dict):
            initial_payload = {"status": initial_status, **info}
        else:
            initial_payload = {"status": initial_status}

        yield f"data: {json.dumps(initial_payload)}\n\n"

        # If task was already completed before subscription, exit immediately
        if state in terminal_states or initial_status in terminal_states:
            return

        # 3. Main event listening loop
        while True:
            # Safety Check 1: Client disconnection detection
            if await request.is_disconnected():
                logger.info(f"SSE client disconnected for task: {task_id}")
                break

            # Safety Check 2: Max stream duration check using monotonic clock
            if time.monotonic() - start_time > max_stream_duration:
                timeout_payload = {"status": "failed", "error": "Stream timeout exceeded"}
                yield f"data: {json.dumps(timeout_payload)}\n\n"
                break

            # Fetch incoming Pub/Sub message with a 2-second timeout
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=2.5)

            if message and message.get("type") == "message":
                raw_data = message.get("data")
                data_str = raw_data.decode("utf-8") if isinstance(raw_data, bytes) else str(raw_data)

                yield f"data: {data_str}\n\n"

                # Parse JSON payload to check for terminal state completion
                try:
                    payload = json.loads(data_str)
                    task_status = payload.get("status") or payload.get("state")
                    if task_status in terminal_states:
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                # Send keep-alive comment line when no data message arrives in this iteration
                yield ": keep-alive\n\n"

    except asyncio.CancelledError:
        logger.info(f"SSE stream generator cancelled for task: {task_id}")
    
    finally:
        # Safe cleanup of Pub/Sub resources without masking original exceptions
        try:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
        except Exception:
            logger.warning(f"Failed to cleanly close pubsub channel for task {task_id}", exc_info=True)




@router.get("/status/stream/{task_id}", dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
async def stream_ingestion_status(request: Request, task_id: str):
    """Real-time Server-Sent Events (SSE) endpoint to stream status updates to client."""
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown task.")

    return StreamingResponse(
        _sse_status_generator(request, task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )