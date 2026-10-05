from __future__ import annotations

import asyncio
import hmac
import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.celery_app import celery_app
from src.core.config import settings
from src.services.upload_service import submit_upload

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Extract the raw secret string for PyJWT compatibility
SECRET_KEY = settings.INGEST_API_KEY.get_secret_value()
ALGORITHM = "HS256"


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Validate JWT token and enforce ID_PATTERN regex on the extracted user_id."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[str] = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing sub claim.",
            )

        # Enforce security regex pattern on user_id extracted from JWT
        if not re.match(settings.ID_PATTERN, user_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user_id format in token payload.",
            )

        return user_id

    except jwt.PyJWTError:
        # Fixed lowercase status.http_401_UNAUTHORIZED bug
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
        )


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Reject callers that don't know the shared secret (constant-time comparison)."""
    expected = settings.INGEST_API_KEY.get_secret_value().encode()
    if not hmac.compare_digest(x_api_key.encode(), expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@router.post("/upload", status_code=202, dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def upload_document(
    request: Request,  # Added Request parameter required by slowapi
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    doc_id: Optional[str] = Form(None, pattern=settings.ID_PATTERN),
):
    """Accept a file and queue it for processing asynchronously.

    Returns immediately (202 Accepted) with a task_id.
    """
    task_id = await submit_upload(file, user_id, doc_id)
    return {"task_id": task_id, "status": "queued"}


def _fetch_state(task_id: str):
    """Synchronous helper to query Redis for Celery task state."""
    result = celery_app.AsyncResult(task_id)
    return result.state, result.info


@router.get("/status/{task_id}", dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
async def ingestion_status(request: Request, task_id: str):  # Added Request parameter
    """Poll the status of an ingestion task by its UUID."""
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown task.")

    state, info = await asyncio.to_thread(_fetch_state, task_id)

    if state == "PENDING":
        return {"status": "queued"}
    if state == "STARTED":
        return {"status": "running"}
    if state == "PROGRESS" and isinstance(info, dict):
        return {"status": "running", "done": info.get("done", 0), "total": info.get("total", 0)}
    if state == "RETRY":
        return {"status": "retrying"}
    if state == "SUCCESS" and isinstance(info, dict):
        return info

    # Never leak internal exceptions to the caller
    return {"status": "failed", "error": "Processing failed. Please try again later."}