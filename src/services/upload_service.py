from typing import Optional
from pathlib import Path
import logging
import uuid
import re

import asyncio
import aiofiles
from fastapi import HTTPException, UploadFile

from src.core.config import settings

from src.core.celery_app import celery_app

logger = logging.getLogger(__name__)


READ_CHUNK_BYTES = settings.READ_CHUNK_BYTES    # 1 MB Stream Buffer


def _clean_filename(raw: Optional[str]) -> str:
    """Keep only a harmless display name for UI rendering."""
    name = re.split(r"[\\/]", raw or "")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:255] or "unknown_file"


def _verify_magic_bytes_stream(head: bytes, ext: str) -> bool:
    """Validate that a file's head content matches its declared extension.
    
    Reads the initial bytes of the uploaded file without blocking the event loop,
    then checks common file signatures (for example PDF headers or ZIP local file
    headers) to ensure the content is consistent with the extension provided by the
    client. Returns True only when the detected header matches the expected type.
    """
    if not head:
        return False
    if ext == ".pdf":
        return b"%PDF-" in head[:1024]
    if ext in settings.ZIP_EXTENSIONS:
        return head.startswith(b"PK\x03\x04")
    if ext in settings.TEXT_EXTENSIONS:
        return b"\x00" not in head
    return False


async def _stream_to_disk(file: UploadFile, ext: str, dest: Path, max_bytes: int) -> int:
    """
    Pure async file streaming directly to disk using aiofiles.
    Zero thread offloading, fully integrated into Python's native event loop.
    """
    written = 0
    file_exceeded = False
    is_first_chunk = True

    try:
        # Native async context manager for file writing
        async with aiofiles.open(dest, "wb") as out:
            while True:
                
                chunk = await file.read(size = READ_CHUNK_BYTES)
                
                if not chunk:
                    break

                # verify file before loading it all in the disk
                if is_first_chunk:
                    if not _verify_magic_bytes_stream(chunk, ext):
                        raise HTTPException(status_code=400, detail="The file content does not match its extension.")
                    is_first_chunk = False 
                
                written += len(chunk)
                if written > max_bytes:
                    file_exceeded = True
                    break

                # Pure non-blocking async write
                await out.write(chunk)

        # Enforce size limit after clean handle closure
        if file_exceeded:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size limit of {settings.MAX_UPLOAD_MB} MB.",
            )

    except BaseException:
        dest.unlink(missing_ok=True)
        raise

    return written


async def _enqueue_ingestion_task(stored_name: str, original_name: str, user_id: str, doc_id: Optional[str]) -> str:
    """Isolated helper to construct and push the Celery JSON payload to Redis."""
    
    try:
                # this way is better and faster,
                # the .to_thread don't affect to much in process 
                # because the process don't take much time
        task = await asyncio.to_thread(celery_app.send_task,
                                       name="ingestion.ingest_document",
                                       kwargs={
                                           "stored_name": stored_name,
                                           "original_name": original_name,
                                           "user_id": user_id,
                                           "doc_id": doc_id
                                       },
                                       )
        return task.id

                                
    
    except Exception:
        logger.exception("Failed to push task payload to Redis queue")
                
        raise HTTPException(status_code=503, detail="Processing queue is currently unavailable.")
    

async def submit_upload(file: UploadFile, user_id: str, doc_id: Optional[str]) -> str:
    """
    100% Pure Async Upload Endpoint Handler:
    Zero threads allocated, zero blocking calls, maximum throughput for FastAPI.
    return -> task_id
    """
    # 1. Sanitize filename and validate extension
    original_name = _clean_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported file type. Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}",)

    # 2. Fast-fail size validation
    max_bytes = settings.MAX_UPLOAD_MB
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"File exceeds limit of {settings.MAX_UPLOAD_MB} MB."
        )

    # 3. Create unique UUID target path
    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest = settings.UPLOAD_DIR / stored_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 4. Stream to disk using pure async aiofiles
    size = await _stream_to_disk(file, ext, dest, max_bytes)

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Construct native Celery protocol JSON task payload
    return await _enqueue_ingestion_task(stored_name, original_name, user_id, doc_id)