"""The Celery task. Imported by the WORKER only (never by the API process)."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_ready

from src.core.config import settings
from src.core.celery_app import celery_app
from src.rag.pipeline import IngestionPipeline
from src.core.validation import STORED_NAME_RE, IngestionError

logger = logging.getLogger(__name__)

# Cheap to build: the heavy models inside it are loaded lazily on first use.
_pipeline = IngestionPipeline()

_STALE_UPLOAD_SECONDS = 24 * 3600


def _upload_path(stored_name: str) -> Path:
    """Map a queue message to a file path, accepting ONLY names we generated."""
    if not STORED_NAME_RE.fullmatch(stored_name):
        raise IngestionError("Invalid upload reference.")
    return settings.UPLOAD_DIR / stored_name


def _delete(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # e.g. Windows still holds a handle; the startup purge will remove it later.
        logger.warning("Could not delete %s", path)


@celery_app.task(bind=True, name="ingestion.ingest_document", max_retries=3)
def ingest_document(
    self,
    stored_name: str,
    original_name: str,
    user_id: str,
    doc_id: Optional[str] = None,
) -> dict:
    """Ingest one uploaded file.

    Return values (instead of raised exceptions) are used for EXPECTED failures
    (bad file), so they are not retried and no stack trace reaches the client.
    Unexpected errors (Qdrant down, network...) are retried with exponential backoff.
    """
    try:
        path = _upload_path(stored_name)
    except IngestionError as exc:
        return {"status": "failed", "error": str(exc)}

    if not path.exists():
        return {
            "status": "failed",
            "error": "The uploaded file was not found (it may have expired). Please upload it again.",
        }

    def on_progress(done: int, total: int) -> None:
        self.update_state(state="PROGRESS", meta={"done": done, "total": total})

    try:
        result = _pipeline.ingest_file(
            path=path,
            original_name=original_name,
            user_id=user_id,
            doc_id=doc_id,
            progress_cb=on_progress,
        )

    except IngestionError as exc:
        _delete(path)
        return {"status": "failed", "error": str(exc)}
    
    except SoftTimeLimitExceeded:
        _delete(path)
        logger.error("Ingestion timed out for %s", stored_name)
        return {"status": "failed", "error": "Processing took too long. Try a smaller file."}
    
    except Exception as exc:
        logger.exception(
            "Ingestion failed (attempt %d/%d)", self.request.retries + 1, self.max_retries + 1
        )
        if self.request.retries >= self.max_retries:
            _delete(path)  # last attempt: give up and clean the disk
            raise
    
        # Keep the file for the next attempt. 5s, 10s, 20s ... capped at 60s.
        raise self.retry(exc=exc, countdown=min(5 * 2 ** self.request.retries, 60))

    _delete(path)
    return {"status": "done", "doc_id": result.doc_id, "chunks": result.chunks}


@worker_ready.connect
def _purge_stale_uploads(sender=None, **kwargs) -> None:
    """On worker start, remove uploads that were orphaned (e.g. worker killed mid-job)."""
    cutoff = time.time() - _STALE_UPLOAD_SECONDS
    try:
        entries = list(os.scandir(settings.UPLOAD_DIR))
    except FileNotFoundError:
        return
    for entry in entries:
        if (
            entry.is_file()
            and STORED_NAME_RE.fullmatch(entry.name)
            and entry.stat().st_mtime < cutoff
        ):
            _delete(Path(entry.path))



import json
import redis

redis_sync = redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

def notify_progress(task_id: str, payload: dict):
    # ينشر الرسالة في القناة التي يستمع إليها FastAPI
    redis_sync.publish(f"task_status:{task_id}", json.dumps(payload))