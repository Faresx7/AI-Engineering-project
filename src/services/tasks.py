"""Celery worker task for document ingestion.

Optimized for a single-server Docker Compose environment using shared volumes.
Implements Fail-Fast principles, Clean Code, and a two-layer resource cleanup guarantee.
"""

from __future__ import annotations

from typing import Generator, Optional, Any
from contextlib import contextmanager
import stat as stat_module
from pathlib import Path
import logging
import time
import os


from celery.exceptions import SoftTimeLimitExceeded

from src.rag.ingestion.ingestion_pipeline import IngestionPipeline
from src.core.validation import STORED_NAME_RE, IngestionError
from src.core.celery_app import celery_app
from src.core.config import settings


logger = logging.getLogger(__name__)

# Lazy initialization of the heavy ML pipeline
_pipeline = IngestionPipeline()

_UPLOAD_DIR_RESOLVED = settings.UPLOAD_DIR.resolve(strict=True)
_MAX_UPLOAD_FILE_BYTES = settings.MAX_UPLOAD_MB



class ProgressThrottler:
    """Limits progress update emissions to Redis to prevent network thrashing."""

    def __init__(self, task_instance: Any, min_interval_seconds: float = 1.0) -> None:
        self.task = task_instance
        self.min_interval = min_interval_seconds
        self.last_update_time = 0.0

    def update(self, done: int, total: Optional[int]) -> None:
        current_time = time.monotonic()
        is_completed = total is not None and done >= total

        if is_completed or (current_time - self.last_update_time) >= self.min_interval:
            self.task.update_state(
                state="PROGRESS",
                meta={"done": done, "total": total},
            )
            self.last_update_time = current_time


def validate_file_reference(stored_name: str) -> Path:
    """Fail-fast validation of an uploaded file reference.
 
    SECURITY NOTE — accepted, deliberate TOCTOU window:
    This function returns a Path, not an open handle. Everything it checks
    (type, size, readability) is re-checked by the OS the moment a loader
    later does `open(path)` / `fitz.open(str(path))` / etc. — so nothing
    checked here is a binding guarantee by the time the file is actually
    read; a sufficiently capable attacker with write access to UPLOAD_DIR
    could swap the target between this call returning and the loader
    opening it. This is accepted for now because closing the gap requires
    every loader to read from a pinned file descriptor instead of a path
    (see validate_file_reference's previous fd-based revision), which is
    out of scope until the loaders are updated. Re-evaluate before treating
    UPLOAD_DIR as writable by anything less trusted than this service.
    """
    
    if not STORED_NAME_RE.fullmatch(stored_name):
        logger.warning("Rejected upload reference with invalid format: %r", stored_name)
        raise IngestionError("Invalid upload reference format.")
 
    raw_path = settings.UPLOAD_DIR / stored_name
 
    # resolve(strict=True) follows symlinks, so a symlinked stored_name
    # still gets caught by the relative_to() check below rather than
    # silently being read from outside UPLOAD_DIR.
    try:
        file_path = raw_path.resolve(strict=True)
    
    except FileNotFoundError as exc:
        raise IngestionError("The referenced upload no longer exists.") from exc
    
    except (PermissionError, OSError) as exc:
        logger.error("OS error resolving upload reference %r: %s", stored_name, exc)
        raise IngestionError("Cannot access the uploaded file.") from exc
    
    except RuntimeError as exc:
        logger.error("Symlink loop detected resolving %r: %s", stored_name, exc)
        raise IngestionError("Invalid upload reference (symlink loop).") from exc
 
    try:
        file_path.relative_to(_UPLOAD_DIR_RESOLVED)
    
    except ValueError:
        logger.critical(
                        "SECURITY: path traversal / symlink escape attempt. "
                        "stored_name=%r resolved=%s",
                        stored_name, file_path,
                        )
        raise IngestionError("Invalid upload reference.")
 
    try:
        st = file_path.stat()
    
    except OSError as exc:
        logger.error("Cannot stat upload %r: %s", stored_name, exc)
        raise IngestionError("Cannot access the uploaded file.") from exc
 
    if not stat_module.S_ISREG(st.st_mode):
        raise IngestionError("The provided reference is not a regular file.")
 
    if st.st_size == 0:
        raise IngestionError("The uploaded file is empty (0 bytes).")
 
    if st.st_size > _MAX_UPLOAD_FILE_BYTES:
        raise IngestionError(
            f"Uploaded file exceeds the maximum allowed size of "
            f"{_MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB."
        )
 
    if not os.access(file_path, os.R_OK):
        raise IngestionError("Cannot read the uploaded file (permission denied).")
 
    return file_path


@contextmanager
def guaranteed_file_cleanup(file_path: Path) -> Generator[Path, None, None]:
    """
    Context manager to guarantee local file cleanup.
    The 'finally' block ensures execution even during Hard Timeouts or severe exceptions.
    """
    try:
        yield file_path
    finally:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info("Successfully cleaned up file: %s", file_path.name)
            except OSError as cleanup_err:
                logger.critical(
                    "CRITICAL: OS prevented deletion of file %s: %s", 
                    file_path.name, 
                    cleanup_err
                )


@celery_app.task(bind=True, name="ingestion.ingest_document", max_retries=3)
def ingest_document(
    self: Any,
    stored_name: str,
    original_name: str,
    user_id: str,
    doc_id: Optional[str] = None,
) -> dict[str, Any]:
    """Main ingestion task handling the document processing."""
    
    # 1. Fail-Fast Validation Phase
    try:
        target_path = validate_file_reference(stored_name)
    except IngestionError as exc:
        logger.warning("Fail-Fast triggered for %s: %s", stored_name, exc)
        return {"status": "failed", "error": str(exc)}

    throttler = ProgressThrottler(task_instance=self, min_interval_seconds=1.0)

    # 2. Execution Phase with Guaranteed Cleanup
    try:
        with guaranteed_file_cleanup(target_path) as path:
            logger.info("Starting ingestion for %s", path.name)
            
            result = _pipeline.ingest_file(
                path=path,
                original_name=original_name,
                user_id=user_id,
                doc_id=doc_id,
                progress_cb=throttler.update,
            )

            return {
                "status": "done",
                "doc_id": result.doc_id,
                "chunks": result.chunks,
            }

    except IngestionError as exc:
        # Expected domain errors (e.g., corrupt PDF)
        logger.warning("Domain error during ingestion of %s: %s", stored_name, exc)
        return {"status": "failed", "error": str(exc)}

    except SoftTimeLimitExceeded:
        # Graceful handling of timeouts
        logger.error("Soft timeout exceeded for %s", stored_name)
        return {"status": "failed", "error": "Processing timed out. Please try a smaller file."}

    except Exception as exc:
        # Unexpected infrastructure errors (e.g., Redis down, DB connection lost)
        logger.exception(
            "Unexpected failure for %s (Attempt %d/%d)",
            stored_name,
            self.request.retries + 1,
            self.max_retries + 1,
        )

        if self.request.retries >= self.max_retries:
            raise

        # Exponential backoff retry logic (5s, 10s, 20s... max 60s)
        countdown = min(5 * (2 ** self.request.retries), 60)
        raise self.retry(exc=exc, countdown=countdown)


@celery_app.task(name="ingestion.purge_stale_local_uploads")
def purge_stale_local_uploads() -> None:
    """
    Fallback Periodic Task (Celery Beat):
    Sweeps the upload directory for orphaned files caused by catastrophic crashes (e.g., SIGKILL).
    """
    cutoff_time = time.time() - (24 * 3600)  # 24 Hours

    try:
        entries = list(os.scandir(settings.UPLOAD_DIR))
    except FileNotFoundError:
        logger.warning("Upload directory not found during purge.")
        return

    for entry in entries:
        if (
            entry.is_file()
            and STORED_NAME_RE.fullmatch(entry.name)
            and entry.stat().st_mtime < cutoff_time
        ):
            target_path = Path(entry.path)
            try:
                target_path.unlink(missing_ok=True)
                logger.info("Purged orphaned file: %s", target_path.name)
            except OSError as err:
                logger.error("Failed to purge file %s: %s", target_path.name, err)