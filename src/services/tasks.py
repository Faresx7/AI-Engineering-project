"""Celery worker task for document ingestion.

Optimized for a single-server Docker Compose environment using shared volumes.
Implements Fail-Fast principles, Clean Code, and a two-layer resource cleanup guarantee.
"""

from __future__ import annotations

from typing import Optional, Any
import stat as stat_module
from pathlib import Path
import logging
import random
import time
import uuid
import os


from celery.exceptions import SoftTimeLimitExceeded
import redis

from src.rag.ingestion.ingestion_pipeline import IngestionPipeline
from src.core.validation import STORED_NAME_RE, IngestionError
from src.core.celery_app import celery_app
from src.core.config import settings

from src.rag.constants import (_INGEST_HARD_TIME_LIMIT,
                                _INGEST_SOFT_TIME_LIMIT,
                                _LOCK_TTL_SECONDS,
                                _NON_RETRYABLE_CODE_ERRORS,
                                )


############################################################################

logger = logging.getLogger(__name__)

_pipeline = IngestionPipeline()

_UPLOAD_DIR_RESOLVED = settings.UPLOAD_DIR.resolve(strict=True)
_MAX_UPLOAD_FILE_BYTES = settings.MAX_UPLOAD_MB

_redis_client: Optional["redis.Redis"] = None

################################################################################

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


def _validate_file_reference(stored_name: str) -> Path:
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
        logger.warning(
            "INGEST_FILE_REJECTED_FORMAT | stored_name=%r action=rejected_upload",
            stored_name,
        )
        
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
        logger.error(
            "INGEST_FILE_RESOLVE_ERROR | stored_name=%r err_type=%s err=%r action=rejected_upload",
            stored_name,
            type(exc).__name__,
            str(exc),
        )
        
        raise IngestionError("Cannot access the uploaded file.") from exc
    
    except RuntimeError as exc:
        logger.error(
            "INGEST_FILE_SYMLINK_LOOP | stored_name=%r err_type=%s err=%r action=rejected_upload",
            stored_name,
            type(exc).__name__,
            str(exc),
        )
        
        raise IngestionError("Invalid upload reference (symlink loop).") from exc
 
    try:
        file_path.relative_to(_UPLOAD_DIR_RESOLVED)
    
    except ValueError:
        logger.critical(
            "INGEST_SECURITY_PATH_TRAVERSAL | stored_name=%r resolved_path=%r action=rejected_upload",
            stored_name,
            str(file_path),
        )
        raise IngestionError("Invalid upload reference.")
 
    try:
        st = file_path.stat()
    
    except OSError as exc:
        logger.error(
            "INGEST_FILE_STAT_ERROR | stored_name=%r err_type=%s err=%r action=rejected_upload",
            stored_name,
            type(exc).__name__,
            str(exc),
        )
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


def _remove_file(verified_path: Path) -> None:
    """
    Safely remove a validated uploaded file from the configured upload directory.

    This helper enforces that the target path remains within the application upload
    root before attempting deletion, preventing path traversal attacks and accidental
    deletion outside the intended storage area. It also tolerates transient filesystem
    races by using missing_ok=True and logs permission or lock-related failures without
    raising exceptions, allowing deferred cleanup by a background purge mechanism.

    Args:
        verified_path (Path): The validated upload path to remove. It must point
            inside the configured upload directory; otherwise the deletion is
            rejected as a security measure.

    Returns:
        None: This function does not return a value. It performs cleanup and logs
        any permission, lock, or filesystem issues without raising them.
    """
    try:
        try:
            verified_path.relative_to(_UPLOAD_DIR_RESOLVED)
        
        except ValueError:
            logger.critical(
                "INGEST_SECURITY_DELETE_ESCAPE | target_path=%r action=aborted_deletion",
                str(verified_path),
            )
            return
        
        # [FIX] avoid race condition by using "missing_ok=True"
        verified_path.unlink(missing_ok=True)
        
        logger.debug(
            "INGEST_FILE_CLEANED_UP | stored_name=%r action=purged_local_file",
            verified_path.name,
        )
    except OSError as cleanup_err:
        logger.warning(
            "INGEST_FILE_CLEANUP_FAILED | stored_name=%r err_type=%s err=%r "
            "action=deferred_to_periodic_purge",
            verified_path.name,
            type(cleanup_err).__name__,
            str(cleanup_err),
        )



################################################################################
# Helpers
################################################################################
def _get_redis() -> "redis.Redis":
    """Lazy Redis client for the document lock, the Celery lock URL."""
    global _redis_client
    
    if _redis_client is None:
        url = getattr(settings, "INGEST_LOCK_REDIS_URL", None) or settings.CELERY_BROKER_URL
        _redis_client = redis.Redis.from_url(url, socket_timeout=5, socket_connect_timeout=5)
    return _redis_client


def _lock_key(user_id: str, doc_id: str) -> str:
    """Build the Redis lock key for a document ingestion attempt."""
    return f"ingest:lock:{user_id}:{doc_id}"


def _derive_doc_id(user_id: str, original_name: str) -> str:
    """Must stay identical to the fallback derivation in IngestionPipeline.ingest_file.

    Needed here because the lock has to be acquired BEFORE calling the
    pipeline, so the lock key must be known in advance.
    """
    return f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{user_id}_{original_name}')}"


def _jittered_backoff(retries: int, base: float = 5.0, cap: float = 60.0) -> float:
    """Exponential backoff + random jitter, used for every retry countdown.

    The jitter spreads out retried tasks so they don't all resume at the
    same instant and cause a CPU/IO spike that could blow OTHER tasks'
    soft time limits.

    Args:
        retries: Number of retry attempts already consumed; higher values
            increase the exponential wait time before the next retry.
        base: Base backoff interval in seconds used for the first exponential
            delay step.
        cap: Maximum backoff ceiling in seconds. The exponential value is
            capped at this value before jitter is added.

    Returns:
        A delay in seconds to wait before retrying, combining exponential
        growth with a small random jitter to avoid synchronized retries.
    """
    exponential = min(base * (2 ** retries), cap)
    return round(exponential + random.uniform(0.5, 3.0), 2)

################################################################################

@celery_app.task(bind=True,
                 name="ingestion.ingest_document",
                 max_retries=3,
                 acks_late=True,
                 reject_on_worker_lost=True,
                 soft_time_limit=_INGEST_SOFT_TIME_LIMIT,
                 time_limit=_INGEST_HARD_TIME_LIMIT,
                )
def ingest_document(self: Any,
                    stored_name: str,
                    original_name: str,
                    user_id: str,
                    doc_id: Optional[str] = None,
                    ) -> dict[str, Any]:
    """Main ingestion task: validate → lock → process → cleanup, with safe retries
    Ingest a stored document safely and idempotently.

    This task validates the uploaded file reference, acquires a per-document
    lock to prevent concurrent ingestion races, runs the ingestion pipeline,
    and cleans up temporary files when the task succeeds or fails. It is built
    to be retry-safe so repeated execution does not duplicate documents.

    Idempotency: a deterministic `doc_id` is derived from (user_id,
    original_name) when none is supplied, and the pipeline performs an
    atomic generation swap in Qdrant (see pipeline.py). A retry of this
    exact task — whether triggered by `self.retry()` below or by broker
    redelivery after a worker crash — is therefore safe to re-run: it will
    not create duplicate documents, at worst it repeats already-done work.

    Cross-worker lock: acquired on (user_id, doc_id) before any risky work
    starts, so two workers can never race on the same document's Qdrant
    generation swap (see the helpers above for why that race matters).

    A hard worker crash (SIGKILL/OOM) during ingestion cannot be caught
    here — `acks_late` + `reject_on_worker_lost` ensure the message is
    redelivered in that case, and any generation partially written to
    Qdrant before the crash is cleaned up later by the scheduled
    `purge_orphaned_generations()` sweep, not by this task. The lock's TTL
    similarly self-expires if the crash happened while it was held.

    Inputs:
        stored_name (str): The uploaded file reference stored on disk or in the
            temporary upload area that should be validated and ingested.
        original_name (str): The user's original filename, used to derive a
            stable `doc_id` when one is not supplied explicitly.
        user_id (str): The owning user identifier that scopes the document and
            the ingestion lock.
        doc_id (Optional[str]): Optional explicit document identifier. When
            omitted, a deterministic ID is derived from `(user_id,
            original_name)` so retries and duplicate deliveries stay idempotent.

    Outputs:
        dict[str, Any]: The ingestion result payload returned by the pipeline,
        typically including the document identifier, status, and any generated
        metadata or indexing details produced during the ingest step. If the
        pipeline or task fails in a retryable way, the function raises an
        exception instead of returning a final result.

    Catches:
        - `IngestionError` for validation failures and domain-level ingest
          errors that should not be retried.
        - `SoftTimeLimitExceeded` for task soft time-limit aborts.
        - Exceptions matching `_NON_RETRYABLE_CODE_ERRORS` as internal code
          defects that should fail fast.
        - Generic `Exception` for transient backend/infrastructure failures that
          are retried until the retry budget is exhausted.
        - Redis/lock backend exceptions around acquiring and releasing the
          per-document lock; these are logged and converted to retry or fail-fast
          behavior depending on the retry budget.

    """
    
    
    # 1. Fail-fast validation phase
    try:
        target_path: Path = _validate_file_reference(stored_name)
    
    except IngestionError as exc:
        logger.warning("INGEST_VALIDATION_REJECTED | stored_name=%s err=%s", stored_name, exc)
        raise  # nothing was resolved yet — there is nothing to clean up


    target_doc_id = doc_id or _derive_doc_id(user_id, original_name)
    is_last_attempt = self.request.retries >= self.max_retries

    # 2. Cross-worker lock — prevents two workers racing on one doc
    key = _lock_key(user_id, target_doc_id)

    try:
        got_lock = bool(_get_redis().set(key, "1", nx=True, ex=_LOCK_TTL_SECONDS))

    except Exception as exc:
        if is_last_attempt:
            logger.error(
                "INGEST_LOCK_BACKEND_UNREACHABLE | task_id=%s user_id=%s doc_id=%s "
                "stored_name=%s attempt=%d/%d err_type=%s err=%r action=purged_local_file",
                self.request.id,
                user_id,
                target_doc_id,
                stored_name,
                self.request.retries + 1,
                self.max_retries + 1,
                type(exc).__name__,
                str(exc),
                exc_info=True,
            )

            _remove_file(target_path)
            
            raise IngestionError("Lock backend unreachable; retry budget aborted.") from exc
        
        countdown = _jittered_backoff(self.request.retries)
        
        logger.warning(
            "INGEST_LOCK_BACKEND_UNREACHABLE | RECONNECTING | task_id=%s user_id=%s doc_id=%s "
            "stored_name=%s attempt=%d/%d err_type=%s err=%r",
            self.request.id,
            user_id,
            target_doc_id,
            stored_name,
            self.request.retries + 1,
            self.max_retries + 1,
            type(exc).__name__,
            str(exc),
            )
        
        raise self.retry(exc=exc, countdown=countdown)

    if not got_lock:
        logger.warning("INGEST_LOCK_TAKEN | task_id=%s user_id=%s doc_id=%s "
                       "stored_name=%s — rejecting concurrent upload.",
                        self.request.id,
                        user_id,
                        target_doc_id,
                        stored_name,
                        )
        
        raise IngestionError("This document is currently being processed. Please wait a moment before re-uploading.")



    # 3. Execution phase (lock held)
    throttler = ProgressThrottler(task_instance=self, min_interval_seconds=5.0)

    try:
        logger.info("INGEST_STARTED | stored_name=%s attempt=%d/%d"
                    "task_id=%s user_id=%s doc_id=%s",
                    stored_name,
                    self.request.retries + 1,
                    self.max_retries + 1,
                    self.request.id,
                    user_id,
                    target_doc_id,
                    )

        result = _pipeline.ingest_file(
            path=target_path,
            original_name=original_name,
            user_id=user_id,
            doc_id=target_doc_id,
            progress_cb=throttler.update,
        )

    except IngestionError as exc:
        # Non-retryable domain error (corrupt file, bad magic bytes, budget exceeded).
        # Clean up, then surface the original error untouched.
        logger.warning(
            "INGEST_DOMAIN_ERROR | task_id=%s user_id=%s doc_id=%s stored_name=%s "
            "attempt=%d/%d err_type=%s err=%r action=purged_local_file",
            self.request.id,
            user_id,
            target_doc_id,
            stored_name,
            self.request.retries + 1,
            self.max_retries + 1,
            type(exc).__name__,
            str(exc),
        )
        _remove_file(target_path)

        raise

    except SoftTimeLimitExceeded:
        # Graceful timeout. The pipeline's own exception handling has
        # already rolled back any partially-written generation in Qdrant
        # (see pipeline.py) before this propagates up to us.
        logger.error(
            "INGEST_SOFT_TIMEOUT | task_id=%s user_id=%s doc_id=%s stored_name=%s "
            "attempt=%d/%d err_type=SoftTimeLimitExceeded action=purged_local_file",
            self.request.id,
            user_id,
            target_doc_id,
            stored_name,
            self.request.retries + 1,
            self.max_retries + 1,
        )

        _remove_file(target_path)

        raise

    except _NON_RETRYABLE_CODE_ERRORS as exc:
        # Internal bug: retrying will not fix a code defect. Fail fast.
        logger.exception(
            "INGEST_CODE_BUG | task_id=%s user_id=%s doc_id=%s stored_name=%s "
            "attempt=%d/%d err_type=%s err=%r action=purged_local_file",
            self.request.id,
            user_id,
            target_doc_id,
            stored_name,
            self.request.retries + 1,
            self.max_retries + 1,
            type(exc).__name__,
            str(exc),
        )
        
        _remove_file(target_path)
        
        raise RuntimeError("Internal document processing error; task aborted.") from exc

    except Exception as exc:
        # Transient infrastructure failure (e.g. Qdrant / Gemini connection
        # or timeout issue).
        logger.exception(
            "INGEST_TRANSIENT_FAILURE | task_id=%s user_id=%s doc_id=%s stored_name=%s "
            "attempt=%d/%d err_type=%s err=%r",
            self.request.id,
            user_id,
            target_doc_id,
            stored_name,
            self.request.retries + 1,
            self.max_retries + 1,
            type(exc).__name__,
            str(exc),
        )

        if is_last_attempt:
            logger.critical(
                "INGEST_TRANSIENT_ABORTED | task_id=%s user_id=%s doc_id=%s stored_name=%s "
                "attempt=%d/%d err_type=%s err=%r action=purged_local_file",
                self.request.id,
                user_id,
                target_doc_id,
                stored_name,
                self.request.retries + 1,
                self.max_retries + 1,
                type(exc).__name__,
                str(exc),
            )
            
            _remove_file(target_path)
            
            raise

        # Deliberately keep target_path on disk — the scheduled retry needs
        # to read it.
        
        countdown = _jittered_backoff(self.request.retries)
        
        logger.warning(
            "INGEST_TRANSIENT_RETRYING | task_id=%s user_id=%s doc_id=%s stored_name=%s "
            "attempt=%d/%d retrying_in=%.2fs err_type=%s err=%r action=kept_local_file",
            self.request.id,
            user_id,
            target_doc_id,
            stored_name,
            self.request.retries + 1,
            self.max_retries + 1,
            countdown,
            type(exc).__name__,
            str(exc),
        )
        
        raise self.retry(exc=exc, countdown=countdown)

    else:
        # Ingestion fully succeeded and is already durably committed in
        # Qdrant. This block is structurally outside every `except` clause
        # above, so nothing that happens here — including a failed
        # cleanup — can ever be reinterpreted as one of those failure
        # branches.
        _remove_file(target_path)

        logger.info(
            "INGEST_SUCCEEDED | task_id=%s user_id=%s doc_id=%s stored_name=%s "
            "generation=%s chunks=%d action=purged_local_file",
            self.request.id,
            user_id,
            result.doc_id,
            stored_name,
            result.generation,
            result.chunks,
        )

        return {
            "status": "done",
            "doc_id": result.doc_id,
            "generation": result.generation,
            "chunks": result.chunks,
        }

    finally:
        # Always release the lock so the next attempt doesn't wait out the TTL.
        try:
            _get_redis().delete(key)
        
        except Exception as exc:
            logger.warning(
                "INGEST_LOCK_RELEASE_FAILED | task_id=%s user_id=%s doc_id=%s stored_name=%s "
                "err_type=%s err=%r action=deferred_to_ttl_expiry",
                self.request.id,
                user_id,
                target_doc_id,
                stored_name,
                type(exc).__name__,
                str(exc),
            )



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