"""Batching and bounded retry utilities for the RAG ingestion pipeline.

What This File Does:
    Provides a memory-safe lazy batcher (`_batched`) so large document streams
    are processed in fixed-size chunks without loading everything into RAM, and
    a bounded exponential-backoff retry helper (`_retry`) for transient network
    or external API failures.

Workflow:
    1. ``_batched`` pulls at most ``size`` items at a time from any iterable
       using ``itertools.islice`` and yields them batch by batch.
    2. ``_retry`` executes a callable up to ``attempts`` times, immediately
       re-raising deterministic validation/code errors or Celery soft timeouts,
       and sleeping with jittered exponential backoff between transient failures.

Raises:
    ValueError:
        - In ``_retry`` if ``attempts < 1`` or ``base_delay < 0``.
        - In ``_batched`` if ``size < 1``.
    IngestionError / SoftTimeLimitExceeded:
        - Immediately re-raised by ``_retry`` without retrying.
    Exception:
        - Re-raises the final transient exception once ``attempts`` are exhausted.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional
from itertools import islice
import logging
import random
import time


from src.rag.constants import _NON_RETRYABLE_CODE_ERRORS
from src.core.validation import IngestionError

try:
    from celery.exceptions import SoftTimeLimitExceeded

except ImportError:  # pragma: no cover
    SoftTimeLimitExceeded = ()  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


# ##############################################################################
# Retry helper — bounded attempts, jittered exponential backoff, fail-fast
# ##############################################################################

def _retry(fn: Callable[[], Any],
           *,
           attempts: int,
           base_delay: float,
           what: str,
          ) -> Any:
    """Executes ``fn()`` with bounded retries and jittered exponential backoff.

    Why it matters:
        Recovers automatically from transient network blips with Gemini or Qdrant
        while immediately aborting on deterministic bugs, validation errors, or
        Celery soft timeouts so worker rollbacks are never delayed.

    Args:
        fn: Zero-argument callable to execute.
        attempts: Maximum number of execution attempts (must be >= 1).
        base_delay: Base sleep duration in seconds before exponential scaling.
        what: Human-readable operation label used in warning logs.

    Returns:
        Any: The return value of ``fn()`` on the first successful attempt.

    Errors:
        ValueError: If ``attempts < 1`` or ``base_delay < 0``.
        IngestionError: Immediately propagated without retrying.
        SoftTimeLimitExceeded: Immediately propagated so Celery cleanup runs on time.
        Exception: Re-raises the last exception after exhausting all attempts.
    """
    # Validate retry configuration before executing anything
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}.")

    if base_delay < 0:
        raise ValueError(f"base_delay must be non-negative, got {base_delay}.")

    last_exc: Optional[BaseException] = None

    # Attempt execution up to the configured limit with exponential backoff + jitter
    for attempt in range(1, attempts + 1):
        try:
            return fn()

        except ((IngestionError,) + ((SoftTimeLimitExceeded,)
               if isinstance(SoftTimeLimitExceeded, type) else ())
            + _NON_RETRYABLE_CODE_ERRORS):
            # Never retry deterministic validation/code errors or Celery soft timeouts
            raise

        except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised below
            last_exc = exc
            if attempt >= attempts:
                break

            # Add +-20% jitter so concurrent workers do not retry in lockstep
            raw_delay = base_delay * (2 ** (attempt - 1))
            delay = raw_delay * random.uniform(0.8, 1.2)
            logger.warning("%s failed (attempt %d/%d): %s. Retrying in %.2fs.",
                           what,
                           attempt,
                           attempts,
                           exc,
                           delay,
                          )
            time.sleep(delay)

    # Re-raise the final captured exception once all retry attempts are exhausted
    if last_exc is None:  # pragma: no cover - logically unreachable
        raise RuntimeError(f"{what}: all {attempts} attempts failed without capturing an exception.")

    raise last_exc


# ##############################################################################
# Lazy batching helper — O(size) memory regardless of stream length
# ##############################################################################

def _batched(iterable: Iterable[Any],
             size: int,
            ) -> Iterator[list[Any]]:
    """Lazily groups an iterable into lists of at most ``size`` items.

    Why it matters:
        Ensures only one bounded batch exists in memory at any moment regardless
        of how many pages or chunks a document produces.

    Args:
        iterable: Source sequence or generator to group into batches.
        size: Maximum number of items per yielded batch (must be >= 1).

    Returns:
        Iterator[list[Any]]: Yields non-empty lists of up to ``size`` items.

    Errors:
        ValueError: If ``size < 1``.
    """
    # Reject non-positive batch sizes so streams are never silently dropped
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}.")

    it = iter(iterable)

    # Pull and yield one bounded slice at a time until the iterator is exhausted
    while True:
        batch = list(islice(it, size))
        if not batch:
            return

        yield batch