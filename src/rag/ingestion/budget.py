"""Row budget tracking, batching, and retry utilities for RAG ingestion."""
from __future__ import annotations

import logging
import time
from itertools import islice
from threading import Lock
from typing import Any, Callable, Iterable, Iterator, Optional

from src.core.validation import IngestionError

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Retry helper — bounded attempts, exponential backoff, fails fast after that
# ══════════════════════════════════════════════════════════════════════════════

def _retry(
    fn: Callable[[], Any],
    *,
    attempts: int,
    base_delay: float,
    what: str,
) -> Any:
    """Calls fn(), retrying on exception up to `attempts` times with backoff.

    ``IngestionError`` is **never** retried — those represent deterministic
    failures (bad input, count mismatches) where retrying cannot help.

    Re-raises the last exception once attempts are exhausted — this never
    retries forever, so a persistently broken dependency fails the ingestion
    quickly instead of hanging the worker.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except IngestionError:
            # Deterministic / validation errors — retrying can never help.
            raise
        except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised below
            last_exc = exc
            if attempt >= attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs.",
                what, attempt, attempts, exc, delay,
            )
            time.sleep(delay)

    # last_exc is guaranteed non-None here because we only reach this point
    # when every attempt raised an Exception (the for-loop ran at least once
    # with attempts >= 1). We use an explicit guard instead of ``assert`` so
    # the safety net survives ``python -O``.
    if last_exc is None:  # pragma: no cover — logically unreachable
        raise RuntimeError(f"{what}: all {attempts} attempts failed but no exception was captured")
    raise last_exc


def _batched(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Groups an iterable into lists of at most `size` items, pulled lazily.

    Only one batch is ever materialized in memory at a time, regardless of
    how many items the source iterable produces in total.
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


# ══════════════════════════════════════════════════════════════════════════════
# Row budget — soft stop, never causes rollback
# ══════════════════════════════════════════════════════════════════════════════

class _RowBudget:
    """Counts data rows across a whole file and signals when the limit is hit.

    ``spend()`` returns ``True`` when the row is accepted (budget remaining)
    and ``False`` when the budget is exhausted. Callers must check the return
    value and stop producing new rows — but no exception is raised, so an
    exhausted budget never triggers a full rollback of already-stored data.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._left = limit
        self._lock = Lock()
        self._exhaustion_logged = False

    def spend(self) -> bool:
        """Deduct one row from the budget.

        Returns ``True`` if the row was accepted (budget still available),
        ``False`` if the budget is exhausted. The first exhaustion is logged
        at WARNING level.
        """
        with self._lock:
            if self._left <= 0:
                if not self._exhaustion_logged:
                    logger.warning(
                        "Row budget of %d exhausted. Remaining rows will be "
                        "truncated (already-stored data is preserved).",
                        self._limit,
                    )
                    self._exhaustion_logged = True
                return False
            self._left -= 1
            return True
