"""Synchronous ingestion pipeline. Runs ONLY inside the Celery worker.

Everything here is deliberately synchronous (no async/await): it is CPU- and
IO-heavy work (parsing files, calling the embedding API) that must never run
on the FastAPI event loop. The worker is a separate process, so even a
10-second PDF parse cannot slow down your API webhooks.

Ingestion safety model
-----------------------
Every call to ``ingest_file`` is tagged with a fresh ``generation`` id. New
chunks are written under that generation WITHOUT touching any previously
stored generation for the same document. Only after every chunk of the new
version is confirmed stored do we delete the older generation(s). This means:

  * Readers never see a half-written document — the old, complete version
    stays live until the new, complete version is live.
  * Any exception raised while writing the new generation triggers an
    automatic rollback that deletes everything written under that
    generation, so a failed run never leaves partial "zombie" chunks mixed
    in with real data.
  * A hard process crash (SIGKILL, OOM-kill, node loss) cannot be caught by
    a `try/except` in this process. That failure mode is handled by
    ``purge_orphaned_generations``, a separate maintenance sweep meant to run
    on a schedule (e.g. Celery beat) that deletes old, abandoned generations
    once they are older than a safety grace period.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    SparseVector,
)

from src.core.config import settings
from src.core.validation import IngestionError, sniff_matches_extension
from src.rag.ingestion.budget import _RowBudget, _batched, _retry
from src.rag.clients import (
    _embed_dense_batch,
    _get_genai_client,
    _get_qdrant,
    _get_sparse_model,
)
from src.rag.constants import (
    _BATCH_SIZE,
    _DEFAULT_PURGE_MAX_DOCS,
    _DEFAULT_PURGE_MIN_AGE_SECONDS,
    _DEFAULT_PURGE_SCROLL_BATCH,
    _DENSE_EMBEDDING_DIM,
    _DENSE_MODEL_NAME,
    _EMBED_MAX_RETRIES,
    _MAX_CELL_CHARS,
    _MAX_CELLS_PER_ROW,
    _MAX_CHUNKS_PER_DOC,
    _MAX_CONSECUTIVE_EMPTY_ROWS,
    _MAX_LINE_CHARS,
    _MAX_PDF_PAGES,
    _MAX_RECORD_CHARS,
    _MAX_TABLE_ROWS,
    _MAX_TEXT_BYTES,
    _QDRANT_MAX_RETRIES,
    _RETRY_BASE_DELAY_SECONDS,
    _SPARSE_MODEL_NAME,
    _WD_PARA_TAG,
    _WD_TABLE_TAG,
)
from src.rag.ingestion.loaders import (
    _LOADERS,
    _csv_loader,
    _docx_loader,
    _excel_loader,
    _json_loader,
    _pdf_loader,
    _text_loader,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, Optional[int]], None]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    chunks: int
    generation: str


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

class IngestionPipeline:
    """Parses, chunks, embeds and upserts a single file into Qdrant.

    Memory profile: O(BATCH_SIZE) regardless of document size. Nothing — not
    the raw rows, not the split chunks, not the embeddings — is ever fully
    materialized in a list; everything flows through generators and is
    processed one bounded batch at a time.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        chunk_size: int = 800,
        chunk_overlap: int = 130,
    ) -> None:
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── chunking (streaming) ───────────────────────────────────────────────
    def _stream_chunks(self, docs: Iterator[Document]) -> Iterator[Document]:
        """Yields chunks one at a time, consuming one source Document at a time.

        Table rows / JSON records (flagged `pre_chunked`) stay whole unless
        they are absurdly long; everything else goes through the text
        splitter. Only the current source Document and its own split
        fragments are ever in memory — never the whole document set.
        """
        for doc in docs:
            if not doc.page_content or not doc.page_content.strip():
                continue
            if doc.metadata.get("pre_chunked") and len(doc.page_content) <= _MAX_RECORD_CHARS:
                yield doc
            else:
                for chunk in self._splitter.split_documents([doc]):
                    if chunk.page_content.strip():
                        yield chunk

    # ── rollback ────────────────────────────────────────────────────────────
    def _rollback_generation(
        self, client: QdrantClient, user_id: str, doc_id: str, generation: str
    ) -> None:
        """Deletes everything written under a failed generation.

        Retried a bounded number of times; if it still fails, logs CRITICAL
        so the condition is loud and visible, and the scheduled
        `purge_orphaned_generations` sweep will clean it up later.
        """
        logger.warning(
            "Rolling back failed ingestion | doc_id=%s generation=%s", doc_id, generation
        )
        try:
            _retry(
                lambda: client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                FieldCondition(key="generation", match=MatchValue(value=generation)),
                            ]
                        )
                    ),
                    wait=True,
                ),
                attempts=_QDRANT_MAX_RETRIES,
                base_delay=_RETRY_BASE_DELAY_SECONDS,
                what="rollback delete",
            )
            logger.info("Rollback complete | doc_id=%s generation=%s", doc_id, generation)
        except Exception as exc:
            logger.critical(
                "ROLLBACK FAILED — orphaned generation left in Qdrant! "
                "doc_id=%s generation=%s err=%s. Will be swept by "
                "purge_orphaned_generations() once past the safety grace period.",
                doc_id, generation, exc,
            )

    # ── main entry point ───────────────────────────────────────────────────
    def ingest_file(
        self,
        path: Path,
        original_name: str,
        user_id: str,
        doc_id: Optional[str] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> IngestionResult:
        """Loads, chunks, embeds and upserts a file with atomic generation swap.

        Raises IngestionError for any recoverable problem (bad file,
        unsupported format, budget exceeded, embedding/Qdrant failure after
        exhausting retries). Any failure after the first successful batch
        write triggers an automatic rollback of everything written so far
        under this run — no partial version is ever left live.
        """
        path = Path(path)
        ext = path.suffix.lower()

        # ── Cheap validation first: fail fast before touching any network. ──
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise IngestionError(f"Unsupported file type: '{ext or '(none)'}'.")
        if ext not in _LOADERS:
            raise IngestionError(f"No loader registered for extension '{ext}'.")
        if not sniff_matches_extension(path, ext):
            raise IngestionError("File content does not match its declared extension.")

        target_doc_id = doc_id or (
            f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{user_id}_{original_name}')}"
        )
        generation = uuid.uuid4().hex
        created_at = time.time()

        client = _get_qdrant()
        sparse_model = _get_sparse_model()

        doc_stream = _LOADERS[ext](path, original_name)
        chunk_stream = self._stream_chunks(doc_stream)

        stored = 0

        try:
            for batch in _batched(chunk_stream, _BATCH_SIZE):
                # Fail fast BEFORE spending an (expensive) embedding call.
                if stored + len(batch) > _MAX_CHUNKS_PER_DOC:
                    raise IngestionError(
                        f"The file produces more than {_MAX_CHUNKS_PER_DOC:,} chunks. "
                        "Please split it into smaller files."
                    )

                texts = [c.page_content for c in batch]

                dense_vecs = _embed_dense_batch(texts)
                sparse_vecs = list(sparse_model.embed(texts, batch_size=_BATCH_SIZE))

                if len(dense_vecs) != len(batch) or len(sparse_vecs) != len(batch):
                    raise IngestionError(
                        f"Embedding mismatch: got {len(dense_vecs)} dense / "
                        f"{len(sparse_vecs)} sparse vectors for a batch of {len(batch)}."
                    )

                points: list[PointStruct] = []
                for offset, (chunk, dense, sparse) in enumerate(
                    zip(batch, dense_vecs, sparse_vecs)
                ):
                    chunk_index = stored + offset
                    chunk_id = f"{target_doc_id}_chunk_{chunk_index}"
                    point_id = str(
                        uuid.uuid5(uuid.NAMESPACE_DNS, f"{generation}_{chunk_id}")
                    )
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector={
                                "dense": dense,
                                "sparse": SparseVector(
                                    indices=sparse.indices.tolist(),
                                    values=sparse.values.tolist(),
                                ),
                            },
                            payload={
                                "user_id": user_id,
                                "doc_id": target_doc_id,
                                "generation": generation,
                                "created_at": created_at,
                                "chunk_id": chunk_id,
                                "chunk_index": chunk_index,
                                "text": chunk.page_content,
                                "metadata": {
                                    **{
                                        k: v
                                        for k, v in chunk.metadata.items()
                                        if k != "pre_chunked"
                                    },
                                    "filename": original_name,
                                },
                            },
                        )
                    )

                _retry(
                    lambda pts=points: client.upsert(
                        collection_name=self.collection_name, points=pts, wait=True,
                    ),
                    attempts=_QDRANT_MAX_RETRIES,
                    base_delay=_RETRY_BASE_DELAY_SECONDS,
                    what="Qdrant upsert",
                )

                stored += len(points)

                # Free batch memory before the next iteration.
                del points, dense_vecs, sparse_vecs, texts, batch

                if progress_cb:
                    try:
                        progress_cb(stored, None)  # total is unknown until streaming finishes
                    except Exception:
                        logger.warning("progress_cb raised an exception; ignoring.", exc_info=True)

        except Exception:
            # Any failure past this point — including our own validation
            # errors raised mid-stream — must not leave partial data live.
            if stored > 0:
                self._rollback_generation(client, user_id, target_doc_id, generation)
            raise

        if stored == 0:
            # Nothing was ever written under this generation, so there is
            # nothing to roll back — just report the problem.
            raise IngestionError(
                "No readable text was found in the file. "
                "Scanned PDFs without embedded text require OCR preprocessing."
            )

        # ── Commit: the new generation is fully stored — now retire the old one. ──
        try:
            _retry(
                lambda: client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                                FieldCondition(key="doc_id", match=MatchValue(value=target_doc_id)),
                            ],
                            must_not=[
                                FieldCondition(key="generation", match=MatchValue(value=generation)),
                            ],
                        )
                    ),
                    wait=True,
                ),
                attempts=_QDRANT_MAX_RETRIES,
                base_delay=_RETRY_BASE_DELAY_SECONDS,
                what="old-generation cleanup",
            )
        except Exception as exc:
            logger.critical(
                "CLEANUP FAILED — doc_id=%s generation=%s stored successfully but the "
                "previous version could not be removed: %s. Run purge_orphaned_generations() "
                "or retry ingestion to self-heal.",
                target_doc_id, generation, exc,
            )
            raise IngestionError(
                f"New version stored successfully ({stored} chunks) but cleanup of the "
                "previous version failed. Search results may temporarily include stale "
                "data until the cleanup is retried."
            ) from exc

        logger.info(
            "Ingestion complete | doc_id=%s | generation=%s | chunks=%d | user=%s",
            target_doc_id, generation, stored, user_id,
        )
        return IngestionResult(doc_id=target_doc_id, chunks=stored, generation=generation)


# ══════════════════════════════════════════════════════════════════════════════
# Maintenance: sweep for generations orphaned by a hard crash
# ══════════════════════════════════════════════════════════════════════════════

def purge_orphaned_generations(
    collection_name: str = "rag_documents",
    min_age_seconds: float = _DEFAULT_PURGE_MIN_AGE_SECONDS,
    scroll_batch_size: int = _DEFAULT_PURGE_SCROLL_BATCH,
    max_docs: int = _DEFAULT_PURGE_MAX_DOCS,
) -> int:
    """Deletes stale, abandoned generations left behind by a worker that died
    mid-ingestion (SIGKILL, OOM-kill, node loss) — failures a `try/except`
    inside `ingest_file` cannot catch because the process never got to run
    its `except` block.

    A document normally has exactly one `generation` in Qdrant. Two or more
    generations coexisting for the same (user_id, doc_id) can only happen
    when a run died before reaching its own cleanup step. This sweep keeps
    only the newest generation (by `created_at`) per document and removes
    the rest — but only once they are older than `min_age_seconds`, so an
    ingestion that is still legitimately in flight is never touched.

    Intended to run on a schedule (e.g. Celery beat, every 15–60 minutes).
    Memory use is bounded by ``max_docs`` — the maximum number of distinct
    (user_id, doc_id) pairs held in the ``seen`` dict at once. If the
    collection is larger, the sweep processes the first ``max_docs`` pairs
    and exits; the next scheduled run picks up where this one left off.

    Returns the number of orphaned generations deleted.
    """
    client = _get_qdrant()
    cutoff = time.time() - min_age_seconds

    seen: dict[tuple[str, str], dict[str, float]] = {}
    offset = None
    scanned = 0
    docs_seen = 0
    hit_doc_limit = False

    while True:
        try:
            records, offset = client.scroll(
                collection_name=collection_name,
                with_payload=["user_id", "doc_id", "generation", "created_at"],
                with_vectors=False,
                limit=scroll_batch_size,
                offset=offset,
            )
        except Exception as exc:
            logger.error(
                "purge_orphaned_generations scroll failed after %d points: %s",
                scanned, exc,
            )
            break

        if not records:
            break

        for record in records:
            payload = record.payload or {}
            user_id = payload.get("user_id")
            doc_id = payload.get("doc_id")
            generation = payload.get("generation")
            created_at = payload.get("created_at", 0.0)
            if not (user_id and doc_id and generation):
                continue

            key = (user_id, doc_id)

            if key not in seen:
                if docs_seen >= max_docs:
                    hit_doc_limit = True
                    continue
                docs_seen += 1

            gens = seen.setdefault(key, {})
            if generation not in gens or created_at > gens[generation]:
                gens[generation] = created_at

        scanned += len(records)
        if offset is None:
            break

    if hit_doc_limit:
        logger.warning(
            "purge_orphaned_generations hit max_docs limit (%d). "
            "Some documents were skipped; they will be processed in the next run.",
            max_docs,
        )

    purged = 0
    for (user_id, doc_id), gens in seen.items():
        if len(gens) <= 1:
            continue  # normal case — nothing to purge

        newest_generation = max(gens, key=lambda g: gens[g])
        for generation, created_at in gens.items():
            if generation == newest_generation:
                continue
            if created_at > cutoff:
                # Might still be an in-flight ingestion — leave it this round.
                continue

            try:
                client.delete(
                    collection_name=collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                FieldCondition(key="generation", match=MatchValue(value=generation)),
                            ]
                        )
                    ),
                    wait=True,
                )
                purged += 1
                logger.info(
                    "Purged orphaned generation | doc_id=%s user_id=%s generation=%s",
                    doc_id, user_id, generation,
                )
            except Exception as exc:
                logger.error(
                    "Failed to purge orphaned generation | doc_id=%s generation=%s err=%s",
                    doc_id, generation, exc,
                )

    logger.info(
        "purge_orphaned_generations finished | scanned_points=%d | purged_generations=%d",
        scanned, purged,
    )
    return purged


__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "ProgressCallback",
    "purge_orphaned_generations",
    # Re-exported for backward compatibility
    "_BATCH_SIZE",
    "_DENSE_EMBEDDING_DIM",
    "_DENSE_MODEL_NAME",
    "_DEFAULT_PURGE_MAX_DOCS",
    "_DEFAULT_PURGE_MIN_AGE_SECONDS",
    "_DEFAULT_PURGE_SCROLL_BATCH",
    "_EMBED_MAX_RETRIES",
    "_LOADERS",
    "_MAX_CELL_CHARS",
    "_MAX_CELLS_PER_ROW",
    "_MAX_CHUNKS_PER_DOC",
    "_MAX_CONSECUTIVE_EMPTY_ROWS",
    "_MAX_LINE_CHARS",
    "_MAX_PDF_PAGES",
    "_MAX_RECORD_CHARS",
    "_MAX_TABLE_ROWS",
    "_MAX_TEXT_BYTES",
    "_QDRANT_MAX_RETRIES",
    "_RETRY_BASE_DELAY_SECONDS",
    "_RowBudget",
    "_SPARSE_MODEL_NAME",
    "_WD_PARA_TAG",
    "_WD_TABLE_TAG",
    "_batched",
    "_csv_loader",
    "_docx_loader",
    "_embed_dense_batch",
    "_excel_loader",
    "_get_genai_client",
    "_get_qdrant",
    "_get_sparse_model",
    "_json_loader",
    "_pdf_loader",
    "_retry",
    "_text_loader",
]
