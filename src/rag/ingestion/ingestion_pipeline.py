"""Synchronous document ingestion and Qdrant maintenance pipeline.

What This File Does:
    Runs inside the Celery worker (never on the FastAPI event loop) to parse,
    chunk, embed, and store uploaded files in Qdrant, and sweeps away old or
    incomplete document versions left behind by crashed workers.

Workflow:
    1. Validate file extension and magic bytes (`_validate_file_input`).
    2. Stream raw pages/rows and split them into sized chunks (`_stream_chunks`).
    3. Generate dense and sparse embeddings per batch (`_embed_batch`).
    4. Package points with a fresh ``generation`` ID and ``committed=False``
       (`_build_point_structs`), then upsert them into Qdrant (`_upsert_points`).
    5. Flip ``committed=True`` on the new generation (`_commit_generation`) and
       delete previous generations of the same document (`_retire_old_generations`).
    6. Roll back any partial new generation on write errors (`_rollback_generation`),
       while ``purge_orphaned_generations`` periodically cleans up leftovers from
       hard worker crashes (SIGKILL / OOM).

Raises:
    ValueError:
        - In ``IngestionPipeline.__init__`` if ``chunk_size <= 0`` or
          ``chunk_overlap`` is outside ``[0, chunk_size)``.
        - In ``purge_orphaned_generations`` if ``min_age_seconds < 0``,
          ``scroll_batch_size <= 0``, or ``max_docs <= 0``.
    IngestionError:
        - If the file extension is not allowed, has no registered loader, or
          does not match the file's real content (magic bytes).
        - If a document contains non-string content or text splitting fails.
        - If the file exceeds ``_MAX_CHUNKS_PER_DOC`` or yields zero chunks.
        - If embedding counts do not match the batch size.
        - If deleting old generations fails after the new version is stored.
    Exception:
        - Re-raises any unexpected write-phase failure after rolling back
          attempted upserts for the current generation.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Callable, Generator, Iterator, Optional
from qdrant_client.models import (FieldCondition,
                                  FilterSelector,
                                  SparseVector,
                                  PointStruct,
                                  MatchValue,
                                  Filter,
                                 )
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
import logging
import time
import uuid


from src.core.validation import IngestionError, sniff_matches_extension
from src.rag.constants import (_DEFAULT_PURGE_MIN_AGE_SECONDS,
                               _DEFAULT_PURGE_SCROLL_BATCH,
                               _RETRY_BASE_DELAY_SECONDS,
                               _DEFAULT_PURGE_MAX_DOCS,
                               _MAX_CHUNKS_PER_DOC,
                               _QDRANT_MAX_RETRIES,
                               _MAX_RECORD_CHARS,
                               _BATCH_SIZE,
                              )

from src.rag.ingestion.loaders import stream_documents, _LOADERS
from src.rag.ingestion.pipeline_utils import _batched, _retry
from src.rag.clients import (_embed_dense_batch,
                             _get_sparse_model,
                             _get_qdrant,
                            )

from src.core.config import settings


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, Optional[int]], None]


# ##############################################################################
# Pipeline result
# ##############################################################################

@dataclass(frozen=True)
class IngestionResult:
    doc_id: str
    chunks: int
    generation: str


# ##############################################################################
# Main pipeline
# ##############################################################################

class IngestionPipeline:
    """Parses, chunks, embeds and upserts a single file into Qdrant.

    Memory profile: O(BATCH_SIZE) regardless of document size. Nothing — not
    the raw rows, not the split chunks, not the embeddings — is ever fully
    materialized in a list; everything flows through generators and is
    processed one bounded batch at a time.
    """

    def __init__(self,
                 collection_name: str = "rag_documents",
                 chunk_size: int = settings.CHUNK_SIZE,
                 chunk_overlap: int = settings.CHUNK_OVERLAP,
                ) -> None:
        """Sets up the pipeline settings and prepares the text splitter.

        Why it matters:
            Validates chunk settings early so the worker fails immediately
            instead of wasting time loading or embedding files with invalid sizes.

        Args:
            collection_name: Name of the target Qdrant collection.
            chunk_size: Maximum characters allowed per text chunk.
            chunk_overlap: Overlapping characters shared between adjacent chunks.

        Returns:
            None.

        Errors:
            ValueError: If chunk_size is not positive, or if chunk_overlap is
                negative or greater than or equal to chunk_size.
        """
        # Validate chunking parameters before initializing anything
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap must be in [0, chunk_size), "
                             f"got {chunk_overlap} for chunk_size={chunk_size}."
                            )

        # Store settings and initialize the recursive character splitter
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap,
                                                        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
                                                       )

    # chunking (streaming)
    def _stream_chunks(self,
                       docs: Iterator[Document],
                      ) -> Generator[Document, None, None]:
        """Reads raw documents one by one and yields clean, sized text chunks.

        Why it matters:
            Keeps memory usage tiny by streaming chunks through a generator
            instead of holding the whole document in RAM, and guarantees the
            upstream file stream is closed even if an error happens midway.

        Args:
            docs: An iterator yielding raw Document objects from a file loader.

        Returns:
            Generator[Document, None, None]: Yields cleaned Document chunks one by one.

        Errors:
            IngestionError: If a document has non-string content or if text
                splitting fails unexpectedly.
            Exception: Catches and logs errors while closing the upstream stream
                so they never hide the primary exception.
        """
        # Initialize counters to track chunking progress and log a summary at the end
        docs_processed = 0
        empty_docs_skipped = 0
        oversized_records_split = 0
        output_chunks = 0
        last_source = "unknown"

        try:
            # Loop over incoming raw documents one at a time
            for doc in docs:
                docs_processed += 1
                raw_meta = doc.metadata if isinstance(doc.metadata, dict) else {}
                source = str(raw_meta.get("source") or "unknown")
                last_source = source

                page = raw_meta.get("page")
                block = raw_meta.get("block")
                location = page if page is not None else (block if block is not None else "n/a")

                # Ensure page_content is a valid string before processing
                if not isinstance(doc.page_content, str):
                    logger.error("INGEST_CHUNK_INVALID_DOC | reason=page_content_not_string "
                                 "file=%s location=%s content_type=%s action=raising",
                                 source,
                                 location,
                                 type(doc.page_content).__name__,
                                )

                    raise IngestionError(f"Invalid document content type in '{source}'.")

                # Strip extra whitespace and skip completely empty pages or rows
                clean_content = doc.page_content.strip()
                if not clean_content:
                    empty_docs_skipped += 1
                    logger.debug("INGEST_CHUNK_EMPTY_DOC | file=%s location=%s action=skipping",
                                 source,
                                 location,
                                )

                    continue

                # Keep pre-chunked items (like table rows) intact if they fit the size limit
                is_pre_chunked = bool(raw_meta.get("pre_chunked"))
                if is_pre_chunked and len(clean_content) <= _MAX_RECORD_CHARS:
                    output_chunks += 1

                    yield Document(page_content=clean_content, metadata=dict(raw_meta))

                    continue

                # Warn if a pre-chunked item is too large and needs to be split anyway
                if is_pre_chunked:
                    oversized_records_split += 1
                    logger.warning("INGEST_CHUNK_OVERSIZED_RECORD | file=%s location=%s chars=%d "
                                   "max_record_chars=%d action=splitting_record",
                                   source,
                                   location,
                                   len(clean_content),
                                   _MAX_RECORD_CHARS,
                                  )

                # Split regular text or oversized records into smaller chunks
                normalized_doc = Document(page_content=clean_content, metadata=dict(raw_meta))

                try:
                    split_docs = self._splitter.split_documents([normalized_doc])

                except Exception as exc:
                    logger.exception("INGEST_CHUNK_SPLIT_FAILED | file=%s location=%s chars=%d "
                                     "err_type=%s err=%r action=raising",
                                     source,
                                     location,
                                     len(clean_content),
                                     type(exc).__name__,
                                     exc,
                                    )

                    raise IngestionError(f"Failed to split text content in '{source}'.") from exc

                # Clean and yield each generated sub-chunk with its sub-chunk index
                sub_chunk_idx = 0

                for chunk in split_docs:
                    chunk_text = chunk.page_content.strip()
                    if not chunk_text:
                        continue

                    sub_chunk_idx += 1
                    output_chunks += 1
                    chunk_meta = dict(chunk.metadata) if isinstance(chunk.metadata, dict) else {}
                    chunk_meta["sub_chunk"] = sub_chunk_idx

                    yield Document(page_content=chunk_text, metadata=chunk_meta)

            # Log a single-line summary after finishing the entire file stream
            logger.info("INGEST_CHUNKING_COMPLETED | file=%s docs_processed=%d empty_skipped=%d "
                        "oversized_split=%d output_chunks=%d action=done",
                        last_source,
                        docs_processed,
                        empty_docs_skipped,
                        oversized_records_split,
                        output_chunks,
                       )

        finally:
            # Always close the upstream generator so underlying file handles do not leak
            close_fn = getattr(docs, "close", None)
            if callable(close_fn):
                try:
                    close_fn()

                except Exception as close_exc:
                    logger.warning("INGEST_CHUNK_STREAM_CLOSE_FAILED | file=%s err_type=%s err=%r "
                                   "action=upstream_resources_may_leak",
                                   last_source,
                                   type(close_exc).__name__,
                                   str(close_exc),
                                  )

    # rollback
    def _rollback_generation(self,
                             client: QdrantClient,
                             user_id: str,
                             doc_id: str,
                             generation: str,
                            ) -> None:
        """Deletes all points written under a failed generation from Qdrant.

        Why it matters:
            Prevents partial or broken uploads from staying in the database
            when an error happens midway through ingestion.

        Args:
            client: Active QdrantClient instance.
            user_id: Owner identifier for tenant isolation.
            doc_id: Target document identifier.
            generation: The specific failed generation ID to delete.

        Returns:
            None.

        Errors:
            Exception: Catches all deletion errors after exhausting retries and
                logs CRITICAL so the scheduled sweep can clean it up later.
        """
        # Log that rollback has started for this failed generation
        logger.warning("INGEST_ROLLBACK_STARTED | doc_id=%s generation=%s action=deleting_generation",
                       doc_id,
                       generation,
                      )

        # Attempt to delete all points matching user_id, doc_id, and generation with retries
        try:
            _retry(lambda: client.delete(collection_name=self.collection_name,
                                         points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id)),
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

            logger.info("INGEST_ROLLBACK_COMPLETE | doc_id=%s generation=%s action=done",
                        doc_id,
                        generation,
                       )

        except Exception as exc:
            # If Qdrant still fails after retries, log CRITICAL and leave it to the orphan sweep
            logger.critical("INGEST_ROLLBACK_FAILED | doc_id=%s generation=%s err_type=%s err=%r "
                            "action=orphan_left_for_purge_sweep",
                            doc_id,
                            generation,
                            type(exc).__name__,
                            str(exc),
                           )

    # validation
    def _validate_file_input(self,
                             path: Path,
                             original_name: str,
                            ) -> str:
        """Checks that the file extension is allowed, supported, and matches its real content.

        Why it matters:
            Stops invalid, unsupported, or spoofed files immediately before
            wasting API calls or network bandwidth.

        Args:
            path: Local filesystem path to the uploaded file.
            original_name: Original filename provided by the user.

        Returns:
            str: The validated lowercase file extension (e.g., '.pdf').

        Errors:
            IngestionError: If the extension is not allowed, has no registered
                loader, or does not match the file's magic bytes.
        """
        # Extract the normalized lowercase extension and check if it is in the allowlist
        ext = Path(original_name).suffix.lower() or path.suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            logger.warning("INGEST_UNSUPPORTED_EXTENSION | original_name=%s ext=%s action=rejected",
                           original_name,
                           ext or "(none)",
                          )

            raise IngestionError(f"Unsupported file type: '{ext or '(none)'}'.")

        # Check if a parser/loader is registered for this extension
        if ext not in _LOADERS:
            logger.error("INGEST_NO_LOADER_REGISTERED | original_name=%s ext=%s action=rejected",
                         original_name,
                         ext,
                        )

            raise IngestionError(f"No loader registered for extension '{ext}'.")

        # Verify actual file signature (magic bytes) matches the claimed extension
        if not sniff_matches_extension(path, ext):
            logger.warning("INGEST_EXTENSION_MISMATCH | original_name=%s claimed_ext=%s action=rejected",
                           original_name,
                           ext,
                          )

            raise IngestionError("File content does not match its declared extension.")

        return ext

    # embedding & point construction
    def _embed_batch(self,
                     batch: list[Document],
                     sparse_model,
                     target_doc_id: str,
                    ) -> tuple[list[list[float]], list]:
        """Generates dense and sparse vector embeddings for a batch of chunks.

        Why it matters:
            Produces both semantic (dense) and keyword (sparse) vectors needed
            for hybrid search, and verifies vector counts match chunk counts.

        Args:
            batch: List of Document chunks to embed.
            sparse_model: Loaded sparse embedding model instance.
            target_doc_id: Document ID used for error logging.

        Returns:
            tuple[list[list[float]], list]: A tuple of (dense_vectors, sparse_vectors).

        Errors:
            IngestionError: If the number of returned dense or sparse vectors
                does not match the batch size.
        """
        # Extract raw text strings and generate both dense and sparse vectors
        texts = [c.page_content for c in batch]
        dense_vecs = _embed_dense_batch(texts)
        sparse_vecs = list(sparse_model.embed(texts, batch_size=_BATCH_SIZE))
        if len(dense_vecs) != len(batch) or len(sparse_vecs) != len(batch):
            logger.error("INGEST_EMBEDDING_MISMATCH | doc_id=%s batch_size=%d "
                         "dense_count=%d sparse_count=%d action=rejected",
                         target_doc_id,
                         len(batch),
                         len(dense_vecs),
                         len(sparse_vecs),
                        )

            raise IngestionError(f"Embedding mismatch: got {len(dense_vecs)} dense / "
                                 f"{len(sparse_vecs)} sparse vectors for a batch of {len(batch)}."
                                )

        return dense_vecs, sparse_vecs

    def _build_point_structs(self,
                             batch: list[Document],
                             dense_vecs: list[list[float]],
                             sparse_vecs: list,
                             user_id: str,
                             target_doc_id: str,
                             generation: str,
                             created_at: float,
                             start_index: int,
                             original_name: str,
                            ) -> list[PointStruct]:
        """Packages chunks, vectors, and metadata into Qdrant PointStruct objects.

        Why it matters:
            Assigns deterministic point IDs and tags every point with
            ``committed=False`` so incomplete uploads are safely isolated until
            the whole file finishes.

        Args:
            batch: List of Document chunks in the current batch.
            dense_vecs: Dense embedding vectors matching the batch.
            sparse_vecs: Sparse embedding vectors matching the batch.
            user_id: Owner user identifier.
            target_doc_id: Target document identifier.
            generation: Unique generation ID for this ingestion run.
            created_at: Unix timestamp when ingestion started.
            start_index: Starting chunk index offset for this batch.
            original_name: Original filename to store in metadata.

        Returns:
            list[PointStruct]: Ready-to-upsert Qdrant point records.

        Errors:
            None.
        """
        points: list[PointStruct] = []

        # Combine each chunk with its dense/sparse vectors and build a Qdrant PointStruct
        for offset, (chunk, dense, sparse) in enumerate(zip(batch, dense_vecs, sparse_vecs)):
            chunk_index = start_index + offset
            chunk_id = f"{target_doc_id}_chunk_{chunk_index}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{generation}_{chunk_id}"))

            points.append(PointStruct(id=point_id,
                                      vector={"dense": dense,
                                              "sparse": SparseVector(indices=sparse.indices.tolist(),
                                                                     values=sparse.values.tolist(),
                                                                    ),
                                             },
                                      payload={"user_id": user_id,
                                               "doc_id": target_doc_id,
                                               "generation": generation,
                                               "created_at": created_at,
                                               "committed": False,
                                               "chunk_id": chunk_id,
                                               "chunk_index": chunk_index,
                                               "text": chunk.page_content,
                                               "metadata": {**{k: v
                                                               for k, v in chunk.metadata.items()
                                                               if k != "pre_chunked"
                                                              },
                                                            "filename": original_name,
                                                           },
                                              },
                                     )
                         )

        return points

    def _commit_generation(self,
                           client: QdrantClient,
                           user_id: str,
                           target_doc_id: str,
                           generation: str,
                          ) -> None:
        """Flips ``committed=True`` on all points of a newly finished generation.

        Why it matters:
            Marks the new document version as complete and safe for search and
            maintenance sweeps before deleting the previous version.

        Args:
            client: Active QdrantClient instance.
            user_id: Owner user identifier.
            target_doc_id: Target document identifier.
            generation: Generation ID to mark as committed.

        Returns:
            None.

        Errors:
            Exception: Propagates Qdrant errors if ``set_payload`` still fails
                after exhausting all retry attempts.
        """
        # Update payload committed flag to True for all points in this generation
        _retry(lambda: client.set_payload(collection_name=self.collection_name,
                                          payload={"committed": True},
                                          points=FilterSelector(filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                                                                                    FieldCondition(key="doc_id", match=MatchValue(value=target_doc_id)),
                                                                                    FieldCondition(key="generation", match=MatchValue(value=generation)),
                                                                                   ]
                                                                             )
                                                               ),
                                          wait=True,
                                         ),
               attempts=_QDRANT_MAX_RETRIES,
               base_delay=_RETRY_BASE_DELAY_SECONDS,
               what="generation commit",
              )

    # storage & commit helpers
    def _upsert_points(self,
                       client: QdrantClient,
                       points: list[PointStruct],
                      ) -> None:
        """Writes a single batch of points into Qdrant with bounded retries.

        Why it matters:
            Protects batch uploads against transient network hiccups while
            waiting for Qdrant to confirm persistence.

        Args:
            client: Active QdrantClient instance.
            points: Batch of PointStruct records to insert or update.

        Returns:
            None.

        Errors:
            Exception: Propagates Qdrant errors if ``upsert`` fails after all retries.
        """
        # Upsert the batch of points into Qdrant and wait for confirmation
        _retry(lambda pts=points: client.upsert(collection_name=self.collection_name,
                                                points=pts,
                                                wait=True,
                                               ),
               attempts=_QDRANT_MAX_RETRIES,
               base_delay=_RETRY_BASE_DELAY_SECONDS,
               what="Qdrant upsert",
              )

    def _notify_progress(self,
                         progress_cb: Optional[ProgressCallback],
                         stored: int,
                         target_doc_id: str,
                        ) -> None:
        """Safely reports how many chunks have been stored so far to a progress callback.

        Why it matters:
            Allows UI or SSE progress bars to update without letting a broken
            callback crash an otherwise healthy document ingestion.

        Args:
            progress_cb: Optional callback function accepting (stored, total).
            stored: Number of chunks stored so far.
            target_doc_id: Document ID used for warning logs if the callback fails.

        Returns:
            None.

        Errors:
            Exception: Catches and logs any exception raised by ``progress_cb``.
        """
        # Skip if no progress callback was provided
        if not progress_cb:
            return

        # Invoke the callback safely so UI/progress errors never abort ingestion
        try:
            progress_cb(stored, None)

        except Exception as exc:
            logger.warning("INGEST_PROGRESS_CALLBACK_FAILED | doc_id=%s err_type=%s action=ignored",
                           target_doc_id,
                           type(exc).__name__,
                           exc_info=True,
                          )

    def _retire_old_generations(self,
                                client: QdrantClient,
                                user_id: str,
                                target_doc_id: str,
                                generation: str,
                                stored: int,
                               ) -> None:
        """Deletes all older generations of a document after the new one is committed.

        Why it matters:
            Completes the atomic swap by removing outdated versions of the same
            document so search results never mix old and new chunks.

        Args:
            client: Active QdrantClient instance.
            user_id: Owner user identifier.
            target_doc_id: Target document identifier.
            generation: The newly committed generation ID to keep.
            stored: Total number of chunks stored in the new generation.

        Returns:
            None.

        Errors:
            IngestionError: Raised if Qdrant fails to delete old generations
                after exhausting retries, so the caller knows stale data remains.
        """
        # Delete all points for this user and document except the newly committed generation
        try:
            _retry(lambda: client.delete(collection_name=self.collection_name,
                                         points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                                                                                            FieldCondition(key="doc_id", match=MatchValue(value=target_doc_id)),
                                                                                           ],
                                                                                      must_not=[FieldCondition(key="generation", match=MatchValue(value=generation)),
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
            # Log CRITICAL and raise IngestionError so stale data is never silently ignored
            logger.critical("INGEST_OLD_GENERATION_CLEANUP_FAILED | doc_id=%s generation=%s "
                            "stored_chunks=%d err_type=%s err=%r action=deferred_to_orphan_sweep",
                            target_doc_id,
                            generation,
                            stored,
                            type(exc).__name__,
                            str(exc),
                           )

            raise IngestionError(f"New version stored successfully ({stored} chunks) but cleanup of the "
                                 "previous version failed. Search results may temporarily include stale "
                                 "data until the cleanup is retried."
                                ) from exc

    # main entry point
    def ingest_file(self,
                    path: Path,
                    original_name: str,
                    user_id: str,
                    doc_id: Optional[str] = None,
                    progress_cb: Optional[ProgressCallback] = None,
                   ) -> IngestionResult:
        """Runs the full ingestion flow: validate, stream, chunk, embed, upsert, and swap.

        Why it matters:
            Guarantees atomic document updates in Qdrant — either the full new
            version is stored and replaces the old one, or any partial write is
            rolled back without harming existing live data.

        Args:
            path: Local filesystem path to the file to ingest.
            original_name: Original filename uploaded by the user.
            user_id: Owner user identifier.
            doc_id: Optional explicit document ID; derived deterministically if omitted.
            progress_cb: Optional callback invoked after each stored batch.

        Returns:
            IngestionResult: Summary containing ``doc_id``, ``chunks`` count, and ``generation``.

        Errors:
            IngestionError: If file validation fails, chunk limit is exceeded,
                no readable text is extracted, or old-generation cleanup fails.
            Exception: Any write-phase error triggers an automatic rollback if
                an upsert was attempted, then re-raises the original exception.
        """
        # Normalize path and validate file extension and magic bytes first
        path = Path(path)
        self._validate_file_input(path, original_name)

        # Prepare document identity, fresh generation ID, timestamp, and clients
        target_doc_id = doc_id or f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, f'{user_id}_{original_name}')}"
        generation = uuid.uuid4().hex
        created_at = time.time()

        client = _get_qdrant()
        sparse_model = _get_sparse_model()

        # Initialize the lazy document stream and write-phase tracking variables
        doc_stream = stream_documents(path, original_name)

        stored = 0
        upsert_attempted = False

        # Stream, chunk, embed, and upsert batches inside a rollback-protected block
        try:
            with closing(self._stream_chunks(doc_stream)) as chunk_stream:
                for batch in _batched(chunk_stream, _BATCH_SIZE):
                    # Reject the file before spending embedding calls if it exceeds the chunk cap
                    if stored + len(batch) > _MAX_CHUNKS_PER_DOC:
                        logger.warning("INGEST_CHUNK_LIMIT_EXCEEDED | doc_id=%s produced_chunks=%d "
                                       "max_chunks=%d action=rejected",
                                       target_doc_id,
                                       stored + len(batch),
                                       _MAX_CHUNKS_PER_DOC,
                                      )

                        raise IngestionError(f"The file produces more than {_MAX_CHUNKS_PER_DOC:,} chunks. "
                                             "Please split it into smaller files."
                                            )

                    # Embed the current batch and build Qdrant PointStruct records
                    dense_vecs, sparse_vecs = self._embed_batch(batch,
                                                                sparse_model,
                                                                target_doc_id,
                                                               )

                    points = self._build_point_structs(batch=batch,
                                                       dense_vecs=dense_vecs,
                                                       sparse_vecs=sparse_vecs,
                                                       user_id=user_id,
                                                       target_doc_id=target_doc_id,
                                                       generation=generation,
                                                       created_at=created_at,
                                                       start_index=stored,
                                                       original_name=original_name,
                                                      )

                    # Mark upsert_attempted before sending so even lost responses get rolled back
                    upsert_attempted = True
                    self._upsert_points(client, points)

                    stored += len(points)
                    self._notify_progress(progress_cb, stored, target_doc_id)

        except Exception:
            # Roll back anything written under this generation if the write phase failed
            if upsert_attempted:
                self._rollback_generation(client,
                                          user_id,
                                          target_doc_id,
                                          generation,
                                         )

            raise

        # Reject files that produced zero readable text chunks
        if stored == 0:
            logger.warning("INGEST_NO_TEXT_EXTRACTED | doc_id=%s original_name=%s action=rejected",
                           target_doc_id,
                           original_name,
                          )

            raise IngestionError("No readable text was found in the file. "
                                 "Scanned PDFs without embedded text require OCR preprocessing."
                                )

        # Mark the completed generation as committed and remove older generations
        self._commit_generation(client,
                                user_id,
                                target_doc_id,
                                generation,
                               )
        self._retire_old_generations(client,
                                     user_id,
                                     target_doc_id,
                                     generation,
                                     stored,
                                    )

        # Log final ingestion completion and return structured result
        logger.info("INGEST_FILE_COMPLETE | doc_id=%s generation=%s chunks=%d user_id=%s action=done",
                    target_doc_id,
                    generation,
                    stored,
                    user_id,
                   )

        return IngestionResult(doc_id=target_doc_id,
                               chunks=stored,
                               generation=generation,
                              )


# ##############################################################################
# Maintenance: sweep for generations orphaned by a hard crash
# ##############################################################################

@dataclass
class _GenInfo:
    """Tracks the latest timestamp and commit status observed for a document generation."""
    created_at: float
    committed: bool


def _collect_document_generations(client: QdrantClient,
                                  collection_name: str,
                                  scroll_batch_size: int,
                                  max_docs: int,
                                 ) -> tuple[dict[tuple[str, str], dict[str, _GenInfo]], int, bool]:
    """Scrolls Qdrant payloads to build a lightweight map of documents and their generations.

    Why it matters:
        Fetches only minimal payload fields (without heavy vectors or text) and
        caps tracked documents at ``max_docs`` so maintenance sweeps stay fast
        and never exhaust worker memory.

    Args:
        client: Active QdrantClient instance.
        collection_name: Name of the Qdrant collection to scan.
        scroll_batch_size: Number of points fetched per scroll request.
        max_docs: Maximum distinct (user_id, doc_id) pairs to track in memory.

    Returns:
        tuple[dict[tuple[str, str], dict[str, _GenInfo]], int, bool]:
            A tuple of (seen_map, scanned_points_count, hit_doc_limit_flag).

    Errors:
        Exception: Catches scroll failures after retries, logs the error, and
            breaks out of the loop to allow cleaning up the partial scan.
    """
    # Initialize in-memory map, pagination offset, and scan counters
    seen: dict[tuple[str, str], dict[str, _GenInfo]] = {}
    offset = None
    scanned = 0
    docs_seen = 0
    hit_doc_limit = False

    # Page through Qdrant points using scroll without fetching vectors or chunk text
    while True:
        try:
            records, offset = _retry(lambda off=offset: client.scroll(collection_name=collection_name,
                                                                      with_payload=["user_id", "doc_id", "generation", "created_at", "committed"],
                                                                      with_vectors=False,
                                                                      limit=scroll_batch_size,
                                                                      offset=off,
                                                                     ),
                                     attempts=_QDRANT_MAX_RETRIES,
                                     base_delay=_RETRY_BASE_DELAY_SECONDS,
                                     what="purge scroll",
                                    )

        except Exception as exc:
            # Stop scrolling on repeated failure and proceed with whatever was collected so far
            logger.error("INGEST_PURGE_GEN_SCROLL_FAILED | scanned_points=%d err_type=%s err=%r "
                         "action=using_partial_scan",
                         scanned,
                         type(exc).__name__,
                         str(exc),
                        )
            break

        if not records:
            break

        # Extract identity, timestamp, and commit status from each record's payload
        for record in records:
            payload = record.payload or {}
            user_id = payload.get("user_id")
            doc_id = payload.get("doc_id")
            generation = payload.get("generation")
            if not (user_id and doc_id and generation):
                continue

            raw_created_at = payload.get("created_at")
            created_at = float(raw_created_at) if isinstance(raw_created_at, (int, float)) else 0.0
            raw_committed = payload.get("committed")
            is_committed = True if raw_committed is None else bool(raw_committed)

            # Enforce the max_docs memory cap for newly discovered documents
            key = (user_id, doc_id)
            if key not in seen:
                if docs_seen >= max_docs:
                    hit_doc_limit = True

                    continue

                docs_seen += 1

            # Record or update the generation's latest timestamp and commit state
            gens = seen.setdefault(key, {})
            if generation not in gens:
                gens[generation] = _GenInfo(created_at=created_at, committed=is_committed)

            else:
                if created_at > gens[generation].created_at:
                    gens[generation].created_at = created_at

                if not is_committed:
                    gens[generation].committed = False

        scanned += len(records)

        if offset is None:
            break

    return seen, scanned, hit_doc_limit


def _delete_orphaned_generation(client: QdrantClient,
                                collection_name: str,
                                user_id: str,
                                doc_id: str,
                                generation: str,
                               ) -> bool:
    """Deletes all Qdrant points belonging to one stale or abandoned generation.

    Why it matters:
        Isolates deletion failures per generation so one failed delete does not
        stop the sweep from cleaning up other orphaned generations.

    Args:
        client: Active QdrantClient instance.
        collection_name: Name of the Qdrant collection.
        user_id: Owner user identifier.
        doc_id: Target document identifier.
        generation: Orphaned generation ID to delete.

    Returns:
        bool: True if deletion succeeded, False if it failed after retries.

    Errors:
        Exception: Catches deletion errors after retries and logs an error so
            the sweep can continue and retry on the next run.
    """
    # Delete all points matching the specific orphaned generation with bounded retries
    try:
        _retry(lambda: client.delete(collection_name=collection_name,
                                     points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                                                                                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                                                                        FieldCondition(key="generation", match=MatchValue(value=generation)),
                                                                                       ]
                                                                                 )
                                                                   ),
                                     wait=True,
                                    ),
               attempts=_QDRANT_MAX_RETRIES,
               base_delay=_RETRY_BASE_DELAY_SECONDS,
               what="orphan generation delete",
              )
        logger.info("INGEST_PURGE_GEN_DELETED | doc_id=%s user_id=%s generation=%s action=purged",
                    doc_id,
                    user_id,
                    generation,
                   )

        return True

    except Exception as exc:
        # Log the failure and return False so the rest of the sweep keeps running
        logger.error("INGEST_PURGE_GEN_DELETE_FAILED | doc_id=%s generation=%s err_type=%s "
                     "err=%r action=deferred_to_next_sweep",
                     doc_id,
                     generation,
                     type(exc).__name__,
                     str(exc),
                    )

        return False


def purge_orphaned_generations(collection_name: str = "rag_documents",
                               min_age_seconds: float = _DEFAULT_PURGE_MIN_AGE_SECONDS,
                               scroll_batch_size: int = _DEFAULT_PURGE_SCROLL_BATCH,
                               max_docs: int = _DEFAULT_PURGE_MAX_DOCS,
                              ) -> int:
    """Sweeps Qdrant to remove stale or uncommitted generations left by crashed workers.

    Why it matters:
        Recovers storage and prevents stale search results after hard crashes
        (OOM-kill, SIGKILL) that bypass ``ingest_file``'s rollback handler, while
        protecting active uploads via ``min_age_seconds``.

    Args:
        collection_name: Name of the Qdrant collection to sweep.
        min_age_seconds: Minimum age in seconds before a generation can be deleted.
        scroll_batch_size: Number of points fetched per scroll page.
        max_docs: Maximum distinct (user_id, doc_id) pairs processed per sweep.

    Returns:
        int: Total number of orphaned generations deleted during this run.

    Errors:
        ValueError: If ``min_age_seconds`` is negative, or if ``scroll_batch_size``
            or ``max_docs`` is not positive.
    """
    # Validate sweep configuration arguments before touching Qdrant
    if min_age_seconds < 0:
        raise ValueError(f"min_age_seconds must be non-negative, got {min_age_seconds}.")

    if scroll_batch_size <= 0:
        raise ValueError(f"scroll_batch_size must be positive, got {scroll_batch_size}.")

    if max_docs <= 0:
        raise ValueError(f"max_docs must be positive, got {max_docs}.")

    # Compute the safety cutoff timestamp and scan Qdrant for document generations
    client = _get_qdrant()
    cutoff = time.time() - min_age_seconds

    seen, scanned, hit_doc_limit = _collect_document_generations(client=client,
                                                                 collection_name=collection_name,
                                                                 scroll_batch_size=scroll_batch_size,
                                                                 max_docs=max_docs,
                                                                )

    if hit_doc_limit:
        logger.warning("INGEST_PURGE_GEN_DOC_LIMIT_REACHED | max_docs=%d "
                       "action=remaining_docs_skipped_this_run",
                       max_docs,
                      )

    # For each document, keep only the newest committed generation and purge old ones
    purged = 0

    for (user_id, doc_id), gens in seen.items():
        committed_gens = {g: info for g, info in gens.items() if info.committed}
        newest_committed = (max(committed_gens, key=lambda g: committed_gens[g].created_at)
                            if committed_gens
                            else None
                           )

        for generation, info in gens.items():
            # Never delete the newest live committed generation
            if generation == newest_committed:
                continue

            # Skip recent generations that might still be actively uploading
            if info.created_at > cutoff:
                continue

            if _delete_orphaned_generation(client,
                                           collection_name,
                                           user_id,
                                           doc_id,
                                           generation,
                                          ):
                purged += 1

    # Log the final sweep summary and return the count of deleted generations
    logger.info("INGEST_PURGE_GEN_SWEEP_COMPLETE | scanned_points=%d purged_generations=%d action=done",
                scanned,
                purged,
               )

    return purged
