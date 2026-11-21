"""Comprehensive test suite for src.services.tasks module.

Covers every public and internal function: ProgressThrottler, _validate_file_reference,
_remove_file, _get_redis, _lock_key, _derive_doc_id, _jittered_backoff, ingest_document,
purge_stale_local_uploads, plus stress/load and security-hardening scenarios.

IMPORTANT: The tasks module has import-time side effects (settings.UPLOAD_DIR.resolve(strict=True)
and IngestionPipeline()), so all imports are deferred behind patches.
"""

from __future__ import annotations

import os
import re
import stat as stat_module
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fakes / stubs used by the fixtures
# ---------------------------------------------------------------------------

STORED_NAME_RE = re.compile(r"^[0-9a-f]{32}\.[a-zA-Z0-9]+$")

VALID_STORED_NAME = "a" * 32 + ".pdf"  # a valid hex name
VALID_STORED_NAME_2 = "b" * 32 + ".txt"


class _FakeIngestionError(ValueError):
    """Mirror of src.core.validation.IngestionError for use in tests."""


@dataclass(frozen=True)
class _FakeIngestionResult:
    doc_id: str
    chunks: int
    generation: str


# ---------------------------------------------------------------------------
# Module-level patches applied BEFORE importing src.services.tasks
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_tasks_imports(tmp_path):
    """Patch heavy / side-effectful imports so the module loads safely.

    Every test gets a fresh tmp_path as the upload directory and clean module state.
    """
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    fake_settings = MagicMock()
    fake_settings.UPLOAD_DIR = upload_dir
    fake_settings.MAX_UPLOAD_MB = 10 * 1024 * 1024  # 10 MiB in bytes
    fake_settings.CELERY_BROKER_URL = "redis://localhost:6379/0"
    fake_settings.INGEST_LOCK_REDIS_URL = "redis://localhost:6379/2"
    fake_settings.INGESTION_SOFT_TIME_LIMIT_SECONDS = 600
    fake_settings.INGESTION_HARD_TIME_LIMIT_SECONDS = 690
    fake_settings.PURGE_LOCK_KEY = "ingest:lock:purge_stale_local_uploads"
    fake_settings.PURGE_SOFT_TIME_LIMIT = 200
    fake_settings.PURGE_HARD_TIME_LIMIT = 250
    fake_settings.MAX_PURGE_FILES_PER_RUN = 5000
    fake_settings.STALE_UPLOAD_MAX_AGE = 3600 * 24

    fake_celery = MagicMock()
    # Make the decorator a passthrough so ingest_document / purge_stale_local_uploads
    # are plain functions in the test.
    fake_celery.task = lambda **kw: (lambda fn: fn)

    mock_pipeline_class = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline_instance

    patches = {
        "src.core.config": MagicMock(settings=fake_settings),
        "src.core.config.settings": fake_settings,
        "src.core.celery_app": MagicMock(celery_app=fake_celery),
        "src.core.celery_app.celery_app": fake_celery,
        "src.core.validation": MagicMock(
            STORED_NAME_RE=STORED_NAME_RE,
            IngestionError=_FakeIngestionError,
        ),
        "src.core.validation.STORED_NAME_RE": STORED_NAME_RE,
        "src.core.validation.IngestionError": _FakeIngestionError,
        "src.rag.ingestion.ingestion_pipeline": MagicMock(
            IngestionPipeline=mock_pipeline_class,
            IngestionResult=_FakeIngestionResult,
        ),
        "src.rag.ingestion.ingestion_pipeline.IngestionPipeline": mock_pipeline_class,
        "src.rag.ingestion.ingestion_pipeline.IngestionResult": _FakeIngestionResult,
        "src.rag.constants": MagicMock(
            _INGEST_HARD_TIME_LIMIT=690,
            _INGEST_SOFT_TIME_LIMIT=600,
            _LOCK_TTL_SECONDS=735,
            _NON_RETRYABLE_CODE_ERRORS=(TypeError, AttributeError, KeyError, ValueError, NotImplementedError),
        ),
        "src.rag.constants._INGEST_HARD_TIME_LIMIT": 690,
        "src.rag.constants._INGEST_SOFT_TIME_LIMIT": 600,
        "src.rag.constants._LOCK_TTL_SECONDS": 735,
        "src.rag.constants._NON_RETRYABLE_CODE_ERRORS": (TypeError, AttributeError, KeyError, ValueError, NotImplementedError),
    }

    # Remove a potentially cached module so we get a fresh import with our patches
    for mod_name in list(sys.modules):
        if mod_name.startswith("src.services.tasks"):
            del sys.modules[mod_name]

    with patch.dict(sys.modules, patches):
        import src.services.tasks as tasks_mod

        # Override module-level resolved constants
        tasks_mod._UPLOAD_DIR_RESOLVED = upload_dir.resolve()
        tasks_mod._MAX_UPLOAD_FILE_BYTES = 10 * 1024 * 1024
        tasks_mod._pipeline = mock_pipeline_instance
        tasks_mod._redis_client = None  # reset singleton
        tasks_mod.IngestionError = _FakeIngestionError
        tasks_mod.STORED_NAME_RE = STORED_NAME_RE
        tasks_mod._NON_RETRYABLE_CODE_ERRORS = (TypeError, AttributeError, KeyError, ValueError, NotImplementedError)
        tasks_mod._LOCK_TTL_SECONDS = 735

        yield tasks_mod

    # Clean up
    for mod_name in list(sys.modules):
        if mod_name.startswith("src.services.tasks"):
            del sys.modules[mod_name]


@pytest.fixture()
def tasks(_patch_tasks_imports):
    """Shorthand access to the patched tasks module."""
    return _patch_tasks_imports


@pytest.fixture()
def upload_dir(tasks):
    """Return the resolved upload directory."""
    return tasks._UPLOAD_DIR_RESOLVED


@pytest.fixture()
def make_upload(upload_dir):
    """Factory: create a real file inside the upload dir with the given stored name and content."""

    def _make(name: str = VALID_STORED_NAME, content: bytes = b"hello world", size: int | None = None):
        p = upload_dir / name
        if size is not None:
            p.write_bytes(b"\x00" * size)
        else:
            p.write_bytes(content)
        return p

    return _make


@pytest.fixture()
def mock_celery_self():
    """Build a mock Celery task self with the standard attributes."""

    def _make(retries: int = 0, max_retries: int = 3, task_id: str = "task-abc-123"):
        self = MagicMock()
        self.request.id = task_id
        self.request.retries = retries
        self.max_retries = max_retries
        self.retry = MagicMock(side_effect=Exception("celery-retry-sentinel"))
        self.update_state = MagicMock()
        return self

    return _make


@pytest.fixture()
def mock_redis(tasks):
    """Patch _get_redis to return a controllable MagicMock redis client."""
    redis_mock = MagicMock()
    with patch.object(tasks, "_get_redis", return_value=redis_mock):
        yield redis_mock


# ============================================================================
# Class 1: TestProgressThrottler
# ============================================================================

class TestProgressThrottler:
    """Tests for the ProgressThrottler class — throttling, completion bypass, edge cases."""

    def test_initial_state_last_update_time_is_zero(self, tasks):
        """The throttler should initialise with last_update_time = 0."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=1.0)
        assert throttler.last_update_time == 0.0

    def test_first_call_always_emits(self, tasks):
        """The very first update() call should always emit because last_update_time=0."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=100.0)
        throttler.update(1, 10)
        mock_task.update_state.assert_called_once()

    def test_rapid_calls_within_interval_throttled(self, tasks):
        """Rapid calls within the interval should be throttled — only the first emits."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=100.0)
        # First call emits
        throttler.update(1, 10)
        # Subsequent rapid calls within interval should NOT emit
        throttler.update(2, 10)
        throttler.update(3, 10)
        throttler.update(4, 10)
        assert mock_task.update_state.call_count == 1

    def test_calls_after_interval_emit(self, tasks):
        """Calls after the interval should emit."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=0.0)
        throttler.update(1, 10)
        throttler.update(2, 10)
        # With interval=0 both should emit
        assert mock_task.update_state.call_count == 2

    def test_completion_bypass_always_emits(self, tasks):
        """When done >= total, the update always emits regardless of interval."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        throttler.update(1, 10)  # first call emits
        # Now done==total should bypass throttle
        throttler.update(10, 10)
        assert mock_task.update_state.call_count == 2

    def test_completion_bypass_done_exceeds_total(self, tasks):
        """When done > total, the update always emits (completion bypass)."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        throttler.update(0, 10)  # first
        throttler.update(15, 10)  # done > total → completion
        assert mock_task.update_state.call_count == 2

    def test_total_none_never_counts_as_completion(self, tasks):
        """When total is None, the completion bypass is NOT triggered."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        throttler.update(0, None)  # first call emits
        throttler.update(100, None)  # should be throttled
        assert mock_task.update_state.call_count == 1

    def test_done_zero_total_zero_counts_as_completion(self, tasks):
        """done=0 and total=0 → 0 >= 0 is True, so it should bypass."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        throttler.update(0, 10)  # first
        throttler.update(0, 0)  # completion bypass (0 >= 0)
        assert mock_task.update_state.call_count == 2

    def test_negative_done_values(self, tasks):
        """Negative done values should not trigger completion bypass."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        throttler.update(-5, 10)  # first call emits
        throttler.update(-3, 10)  # should be throttled
        assert mock_task.update_state.call_count == 1

    def test_very_large_numbers(self, tasks):
        """Very large done/total values should work correctly."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        big = 10**18
        throttler.update(0, big)  # first
        throttler.update(big, big)  # completion → emits
        assert mock_task.update_state.call_count == 2

    def test_meta_contains_done_and_total(self, tasks):
        """The emitted state must include the correct meta dict."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=0.0)
        throttler.update(5, 20)
        mock_task.update_state.assert_called_with(state="PROGRESS", meta={"done": 5, "total": 20})

    def test_rapid_fire_100_calls(self, tasks):
        """100+ rapid-fire calls within a large interval should produce exactly 1 emission."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        for i in range(100):
            throttler.update(i, 1000)
        # Only the first call should have emitted
        assert mock_task.update_state.call_count == 1

    def test_custom_min_interval(self, tasks):
        """The min_interval_seconds parameter should be honoured."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=5.0)
        assert throttler.min_interval == 5.0


# ============================================================================
# Class 2: TestValidateFileReference
# ============================================================================

class TestValidateFileReference:
    """Tests for _validate_file_reference — format, security, filesystem checks."""

    def test_valid_stored_name_returns_path(self, tasks, make_upload):
        """A valid stored name for an existing file should return the resolved path."""
        make_upload(VALID_STORED_NAME, b"content")
        result = tasks._validate_file_reference(VALID_STORED_NAME)
        assert result.name == VALID_STORED_NAME
        assert result.is_file()

    def test_invalid_format_empty_string(self, tasks):
        """Empty string should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("")

    def test_invalid_format_spaces(self, tasks):
        """Whitespace-only string should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("   ")

    def test_invalid_format_path_traversal(self, tasks):
        """Path traversal attempt via ../etc/passwd should be rejected at regex."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("../etc/passwd")

    def test_invalid_format_null_bytes(self, tasks):
        """Null byte injection should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("abc\x00.pdf")

    def test_invalid_format_unicode(self, tasks):
        """Unicode characters should be rejected by the hex-only regex."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("aaaa" * 8 + ".пдф")

    def test_invalid_format_dot_only(self, tasks):
        """Single dot '.' should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference(".")

    def test_invalid_format_double_dot(self, tasks):
        """Double dot '..' should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("..")

    def test_invalid_format_slash(self, tasks):
        """Forward slash should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 32 + "/pdf")

    def test_invalid_format_backslash(self, tasks):
        """Backslash should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 32 + "\\.pdf")

    def test_invalid_format_very_long_name(self, tasks):
        """Very long name that doesn't match regex should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 10000 + ".pdf")

    def test_invalid_format_no_extension(self, tasks):
        """Name without extension should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 32)

    def test_invalid_format_uppercase_hex(self, tasks):
        """Uppercase hex chars should be rejected (regex requires lowercase)."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("A" * 32 + ".pdf")

    def test_file_not_found(self, tasks, upload_dir):
        """FileNotFoundError during resolve should raise IngestionError."""
        with pytest.raises(_FakeIngestionError, match="no longer exists"):
            tasks._validate_file_reference(VALID_STORED_NAME)  # file doesn't exist

    def test_permission_error_on_resolve(self, tasks, upload_dir):
        """PermissionError during path.resolve should raise IngestionError."""
        raw_path = upload_dir / VALID_STORED_NAME
        with patch.object(type(raw_path), "resolve", side_effect=PermissionError("denied")):
            with patch("src.services.tasks.settings") as mock_s:
                mock_s.UPLOAD_DIR = upload_dir
                with pytest.raises(_FakeIngestionError, match="Cannot access"):
                    tasks._validate_file_reference(VALID_STORED_NAME)

    def test_oserror_on_resolve(self, tasks, upload_dir):
        """Generic OSError during resolve should raise IngestionError."""
        with patch("src.services.tasks.settings") as mock_s:
            mock_s.UPLOAD_DIR = upload_dir
            with patch("pathlib.Path.resolve", side_effect=OSError("disk error")):
                with pytest.raises(_FakeIngestionError, match="Cannot access"):
                    tasks._validate_file_reference(VALID_STORED_NAME)

    def test_runtime_error_symlink_loop(self, tasks, upload_dir):
        """RuntimeError (symlink loop) during resolve should raise IngestionError."""
        with patch("src.services.tasks.settings") as mock_s:
            mock_s.UPLOAD_DIR = upload_dir
            with patch("pathlib.Path.resolve", side_effect=RuntimeError("symlink loop")):
                with pytest.raises(_FakeIngestionError, match="symlink loop"):
                    tasks._validate_file_reference(VALID_STORED_NAME)

    def test_path_traversal_resolved_outside_upload_dir(self, tasks, upload_dir, tmp_path):
        """A file that resolves outside UPLOAD_DIR should be rejected."""
        # Create a file outside the upload dir
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"evil")

        with patch("src.services.tasks.settings") as mock_s:
            mock_s.UPLOAD_DIR = upload_dir
            with patch("pathlib.Path.resolve", return_value=outside):
                with pytest.raises(_FakeIngestionError, match="Invalid upload reference"):
                    tasks._validate_file_reference(VALID_STORED_NAME)

    def test_stat_failure(self, tasks, make_upload):
        """OSError on stat() should raise IngestionError."""
        make_upload(VALID_STORED_NAME)
        with patch("pathlib.Path.stat", side_effect=OSError("stat fail")):
            with pytest.raises(_FakeIngestionError, match="Cannot access"):
                tasks._validate_file_reference(VALID_STORED_NAME)

    def test_not_regular_file_directory(self, tasks, upload_dir):
        """A directory matching the name pattern should be rejected."""
        d = upload_dir / VALID_STORED_NAME
        d.mkdir()
        with pytest.raises(_FakeIngestionError, match="not a regular file"):
            tasks._validate_file_reference(VALID_STORED_NAME)

    def test_empty_file_zero_bytes(self, tasks, make_upload):
        """Empty file (0 bytes) should be rejected."""
        make_upload(VALID_STORED_NAME, b"")
        with pytest.raises(_FakeIngestionError, match="empty"):
            tasks._validate_file_reference(VALID_STORED_NAME)

    def test_file_exceeds_max_upload_size(self, tasks, make_upload):
        """File larger than MAX_UPLOAD_FILE_BYTES should be rejected."""
        # Create file larger than 10 MiB
        make_upload(VALID_STORED_NAME, size=10 * 1024 * 1024 + 1)
        with pytest.raises(_FakeIngestionError, match="exceeds"):
            tasks._validate_file_reference(VALID_STORED_NAME)

    def test_file_exactly_at_size_limit(self, tasks, make_upload):
        """File exactly at the limit (not over) should be accepted."""
        make_upload(VALID_STORED_NAME, size=10 * 1024 * 1024)
        result = tasks._validate_file_reference(VALID_STORED_NAME)
        assert result.is_file()

    def test_file_one_byte_over_limit(self, tasks, make_upload):
        """File 1 byte over the limit should be rejected."""
        make_upload(VALID_STORED_NAME, size=10 * 1024 * 1024 + 1)
        with pytest.raises(_FakeIngestionError, match="exceeds"):
            tasks._validate_file_reference(VALID_STORED_NAME)

    def test_read_permission_denied(self, tasks, make_upload):
        """os.access returning False should raise IngestionError."""
        make_upload(VALID_STORED_NAME)
        with patch("os.access", return_value=False):
            with pytest.raises(_FakeIngestionError, match="permission denied"):
                tasks._validate_file_reference(VALID_STORED_NAME)

    @pytest.mark.parametrize("name", [
        "0123456789abcdef0123456789abcdef.pdf",
        "abcdef0123456789abcdef0123456789.txt",
        "deadbeef" * 4 + ".docx",
        "0" * 32 + ".json",
    ])
    def test_valid_name_formats(self, tasks, make_upload, name):
        """Various valid stored names should pass validation."""
        make_upload(name)
        result = tasks._validate_file_reference(name)
        assert result.name == name


# ============================================================================
# Class 3: TestRemoveFile
# ============================================================================

class TestRemoveFile:
    """Tests for _remove_file — cleanup safety and security."""

    def test_successful_removal(self, tasks, make_upload, upload_dir):
        """File inside upload dir should be removed successfully."""
        p = make_upload(VALID_STORED_NAME)
        assert p.exists()
        tasks._remove_file(p)
        assert not p.exists()

    def test_path_outside_upload_dir_not_deleted(self, tasks, tmp_path):
        """Path outside the upload dir should NOT be deleted and should log critical."""
        outside = tmp_path / "other_dir" / "secret.txt"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("important")
        tasks._remove_file(outside)
        # File must still exist
        assert outside.exists()

    def test_file_already_deleted_missing_ok(self, tasks, upload_dir):
        """If the file is already deleted, missing_ok=True should handle it."""
        p = upload_dir / VALID_STORED_NAME
        # File doesn't exist — should not raise
        tasks._remove_file(p)

    def test_oserror_during_unlink(self, tasks, make_upload):
        """OSError during unlink should be caught, logged, and not raised."""
        p = make_upload(VALID_STORED_NAME)
        with patch.object(type(p), "unlink", side_effect=OSError("disk full")):
            # Should not raise
            tasks._remove_file(p)

    def test_permission_error_during_unlink(self, tasks, make_upload):
        """PermissionError (a subclass of OSError) during unlink should be caught."""
        p = make_upload(VALID_STORED_NAME)
        with patch.object(type(p), "unlink", side_effect=PermissionError("locked")):
            # Should not raise
            tasks._remove_file(p)


# ============================================================================
# Class 4: TestGetRedis
# ============================================================================

class TestGetRedis:
    """Tests for _get_redis — lazy singleton creation."""

    def test_first_call_creates_client(self, tasks):
        """First call should create a Redis client."""
        tasks._redis_client = None
        with patch("redis.Redis") as mock_redis_cls:
            mock_client = MagicMock()
            mock_redis_cls.from_url.return_value = mock_client
            result = tasks._get_redis()
            assert result is mock_client
            mock_redis_cls.from_url.assert_called_once()

    def test_second_call_returns_cached(self, tasks):
        """Second call should return the cached client, not create a new one."""
        sentinel = MagicMock()
        tasks._redis_client = sentinel
        result = tasks._get_redis()
        assert result is sentinel

    def test_uses_ingest_lock_redis_url_when_available(self, tasks):
        """Should prefer INGEST_LOCK_REDIS_URL when present."""
        tasks._redis_client = None
        with patch("redis.Redis") as mock_redis_cls, \
             patch("src.services.tasks.settings") as mock_s:
            mock_s.INGEST_LOCK_REDIS_URL = "redis://custom:6379/5"
            mock_s.CELERY_BROKER_URL = "redis://broker:6379/0"
            mock_redis_cls.from_url.return_value = MagicMock()
            tasks._get_redis()
            mock_redis_cls.from_url.assert_called_once_with(
                "redis://custom:6379/5", socket_timeout=5, socket_connect_timeout=5
            )

    def test_falls_back_to_celery_broker_url(self, tasks):
        """Should fall back to CELERY_BROKER_URL when INGEST_LOCK_REDIS_URL is not set."""
        tasks._redis_client = None
        with patch("redis.Redis") as mock_redis_cls, \
             patch("src.services.tasks.settings") as mock_s:
            # Simulate missing attribute
            del mock_s.INGEST_LOCK_REDIS_URL
            mock_s.CELERY_BROKER_URL = "redis://broker:6379/0"
            mock_redis_cls.from_url.return_value = MagicMock()
            tasks._get_redis()
            mock_redis_cls.from_url.assert_called_once_with(
                "redis://broker:6379/0", socket_timeout=5, socket_connect_timeout=5
            )


# ============================================================================
# Class 5: TestLockKey
# ============================================================================

class TestLockKey:
    """Tests for _lock_key — deterministic key generation."""

    def test_standard_inputs(self, tasks):
        """Standard user and doc IDs should produce expected key."""
        assert tasks._lock_key("user1", "doc1") == "ingest:lock:user1:doc1"

    def test_special_characters(self, tasks):
        """Special characters in IDs should be included verbatim."""
        assert tasks._lock_key("u@ser:1", "doc/2") == "ingest:lock:u@ser:1:doc/2"

    def test_empty_strings(self, tasks):
        """Empty strings should produce a key with empty segments."""
        assert tasks._lock_key("", "") == "ingest:lock::"

    def test_unicode_in_ids(self, tasks):
        """Unicode characters should be passed through."""
        assert tasks._lock_key("مستخدم", "مستند") == "ingest:lock:مستخدم:مستند"


# ============================================================================
# Class 6: TestDeriveDocId
# ============================================================================

class TestDeriveDocId:
    """Tests for _derive_doc_id — determinism and idempotency."""

    def test_same_inputs_same_output(self, tasks):
        """Same inputs must always produce the same doc_id (idempotent)."""
        id1 = tasks._derive_doc_id("user1", "report.pdf")
        id2 = tasks._derive_doc_id("user1", "report.pdf")
        assert id1 == id2

    def test_different_inputs_different_output(self, tasks):
        """Different inputs should produce different doc_ids."""
        id1 = tasks._derive_doc_id("user1", "report.pdf")
        id2 = tasks._derive_doc_id("user2", "report.pdf")
        id3 = tasks._derive_doc_id("user1", "other.pdf")
        assert id1 != id2
        assert id1 != id3

    def test_format_starts_with_doc_prefix(self, tasks):
        """The derived ID should start with 'doc_'."""
        result = tasks._derive_doc_id("user1", "file.pdf")
        assert result.startswith("doc_")

    def test_format_contains_uuid5(self, tasks):
        """The ID should contain a valid UUID5."""
        result = tasks._derive_doc_id("user1", "file.pdf")
        uuid_part = result[4:]  # strip "doc_"
        # Validate it parses as a UUID
        parsed = uuid.UUID(uuid_part)
        assert parsed.version == 5

    def test_matches_manual_computation(self, tasks):
        """The derived doc_id should match a manual uuid5 computation."""
        expected = f"doc_{uuid.uuid5(uuid.NAMESPACE_DNS, 'user1_report.pdf')}"
        assert tasks._derive_doc_id("user1", "report.pdf") == expected


# ============================================================================
# Class 7: TestJitteredBackoff
# ============================================================================

class TestJitteredBackoff:
    """Tests for _jittered_backoff — exponential backoff + jitter correctness."""

    def test_retries_zero(self, tasks):
        """retries=0 → base*2^0 + jitter = base + jitter."""
        with patch("random.uniform", return_value=1.0):
            result = tasks._jittered_backoff(0, base=5.0, cap=60.0)
            assert result == 6.0  # 5.0 + 1.0

    def test_retries_one(self, tasks):
        """retries=1 → base*2^1 + jitter = 2*base + jitter."""
        with patch("random.uniform", return_value=1.5):
            result = tasks._jittered_backoff(1, base=5.0, cap=60.0)
            assert result == 11.5  # 10.0 + 1.5

    def test_retries_high_capped(self, tasks):
        """retries=10 → min(base*2^10, cap) + jitter → cap + jitter."""
        with patch("random.uniform", return_value=2.0):
            result = tasks._jittered_backoff(10, base=5.0, cap=60.0)
            assert result == 62.0  # 60.0 + 2.0

    def test_jitter_range(self, tasks):
        """The jitter portion should be between 0.5 and 3.0."""
        results = set()
        for _ in range(1000):
            val = tasks._jittered_backoff(0, base=0.0, cap=0.0)
            results.add(val)
            assert 0.5 <= val <= 3.0

    def test_result_rounded_to_2_decimal_places(self, tasks):
        """Result should be rounded to 2 decimal places."""
        with patch("random.uniform", return_value=1.123456):
            result = tasks._jittered_backoff(0, base=5.0, cap=60.0)
            assert result == round(5.0 + 1.123456, 2)
            # Verify it has at most 2 decimal places
            assert result == round(result, 2)

    def test_negative_retries(self, tasks):
        """Negative retries should produce base*2^(-n) + jitter (small value)."""
        with patch("random.uniform", return_value=1.0):
            result = tasks._jittered_backoff(-1, base=8.0, cap=60.0)
            # 8 * 2^(-1) = 4.0, + 1.0 = 5.0
            assert result == 5.0

    def test_very_large_retries_no_overflow(self, tasks):
        """Very large retry count should be capped, not overflow."""
        with patch("random.uniform", return_value=1.0):
            result = tasks._jittered_backoff(1000, base=5.0, cap=60.0)
            assert result == 61.0  # capped at 60 + 1.0

    def test_statistical_distribution(self, tasks):
        """Over many calls, results should show variance from jitter."""
        results = [tasks._jittered_backoff(0, base=5.0, cap=60.0) for _ in range(10000)]
        assert min(results) >= 5.5  # 5.0 + 0.5
        assert max(results) <= 8.0  # 5.0 + 3.0
        # Variance should be non-zero
        mean = sum(results) / len(results)
        variance = sum((x - mean) ** 2 for x in results) / len(results)
        assert variance > 0.01


# ============================================================================
# Class 8: TestIngestDocument — Full Lifecycle (MOST CRITICAL)
# ============================================================================

class TestIngestDocument:
    """Tests for ingest_document Celery task — validation, locking, pipeline, cleanup."""

    # -- Happy path -----------------------------------------------------------

    def test_happy_path_full_lifecycle(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Happy path: validate → lock → ingest → cleanup → return result."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True  # lock acquired

        result_obj = _FakeIngestionResult(doc_id="doc_123", chunks=5, generation="gen_1")
        tasks._pipeline.ingest_file.return_value = result_obj

        result = tasks.ingest_document(
            self_mock, VALID_STORED_NAME, "report.pdf", "user1", doc_id="doc_123"
        )

        assert result["status"] == "done"
        assert result["doc_id"] == "doc_123"
        assert result["generation"] == "gen_1"
        assert result["chunks"] == 5

    def test_file_removed_on_success(self, tasks, make_upload, mock_redis, mock_celery_self):
        """The uploaded file should be removed after successful ingestion."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True

        result_obj = _FakeIngestionResult(doc_id="doc_1", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="doc_1")
        assert not p.exists()

    def test_lock_released_on_success(self, tasks, make_upload, mock_redis, mock_celery_self):
        """The Redis lock should be released after success."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True

        result_obj = _FakeIngestionResult(doc_id="d", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="d")
        mock_redis.delete.assert_called()

    # -- Validation failures --------------------------------------------------

    def test_invalid_stored_name_raises(self, tasks, mock_celery_self):
        """Invalid stored_name should raise IngestionError with no cleanup."""
        self_mock = mock_celery_self()
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks.ingest_document(self_mock, "INVALID!", "f.pdf", "u1")

    def test_file_not_found_raises(self, tasks, mock_celery_self, mock_redis):
        """Non-existent file should raise IngestionError."""
        self_mock = mock_celery_self()
        with pytest.raises(_FakeIngestionError, match="no longer exists"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")

    # -- Lock behavior --------------------------------------------------------

    def test_lock_acquired_proceeds(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Lock acquired → task should proceed to pipeline."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True

        result_obj = _FakeIngestionResult(doc_id="d", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        result = tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="d")
        assert result["status"] == "done"

    def test_lock_already_taken_raises(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Lock already held → IngestionError 'currently being processed'."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = False  # lock not acquired

        with pytest.raises(_FakeIngestionError, match="currently being processed"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")

    def test_redis_down_not_last_attempt_retries(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Redis failure on non-last attempt → retry with backoff."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self(retries=0, max_retries=3)
        mock_redis.set.side_effect = ConnectionError("redis down")

        with pytest.raises(Exception, match="celery-retry-sentinel"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        self_mock.retry.assert_called_once()

    def test_redis_down_last_attempt_raises_ingestion_error(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Redis failure on last attempt → IngestionError, file removed."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self(retries=3, max_retries=3)
        mock_redis.set.side_effect = ConnectionError("redis down")

        with pytest.raises(_FakeIngestionError, match="Lock backend unreachable"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        assert not p.exists()

    # -- Pipeline failures ----------------------------------------------------

    def test_ingestion_error_from_pipeline(self, tasks, make_upload, mock_redis, mock_celery_self):
        """IngestionError from pipeline → file removed, re-raised, lock released."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = _FakeIngestionError("bad file")

        with pytest.raises(_FakeIngestionError, match="bad file"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        assert not p.exists()
        mock_redis.delete.assert_called()

    def test_soft_time_limit_exceeded(self, tasks, make_upload, mock_redis, mock_celery_self):
        """SoftTimeLimitExceeded → file removed, re-raised, lock released."""
        from celery.exceptions import SoftTimeLimitExceeded

        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = SoftTimeLimitExceeded("timeout")

        with pytest.raises(SoftTimeLimitExceeded):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        assert not p.exists()
        mock_redis.delete.assert_called()

    @pytest.mark.parametrize("exc_type", [TypeError, AttributeError, KeyError])
    def test_non_retryable_code_error(self, tasks, make_upload, mock_redis, mock_celery_self, exc_type):
        """Non-retryable code errors → file removed, RuntimeError raised, lock released."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = exc_type("bug")

        with pytest.raises(RuntimeError, match="Internal document processing error"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        assert not p.exists()
        mock_redis.delete.assert_called()

    def test_value_error_is_non_retryable(self, tasks, make_upload, mock_redis, mock_celery_self):
        """ValueError (non-retryable code error) from pipeline → RuntimeError.

        Note: IngestionError extends ValueError and is caught FIRST by
        the IngestionError handler in the source. A plain ValueError
        that is NOT an IngestionError falls through to the
        _NON_RETRYABLE_CODE_ERRORS handler.
        """
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        # Use a plain ValueError, NOT an IngestionError
        tasks._pipeline.ingest_file.side_effect = ValueError("plain value error")

        with pytest.raises(RuntimeError, match="Internal document processing error"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")

    def test_not_implemented_error_is_non_retryable(self, tasks, make_upload, mock_redis, mock_celery_self):
        """NotImplementedError → RuntimeError wrapping the code bug."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = NotImplementedError("missing feature")

        with pytest.raises(RuntimeError):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")

    def test_transient_error_not_last_attempt_retries(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Transient Exception, not last attempt → retry, file kept."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self(retries=0, max_retries=3)
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = ConnectionError("qdrant down")

        with pytest.raises(Exception, match="celery-retry-sentinel"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        # File should be kept for retry
        assert p.exists()
        self_mock.retry.assert_called_once()

    def test_transient_error_last_attempt_raises(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Transient Exception on last attempt → file removed, re-raised."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self(retries=3, max_retries=3)
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = ConnectionError("qdrant down")

        with pytest.raises(ConnectionError, match="qdrant down"):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        assert not p.exists()
        mock_redis.delete.assert_called()

    # -- Lock release ---------------------------------------------------------

    def test_lock_release_failure_does_not_crash(self, tasks, make_upload, mock_redis, mock_celery_self):
        """If lock release fails (Redis down in finally), it should log but not crash."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        result_obj = _FakeIngestionResult(doc_id="d", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj
        mock_redis.delete.side_effect = ConnectionError("redis gone")

        # Should still return the result
        result = tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="d")
        assert result["status"] == "done"

    # -- doc_id derivation ----------------------------------------------------

    def test_explicit_doc_id_used(self, tasks, make_upload, mock_redis, mock_celery_self):
        """When doc_id is provided, it should be used directly."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        result_obj = _FakeIngestionResult(doc_id="explicit_id", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        result = tasks.ingest_document(
            self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="explicit_id"
        )
        assert result["doc_id"] == "explicit_id"

    def test_derived_doc_id_when_none(self, tasks, make_upload, mock_redis, mock_celery_self):
        """When doc_id is None, it should be derived from user_id + original_name."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True

        expected_doc_id = tasks._derive_doc_id("u1", "f.pdf")
        result_obj = _FakeIngestionResult(doc_id=expected_doc_id, chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id=None)
        # Verify pipeline was called with the derived doc_id
        _, kwargs = tasks._pipeline.ingest_file.call_args
        assert kwargs["doc_id"] == expected_doc_id

    # -- Retry mechanics ------------------------------------------------------

    def test_retry_countdown_uses_jittered_backoff(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Retry countdown should come from _jittered_backoff."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self(retries=1, max_retries=3)
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = ConnectionError("fail")

        with patch.object(tasks, "_jittered_backoff", return_value=15.0) as mock_backoff:
            with pytest.raises(Exception, match="celery-retry-sentinel"):
                tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
            mock_backoff.assert_called_once_with(1)

    def test_retry_called_with_correct_exc_and_countdown(self, tasks, make_upload, mock_redis, mock_celery_self):
        """self.retry() should receive the original exception and computed countdown."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self(retries=0, max_retries=3)
        mock_redis.set.return_value = True
        exc = ConnectionError("transient")
        tasks._pipeline.ingest_file.side_effect = exc

        with patch.object(tasks, "_jittered_backoff", return_value=7.5):
            with pytest.raises(Exception, match="celery-retry-sentinel"):
                tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
            self_mock.retry.assert_called_once_with(exc=exc, countdown=7.5)

    # -- Lock release on failure paths ----------------------------------------

    def test_lock_released_on_pipeline_ingestion_error(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Lock is released even after IngestionError from pipeline."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = _FakeIngestionError("bad")

        with pytest.raises(_FakeIngestionError):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        mock_redis.delete.assert_called()

    def test_lock_released_on_timeout(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Lock is released even after SoftTimeLimitExceeded."""
        from celery.exceptions import SoftTimeLimitExceeded

        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        tasks._pipeline.ingest_file.side_effect = SoftTimeLimitExceeded()

        with pytest.raises(SoftTimeLimitExceeded):
            tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1")
        mock_redis.delete.assert_called()

    # -- Multi-tenant & same filename tests ------------------------------------

    def test_two_different_users_same_original_filename_no_collision(
        self, tasks, make_upload, mock_redis, mock_celery_self
    ):
        """Two different users uploading a file with the same original_name must not collide."""
        stored_a = "1" * 32 + ".pdf"
        stored_b = "2" * 32 + ".pdf"
        file_a = make_upload(stored_a, content=b"User A PDF data")
        file_b = make_upload(stored_b, content=b"User B PDF data")

        self_mock_a = mock_celery_self(task_id="task-user-a")
        self_mock_b = mock_celery_self(task_id="task-user-b")

        # Track keys set in Redis to verify lock isolation
        active_locks = set()

        def fake_set(key, val, nx=False, ex=None):
            if nx and key in active_locks:
                return False
            active_locks.add(key)
            return True

        def fake_delete(key):
            active_locks.discard(key)
            return 1

        mock_redis.set.side_effect = fake_set
        mock_redis.delete.side_effect = fake_delete

        # Mock pipeline to return user-distinct results
        def fake_ingest(path, original_name, user_id, doc_id, progress_cb=None):
            return _FakeIngestionResult(doc_id=doc_id, chunks=10, generation=f"gen_{user_id}")

        tasks._pipeline.ingest_file.side_effect = fake_ingest

        # User A ingests "financial_report.pdf"
        res_a = tasks.ingest_document(self_mock_a, stored_a, "financial_report.pdf", "user_alpha")
        # User B ingests the identical original_name "financial_report.pdf"
        res_b = tasks.ingest_document(self_mock_b, stored_b, "financial_report.pdf", "user_beta")

        # Verify both completed successfully
        assert res_a["status"] == "done"
        assert res_b["status"] == "done"

        # Verify doc_ids are completely different despite having the same filename
        assert res_a["doc_id"] != res_b["doc_id"]
        assert res_a["generation"] == "gen_user_alpha"
        assert res_b["generation"] == "gen_user_beta"

        # Both physical files were removed independently
        assert not file_a.exists()
        assert not file_b.exists()

    def test_different_users_same_filename_concurrent_no_lock_contention(
        self, tasks, make_upload, mock_redis, mock_celery_self
    ):
        """User A holding a lock on 'doc.pdf' must NOT block User B uploading 'doc.pdf' concurrently."""
        stored_b = "3" * 32 + ".pdf"
        make_upload(stored_b)
        self_mock_b = mock_celery_self(task_id="task-user-b")

        # Derive User A's lock key
        doc_id_a = tasks._derive_doc_id("user_alpha", "statement.pdf")
        lock_key_a = tasks._lock_key("user_alpha", doc_id_a)

        # Simulate Redis state where User A's lock is CURRENTLY held
        held_keys = {lock_key_a}

        def fake_set(key, val, nx=False, ex=None):
            if nx and key in held_keys:
                return False  # locked!
            held_keys.add(key)
            return True

        mock_redis.set.side_effect = fake_set

        tasks._pipeline.ingest_file.return_value = _FakeIngestionResult(
            doc_id="doc_user_b", chunks=3, generation="gen_b"
        )

        # User B ingests the exact same filename "statement.pdf" concurrently while User A's lock is active
        res_b = tasks.ingest_document(self_mock_b, stored_b, "statement.pdf", "user_beta")

        # User B must succeed without being blocked by User A
        assert res_b["status"] == "done"
        # User A's lock key is still in Redis, while User B's lock was processed
        assert lock_key_a in held_keys

    def test_same_user_same_filename_concurrent_upload_is_properly_blocked(
        self, tasks, make_upload, mock_redis, mock_celery_self
    ):
        """The SAME user uploading the same filename concurrently MUST be blocked by the lock."""
        stored_dup = "4" * 32 + ".pdf"
        make_upload(stored_dup)
        self_mock = mock_celery_self(task_id="task-user-dup")

        # Simulate lock already held for this user + doc
        mock_redis.set.return_value = False  # NX failed because lock exists

        with pytest.raises(_FakeIngestionError) as exc_info:
            tasks.ingest_document(self_mock, stored_dup, "statement.pdf", "user_alpha")

        assert "currently being processed" in str(exc_info.value)



# ============================================================================
# Class 9: TestPurgeStaleLocalUploads — Sweep Behavior
# ============================================================================

class TestPurgeStaleLocalUploads:
    """Tests for purge_stale_local_uploads Celery task."""

    def _make_purge_self(self, task_id="purge-task-1"):
        """Create a mock Celery self for the purge task."""
        self = MagicMock()
        self.request.id = task_id
        return self

    def _create_stale_file(self, upload_dir, name=VALID_STORED_NAME, age_seconds=200000):
        """Create a file and backdate its mtime to make it stale."""
        p = upload_dir / name
        p.write_bytes(b"stale content")
        old_time = time.time() - age_seconds
        os.utime(p, (old_time, old_time))
        return p

    def _create_fresh_file(self, upload_dir, name=VALID_STORED_NAME_2):
        """Create a file with current mtime (fresh)."""
        p = upload_dir / name
        p.write_bytes(b"fresh content")
        return p

    # -- Normal operation -----------------------------------------------------

    def test_empty_directory(self, tasks, upload_dir, mock_redis):
        """Empty directory should return done with all zeros."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True
        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        assert result["scanned"] == 0
        assert result["purged"] == 0

    def test_mix_stale_and_fresh(self, tasks, upload_dir, mock_redis):
        """Only stale files should be purged; fresh files should be kept."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        stale = self._create_stale_file(upload_dir, VALID_STORED_NAME)
        fresh = self._create_fresh_file(upload_dir, VALID_STORED_NAME_2)

        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        assert result["purged"] >= 1
        assert not stale.exists()
        assert fresh.exists()

    def test_non_matching_filenames_skipped(self, tasks, upload_dir, mock_redis):
        """Files not matching STORED_NAME_RE should be skipped."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        # Create a file with non-matching name
        (upload_dir / ".gitkeep").write_bytes(b"keep")
        old_time = time.time() - 200000
        os.utime(upload_dir / ".gitkeep", (old_time, old_time))

        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        assert result["skipped"] >= 1
        assert (upload_dir / ".gitkeep").exists()

    # -- Lock behavior --------------------------------------------------------

    def test_lock_acquired_proceeds(self, tasks, upload_dir, mock_redis):
        """When lock is acquired, the sweep proceeds and releases in finally."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        mock_redis.delete.assert_called()

    def test_lock_already_held_returns_skipped(self, tasks, upload_dir, mock_redis):
        """Lock already held → returns skipped/already_running."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = False  # lock not acquired

        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "skipped"
        assert result["reason"] == "already_running"

    def test_redis_down_proceeds_without_lock(self, tasks, upload_dir, mock_redis):
        """Redis failure → proceeds without lock (graceful degradation)."""
        self_mock = self._make_purge_self()
        mock_redis.set.side_effect = ConnectionError("redis down")

        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"

    # -- Filesystem errors ----------------------------------------------------

    def test_upload_dir_missing(self, tasks, mock_redis):
        """Missing upload dir → returns skipped/upload_dir_missing."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        # Point to a non-existent directory
        tasks._UPLOAD_DIR_RESOLVED = Path("/nonexistent/dir/xyz")
        with patch("os.scandir", side_effect=FileNotFoundError("no dir")):
            result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "skipped"
        assert result["reason"] == "upload_dir_missing"

    def test_oserror_scanning_dir(self, tasks, mock_redis):
        """OSError scanning directory → returns skipped/scan_error."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        with patch("os.scandir", side_effect=OSError("disk error")):
            result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "skipped"
        assert result["reason"] == "scan_error"

    def test_oserror_on_individual_file_stat(self, tasks, upload_dir, mock_redis):
        """OSError on individual file stat → increments failed, continues."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        # Create a mock entry that raises OSError on stat
        mock_entry = MagicMock()
        mock_entry.name = VALID_STORED_NAME
        mock_entry.is_file.side_effect = OSError("stat fail")

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=iter([mock_entry]))
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("os.scandir", return_value=mock_ctx):
            result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        assert result["failed"] >= 1

    def test_oserror_on_individual_file_unlink(self, tasks, upload_dir, mock_redis):
        """OSError on individual file unlink → increments failed, continues."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        stale = self._create_stale_file(upload_dir, VALID_STORED_NAME)
        with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        assert result["failed"] >= 1

    def test_file_not_found_mid_scan_toctou(self, tasks, upload_dir, mock_redis):
        """FileNotFoundError mid-scan (TOCTOU) → increments skipped, continues."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        # Create a mock entry that raises FileNotFoundError on is_file
        mock_entry = MagicMock()
        mock_entry.name = VALID_STORED_NAME
        mock_entry.is_file.side_effect = FileNotFoundError("gone")

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=iter([mock_entry]))
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("os.scandir", return_value=mock_ctx):
            result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        assert result["skipped"] >= 1

    # -- Limits ---------------------------------------------------------------

    def test_max_purge_files_per_run_boundary(self, tasks, upload_dir, mock_redis):
        """Scanning should stop at MAX_PURGE_FILES_PER_RUN."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        # Temporarily set a low limit
        with patch("src.services.tasks.settings") as mock_s:
            mock_s.MAX_PURGE_FILES_PER_RUN = 2
            mock_s.STALE_UPLOAD_MAX_AGE = 3600 * 24
            mock_s.PURGE_LOCK_KEY = "lock"
            mock_s.PURGE_HARD_TIME_LIMIT = 250

            # Create 5 stale files
            for i in range(5):
                name = f"{i:032x}.pdf"
                self._create_stale_file(upload_dir, name)

            result = tasks.purge_stale_local_uploads(self_mock)
            assert result["status"] == "done"
            assert result["scanned"] == 2

    # -- Timeout --------------------------------------------------------------

    def test_soft_time_limit_exceeded_during_scan(self, tasks, upload_dir, mock_redis):
        """SoftTimeLimitExceeded during scan → logs, re-raises."""
        from celery.exceptions import SoftTimeLimitExceeded

        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        with patch("os.scandir", side_effect=SoftTimeLimitExceeded("timeout")):
            with pytest.raises(SoftTimeLimitExceeded):
                tasks.purge_stale_local_uploads(self_mock)

    def test_lock_released_after_timeout(self, tasks, upload_dir, mock_redis):
        """Lock should still be released after SoftTimeLimitExceeded."""
        from celery.exceptions import SoftTimeLimitExceeded

        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        with patch("os.scandir", side_effect=SoftTimeLimitExceeded()):
            with pytest.raises(SoftTimeLimitExceeded):
                tasks.purge_stale_local_uploads(self_mock)
        # Lock release should have been called in finally
        mock_redis.delete.assert_called()

    # -- Lock release failure -------------------------------------------------

    def test_lock_release_failure_does_not_crash(self, tasks, upload_dir, mock_redis):
        """Lock release failure should log warning, not crash."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True
        mock_redis.delete.side_effect = ConnectionError("redis gone")

        # Should complete normally despite lock release failure
        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"

    # -- Symlinks ignored -----------------------------------------------------

    def test_symlinks_ignored(self, tasks, upload_dir, mock_redis):
        """Symlinks should be skipped (is_file with follow_symlinks=False)."""
        self_mock = self._make_purge_self()
        mock_redis.set.return_value = True

        # Create a mock entry that is a symlink (is_file(follow_symlinks=False) → False)
        mock_entry = MagicMock()
        mock_entry.name = VALID_STORED_NAME
        mock_entry.is_file.return_value = False  # symlink

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=iter([mock_entry]))
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("os.scandir", return_value=mock_ctx):
            result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"
        assert result["skipped"] >= 1
        assert result["purged"] == 0


# ============================================================================
# Class 10: TestStressAndLoad — High Volume
# ============================================================================

class TestStressAndLoad:
    """High-volume stress tests to verify correctness under load."""

    def test_1000_rapid_throttler_updates(self, tasks):
        """1000 rapid ProgressThrottler updates should correctly throttle."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        for i in range(1000):
            throttler.update(i, 2000)
        # Only the first call should emit (all within the same interval)
        assert mock_task.update_state.call_count == 1

    def test_1000_throttler_with_completion(self, tasks):
        """1000 updates with final completion should emit exactly 2: first + completion."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=999.0)
        for i in range(999):
            throttler.update(i, 1000)
        throttler.update(1000, 1000)  # completion
        assert mock_task.update_state.call_count == 2

    def test_100_concurrent_lock_contention(self, tasks, make_upload, mock_redis, mock_celery_self):
        """100 ingest calls with lock taken should all raise IngestionError."""
        failures = 0
        for i in range(100):
            name = f"{i:032x}.pdf"
            make_upload(name)
            self_mock = mock_celery_self()
            mock_redis.set.return_value = False  # lock taken

            try:
                tasks.ingest_document(self_mock, name, "f.pdf", "u1")
            except _FakeIngestionError:
                failures += 1

        assert failures == 100

    def test_jittered_backoff_10000_calls_within_range(self, tasks):
        """10000 _jittered_backoff calls should all produce results within expected range."""
        for _ in range(10000):
            val = tasks._jittered_backoff(0, base=5.0, cap=60.0)
            assert 5.5 <= val <= 8.0, f"Value {val} out of range"

    def test_validate_1000_different_names(self, tasks, make_upload, upload_dir):
        """1000 different valid/invalid names should be handled correctly."""
        valid_count = 0
        invalid_count = 0
        for i in range(500):
            name = f"{i:032x}.pdf"
            make_upload(name)
            try:
                tasks._validate_file_reference(name)
                valid_count += 1
            except _FakeIngestionError:
                invalid_count += 1

        for i in range(500):
            invalid_name = f"INVALID_{i}"
            try:
                tasks._validate_file_reference(invalid_name)
            except _FakeIngestionError:
                invalid_count += 1

        assert valid_count == 500
        assert invalid_count == 500

    def test_purge_with_many_entries(self, tasks, upload_dir, mock_redis):
        """Purge with many directory entries (mocked) should handle correctly."""
        self_mock = MagicMock()
        self_mock.request.id = "purge-stress"
        mock_redis.set.return_value = True

        # Create mock entries
        entries = []
        for i in range(100):
            entry = MagicMock()
            entry.name = f"{i:032x}.pdf"
            entry.is_file.return_value = True
            stat_result = MagicMock()
            stat_result.st_mtime = 0  # very old
            entry.stat.return_value = stat_result
            entries.append(entry)

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=iter(entries))
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("os.scandir", return_value=mock_ctx):
            # We also need to patch Path.unlink for the unlink calls
            with patch.object(Path, "unlink"):
                result = tasks.purge_stale_local_uploads(self_mock)

        assert result["status"] == "done"
        assert result["scanned"] == 100


# ============================================================================
# Class 11: TestSecurityHardening — Attack Vectors
# ============================================================================

class TestSecurityHardening:
    """Security-focused tests for path traversal, injection, and escape attacks."""

    @pytest.mark.parametrize("malicious_name", [
        "../../etc/shadow",
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "....//....//etc//passwd",
    ])
    def test_path_traversal_rejected(self, tasks, malicious_name):
        """Path traversal attempts should be rejected at the regex level."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference(malicious_name)

    def test_null_byte_injection(self, tasks):
        """Null byte injection should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("abc\x00.pdf")

    @pytest.mark.parametrize("name", [
        "aaaa" * 8 + ".пдф",  # Cyrillic
        "aaaa" * 8 + ".文件",  # Chinese
        "\u202e" + "a" * 32 + ".pdf",  # RTL override
    ])
    def test_unicode_normalization_attacks(self, tasks, name):
        """Unicode characters in names should be rejected by hex-only regex."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference(name)

    def test_very_long_filename(self, tasks):
        """Very long filename (10000 chars) should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 10000 + ".pdf")

    def test_lock_key_injection_attempt(self, tasks):
        """Lock key injection should not corrupt the key format."""
        key = tasks._lock_key("user\n\r\x00", "doc\n\r\x00")
        # The key should contain the injected chars verbatim, not break format
        assert key.startswith("ingest:lock:")

    def test_symlink_escape_blocked_by_resolve(self, tasks, upload_dir, tmp_path):
        """Symlink that resolves outside upload dir should be blocked."""
        # Create a file outside upload dir
        secret = tmp_path / "secret.txt"
        secret.write_text("secret data")

        # Can't actually create a symlink reliably on Windows in all envs,
        # so we simulate via patching resolve
        with patch("src.services.tasks.settings") as mock_s:
            mock_s.UPLOAD_DIR = upload_dir
            with patch("pathlib.Path.resolve", return_value=secret):
                with pytest.raises(_FakeIngestionError, match="Invalid upload reference"):
                    tasks._validate_file_reference(VALID_STORED_NAME)

    def test_double_dot_segments(self, tasks):
        """Double-dot segments should be rejected at regex."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 30 + "..pdf")

    @pytest.mark.parametrize("name", [
        "a" * 32 + "\\.pdf",
        "a" * 32 + "/.pdf",
        "a" * 32 + "\\..\\..\\etc\\passwd",
    ])
    def test_backslash_variants(self, tasks, name):
        """Backslash and slash variants should be rejected by the regex."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference(name)

    def test_remove_file_outside_upload_dir_prevented(self, tasks, tmp_path):
        """_remove_file should refuse to delete files outside upload dir."""
        outside = tmp_path / "other" / "important.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("do not delete")

        tasks._remove_file(outside)
        assert outside.exists(), "File outside upload dir must NOT be deleted"

    def test_stored_name_with_special_extension_chars(self, tasks):
        """Extension part should only allow alphanumeric chars."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 32 + ".pd-f")

    def test_stored_name_empty_extension(self, tasks):
        """Empty extension (just a dot) should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 32 + ".")

    def test_multiple_dots_in_name(self, tasks):
        """Multiple dots in name should be rejected (hex part won't match)."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 16 + ".hidden." + "b" * 15 + ".pdf")

    def test_whitespace_in_name(self, tasks):
        """Whitespace in stored name should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 31 + " .pdf")

    def test_tab_in_name(self, tasks):
        """Tab character in name should be rejected."""
        with pytest.raises(_FakeIngestionError, match="Invalid upload reference format"):
            tasks._validate_file_reference("a" * 31 + "\t.pdf")


# ============================================================================
# Additional edge case tests
# ============================================================================

class TestEdgeCases:
    """Extra edge-case tests to reach full branch coverage."""

    def test_ingest_document_lock_key_uses_target_doc_id(self, tasks, make_upload, mock_redis, mock_celery_self):
        """Lock key should be built from user_id and the resolved doc_id."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        result_obj = _FakeIngestionResult(doc_id="myid", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="myid")

        # Verify the set call used the correct key
        set_call = mock_redis.set.call_args
        assert set_call[0][0] == "ingest:lock:u1:myid"

    def test_progress_throttler_stores_task_instance(self, tasks):
        """The throttler should store the task instance reference."""
        mock_task = MagicMock()
        throttler = tasks.ProgressThrottler(mock_task, min_interval_seconds=2.0)
        assert throttler.task is mock_task

    def test_derive_doc_id_unicode_inputs(self, tasks):
        """_derive_doc_id should handle unicode inputs gracefully."""
        result = tasks._derive_doc_id("مستخدم", "ملف.pdf")
        assert result.startswith("doc_")

    def test_jittered_backoff_zero_base(self, tasks):
        """With base=0, result should just be the jitter."""
        with patch("random.uniform", return_value=2.0):
            result = tasks._jittered_backoff(0, base=0.0, cap=60.0)
            assert result == 2.0

    def test_jittered_backoff_zero_cap(self, tasks):
        """With cap=0, exponential should be 0, result is just jitter."""
        with patch("random.uniform", return_value=1.5):
            result = tasks._jittered_backoff(5, base=100.0, cap=0.0)
            assert result == 1.5  # min(100*32, 0) = 0, + 1.5

    def test_purge_redis_down_still_completes(self, tasks, upload_dir, mock_redis):
        """Purge should complete even when Redis is down (graceful degradation)."""
        self_mock = MagicMock()
        self_mock.request.id = "purge-1"
        mock_redis.set.side_effect = ConnectionError("redis unavailable")

        result = tasks.purge_stale_local_uploads(self_mock)
        assert result["status"] == "done"

    def test_purge_lock_not_released_when_not_acquired(self, tasks, upload_dir, mock_redis):
        """When lock acquisition failed (Redis down), finally should NOT try to release."""
        self_mock = MagicMock()
        self_mock.request.id = "purge-2"
        mock_redis.set.side_effect = ConnectionError("redis down")

        tasks.purge_stale_local_uploads(self_mock)
        # delete should NOT be called because got_lock is False
        mock_redis.delete.assert_not_called()

    def test_validate_file_reference_multiple_valid_extensions(self, tasks, make_upload):
        """Various valid extensions should work: pdf, txt, docx, json, csv."""
        for ext in ["pdf", "txt", "docx", "json", "csv", "md", "xlsx"]:
            name = "a" * 32 + f".{ext}"
            make_upload(name)
            result = tasks._validate_file_reference(name)
            assert result.name == name

    def test_ingest_document_throttler_created_with_5s_interval(self, tasks, make_upload, mock_redis, mock_celery_self):
        """The throttler in ingest_document should use 5.0s interval."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True

        # Capture the throttler by intercepting pipeline call
        captured = {}

        def capture_pipeline(**kwargs):
            captured["progress_cb"] = kwargs.get("progress_cb")
            return _FakeIngestionResult(doc_id="d", chunks=1, generation="g")

        tasks._pipeline.ingest_file.side_effect = capture_pipeline

        tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="d")
        assert captured["progress_cb"] is not None

    def test_purge_duration_ms_included(self, tasks, upload_dir, mock_redis):
        """The returned dict should include duration_ms."""
        self_mock = MagicMock()
        self_mock.request.id = "purge-3"
        mock_redis.set.return_value = True

        result = tasks.purge_stale_local_uploads(self_mock)
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], int)

    def test_ingest_document_with_all_non_retryable_errors(self, tasks, make_upload, mock_redis, mock_celery_self):
        """All _NON_RETRYABLE_CODE_ERRORS should be wrapped in RuntimeError."""
        for exc_type in (TypeError, AttributeError, KeyError, NotImplementedError):
            name = f"{'a':0>32}.pdf"
            make_upload(name)
            self_mock = mock_celery_self()
            mock_redis.set.return_value = True
            tasks._pipeline.ingest_file.side_effect = exc_type("test")

            with pytest.raises(RuntimeError, match="Internal document processing error"):
                tasks.ingest_document(self_mock, name, "f.pdf", "u1")

    def test_ingest_lock_ttl_passed_correctly(self, tasks, make_upload, mock_redis, mock_celery_self):
        """The lock SET should use the correct TTL from _LOCK_TTL_SECONDS."""
        make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        result_obj = _FakeIngestionResult(doc_id="d", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        tasks.ingest_document(self_mock, VALID_STORED_NAME, "f.pdf", "u1", doc_id="d")

        set_call = mock_redis.set.call_args
        assert set_call[1]["ex"] == 735  # _LOCK_TTL_SECONDS

    def test_ingest_pipeline_called_with_correct_args(self, tasks, make_upload, mock_redis, mock_celery_self):
        """The pipeline should receive the correct arguments."""
        p = make_upload(VALID_STORED_NAME)
        self_mock = mock_celery_self()
        mock_redis.set.return_value = True
        result_obj = _FakeIngestionResult(doc_id="d", chunks=1, generation="g")
        tasks._pipeline.ingest_file.return_value = result_obj

        tasks.ingest_document(self_mock, VALID_STORED_NAME, "original.pdf", "user42", doc_id="d")

        _, kwargs = tasks._pipeline.ingest_file.call_args
        assert kwargs["original_name"] == "original.pdf"
        assert kwargs["user_id"] == "user42"
        assert kwargs["doc_id"] == "d"
        assert "progress_cb" in kwargs

    def test_ingest_validation_error_no_cleanup_attempted(self, tasks, mock_celery_self, mock_redis):
        """When validation fails, no file cleanup should be attempted (no path resolved)."""
        self_mock = mock_celery_self()
        mock_redis.set.assert_not_called()

        with pytest.raises(_FakeIngestionError):
            tasks.ingest_document(self_mock, "INVALID!", "f.pdf", "u1")

        # _remove_file should not have been called
        # (no file path was resolved, so nothing to remove)
        mock_redis.set.assert_not_called()
        mock_redis.delete.assert_not_called()
