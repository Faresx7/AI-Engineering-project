"""
tests/unit/test_cache.py
========================
Unit tests for src.core.cache.MessageCache

REAL-LIFE CONDITIONS SIMULATED:
  1. Normal message deduplication (Duplicate delivery from Meta)
  2. LRU eviction when the cache fills to capacity (memory pressure)
  3. get() promotes an element to MRU position (access-order verification)
  4. Concurrent duplicate insertions (race-condition guard from two threads)
  5. Boundary: max_size=1 (extreme-small cache)
  6. Large-volume stress: 10 000 sequential insertions
  7. None / empty key guards (defensive usage)
"""

import threading

import pytest

from src.core.cache import MessageCache


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fill_cache(cache: MessageCache, n: int, prefix: str = "msg") -> list[str]:
    """Insert n unique items and return the list of keys."""
    keys = [f"{prefix}_{i}" for i in range(n)]
    for key in keys:
        cache.set_if_absent(key, f"text_{key}")
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# Basic set_if_absent behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestSetIfAbsent:

    def test_returns_true_on_first_insertion(self):
        """New keys must be accepted and True returned."""
        cache = MessageCache(max_size=100)
        result = cache.set_if_absent("mid_001", "Hello world")
        assert result is True

    def test_returns_false_on_duplicate(self):
        """
        REAL-LIFE: Meta sometimes delivers the same webhook event twice
        (within milliseconds). The cache must silently reject duplicates.
        """
        cache = MessageCache(max_size=100)
        cache.set_if_absent("mid_001", "Hello")
        result = cache.set_if_absent("mid_001", "Hello again")
        assert result is False

    def test_different_keys_are_both_accepted(self):
        """Two distinct message IDs must both be stored."""
        cache = MessageCache(max_size=100)
        assert cache.set_if_absent("mid_A", "msg A") is True
        assert cache.set_if_absent("mid_B", "msg B") is True

    def test_value_is_stored_correctly(self):
        """The value inserted via set_if_absent must be retrievable via get()."""
        cache = MessageCache(max_size=100)
        cache.set_if_absent("mid_XYZ", "Stored text")
        assert cache.get("mid_XYZ") == "Stored text"


# ─────────────────────────────────────────────────────────────────────────────
# LRU eviction policy
# ─────────────────────────────────────────────────────────────────────────────

class TestEviction:

    def test_oldest_entry_is_evicted_when_full(self):
        """
        REAL-LIFE: Under sustained high-volume traffic the cache fills up.
        The oldest (least-recently-used) entry must be silently dropped to
        keep memory bounded.
        """
        cache = MessageCache(max_size=3)
        cache.set_if_absent("A", "a")
        cache.set_if_absent("B", "b")
        cache.set_if_absent("C", "c")
        # D pushes A out
        cache.set_if_absent("D", "d")

        assert cache.get("A") is None          # evicted
        assert cache.get("B") == "b"           # still present
        assert cache.get("D") == "d"           # newest

    def test_max_size_never_exceeded(self):
        """
        REAL-LIFE STRESS: Insert 10 000 items into a max_size=5000 cache.
        Internal storage must never exceed 5 000 entries.
        """
        cache = MessageCache(max_size=5000)
        for i in range(10_000):
            cache.set_if_absent(f"key_{i}", f"val_{i}")
        # OrderedDict len = exact size
        assert len(cache._cache) <= 5000

    def test_single_slot_cache_keeps_only_latest(self):
        """
        BOUNDARY: max_size=1 means every new entry evicts the previous one.
        """
        cache = MessageCache(max_size=1)
        cache.set_if_absent("first", "value_1")
        cache.set_if_absent("second", "value_2")

        assert cache.get("first") is None
        assert cache.get("second") == "value_2"


# ─────────────────────────────────────────────────────────────────────────────
# get() behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestGet:

    def test_get_missing_key_returns_none(self):
        """Looking up a key that was never inserted must return None."""
        cache = MessageCache()
        assert cache.get("non_existent_mid") is None

    def test_get_promotes_item_to_mru(self):
        """
        REAL-LIFE: A user replies to a bot message. The reply-context lookup
        via get() must promote that item so it survives subsequent evictions.
        """
        cache = MessageCache(max_size=3)
        cache.set_if_absent("A", "a")
        cache.set_if_absent("B", "b")
        cache.set_if_absent("C", "c")

        # Access A — promotes it to MRU
        cache.get("A")

        # D evicts B (now the least-recently-used)
        cache.set_if_absent("D", "d")

        assert cache.get("B") is None   # B was evicted
        assert cache.get("A") == "a"    # A survived because it was accessed

    def test_repeated_get_does_not_alter_value(self):
        """get() is read-only; multiple reads must return the same value."""
        cache = MessageCache()
        cache.set_if_absent("mid", "hello")
        assert cache.get("mid") == "hello"
        assert cache.get("mid") == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safety (race condition simulation)
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_concurrent_inserts_with_same_key_accept_only_once(self):
        """
        REAL-LIFE RACE CONDITION: Two goroutines (or asyncio tasks) receive
        the identical message simultaneously.  Only ONE should be processed.
        The result list must contain exactly one True.
        """
        cache = MessageCache(max_size=1000)
        results = []
        lock = threading.Lock()

        def insert():
            outcome = cache.set_if_absent("shared_mid", "shared_text")
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=insert) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one insertion must have succeeded
        assert results.count(True) == 1
        assert results.count(False) == 19

    def test_concurrent_inserts_with_unique_keys_all_succeed(self):
        """
        REAL-LIFE: Burst of 100 simultaneous unique messages.
        Every unique key must be accepted exactly once.
        """
        cache = MessageCache(max_size=1000)
        results = []
        lock = threading.Lock()

        def insert(i):
            outcome = cache.set_if_absent(f"key_{i}", f"val_{i}")
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=insert, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is True for r in results)
        assert len(cache._cache) == 100


# ─────────────────────────────────────────────────────────────────────────────
# Edge: unusual inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeInputs:

    def test_empty_string_key_is_stored(self):
        """Empty-string keys are unusual but must not crash the cache."""
        cache = MessageCache()
        assert cache.set_if_absent("", "some_value") is True
        assert cache.get("") == "some_value"

    def test_unicode_key_and_value(self):
        """Arabic / Unicode text must round-trip through the cache correctly."""
        cache = MessageCache()
        cache.set_if_absent("mid_arabic", "مرحباً بالعالم")
        assert cache.get("mid_arabic") == "مرحباً بالعالم"

    def test_very_long_key(self):
        """Keys the length of a real wamid (64 chars) must be accepted."""
        long_key = "wamid." + "a" * 58
        cache = MessageCache()
        cache.set_if_absent(long_key, "long key message")
        assert cache.get(long_key) == "long key message"
