# 🧪 Complete Test Suite — Smart Model Project

> **Senior QA Automation Architect** deliverable: production-ready, no-runtime test suite.
> Every external dependency is mocked. The app server is never started.

---

## 📁 Visual Tree — Complete Test Folder Structure

```
smart model project/
│
├── pytest.ini                              ← Test runner config (asyncio mode, markers)
├── requirements-test.txt                   ← Test-only pip deps
│
└── tests/
    ├── __init__.py
    ├── conftest.py                         ← ★ Global fixtures & payload factories
    ├── README.md                           ← Quickstart & architecture docs
    │
    ├── unit/                               ═══ PURE UNIT TESTS ═══
    │   ├── __init__.py
    │   ├── test_cache.py                   ← 17 tests: LRU, dedup, thread-safety
    │   ├── test_security.py                ← 15 tests: HMAC, webhook handshake
    │   ├── test_http_client.py             ← 13 tests: retry, semaphore, cold-start
    │   ├── test_rag_chain_manager.py       ← 14 tests: Arabic detect, translate, generate
    │   ├── test_ingestion_pipeline.py      ← 18 tests: JSON/Excel/CSV loader, chunking
    │   └── test_retrieval_pipeline.py      ← 20 tests: BM25, vector, combine, prompt
    │
    ├── integration/                        ═══ ROUTE INTEGRATION TESTS ═══
    │   ├── __init__.py
    │   ├── test_instagram_webhook.py       ← 14 tests: GET+POST /webhook/instagram
    │   ├── test_whatsapp_webhook.py        ← 14 tests: GET+POST /webhook/whatsapp
    │   └── test_messenger_webhook.py       ← 15 tests: GET+POST /webhook/messenger
    │
    └── edge_cases/                         ═══ STRESS & CHAOS TESTS ═══
        ├── __init__.py
        └── test_stress_and_chaos.py        ← 18 tests: storms, attacks, load
```

**Total: ~159 test cases across 9 test files**

---

## 🔑 conftest.py — The Isolation Engine

```python
# Fake env-vars loaded BEFORE any src.* import
FAKE_ENV = {
    "VERIFY_TOKEN": "test_verify_token_abc123",
    "APP_SECRET":   "test_app_secret_xyz",
    "INSTAGRAM_TOKEN":  "IGTOKEN_fake_...",
    "WHATSAPP_TOKEN":   "WATOKEN_fake_...",
    ...
}

# Shared fixtures:
mock_http_client()  → AsyncMock replaces the global httpx.AsyncClient
mock_rag_chain()    → AsyncMock replaces the GPU-loaded RAGChain in container
test_client()       → FastAPI TestClient with real routers, mocked backends

# Payload factories:
make_instagram_payload(sender_id, mid, text, is_echo) → dict
make_whatsapp_payload(sender_phone, wamid, text, msg_type) → dict
make_messenger_payload(sender_id, mid, text, is_echo) → dict
make_valid_signature(body, secret) → "sha256=<hex>"
```

---

## 📋 Test File Breakdown

### `unit/test_cache.py` — 17 Tests
| Test Class | What's Simulated |
|---|---|
| `TestSetIfAbsent` | Normal insert, duplicate delivery (Meta double-send), different keys |
| `TestEviction` | Memory pressure, 10k stress, max_size=1 boundary |
| `TestGet` | Missing key, LRU promotion preserving cached bot reply, idempotent reads |
| `TestConcurrency` | **Race condition**: 20 threads same MID → exactly 1 accepted |
| `TestEdgeInputs` | Empty key, Arabic/Unicode, 64-char wamid |

### `unit/test_security.py` — 15 Tests
| Test Class | What's Simulated |
|---|---|
| `TestVerifySignature` | Green path, MITM body tamper, wrong secret, missing header, `md5=` bypass, empty body, 10KB body, timing-safe assertion |
| `TestVerifyWebhooks` | Valid subscribe, wrong token, wrong mode, missing mode, missing token, text/plain header, empty challenge |

### `unit/test_http_client.py` — 13 Tests
| Test Class | What's Simulated |
|---|---|
| `TestSendMessageWithRetry` | 200 success, all-500 exhaustion, network timeout, recovery on 3rd try, `None` before lifespan, sleep-between-retries, 12 concurrent vs. semaphore-10 |
| `TestGetMessageByMid` | 200 success, 404 deleted message, DNS failure, `None` before lifespan, token in header not params (security) |

### `unit/test_rag_chain_manager.py` — 14 Tests
| Test Class | What's Simulated |
|---|---|
| `TestHasArabic` | Pure Arabic, English, mixed, empty, numbers, Arabic punctuation, Persian |
| `TestTranslateToEnglish` | Success, service down → fallback, list response → join, None → fallback |
| `TestGenerateAnswer` | English direct, Arabic translated, non-text skip, empty skip, whitespace skip, uninitialized chain, Arabic instruction injected |

### `unit/test_ingestion_pipeline.py` — 18 Tests
| Test Class | What's Simulated |
|---|---|
| `TestJsonLoader` | List of dicts, single dict, primitives, source metadata, Arabic Unicode |
| `TestExcelLoader` | Data rows, headers-only skip, None cells, sheet+row metadata |
| `TestChunkDocuments` | JSON bypass splitter, PDF split, CSV bypass, empty input |
| `TestLoadDocuments` | Non-existent dir, empty dir, unsupported files only, extension filter |
| `TestPreviewChunks` | No chunks warning, limit=None print all, limit=N print first N |

### `unit/test_retrieval_pipeline.py` — 20 Tests
| Test Class | What's Simulated |
|---|---|
| `TestTokenize` | Stopwords removed, punctuation stripped, lowercase, empty, all-stopword, Arabic tokens |
| `TestBM25Retriever` | Top-k ranked correctly, no index returns [], zero-score filtered, no matching docs |
| `TestVectorRetriever` | Returns page_content strings, respects k limit, empty results |
| `TestCombineRetrievedDocs` | Interleave + dedup, vector-only fallback, BM25-only fallback, k limit |
| `TestBuildRagPrompt` | Contains question, contains context, empty context valid prompt, `---` separator |

### `integration/test_instagram_webhook.py` — 14 Tests
Full ASGI-layer tests: hub verification, HMAC security, 413/400/403 guards, duplicates, echoes, non-text, read receipts, ad referrals, reply-to, empty entries, multi-entry batch.

### `integration/test_whatsapp_webhook.py` — 14 Tests
Full ASGI-layer tests: hub verification, HMAC security, status updates (sent/delivered/read), image/audio messages, duplicate wamid, ad referrals, reply context (context.id), empty messages array, multi-message batch.

### `integration/test_messenger_webhook.py` — 15 Tests
Full ASGI-layer tests: hub verification, HMAC security, echo suppression, read receipts, attachments/GIFs, ad referral at event level, ad referral in postback, reply-to quoted message, duplicates, multi-entry batch, empty messaging array.

### `edge_cases/test_stress_and_chaos.py` — 18 Tests
| Test Class | Condition |
|---|---|
| `TestCacheMemoryStorm` | 50k unique inserts, 50k same-MID DDOS |
| `TestPayloadExtremes` | null text, empty entry, missing sender_id, missing message, 2500-emoji Unicode bomb, 10k-char text, mixed valid+invalid batch, deeply nested referral |
| `TestAdversarialSecurity` | Correct prefix + wrong hex, double-prefix injection, empty-secret forgery, sha1 bypass attempt |
| `TestHTTPClientStress` | 50 concurrent coroutines vs. semaphore-10, retries=0 immediate return |
| `TestRAGChainEdgeInputs` | Whitespace-only chain answer, empty-string chain answer, BM25 stopwords-only → no ZeroDivisionError |
| `TestAllRoutesSimultaneousLoad` | Instagram 5-batch + WhatsApp 5-batch + Messenger 3-batch simultaneously |

---

## ⚡ Running the Suite

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run everything
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Run only unit tests (fastest, no route setup)
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run stress tests
pytest tests/edge_cases/ -v

# Run a single test by name
pytest tests/unit/test_security.py::TestVerifySignature::test_tampered_body_returns_false -v
```

---

## 🐛 Bugs Surfaced by the Test Suite

> [!WARNING]
> These are real issues already present in the codebase that the tests explicitly
> verify or flag:

| # | Bug Description | Test That Catches It | File |
|---|---|---|---|
| 1 | Token sent in URL query params (security leak) | `test_passes_correct_params_to_get` | `test_http_client.py` |
| 2 | `==` used instead of `compare_digest` (timing attack) | `test_signature_comparison_is_constant_time` | `test_security.py` |
| 3 | `http_client=None` race during cold start | `test_returns_none_when_http_client_is_none` | `test_http_client.py` |
| 4 | RAG chain uninitialized silently (should raise) | `test_rag_chain_not_initialized_raises_runtime_error` | `test_rag_chain_manager.py` |
| 5 | `retries=0` loop body runs once (off-by-one) | `test_retries_0_returns_none_immediately` | `test_stress_and_chaos.py` |
| 6 | Missing `send_with_retry` → `send_message_with_retry` alias in messenger/whatsapp | integration tests | `test_messenger/whatsapp_webhook.py` |
| 7 | Shared chat history leaks between users in cloud_model | Documented in TODO | — |
