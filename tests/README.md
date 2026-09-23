# 🧪 Test Suite — Smart Model Project

Production-ready, no-runtime test suite for the **Smart Model** FastAPI chatbot
(Instagram · WhatsApp · Messenger webhooks + RAG pipeline).

---

## 📁 Folder Structure

```
tests/
├── conftest.py                          ← Global fixtures, factories, env injection
│
├── unit/                                ← Pure isolated logic tests (no I/O)
│   ├── __init__.py
│   ├── test_cache.py                    ← MessageCache: LRU, dedup, thread-safety
│   ├── test_security.py                 ← verify_signature(), verify_webhooks()
│   ├── test_http_client.py              ← send_message_with_retry(), get_message_by_mid()
│   ├── test_rag_chain_manager.py        ← has_arabic(), translate_to_english(), generate_answer()
│   ├── test_ingestion_pipeline.py       ← JSON/Excel/CSV loaders, chunking, preview
│   └── test_retrieval_pipeline.py       ← Tokenize, BM25, vector, combine, RAG prompt
│
├── integration/                         ← FastAPI TestClient route tests
│   ├── __init__.py
│   ├── test_instagram_webhook.py        ← GET/POST /webhook/instagram (12 scenarios)
│   ├── test_whatsapp_webhook.py         ← GET/POST /webhook/whatsapp  (13 scenarios)
│   └── test_messenger_webhook.py        ← GET/POST /webhook/messenger  (14 scenarios)
│
└── edge_cases/                          ← Stress, chaos, adversarial tests
    ├── __init__.py
    └── test_stress_and_chaos.py         ← 50k cache storms, DDOS, Unicode bombs, etc.
```

---

## ⚡ Quick Start

```bash
# 1. Install test dependencies (one-time)
pip install -r requirements-test.txt

# 2. Run the full suite
pytest

# 3. Run only unit tests
pytest tests/unit/ -v

# 4. Run only integration tests
pytest tests/integration/ -v

# 5. Run stress / edge-case tests
pytest tests/edge_cases/ -v

# 6. Run with coverage report
pytest --cov=src --cov-report=term-missing

# 7. Run a specific file
pytest tests/unit/test_cache.py -v

# 8. Run a specific test class
pytest tests/unit/test_security.py::TestVerifySignature -v
```

---

## 🏗️ Architecture Decisions

### No-Runtime Isolation
The entire suite runs **without** launching uvicorn, loading HuggingFace models,
connecting to Chroma DB, or calling any real Meta API:

| Real Component | Test Replacement |
|---|---|
| `uvicorn` server | `fastapi.testclient.TestClient` |
| `httpx.AsyncClient` | `AsyncMock` via `mock_http_client` fixture |
| `RAGChain` (GPU model) | `AsyncMock` via `mock_rag_chain` fixture |
| `HuggingFaceEmbeddings` | `unittest.mock.patch` |
| `Chroma` vector DB | `unittest.mock.patch` |
| `MyMemoryTranslator` | `unittest.mock.patch` |
| `.env` secrets | `os.environ.setdefault(...)` in `conftest.py` |
| Meta Graph API | `mock_http_client.post/get.return_value` |

### Real-Life Conditions Simulated

| Category | Test File | Scenarios |
|---|---|---|
| **Security** | `test_security.py` | HMAC tampering, timing attacks, wrong prefix, sha1 bypass |
| **Deduplication** | `test_cache.py` | Duplicate delivery, concurrent race condition |
| **Memory Pressure** | `test_stress_and_chaos.py` | 50k inserts, LRU eviction |
| **Network Failures** | `test_http_client.py` | Timeout, 500 errors, retry recovery |
| **Payload Attacks** | `test_stress_and_chaos.py` | Unicode bombs, oversized payloads, malformed JSON |
| **Cold Start** | `test_http_client.py` | `http_client=None` before lifespan completes |
| **RAG Pipeline** | `test_retrieval_pipeline.py` | BM25 with stopwords only, empty context |
| **Translation** | `test_rag_chain_manager.py` | API down, list response, Arabic detection |
| **Ingestion** | `test_ingestion_pipeline.py` | Missing dir, empty dir, unsupported files |

---

## 📊 Coverage Map

| Module | Covered By |
|---|---|
| `src/core/cache.py` | `unit/test_cache.py` |
| `src/core/security.py` | `unit/test_security.py` |
| `src/core/http_client.py` | `unit/test_http_client.py` |
| `src/core/config.py` | `conftest.py` (env injection) |
| `src/core/container.py` | `unit/test_rag_chain_manager.py` |
| `src/services/rag_chain_manager.py` | `unit/test_rag_chain_manager.py` |
| `src/rag/ingestion_pipeline.py` | `unit/test_ingestion_pipeline.py` |
| `src/rag/retrieval_pipeline.py` | `unit/test_retrieval_pipeline.py` |
| `src/rag/rag_chain.py` | `edge_cases/test_stress_and_chaos.py` |
| `src/routes/instagram.py` | `integration/test_instagram_webhook.py` |
| `src/routes/whatsapp.py` | `integration/test_whatsapp_webhook.py` |
| `src/routes/messenger.py` | `integration/test_messenger_webhook.py` |
| `main.py` | `conftest.py` (router assembly) |

---

## 🐛 Known Bugs Caught by Tests

| Bug | Test | File |
|---|---|---|
| Token sent in query params (security leak) | `test_passes_correct_params_to_get` | `test_http_client.py` |
| Signature using `==` instead of `compare_digest` | `test_signature_comparison_is_constant_time` | `test_security.py` |
| Missing `retries=0` immediate-return | `test_retries_0_returns_none_immediately` | `test_stress_and_chaos.py` |
| RAG chain cold-start RuntimeError | `test_rag_chain_not_initialized_raises_runtime_error` | `test_rag_chain_manager.py` |
| `http_client=None` before lifespan | `test_returns_none_when_http_client_is_none` | `test_http_client.py` |
