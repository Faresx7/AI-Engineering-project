from unittest.mock import AsyncMock, patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import jwt
from src.routes.main_route import router as main_router
from src.routes.ingestion import router as ingestion_router
from src.core.config import settings

test_app = FastAPI()
test_app.include_router(main_router)
test_app.include_router(ingestion_router)


@pytest.fixture
def client():
    with TestClient(test_app) as c:
        yield c


def test_index_page_returns_html(client):
    """GET / should serve the upload dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "منظومة رفع ومعالجة المستندات" in response.text
    assert "dropzone" in response.text


def test_upload_page_returns_html(client):
    """GET /upload should also serve the upload dashboard HTML."""
    response = client.get("/upload")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "بدء الرفع والمعالجة" in response.text


def test_auth_token_generator(client):
    """GET /auth/token should generate a valid, decodable JWT token."""
    response = client.get("/auth/token?user_id=test_engineer")
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user_id"] == "test_engineer"

    # Decode using JWT_SECRET_KEY to verify signature
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    decoded = jwt.decode(data["access_token"], secret, algorithms=["HS256"])
    assert decoded["sub"] == "test_engineer"


def test_api_config_endpoint(client):
    """GET /api/config should expose allowed file extensions and max upload size."""
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert ".pdf" in data["allowed_extensions"]
    assert ".docx" in data["allowed_extensions"]
    assert data["max_upload_mb"] > 0
    assert "default_api_key" in data


def test_frontend_upload_flow(client, tmp_path):
    """Verify that a token obtained from /auth/token can successfully upload a file to /ingestion/upload."""
    # 1. Get token
    tok_resp = client.get("/auth/token?user_id=user_alpha")
    token = tok_resp.json()["access_token"]
    api_key = settings.INGEST_API_KEY.get_secret_value()

    # 2. Mock submit_upload to avoid requiring a real Celery worker during test
    with patch("src.routes.ingestion.submit_upload", new_callable=AsyncMock) as mock_submit:
        mock_submit.return_value = "550e8400-e29b-41d4-a716-446655440000"

        # 3. Send file upload request as the frontend does
        file_content = b"%PDF-1.4 test document content"
        files = {"file": ("test_contract.pdf", file_content, "application/pdf")}
        data = {"doc_id": "test_doc_001"}
        headers = {
            "x-api-key": api_key,
            "Authorization": f"Bearer {token}",
        }

        response = client.post("/ingestion/upload", files=files, data=data, headers=headers)
        assert response.status_code == 202
        res_data = response.json()
        assert res_data["status"] == "queued"
        assert res_data["task_id"] == "550e8400-e29b-41d4-a716-446655440000"
        mock_submit.assert_awaited_once()
