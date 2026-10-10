from pathlib import Path

from pydantic import SecretStr
from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    VERIFY_TOKEN: str 
    APP_SECRET: SecretStr

    GEMINI_API_KEY: SecretStr
    
    INSTAGRAM_TOKEN: SecretStr
    
    WHATSAPP_TOKEN: SecretStr 
    WHATSAPP_APP_SECRET: SecretStr
    PHONE_NUMBER_ID: SecretStr
    
    MESSENGER_TOKEN: SecretStr
    MESSENGER_APP_SECRET: SecretStr
    FB_GRAPH_API_VERSION: str = "v26.0"

    # ingestion
    INGEST_API_KEY: SecretStr
    ID_PATTERN: ClassVar[str]= r"^[\w\-.:@]{1,128}$"
    UPLOAD_DIR: Path = PROJECT_ROOT / "storage" / "uploads"
    MAX_UPLOAD_MB: ClassVar[int] = 10 * 1024 * 1024  # in megabytes (MB)

    JWT_SECRET_KEY: SecretStr

    # celery & redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # File handling configurations (Application Constants)
    ALLOWED_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {".pdf", ".docx", ".xlsx", ".json", ".txt", ".md", ".csv"}
    )
    ZIP_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".docx", ".xlsx"})
    TEXT_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".json", ".txt", ".md", ".csv"})
    
    # Guard against "zip bombs": a tiny .xlsx/.docx that expands to gigabytes.
    MAX_UNCOMPRESSED_BYTES: ClassVar[int] = 200 * 1024 * 1024

    MAX_CONCURRENT_UPLOADS: ClassVar[int] = 20

    READ_CHUNK_BYTES: ClassVar[int]= 1024 * 1024

settings = Settings()  # type: ignore
