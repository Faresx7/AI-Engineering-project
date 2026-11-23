from pathlib import Path

from pydantic import SecretStr
from typing import ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    
    # token, API'S and messaging
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
    MAX_UPLOAD_MB: ClassVar[int] = 15 * 1024 * 1024  # in megabytes (15MB)

    JWT_SECRET_KEY: SecretStr
    
    # chunking settings 
    # used in ingestion_pipeline.py
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 130

    # Maximum character length per yielded slice when streaming MarkItDown output.
    # loader.py
    _STREAM_SLICE_CHARS: int = 4_000

    # Guard against "zip bombs": a tiny .xlsx/.docx that expands to gigabytes.
    # used in validation.py
    MAX_UNCOMPRESSED_BYTES: ClassVar[int] = 200 * 1024 * 1024

    # used in upload_service.py
    READ_CHUNK_BYTES: ClassVar[int]= 1024 * 1024

    # celery & redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    INGEST_LOCK_REDIS_URL: str = "redis://localhost:6379/2"
    PURGE_LOCK_KEY: str = "ingest:lock:purge_stale_local_uploads"



    # timeouts 
    # ingestion timeouts 
    INGESTION_SOFT_TIME_LIMIT_SECONDS: int = 600
    INGESTION_HARD_TIME_LIMIT_SECONDS: int = 690
    STALE_UPLOAD_MAX_AGE: int = 3600 * 24 # deletes file 24 hours old 

    # purge timeouts
    PURGE_FILE_SOFT_TIME_LIMIT: int = 200
    PURGE_FILE_HARD_TIME_LIMIT: int = 250

    PURGE_GEN_SOFT_TIME_LIMIT: int = 120
    PURGE_GEN_HARD_TIME_LIMIT: int = 150

    # Interval in seconds between periodic orphan-generation cleanup runs (75 minutes)
    PURGE_GEN_SCHEDULE: int = 75 * 60
    PURGE_GEN_EXPIRES_MESSAGE: int = PURGE_GEN_SCHEDULE - (5 * 60)

    MAX_PURGE_FILES_PER_RUN: int = 5000



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
    

settings = Settings()  # type: ignore
