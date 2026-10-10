"""Light-weight validation helpers shared by the API process and the Celery worker.

IMPORTANT: this module is imported by the FastAPI process too, so it must stay
free of heavy imports (fastembed, pypdf, openpyxl, ...). Importing those inside
the API would slow its startup and waste RAM in a process that never uses them.
"""
from __future__ import annotations

from pathlib import Path
import zipfile


from src.core.config import settings

class IngestionError(ValueError):
    """The problem is the uploaded file itself (corrupt, unsupported, too big...).

    These errors are shown to the user and are NEVER retried, because trying the
    same bad file again can't succeed.
    """


def sniff_matches_extension(path: Path, ext: str) -> bool:
    """Check the first bytes ("magic bytes") against the claimed extension.

    This does not need libmagic (which is painful to install on Windows).
    It cannot prove a file is harmless; it just stops an obviously mislabeled
    file before any parser touches it.
    """
    with open(path, "rb") as fh:
        head = fh.read(8192)
    if not head:
        return False
    if ext == ".pdf":
        # The PDF spec allows the header anywhere in the first 1024 bytes.
        return b"%PDF-" in head[:1024]
    if ext in settings.ZIP_EXTENSIONS:
        return head.startswith(b"PK\x03\x04")
    if ext in settings.TEXT_EXTENSIONS:
        # Real text files never contain NUL bytes; binary files almost always do.
        return b"\x00" not in head
    return False


def check_zip_expansion(path: Path) -> None:
    """Reject .docx/.xlsx files whose declared uncompressed size is absurdly large."""
    try:
        with zipfile.ZipFile(path) as archive:
            total = sum(info.file_size for info in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise IngestionError("The file is corrupt or not a valid Office document.") from exc
    if total > settings.MAX_UNCOMPRESSED_BYTES:
        raise IngestionError("The file expands to too much data and was rejected.")
