"""Shared ingestion types and helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = (
    SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS
)


@dataclass(frozen=True)
class IngestedDocument:
    filename: str
    content_hash: str
    raw_text: str
    source_type: str


class IngestionError(Exception):
    """Raised when a file cannot be ingested."""


def content_hash(data: bytes) -> str:
    """Return a SHA-256 hex digest for raw file bytes."""
    return hashlib.sha256(data).hexdigest()


def guess_source_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return "text"
    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return "pdf"
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    raise IngestionError(f"Unsupported file type: {suffix or '(no extension)'}")
