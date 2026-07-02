"""Dispatch ingestion by file type and persist documents with content-hash caching."""

from __future__ import annotations

from pathlib import Path

from local_ai_cpu.config import MAX_UPLOAD_MB
from local_ai_cpu.ingestion import ocr as ocr_ingest
from local_ai_cpu.ingestion import pdf as pdf_ingest
from local_ai_cpu.ingestion import text as text_ingest
from local_ai_cpu.ingestion.base import (
    IngestedDocument,
    IngestionError,
    content_hash,
    guess_source_type,
)
from local_ai_cpu.processing.normalize import normalize_text
from local_ai_cpu.storage.repository import DocumentRecord, Repository


def ingest_bytes(filename: str, data: bytes) -> IngestedDocument:
    """Parse uploaded bytes into normalized text."""
    _validate_size(data)
    source_type = guess_source_type(filename)
    raw_text = _extract_by_type(source_type, data)
    normalized = normalize_text(raw_text)
    if not normalized.strip():
        raise IngestionError("No text content could be extracted from the file.")

    return IngestedDocument(
        filename=filename,
        content_hash=content_hash(data),
        raw_text=normalized,
        source_type=source_type,
    )


def ingest_file(path: Path) -> IngestedDocument:
    """Parse a local file into normalized text."""
    data = path.read_bytes()
    return ingest_bytes(path.name, data)


def persist_document(
    repo: Repository,
    ingested: IngestedDocument,
) -> tuple[int, DocumentRecord, bool]:
    """
    Store an ingested document, reusing existing rows when the content hash matches.

    Returns (document_id, document_record, cache_hit).
    """
    existing = repo.get_document_by_hash(ingested.content_hash)
    document_id = repo.save_document(
        ingested.filename,
        ingested.content_hash,
        ingested.raw_text,
    )
    record = repo.get_document(document_id)
    assert record is not None
    cache_hit = existing is not None
    return document_id, record, cache_hit


def _validate_size(data: bytes) -> None:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise IngestionError(
            f"File exceeds the {MAX_UPLOAD_MB} MB upload limit "
            f"({len(data) / (1024 * 1024):.1f} MB)."
        )


def _extract_by_type(source_type: str, data: bytes) -> str:
    if source_type == "text":
        return text_ingest.extract_text(data)
    if source_type == "pdf":
        return pdf_ingest.extract_text(data)
    if source_type == "image":
        return ocr_ingest.extract_text(data)
    raise IngestionError(f"Unsupported source type: {source_type}")
