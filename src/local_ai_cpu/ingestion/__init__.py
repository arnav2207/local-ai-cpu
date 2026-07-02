"""Document ingestion adapters."""

from local_ai_cpu.ingestion.base import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
    IngestedDocument,
    IngestionError,
    content_hash,
    guess_source_type,
)
from local_ai_cpu.ingestion.dispatcher import ingest_bytes, ingest_file, persist_document
from local_ai_cpu.ingestion.ocr import is_tesseract_available

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_PDF_EXTENSIONS",
    "SUPPORTED_TEXT_EXTENSIONS",
    "IngestedDocument",
    "IngestionError",
    "content_hash",
    "guess_source_type",
    "ingest_bytes",
    "ingest_file",
    "is_tesseract_available",
    "persist_document",
]
