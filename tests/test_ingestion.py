from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from local_ai_cpu.ingestion import IngestionError, ingest_bytes, ingest_file, persist_document
from local_ai_cpu.processing import chunk_text, normalize_text


def test_ingest_contact_example(contact_text: bytes) -> None:
    ingested = ingest_bytes("contact.txt", contact_text)
    assert "Jane Doe" in ingested.raw_text
    assert ingested.source_type == "text"


def test_persist_document_reports_cache_hit(repo, contact_text: bytes) -> None:
    ingested = ingest_bytes("contact.txt", contact_text)
    _, _, first_hit = persist_document(repo, ingested)
    _, _, second_hit = persist_document(repo, ingested)
    assert first_hit is False
    assert second_hit is True


def test_ingest_pdf_bytes() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Invoice total: 42.00")
    data = document.tobytes()
    document.close()

    ingested = ingest_bytes("invoice.pdf", data)
    assert "42.00" in ingested.raw_text
    assert ingested.source_type == "pdf"


def test_ingest_rejects_oversized_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_ai_cpu.ingestion.dispatcher.MAX_UPLOAD_MB", 0)
    with pytest.raises(IngestionError):
        ingest_bytes("big.txt", b"too large")


def test_normalize_and_chunk_text() -> None:
    text = normalize_text("Line one\r\n\r\nLine two. " * 100)
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_ingest_file_reads_example(examples_dir: Path) -> None:
    ingested = ingest_file(examples_dir / "product_listing.txt")
    assert "SolarEdge Mini Sensor" in ingested.raw_text
