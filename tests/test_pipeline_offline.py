from __future__ import annotations

import pytest

from local_ai_cpu.extraction import is_model_available
from local_ai_cpu.extraction.llm import reset_llm_cache
from local_ai_cpu.pipeline import run_pipeline
from local_ai_cpu.storage import schema_hash


def test_pipeline_uses_extraction_cache_without_model(
    repo,
    contact_schema: dict,
    contact_text: bytes,
) -> None:
    schema_id = repo.save_schema("contact", contact_schema)
    sh = schema_hash(contact_schema)

    from local_ai_cpu.ingestion import ingest_bytes, persist_document

    ingested = ingest_bytes("contact.txt", contact_text)
    persist_document(repo, ingested)
    repo.upsert_cache(
        ingested.content_hash,
        sh,
        {"name": "Jane Doe", "email": "jane.doe@example.com"},
    )

    result = run_pipeline(repo, schema_id, "contact.txt", contact_text)
    assert result.error is None
    assert result.extraction_cache_hit is True
    assert result.validation_ok is True
    assert result.result["email"] == "jane.doe@example.com"


def test_pipeline_fails_gracefully_when_model_missing(
    repo,
    contact_schema: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_ai_cpu.pipeline.is_model_available", lambda *_: False)
    schema_id = repo.save_schema("contact", contact_schema)
    result = run_pipeline(repo, schema_id, "note.txt", b"Name: Bob\nEmail: bob@example.com")
    assert result.error is not None
    assert result.validation_ok is False
    job = repo.get_job(result.job_id)
    assert job is not None and job.status == "failed"


@pytest.mark.slow
def test_live_extraction_with_local_model(
    repo,
    contact_schema: dict,
    contact_text: bytes,
    examples_dir,
) -> None:
    if not is_model_available():
        pytest.skip("Local GGUF model not downloaded")

    reset_llm_cache()
    schema_id = repo.save_schema("contact", contact_schema)
    text = (examples_dir / "contact.txt").read_bytes()
    result = run_pipeline(repo, schema_id, "contact.txt", text)

    assert result.error is None, result.error
    assert result.result
    assert "email" in result.result
    assert result.metrics.duration_seconds >= 0

    # Cached rerun should be fast and skip inference metrics growth
    cached = run_pipeline(repo, schema_id, "contact.txt", text)
    assert cached.extraction_cache_hit is True
    assert cached.validation_ok is True
