from __future__ import annotations

from local_ai_cpu.storage import Repository, schema_hash


def test_repository_round_trip(repo: Repository, contact_schema: dict) -> None:
    schema_id = repo.save_schema("contact", contact_schema)
    document_id = repo.save_document("contact.txt", "hash123", "Jane Doe")
    job_id = repo.create_job(schema_id, document_id)

    repo.update_job(job_id, status="processing")
    repo.save_extraction(
        job_id,
        {"name": "Jane Doe", "email": "jane@example.com"},
        validation_ok=True,
    )
    repo.update_job(job_id, status="done", finished=True)

    schema = repo.get_schema(schema_id)
    document = repo.get_document(document_id)
    job = repo.get_job(job_id)
    extraction = repo.get_extraction_by_job(job_id)

    assert schema is not None and schema.name == "contact"
    assert document is not None and document.raw_text == "Jane Doe"
    assert job is not None and job.status == "done"
    assert extraction is not None and extraction.validation_ok is True


def test_cache_lookup(repo: Repository, contact_schema: dict) -> None:
    sh = schema_hash(contact_schema)
    repo.upsert_cache("doc-hash", sh, {"name": "Jane Doe", "email": "jane@example.com"})
    cached = repo.get_cache("doc-hash", sh)
    assert cached == {"name": "Jane Doe", "email": "jane@example.com"}


def test_list_documents(repo: Repository) -> None:
    repo.save_document("a.txt", "hash-a", "A")
    repo.save_document("b.txt", "hash-b", "B")
    documents = repo.list_documents()
    assert len(documents) == 2
    assert documents[0].filename == "b.txt"
