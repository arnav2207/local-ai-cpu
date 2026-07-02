"""End-to-end ingest, extract, and persist workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_ai_cpu.extraction.extractor import ExtractionResult, extract_structured
from local_ai_cpu.extraction.llm import ModelNotFoundError, is_model_available
from local_ai_cpu.extraction.schema import SchemaValidationError, validate_schema_definition
from local_ai_cpu.ingestion.base import IngestionError
from local_ai_cpu.ingestion.dispatcher import ingest_bytes, persist_document
from local_ai_cpu.metrics.monitor import ResourceMetrics, ResourceMonitor
from local_ai_cpu.storage.repository import Repository, schema_hash


@dataclass(frozen=True)
class PipelineResult:
    job_id: int
    schema_id: int
    document_id: int
    result: dict[str, Any]
    validation_ok: bool
    extraction_cache_hit: bool
    document_cache_hit: bool
    repair_attempts: int
    chunk_count: int
    metrics: ResourceMetrics
    error: str | None = None


def run_pipeline(
    repo: Repository,
    schema_id: int,
    filename: str,
    data: bytes,
    *,
    model_path: Path | None = None,
) -> PipelineResult:
    """Ingest a file, extract structured data for a schema, and persist the job."""
    schema_record = repo.get_schema(schema_id)
    if schema_record is None:
        raise ValueError(f"Schema {schema_id} not found.")

    ingested = ingest_bytes(filename, data)
    document_id, _, document_cache_hit = persist_document(repo, ingested)
    return _run_extraction_for_document(
        repo,
        schema_id=schema_id,
        document_id=document_id,
        json_schema=schema_record.json_schema,
        source_text=ingested.raw_text,
        content_hash=ingested.content_hash,
        document_cache_hit=document_cache_hit,
        model_path=model_path,
    )


def run_pipeline_for_document(
    repo: Repository,
    schema_id: int,
    document_id: int,
    *,
    model_path: Path | None = None,
) -> PipelineResult:
    """Run extraction for an already-ingested document."""
    schema_record = repo.get_schema(schema_id)
    if schema_record is None:
        raise ValueError(f"Schema {schema_id} not found.")

    document = repo.get_document(document_id)
    if document is None:
        raise ValueError(f"Document {document_id} not found.")

    return _run_extraction_for_document(
        repo,
        schema_id=schema_id,
        document_id=document_id,
        json_schema=schema_record.json_schema,
        source_text=document.raw_text,
        content_hash=document.content_hash,
        document_cache_hit=True,
        model_path=model_path,
    )


def _run_extraction_for_document(
    repo: Repository,
    *,
    schema_id: int,
    document_id: int,
    json_schema: dict[str, Any],
    source_text: str,
    content_hash: str,
    document_cache_hit: bool,
    model_path: Path | None,
) -> PipelineResult:
    validate_schema_definition(json_schema)

    job_id = repo.create_job(schema_id, document_id)
    repo.update_job(job_id, status="processing")

    schema_hash_value = schema_hash(json_schema)
    cached = repo.get_cache(content_hash, schema_hash_value)
    metrics = ResourceMetrics(duration_seconds=0.0, peak_cpu_percent=0.0, peak_rss_mb=0.0)

    try:
        if cached is not None:
            extraction = ExtractionResult(
                data=cached,
                validation_ok=True,
                raw_response="",
                repair_attempts=0,
                chunk_count=1,
            )
            extraction_cache_hit = True
        else:
            if not is_model_available(model_path):
                raise ModelNotFoundError(
                    "Local model is not available. Run: uv run python scripts/download_model.py"
                )
            with ResourceMonitor() as monitor:
                extraction = extract_structured(
                    source_text,
                    json_schema,
                    model_path=model_path,
                )
            assert monitor.metrics is not None
            metrics = monitor.metrics
            extraction_cache_hit = False
            if extraction.validation_ok:
                repo.upsert_cache(content_hash, schema_hash_value, extraction.data)

        repo.save_extraction(
            job_id,
            extraction.data,
            validation_ok=extraction.validation_ok,
        )
        repo.update_job(job_id, status="done", finished=True)

        return PipelineResult(
            job_id=job_id,
            schema_id=schema_id,
            document_id=document_id,
            result=extraction.data,
            validation_ok=extraction.validation_ok,
            extraction_cache_hit=extraction_cache_hit,
            document_cache_hit=document_cache_hit,
            repair_attempts=extraction.repair_attempts,
            chunk_count=extraction.chunk_count,
            metrics=metrics,
        )
    except (IngestionError, SchemaValidationError, ModelNotFoundError, ValueError) as exc:
        repo.update_job(job_id, status="failed", error=str(exc), finished=True)
        return PipelineResult(
            job_id=job_id,
            schema_id=schema_id,
            document_id=document_id,
            result={},
            validation_ok=False,
            extraction_cache_hit=False,
            document_cache_hit=document_cache_hit,
            repair_attempts=0,
            chunk_count=0,
            metrics=metrics,
            error=str(exc),
        )
    except Exception as exc:
        repo.update_job(job_id, status="failed", error=str(exc), finished=True)
        raise
