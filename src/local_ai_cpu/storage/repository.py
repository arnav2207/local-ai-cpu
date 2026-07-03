"""CRUD operations for schemas, documents, jobs, extractions, and cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from local_ai_cpu.storage.database import connect, init_db

JobStatus = str  # pending | processing | done | failed


@dataclass(frozen=True)
class SchemaRecord:
    id: int
    name: str
    json_schema: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    filename: str
    content_hash: str
    raw_text: str
    created_at: str


@dataclass(frozen=True)
class JobRecord:
    id: int
    schema_id: int
    document_id: int
    status: JobStatus
    error: str | None
    created_at: str
    finished_at: str | None


@dataclass(frozen=True)
class ExtractionRecord:
    id: int
    job_id: int
    result_json: dict[str, Any]
    validation_ok: bool
    created_at: str


def schema_hash(json_schema: dict[str, Any]) -> str:
    """Stable hash for a JSON Schema object."""
    canonical = json.dumps(json_schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Repository:
    """SQLite-backed persistence for extraction workflows."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = init_db(db_path)

    def save_schema(self, name: str, json_schema: dict[str, Any]) -> int:
        payload = json.dumps(json_schema)
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO schemas (name, json_schema) VALUES (?, ?)",
                (name, payload),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def get_schema(self, schema_id: int) -> SchemaRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, name, json_schema, created_at FROM schemas WHERE id = ?",
                (schema_id,),
            ).fetchone()
        return _row_to_schema(row) if row else None

    def list_schemas(self) -> list[SchemaRecord]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, name, json_schema, created_at FROM schemas ORDER BY id DESC"
            ).fetchall()
        return [_row_to_schema(row) for row in rows]

    def save_document(
        self,
        filename: str,
        content_hash: str,
        raw_text: str,
    ) -> int:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO documents (filename, content_hash, raw_text)
                VALUES (?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    filename = excluded.filename,
                    raw_text = excluded.raw_text
                """,
                (filename, content_hash, raw_text),
            )
            row = conn.execute(
                "SELECT id FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        assert row is not None
        return int(row["id"])

    def get_document(self, document_id: int) -> DocumentRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, filename, content_hash, raw_text, created_at
                FROM documents WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        return _row_to_document(row) if row else None

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, filename, content_hash, raw_text, created_at
                FROM documents WHERE content_hash = ?
                """,
                (content_hash,),
            ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self, limit: int = 50) -> list[DocumentRecord]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, filename, content_hash, raw_text, created_at
                FROM documents ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_document(row) for row in rows]

    def create_job(self, schema_id: int, document_id: int) -> int:
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO jobs (schema_id, document_id, status)
                VALUES (?, ?, 'pending')
                """,
                (schema_id, document_id),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def update_job(
        self,
        job_id: int,
        *,
        status: JobStatus | None = None,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []

        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if finished:
            fields.append("finished_at = datetime('now')")

        if not fields:
            return

        values.append(job_id)
        query = f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?"  # nosec B608
        with connect(self.db_path) as conn:
            conn.execute(query, values)

    def get_job(self, job_id: int) -> JobRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, schema_id, document_id, status, error, created_at, finished_at
                FROM jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        return _row_to_job(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, schema_id, document_id, status, error, created_at, finished_at
                FROM jobs ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def save_extraction(
        self,
        job_id: int,
        result_json: dict[str, Any],
        *,
        validation_ok: bool,
    ) -> int:
        payload = json.dumps(result_json)
        with connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO extractions (job_id, result_json, validation_ok)
                VALUES (?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    result_json = excluded.result_json,
                    validation_ok = excluded.validation_ok,
                    created_at = datetime('now')
                """,
                (job_id, payload, int(validation_ok)),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def get_extraction_by_job(self, job_id: int) -> ExtractionRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, job_id, result_json, validation_ok, created_at
                FROM extractions WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        return _row_to_extraction(row) if row else None

    def get_cache(self, content_hash: str, schema_hash_value: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT result_json FROM cache
                WHERE content_hash = ? AND schema_hash = ?
                """,
                (content_hash, schema_hash_value),
            ).fetchone()
        if row is None:
            return None
        return cast(dict[str, Any], json.loads(row["result_json"]))

    def upsert_cache(
        self,
        content_hash: str,
        schema_hash_value: str,
        result_json: dict[str, Any],
    ) -> None:
        payload = json.dumps(result_json)
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO cache (content_hash, schema_hash, result_json)
                VALUES (?, ?, ?)
                ON CONFLICT(content_hash, schema_hash) DO UPDATE SET
                    result_json = excluded.result_json,
                    created_at = datetime('now')
                """,
                (content_hash, schema_hash_value, payload),
            )


def _row_to_schema(row: Any) -> SchemaRecord:
    return SchemaRecord(
        id=int(row["id"]),
        name=row["name"],
        json_schema=json.loads(row["json_schema"]),
        created_at=row["created_at"],
    )


def _row_to_document(row: Any) -> DocumentRecord:
    return DocumentRecord(
        id=int(row["id"]),
        filename=row["filename"],
        content_hash=row["content_hash"],
        raw_text=row["raw_text"],
        created_at=row["created_at"],
    )


def _row_to_job(row: Any) -> JobRecord:
    return JobRecord(
        id=int(row["id"]),
        schema_id=int(row["schema_id"]),
        document_id=int(row["document_id"]),
        status=row["status"],
        error=row["error"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
    )


def _row_to_extraction(row: Any) -> ExtractionRecord:
    return ExtractionRecord(
        id=int(row["id"]),
        job_id=int(row["job_id"]),
        result_json=json.loads(row["result_json"]),
        validation_ok=bool(row["validation_ok"]),
        created_at=row["created_at"],
    )
