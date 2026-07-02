"""Structured extraction with schema validation and repair retries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_ai_cpu.extraction.llm import ModelNotFoundError, generate_chat, is_model_available
from local_ai_cpu.extraction.prompts import (
    build_extraction_prompt,
    build_merge_prompt,
    build_repair_prompt,
)
from local_ai_cpu.extraction.schema import (
    SchemaValidationError,
    parse_json_response,
    validate_instance,
    validate_schema_definition,
)
from local_ai_cpu.processing.chunker import chunk_text

MAX_REPAIR_ATTEMPTS = 2


@dataclass(frozen=True)
class ExtractionResult:
    data: dict[str, Any]
    validation_ok: bool
    raw_response: str
    repair_attempts: int
    chunk_count: int


def extract_structured(
    source_text: str,
    json_schema: dict[str, Any],
    *,
    model_path: Path | None = None,
    max_repairs: int = MAX_REPAIR_ATTEMPTS,
) -> ExtractionResult:
    """Extract structured JSON from text using the local SLM with validation and repair."""
    validate_schema_definition(json_schema)

    chunks = chunk_text(source_text)
    if not chunks:
        raise SchemaValidationError("Source text is empty after chunking.")

    if len(chunks) == 1:
        return _extract_single(
            chunks[0],
            json_schema,
            model_path=model_path,
            max_repairs=max_repairs,
        )

    partials: list[dict[str, Any]] = []
    total_repairs = 0
    last_response = ""

    for chunk in chunks:
        chunk_result = _extract_single(
            chunk,
            json_schema,
            model_path=model_path,
            max_repairs=max_repairs,
        )
        partials.append(chunk_result.data)
        total_repairs += chunk_result.repair_attempts
        last_response = chunk_result.raw_response

    merged = _merge_partials(partials, json_schema, model_path=model_path)
    errors = validate_instance(merged.data, json_schema)
    return ExtractionResult(
        data=merged.data,
        validation_ok=not errors,
        raw_response=merged.raw_response or last_response,
        repair_attempts=total_repairs + merged.repair_attempts,
        chunk_count=len(chunks),
    )


def _extract_single(
    source_text: str,
    json_schema: dict[str, Any],
    *,
    model_path: Path | None = None,
    max_repairs: int,
) -> ExtractionResult:
    system_prompt, user_prompt = build_extraction_prompt(source_text, json_schema)
    raw_response = generate_chat(system_prompt, user_prompt, model_path=model_path)

    parsed, errors, repair_attempts = _validate_or_repair(
        source_text=source_text,
        json_schema=json_schema,
        raw_response=raw_response,
        model_path=model_path,
        max_repairs=max_repairs,
    )

    if not isinstance(parsed, dict):
        raise SchemaValidationError("Extracted JSON must be an object at the top level.")

    return ExtractionResult(
        data=parsed,
        validation_ok=not errors,
        raw_response=raw_response,
        repair_attempts=repair_attempts,
        chunk_count=1,
    )


def _validate_or_repair(
    *,
    source_text: str,
    json_schema: dict[str, Any],
    raw_response: str,
    model_path: Path | None,
    max_repairs: int,
) -> tuple[Any, list[str], int]:
    repair_attempts = 0
    current_response = raw_response

    while True:
        try:
            parsed = parse_json_response(current_response)
        except SchemaValidationError as exc:
            errors = [str(exc)]
            parsed = None
        else:
            errors = validate_instance(parsed, json_schema)

        if not errors:
            return parsed, [], repair_attempts

        if repair_attempts >= max_repairs:
            return parsed or {}, errors, repair_attempts

        system_prompt, user_prompt = build_repair_prompt(
            source_text,
            json_schema,
            current_response,
            errors,
        )
        current_response = generate_chat(system_prompt, user_prompt, model_path=model_path)
        repair_attempts += 1


def _merge_partials(
    partials: list[dict[str, Any]],
    json_schema: dict[str, Any],
    *,
    model_path: Path | None,
) -> ExtractionResult:
    deterministic = _deterministic_merge(partials)
    errors = validate_instance(deterministic, json_schema)
    if not errors:
        return ExtractionResult(
            data=deterministic,
            validation_ok=True,
            raw_response=json.dumps(deterministic),
            repair_attempts=0,
            chunk_count=len(partials),
        )

    system_prompt, user_prompt = build_merge_prompt(partials, json_schema)
    raw_response = generate_chat(system_prompt, user_prompt, model_path=model_path)
    parsed, errors, repair_attempts = _validate_or_repair(
        source_text=json.dumps(partials),
        json_schema=json_schema,
        raw_response=raw_response,
        model_path=model_path,
        max_repairs=MAX_REPAIR_ATTEMPTS,
    )
    if not isinstance(parsed, dict):
        parsed = deterministic

    return ExtractionResult(
        data=parsed,
        validation_ok=not errors,
        raw_response=raw_response,
        repair_attempts=repair_attempts,
        chunk_count=len(partials),
    )


def _deterministic_merge(partials: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for partial in partials:
        merged = _deep_merge(merged, partial)
    return merged


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if key not in result or result[key] in (None, "", [], {}):
            result[key] = value
            continue
        existing = result[key]
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            result[key] = existing + [item for item in value if item not in existing]
        elif value not in (None, "", [], {}):
            result[key] = value
    return result


__all__ = [
    "ExtractionResult",
    "ModelNotFoundError",
    "SchemaValidationError",
    "extract_structured",
    "is_model_available",
]
