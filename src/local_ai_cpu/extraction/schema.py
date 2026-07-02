"""JSON Schema validation and LLM response parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class SchemaValidationError(ValueError):
    """Raised when JSON Schema or extracted data is invalid."""


def validate_schema_definition(json_schema: dict[str, Any]) -> None:
    """Validate that a JSON Schema definition is well-formed."""
    try:
        Draft202012Validator.check_schema(json_schema)
    except SchemaError as exc:
        raise SchemaValidationError(str(exc)) from exc


def validate_instance(instance: Any, json_schema: dict[str, Any]) -> list[str]:
    """Validate extracted data against a JSON Schema, returning error messages."""
    validator = Draft202012Validator(json_schema)
    errors = sorted(validator.iter_errors(instance), key=lambda err: err.path)
    return [error.message for error in errors]


def parse_json_response(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating fenced code blocks."""
    stripped = text.strip()
    if not stripped:
        raise SchemaValidationError("Model returned an empty response.")

    fence_match = _JSON_FENCE.search(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _extract_json_fragment(stripped)
        if candidate is None:
            raise SchemaValidationError("Model response did not contain valid JSON.") from None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"Invalid JSON in model response: {exc}") from exc


def _extract_json_fragment(text: str) -> str | None:
    start_candidates = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not start_candidates:
        return None

    start = min(start_candidates)
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
