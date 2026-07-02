from __future__ import annotations

import pytest

from local_ai_cpu.extraction.schema import (
    SchemaValidationError,
    parse_json_response,
    validate_instance,
    validate_schema_definition,
)


def test_validate_schema_definition_accepts_contact_schema(contact_schema: dict) -> None:
    validate_schema_definition(contact_schema)


def test_validate_schema_definition_rejects_invalid_schema() -> None:
    with pytest.raises(SchemaValidationError):
        validate_schema_definition({"type": "object", "properties": "invalid"})


def test_parse_json_response_accepts_fenced_json() -> None:
    parsed = parse_json_response('```json\n{"name": "Jane"}\n```')
    assert parsed == {"name": "Jane"}


def test_validate_instance_reports_type_errors(contact_schema: dict) -> None:
    errors = validate_instance({"name": 123, "email": "jane@example.com"}, contact_schema)
    assert errors


def test_parse_json_response_raises_for_non_json() -> None:
    with pytest.raises(SchemaValidationError):
        parse_json_response("not json")
