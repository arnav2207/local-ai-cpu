from __future__ import annotations

from local_ai_cpu.app import _fields_to_schema


def test_fields_to_schema_builds_required_properties() -> None:
    schema = _fields_to_schema(
        [
            {"name": "title", "type": "string", "required": True, "description": "Title"},
            {"name": "notes", "type": "string", "required": False, "description": ""},
        ]
    )
    assert schema["type"] == "object"
    assert schema["required"] == ["title"]
    assert schema["properties"]["title"]["description"] == "Title"
