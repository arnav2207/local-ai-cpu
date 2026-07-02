"""Prompt templates for structured extraction."""

from __future__ import annotations

import json
from typing import Any


def build_extraction_prompt(source_text: str, json_schema: dict[str, Any]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for initial extraction."""
    schema_text = json.dumps(json_schema, indent=2)
    system_prompt = (
        "You extract structured data from unstructured text. "
        "Respond with valid JSON only. Do not include markdown fences or commentary."
    )
    user_prompt = (
        "Extract data from the source text into JSON that conforms to this JSON Schema:\n"
        f"{schema_text}\n\n"
        "Source text:\n"
        f"{source_text}\n\n"
        "Return only the JSON object."
    )
    return system_prompt, user_prompt


def build_repair_prompt(
    source_text: str,
    json_schema: dict[str, Any],
    invalid_json: str,
    validation_errors: list[str],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for schema repair retries."""
    schema_text = json.dumps(json_schema, indent=2)
    error_text = "\n".join(f"- {error}" for error in validation_errors)
    system_prompt = (
        "You fix invalid JSON extraction output. "
        "Respond with valid JSON only. Do not include markdown fences or commentary."
    )
    user_prompt = (
        "The previous JSON output was invalid.\n\n"
        f"Validation errors:\n{error_text}\n\n"
        "Invalid output:\n"
        f"{invalid_json}\n\n"
        "Target JSON Schema:\n"
        f"{schema_text}\n\n"
        "Source text:\n"
        f"{source_text}\n\n"
        "Return corrected JSON only."
    )
    return system_prompt, user_prompt


def build_merge_prompt(
    partial_results: list[dict[str, Any]],
    json_schema: dict[str, Any],
) -> tuple[str, str]:
    """Return prompts to merge chunk-level extraction results."""
    schema_text = json.dumps(json_schema, indent=2)
    partial_text = json.dumps(partial_results, indent=2)
    system_prompt = (
        "You merge partial JSON extraction results into one valid JSON object. "
        "Respond with valid JSON only."
    )
    user_prompt = (
        "Merge these partial JSON extraction results into one object that satisfies "
        "the JSON Schema. Prefer the most complete non-null values.\n\n"
        f"JSON Schema:\n{schema_text}\n\n"
        f"Partial results:\n{partial_text}\n\n"
        "Return only the merged JSON object."
    )
    return system_prompt, user_prompt
