"""Plain text and Markdown ingestion."""

from __future__ import annotations


def extract_text(data: bytes) -> str:
    """Decode bytes as UTF-8 text, falling back to latin-1."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode text file")
