"""Normalize extracted text for downstream processing."""

from __future__ import annotations

import re

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_text(text: str) -> str:
    """Clean whitespace and line endings while preserving paragraph structure."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _TRAILING_SPACE.sub("", cleaned)
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()
