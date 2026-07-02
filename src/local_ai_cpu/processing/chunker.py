"""Split long documents into overlapping chunks for local SLM extraction."""

from __future__ import annotations

from local_ai_cpu.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            boundary = _find_boundary(normalized, start, end)
            if boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)

    return chunks


def _find_boundary(text: str, start: int, end: int) -> int:
    window = text[start:end]
    for separator in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
        index = window.rfind(separator)
        if index > len(window) // 2:
            return start + index + len(separator)
    return end
