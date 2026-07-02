"""PDF text extraction via PyMuPDF."""

from __future__ import annotations

import fitz


def extract_text(data: bytes) -> str:
    """Extract text from PDF bytes."""
    with fitz.open(stream=data, filetype="pdf") as document:
        pages = [page.get_text("text") for page in document]
    return "\n".join(pages)
