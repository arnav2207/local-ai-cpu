"""Optional OCR ingestion for image files."""

from __future__ import annotations

import io
import shutil

import pytesseract
from PIL import Image

from local_ai_cpu.ingestion.base import IngestionError


def is_tesseract_available() -> bool:
    """Return True when the Tesseract binary is installed and reachable."""
    if shutil.which("tesseract") is None:
        return False
    try:
        pytesseract.get_tesseract_version()
    except (pytesseract.TesseractError, OSError):
        return False
    return True


def extract_text(data: bytes) -> str:
    """Extract text from image bytes using Tesseract OCR."""
    if not is_tesseract_available():
        raise IngestionError(
            "Tesseract OCR is not installed. Install it to ingest image files, "
            "or upload text/PDF documents instead."
        )

    with Image.open(io.BytesIO(data)) as image:
        rgb_image = image.convert("RGB")
        text = pytesseract.image_to_string(rgb_image)

    if not text.strip():
        raise IngestionError("OCR completed but no text was detected in the image.")

    return text
