#!/usr/bin/env python3
"""Download the default local GGUF model for offline inference."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from local_ai_cpu.config import DEFAULT_MODEL_PATH, DEFAULT_MODEL_URL, ensure_data_dirs


def download_model(
    destination: Path = DEFAULT_MODEL_PATH,
    url: str = DEFAULT_MODEL_URL,
    *,
    force: bool = False,
) -> Path:
    """Download the default model if it is not already present."""
    ensure_data_dirs()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"Model already exists at {destination}")
        return destination

    print(f"Downloading model to {destination}")
    print(f"Source: {url}")

    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0

        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 // total
                    print(f"\rProgress: {percent}% ({downloaded // (1024 * 1024)} MB)", end="")

    print(f"\nSaved model to {destination}")
    return destination


def main() -> int:
    force = "--force" in sys.argv
    try:
        download_model(force=force)
    except httpx.HTTPError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
