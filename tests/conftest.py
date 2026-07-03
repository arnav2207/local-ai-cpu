from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_ai_cpu.storage import Repository

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES_DIR


@pytest.fixture
def repo(tmp_path: Path) -> Repository:
    return Repository(tmp_path / "test.db")


@pytest.fixture
def contact_schema(examples_dir: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((examples_dir / "contact_schema.json").read_text(encoding="utf-8")),
    )


@pytest.fixture
def contact_text(examples_dir: Path) -> bytes:
    return (examples_dir / "contact.txt").read_bytes()
