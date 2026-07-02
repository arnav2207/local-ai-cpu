"""Application configuration and local paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LOCAL_AI_CPU_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("LOCAL_AI_CPU_MODELS_DIR", PROJECT_ROOT / "models"))
DB_PATH = Path(os.environ.get("LOCAL_AI_CPU_DB_PATH", DATA_DIR / "local_ai_cpu.db"))

DEFAULT_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_FILENAME

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

LLM_N_CTX = int(os.environ.get("LOCAL_AI_CPU_N_CTX", "4096"))
LLM_N_THREADS = int(os.environ.get("LOCAL_AI_CPU_N_THREADS", os.cpu_count() or 4))
LLM_N_BATCH = int(os.environ.get("LOCAL_AI_CPU_N_BATCH", "512"))
LLM_MAX_TOKENS = int(os.environ.get("LOCAL_AI_CPU_MAX_TOKENS", "2048"))

CHUNK_SIZE_CHARS = int(os.environ.get("LOCAL_AI_CPU_CHUNK_SIZE", "3000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("LOCAL_AI_CPU_CHUNK_OVERLAP", "200"))

MAX_UPLOAD_MB = int(os.environ.get("LOCAL_AI_CPU_MAX_UPLOAD_MB", "25"))


def ensure_data_dirs() -> None:
    """Create local data and model directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
