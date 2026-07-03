"""Local llama.cpp inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from local_ai_cpu.config import (
    DEFAULT_MODEL_PATH,
    LLM_MAX_TOKENS,
    LLM_N_BATCH,
    LLM_N_CTX,
    LLM_N_THREADS,
    ensure_data_dirs,
)

if TYPE_CHECKING:
    from llama_cpp import Llama

_LLM_INSTANCE: Any | None = None
_LLM_MODEL_PATH: Path | None = None


class ModelNotFoundError(FileNotFoundError):
    """Raised when the local GGUF model file is missing."""


class LlamaCppNotInstalledError(ImportError):
    """Raised when llama-cpp-python is not installed."""


def is_llama_cpp_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


def is_model_available(model_path: Path | None = None) -> bool:
    path = model_path or DEFAULT_MODEL_PATH
    return is_llama_cpp_available() and path.exists()


def _load_llama_class() -> type[Llama]:
    if not is_llama_cpp_available():
        raise LlamaCppNotInstalledError(
            "llama-cpp-python is not installed. Run: uv sync --extra llm"
        )
    from llama_cpp import Llama

    return cast(type[Llama], Llama)


def get_llm(model_path: Path | None = None) -> Llama:
    """Load or reuse a singleton llama.cpp model instance."""
    global _LLM_INSTANCE, _LLM_MODEL_PATH

    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        raise ModelNotFoundError(
            f"Model not found at {path}. Run: uv run python scripts/download_model.py"
        )

    if _LLM_INSTANCE is not None and _LLM_MODEL_PATH == path:
        return _LLM_INSTANCE

    Llama = _load_llama_class()
    ensure_data_dirs()
    _LLM_INSTANCE = Llama(
        model_path=str(path),
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_batch=LLM_N_BATCH,
        verbose=False,
    )
    _LLM_MODEL_PATH = path
    return _LLM_INSTANCE


def reset_llm_cache() -> None:
    """Clear the cached model instance (useful in tests)."""
    global _LLM_INSTANCE, _LLM_MODEL_PATH
    _LLM_INSTANCE = None
    _LLM_MODEL_PATH = None


def generate_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model_path: Path | None = None,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = 0.1,
) -> str:
    """Run a chat completion and return the assistant message content."""
    llm = get_llm(model_path)
    response: Any = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response["choices"][0]["message"]["content"]
    if not isinstance(choice, str):
        raise RuntimeError("Unexpected non-string response from local model.")
    return choice.strip()
