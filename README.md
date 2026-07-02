# local-ai-cpu

Offline-first, CPU-optimized application that transforms unstructured documents into structured JSON using a user-defined schema and a local small language model.

All inference runs on-device. No cloud APIs are required after the initial model download.

## Features

- **Generic schema builder** — define or upload a JSON Schema at runtime
- **Document ingestion** — plain text, Markdown, PDF, and optional image OCR
- **Local SLM extraction** — quantized GGUF models via llama.cpp
- **SQLite persistence** — schemas, documents, jobs, results, and extraction cache
- **Resource metrics** — CPU and memory usage during inference

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Optional: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for image ingestion

## Quick start

```bash
# Install dependencies
uv sync --extra dev

# Download the default model (one-time; requires network)
uv run python scripts/download_model.py

# Launch the Streamlit UI
uv run streamlit run src/local_ai_cpu/app.py
```

## Offline usage

1. Run `scripts/download_model.py` once while online, or copy a `.gguf` file into `models/`.
2. Disconnect from the network — the app loads models and stores data locally only.
3. Re-running the same document + schema uses the local SQLite cache.

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_AI_CPU_DATA_DIR` | `./data` | SQLite DB and app data |
| `LOCAL_AI_CPU_MODELS_DIR` | `./models` | GGUF model files |
| `LOCAL_AI_CPU_N_THREADS` | CPU count | llama.cpp thread count |
| `LOCAL_AI_CPU_MAX_UPLOAD_MB` | `25` | Upload size limit |

## Development

```bash
uv run pytest
uv run ruff check src tests
```

## Hackathon constraints

- All AI inference is local (llama.cpp + GGUF)
- No OpenAI, Anthropic, or other external inference APIs
- Optimized for CPU-only edge hardware

## License

MIT
