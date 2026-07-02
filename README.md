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
uv run pytest -m "not slow"
uv run ruff check src tests
```

Run the optional live model integration test after downloading the GGUF file:

```bash
uv run pytest -m slow
```

## Hackathon demo flow

Use this 3-minute judge demo to show offline structured extraction end to end.

1. **Install and download the model**
   ```bash
   uv sync --extra dev
   uv run python scripts/download_model.py
   ```

2. **Launch the UI**
   ```bash
   uv run streamlit run src/local_ai_cpu/app.py
   ```

3. **Schema Builder tab**
   - Upload [`examples/contact_schema.json`](examples/contact_schema.json), or build fields manually
   - Click **Save schema**

4. **Ingest tab**
   - Upload [`examples/contact.txt`](examples/contact.txt)
   - Confirm the text preview, then click **Save document**

5. **Extract tab**
   - Select the saved schema and document (or upload the file directly)
   - Click **Run extraction**
   - Show structured JSON output and note CPU metrics in the status panel

6. **Results tab**
   - Open the latest job and compare source text vs extracted JSON side by side
   - Download the JSON export

7. **Metrics tab**
   - Highlight peak CPU %, memory footprint, and cache behavior
   - Run extraction again to demonstrate a **cache hit** without LLM latency

8. **Offline resiliency check**
   - Disconnect from the network
   - Re-run extraction on the same file + schema
   - Confirm the app still works using the local model and SQLite cache

### Alternate demo

Repeat steps 3–7 with:
- Schema: [`examples/product_schema.json`](examples/product_schema.json)
- Document: [`examples/product_listing.txt`](examples/product_listing.txt)

## Hackathon constraints

- All AI inference is local (llama.cpp + GGUF)
- No OpenAI, Anthropic, or other external inference APIs
- Optimized for CPU-only edge hardware

## License

MIT
