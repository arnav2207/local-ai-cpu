"""Streamlit UI for offline structured extraction."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from local_ai_cpu import __version__
from local_ai_cpu.config import (
    DB_PATH,
    DEFAULT_MODEL_PATH,
    MAX_UPLOAD_MB,
    ensure_data_dirs,
)
from local_ai_cpu.extraction.llm import get_llm, is_model_available
from local_ai_cpu.extraction.schema import SchemaValidationError, validate_schema_definition
from local_ai_cpu.ingestion import (
    SUPPORTED_EXTENSIONS,
    IngestionError,
    ingest_bytes,
    is_tesseract_available,
    persist_document,
)
from local_ai_cpu.pipeline import PipelineResult, run_pipeline, run_pipeline_for_document
from local_ai_cpu.storage import JobRecord, Repository

FIELD_TYPES = ["string", "number", "integer", "boolean", "array"]


@st.cache_resource
def get_repository() -> Repository:
    ensure_data_dirs()
    return Repository()


@st.cache_resource
def load_local_model():
    if is_model_available():
        return get_llm()
    return None


def main() -> None:
    st.set_page_config(
        page_title="local-ai-cpu",
        page_icon="🧠",
        layout="wide",
    )

    repo = get_repository()
    _init_session_state()

    st.title("local-ai-cpu")
    st.caption(
        f"v{__version__} · Offline-first structured extraction · "
        f"Model: {'ready' if is_model_available() else 'missing'}"
    )

    tabs = st.tabs(["Schema Builder", "Ingest", "Extract", "Results", "Metrics"])
    with tabs[0]:
        _render_schema_builder(repo)
    with tabs[1]:
        _render_ingest(repo)
    with tabs[2]:
        _render_extract(repo)
    with tabs[3]:
        _render_results(repo)
    with tabs[4]:
        _render_metrics(repo)


def _init_session_state() -> None:
    if "last_pipeline_result" not in st.session_state:
        st.session_state.last_pipeline_result = None
    if "selected_document_id" not in st.session_state:
        st.session_state.selected_document_id = None


def _render_schema_builder(repo: Repository) -> None:
    st.subheader("Define a JSON Schema")
    mode = st.radio("Input mode", ["Field builder", "Paste JSON", "Upload JSON"], horizontal=True)

    schema: dict[str, Any] | None = None
    default_name = "custom_schema"

    if mode == "Field builder":
        default_name = st.text_input("Schema name", value="custom_schema")
        st.markdown("Add fields to build a JSON Schema object.")
        edited = st.data_editor(
            [
                {
                    "name": "title",
                    "type": "string",
                    "required": True,
                    "description": "Document title",
                }
            ],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Field name"),
                "type": st.column_config.SelectboxColumn("Type", options=FIELD_TYPES),
                "required": st.column_config.CheckboxColumn("Required"),
                "description": st.column_config.TextColumn("Description"),
            },
            key="schema_field_editor",
        )
        schema = _fields_to_schema(edited)
        st.json(schema)
    elif mode == "Paste JSON":
        default_name = st.text_input("Schema name", value="custom_schema")
        raw = st.text_area(
            "JSON Schema",
            height=240,
            placeholder='{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}',
        )
        if raw.strip():
            try:
                schema = json.loads(raw)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
    else:
        default_name = st.text_input("Schema name", value="custom_schema")
        upload = st.file_uploader("Upload JSON Schema", type=["json"])
        if upload is not None:
            try:
                schema = json.loads(upload.getvalue().decode("utf-8"))
                st.json(schema)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f"Could not read schema file: {exc}")

    if schema is not None:
        try:
            validate_schema_definition(schema)
            st.success("Schema definition is valid.")
        except SchemaValidationError as exc:
            st.error(str(exc))
            return

        if st.button("Save schema", type="primary"):
            schema_id = repo.save_schema(default_name.strip() or "custom_schema", schema)
            st.success(f"Saved schema #{schema_id}: {default_name}")

    saved = repo.list_schemas()
    if saved:
        st.markdown("### Saved schemas")
        st.dataframe(
            [{"id": item.id, "name": item.name, "created_at": item.created_at} for item in saved],
            use_container_width=True,
            hide_index=True,
        )


def _render_ingest(repo: Repository) -> None:
    st.subheader("Ingest documents")
    st.caption(
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))} · "
        f"Max size: {MAX_UPLOAD_MB} MB · "
        f"Tesseract OCR: {'available' if is_tesseract_available() else 'not installed'}"
    )

    upload = st.file_uploader(
        "Upload a document",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
    )
    if upload is None:
        return

    try:
        ingested = ingest_bytes(upload.name, upload.getvalue())
    except IngestionError as exc:
        st.error(str(exc))
        return

    st.text_area("Extracted text preview", ingested.raw_text[:5000], height=240)
    if len(ingested.raw_text) > 5000:
        st.info(f"Preview truncated. Full text length: {len(ingested.raw_text):,} characters.")

    if st.button("Save document", type="primary"):
        document_id, record, cache_hit = persist_document(repo, ingested)
        st.session_state.selected_document_id = document_id
        message = f"Saved document #{document_id} ({record.filename})"
        if cache_hit:
            message += " · content hash matched an existing document"
        st.success(message)

    documents = repo.list_documents()
    if documents:
        st.markdown("### Saved documents")
        st.dataframe(
            [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "chars": len(doc.raw_text),
                    "created_at": doc.created_at,
                }
                for doc in documents
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_extract(repo: Repository) -> None:
    st.subheader("Run extraction")
    schemas = repo.list_schemas()
    documents = repo.list_documents()

    if not schemas:
        st.warning("Create a schema in the Schema Builder tab first.")
        return

    schema_options = {f"{item.name} (#{item.id})": item.id for item in schemas}
    selected_schema_label = st.selectbox("Schema", list(schema_options.keys()))
    schema_id = schema_options[selected_schema_label]

    source_mode = st.radio(
        "Document source",
        ["Saved document", "Upload new file"],
        horizontal=True,
    )

    upload = None
    document_id = None
    if source_mode == "Saved document":
        if not documents:
            st.warning("Ingest a document first, or switch to upload mode.")
            return
        doc_options = {f"{doc.filename} (#{doc.id})": doc.id for doc in documents}
        doc_labels = list(doc_options.keys())
        default_index = 0
        if st.session_state.selected_document_id is not None:
            for index, label in enumerate(doc_labels):
                if doc_options[label] == st.session_state.selected_document_id:
                    default_index = index
                    break
        selected_doc_label = st.selectbox(
            "Document",
            doc_labels,
            index=default_index,
        )
        document_id = doc_options[selected_doc_label]
    else:
        upload = st.file_uploader(
            "Upload for extraction",
            type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        )

    if st.button("Run extraction", type="primary"):
        with st.status("Running local extraction pipeline...", expanded=True) as status:
            st.write("Checking cache and model availability...")
            if not is_model_available():
                st.warning(
                    "Local model not found. Cache hits will still work. "
                    "Download with: `uv run python scripts/download_model.py`"
                )
            else:
                st.write("Loading local model...")
                load_local_model()

            if source_mode == "Upload new file":
                if upload is None:
                    status.update(label="Upload a file first.", state="error")
                    return
                st.write("Ingesting document...")
                result = run_pipeline(repo, schema_id, upload.name, upload.getvalue())
            else:
                assert document_id is not None
                st.write(f"Extracting from document #{document_id}...")
                result = run_pipeline_for_document(repo, schema_id, document_id)

            st.session_state.last_pipeline_result = result
            st.session_state.selected_document_id = result.document_id

            if result.error:
                status.update(label=f"Extraction failed: {result.error}", state="error")
                st.error(result.error)
            elif result.validation_ok:
                label = "Extraction complete"
                if result.extraction_cache_hit:
                    label += " (cache hit)"
                status.update(label=label, state="complete")
                st.json(result.result)
            else:
                status.update(
                    label="Extraction finished with validation warnings",
                    state="complete",
                )
                st.warning("Output did not fully validate against the schema.")
                st.json(result.result)


def _render_results(repo: Repository) -> None:
    st.subheader("Recent results")
    jobs = repo.list_jobs(limit=20)
    if not jobs:
        st.info("No extraction jobs yet.")
        return

    rows = []
    for job in jobs:
        schema = repo.get_schema(job.schema_id)
        document = repo.get_document(job.document_id)
        extraction = repo.get_extraction_by_job(job.id)
        rows.append(
            {
                "job_id": job.id,
                "status": job.status,
                "schema": schema.name if schema else job.schema_id,
                "document": document.filename if document else job.document_id,
                "validated": extraction.validation_ok if extraction else None,
                "created_at": job.created_at,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    job_ids = [job.id for job in jobs]
    selected_job_id = st.selectbox("Inspect job", job_ids)
    selected_job: JobRecord | None = repo.get_job(selected_job_id)
    extraction = repo.get_extraction_by_job(selected_job_id)
    document = repo.get_document(selected_job.document_id) if selected_job else None

    if selected_job and extraction and document:
        left, right = st.columns(2)
        with left:
            st.markdown("**Source text**")
            st.text_area(
                "source",
                document.raw_text[:8000],
                height=320,
                label_visibility="collapsed",
            )
        with right:
            st.markdown("**Structured output**")
            st.json(extraction.result_json)
            st.download_button(
                "Download JSON",
                data=json.dumps(extraction.result_json, indent=2),
                file_name=f"job-{selected_job_id}.json",
                mime="application/json",
            )


def _render_metrics(repo: Repository) -> None:
    st.subheader("Resource and offline status")

    col1, col2, col3 = st.columns(3)
    col1.metric("Local model", "Ready" if is_model_available() else "Missing")
    col2.metric("Database", str(DB_PATH))
    col3.metric("Tesseract OCR", "Ready" if is_tesseract_available() else "Unavailable")

    st.markdown("### Offline mode")
    st.info(
        "This app performs inference locally and stores data in SQLite. "
        "After the GGUF model is downloaded, the extraction flow works without internet access."
    )
    if not is_model_available():
        st.warning(f"Expected model path: `{DEFAULT_MODEL_PATH}`")

    result: PipelineResult | None = st.session_state.get("last_pipeline_result")
    if result is None:
        st.caption("Run an extraction to populate runtime metrics.")
        return

    st.markdown("### Last extraction run")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration (s)", f"{result.metrics.duration_seconds:.2f}")
    m2.metric("Peak CPU %", f"{result.metrics.peak_cpu_percent:.1f}")
    m3.metric("Peak RSS (MB)", f"{result.metrics.peak_rss_mb:.1f}")
    m4.metric("Chunks", result.chunk_count)

    c1, c2, c3 = st.columns(3)
    c1.metric("Document cache hit", "Yes" if result.document_cache_hit else "No")
    c2.metric("Extraction cache hit", "Yes" if result.extraction_cache_hit else "No")
    c3.metric("Repair attempts", result.repair_attempts)

    if result.extraction_cache_hit:
        st.success("Latest run used the local extraction cache (no LLM inference).")


def _fields_to_schema(fields: list[dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in fields:
        name = str(field.get("name", "")).strip()
        if not name:
            continue
        field_type = field.get("type", "string")
        property_schema: dict[str, Any] = {"type": field_type}
        description = str(field.get("description", "")).strip()
        if description:
            property_schema["description"] = description
        properties[name] = property_schema
        if field.get("required"):
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


if __name__ == "__main__":
    main()
