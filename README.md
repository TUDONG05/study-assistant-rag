# Study Assistant

Study Assistant is a Vietnamese-first learning workspace built around an explicit Advanced RAG
pipeline. It is designed to ingest study documents, answer with verified citations, compare
retrieval strategies, and generate sourced summaries, quizzes, and mind maps.

The project intentionally keeps the RAG pipeline visible and testable instead of hiding it behind
an orchestration framework.

## Current milestone

Phase 2 provides the Streamlit application shell, typed configuration, isolated session state,
BYOK input, storage-mode contracts, and placeholder views for the complete product journey.
Document ingestion is introduced in Phase 3, so the uploader remains disabled in this milestone.

## Quick start

Requirements: Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

The app runs in `local` storage mode without cloud credentials. Gemini always uses BYOK: enter a
personal API key in the sidebar. The key remains in the current Streamlit session and is first
validated when an AI feature makes a request.

## Configuration

Configuration precedence is:

1. Environment variables
2. Streamlit secrets
3. Safe non-secret defaults

Supported storage modes:

- `local`: persistent Qdrant data under `.data/qdrant`
- `demo`: in-memory data isolated by browser session
- `cloud`: Qdrant URL and API key, with mandatory workspace filtering in later phases

Never commit `.streamlit/secrets.toml`, `.env`, API keys, or local vector data.

## Roadmap

- Reliable PDF/DOCX/PPTX ingestion with contextual chunks
- Grounded chat and verified chunk-level citations
- Dense baseline, hybrid retrieval, query rewriting, multi-query, and reranking
- Offline evaluation and ablation reporting
- Sourced summary, quiz, and mind-map tools

The evaluation results will be published only after the hold-out benchmark is implemented.
