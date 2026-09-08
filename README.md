# Study Assistant

Study Assistant is a Vietnamese-first learning workspace built around an explicit Advanced RAG
pipeline. It ingests study documents, preserves verifiable source locations, and is designed to
support grounded chat, retrieval comparison, summaries, quizzes, and mind maps.

The project keeps the RAG pipeline visible and testable instead of hiding it behind an
orchestration framework.

## Current milestone

Phase 3 adds reliable PDF, DOCX, and PPTX ingestion:

- defensive file and OpenXML validation with bounded resource usage
- page-, slide-, section-, and sentence-aware chunks with deterministic IDs
- versioned contextual retrieval prefixes kept separate from citation evidence
- Gemini document embeddings in validated batches
- workspace-scoped Qdrant storage with deduplication and staged replacement

The chat and study-tool views remain disabled until their retrieval and grounding phases are
implemented.

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

Upload PDF, DOCX, or PPTX files in **Tài liệu**, then choose **Lập chỉ mục tài liệu**. Deterministic
context is enabled by default. Gemini context enrichment is optional because it adds latency and
API usage; failures fall back to deterministic context without discarding the upload.

## Configuration

Configuration precedence is:

1. Environment variables
2. Streamlit secrets
3. Safe non-secret defaults

Supported storage modes:

- `local`: persistent Qdrant data under `.data/qdrant`
- `demo`: in-memory data isolated by browser session
- `cloud`: Qdrant URL and API key; every operation is filtered by workspace

Important ingestion controls include `MAX_UPLOAD_MB`, `MAX_DOCUMENT_PAGES`, `CHUNK_SIZE_CHARS`,
`CHUNK_OVERLAP_CHARS`, `EMBEDDING_BATCH_SIZE`, and `MAX_LLM_CONTEXT_CHUNKS`.

Never commit `.streamlit/secrets.toml`, `.env`, API keys, or local vector data.

## Roadmap

- Grounded chat and verified chunk-level citations
- Dense baseline, hybrid retrieval, query rewriting, multi-query, and reranking
- Offline evaluation and ablation reporting
- Sourced summary, quiz, and mind-map tools

The evaluation results will be published only after the hold-out benchmark is implemented.
