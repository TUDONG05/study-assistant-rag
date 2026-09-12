# Study Assistant

Study Assistant is a Vietnamese-first learning workspace built around an explicit Advanced RAG
pipeline. It ingests study documents, preserves verifiable source locations, and is designed to
support grounded chat, retrieval comparison, summaries, quizzes, and mind maps.

The project keeps the RAG pipeline visible and testable instead of hiding it behind an
orchestration framework.

## Current milestone

Phase 4 adds grounded chat on top of reliable PDF, DOCX, and PPTX ingestion:

- defensive file and OpenXML validation with bounded resource usage
- page-, slide-, section-, and sentence-aware chunks with deterministic IDs
- versioned contextual retrieval prefixes kept separate from citation evidence
- Gemini Embedding 2 document and question embeddings in validated batches
- workspace-scoped Qdrant storage with deduplication and staged replacement
- dense cosine retrieval restricted to active document versions and the selected document scope
- structured Gemini answers with server-validated, chunk-level inline citations
- deterministic refusal when the selected documents do not contain enough evidence

Hybrid retrieval, reranking, evaluation, and study-tool views remain future phases.

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

Open **Chat**, optionally select a subset of compatible documents, and ask a question. An empty
selection searches every compatible active document. Answers cite source IDs such as `[S1]`; the
source panel resolves each ID to trusted file and page, slide, or section metadata. Retrieval scores
are ranking metadata, not answer-confidence probabilities.

### Phase 4 re-index requirement

Phase 4 corrects the input format for `gemini-embedding-2`. Documents indexed by Phase 3 use an old
pipeline identity and are intentionally excluded from chat. Re-upload and index those documents once
to create compatible vectors; the document registry will activate the replacement version.

## Configuration

Configuration precedence is:

1. Environment variables
2. Streamlit secrets
3. Safe non-secret defaults

Supported storage modes:

- `local`: persistent Qdrant data under `.data/qdrant` for a single-user deployment
- `demo`: in-memory data isolated by browser session
- `cloud`: Qdrant URL and API key; data is session-isolated until persistent identity is added

Important ingestion controls include `MAX_UPLOAD_MB`, `MAX_DOCUMENT_PAGES`, `CHUNK_SIZE_CHARS`,
`CHUNK_OVERLAP_CHARS`, `EMBEDDING_BATCH_SIZE`, and `MAX_LLM_CONTEXT_CHUNKS`. Chat controls include
`RETRIEVAL_TOP_K` (default `6`), `RETRIEVAL_SCORE_THRESHOLD` (provisional default `0.5`),
`MAX_QUESTION_CHARS` (default `4000`), and `MAX_REQUESTS_PER_SESSION`.

Never commit `.streamlit/secrets.toml`, `.env`, API keys, or local vector data.

## Roadmap

- Hybrid retrieval, query rewriting, multi-query, and reranking
- Offline evaluation and ablation reporting
- Sourced summary, quiz, and mind-map tools

The evaluation results will be published only after the hold-out benchmark is implemented.
