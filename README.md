# FedRAMP Readiness Assistant

A Streamlit RAG application that answers FedRAMP authorization questions grounded exclusively in a local document corpus. Supports both PDF documents and FedRAMP's official machine-readable JSON specification (FRMR).

---

## Prerequisites

- Python 3.11
- OpenAI API key — always required (embeddings + optional LLM)
- Anthropic API key — only if `LLM_PROVIDER=anthropic`

---

## Setup

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install ruff   # code formatter (optional but recommended)

# 3. Configure environment
cp .env.example .env
# Open .env and fill in your API keys

# 4. Verify everything works
pytest tests/test_connections.py -v
```

---

## Ingest

Three ingest scripts are available depending on your corpus:

```powershell
# FRMR JSON only — no PDFs needed, works immediately
python scripts/ingest_json.py

# PDFs only — place files in docs/ first
python scripts/ingest.py

# Full corpus: PDFs + FRMR JSON (recommended)
python scripts/ingest_all.py
```

**FRMR JSON** is downloaded automatically from the FedRAMP GitHub repository
and cached at `data/FRMR.documentation.json`. Delete that file to force a
fresh download when a new version releases.

**PDFs** go in `docs/` (gitignored). Re-run the relevant ingest script any
time documents change.

---

## Run

```powershell
streamlit run app/main.py
```

Restart the app after re-ingesting to pick up the new index.

---

## Project Structure

```
fedramp-rag/
├── app/
│   ├── main.py               # Streamlit UI and chat loop
│   └── rag.py                # RAG engine: index loading, querying, citations
├── scripts/
│   ├── ingest.py             # PDF ingestion pipeline
│   ├── ingest_json.py        # FRMR JSON ingestion pipeline
│   └── ingest_all.py         # Runs both pipelines in sequence
├── tests/
│   ├── test_connections.py   # API key + service smoke tests
│   └── test_rag.py           # RAG unit tests (TDD — add as you build)
├── data/
│   ├── chroma_db/            # Persisted vector store (gitignored)
│   └── FRMR.documentation.json  # Cached FRMR spec (gitignored)
├── docs/                     # Place PDF documents here (gitignored)
├── eval/
│   └── test_questions.json   # 10 evaluation questions
├── logs/
│   └── query_log.jsonl       # Per-query log (gitignored, auto-created)
├── .env                      # Your real API keys (gitignored)
├── .env.example              # Template — safe to commit
├── pytest.ini
└── requirements.txt
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name matching the provider |
| `OPENAI_API_KEY` | — | Required always (embeddings) |
| `ANTHROPIC_API_KEY` | — | Required only if `LLM_PROVIDER=anthropic` |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |

---

## How It Works

1. **Ingest** — Documents are chunked, embedded with `text-embedding-3-small`, and stored in a local ChromaDB collection.
   - PDFs: 512-token chunks, page-level metadata preserved
   - FRMR JSON: 384-token chunks, requirement ID and document name preserved

2. **Query** — The user's question is embedded and the top-5 most similar chunks are retrieved.

3. **Generate** — The LLM answers using only the retrieved chunks.

4. **Citations** — Inline citations in every response:
   - PDF source: `[filename.pdf, p.12]`
   - FRMR requirement: `[FRMR-ADS, p.ADS-CSO-PUB]`
   - FRMR definition: `[FRMR, p.FRD-ACV]`

5. **Refusal** — Fixed refusal message when retrieved context does not support an answer.

6. **Logging** — Every query appended to `logs/query_log.jsonl` with timestamp, question, answer, citations, and chunks.

---

## Testing

```powershell
# Smoke tests — verify API keys and services (makes real API calls)
pytest tests/test_connections.py -v

# All tests
pytest -v

# With coverage
pytest --cov=app --cov=scripts --cov-report=term-missing
```

### Known: re-run pip install after requirements.txt changes

If connection tests fail with import errors, re-run:
```powershell
pip install -r requirements.txt
```

---

## Evaluation

`eval/test_questions.json` contains 10 FedRAMP questions covering impact levels, SSP requirements, POA&M, authorization boundary, encryption standards, JAB vs. Agency authorization, penetration testing, and MFA. Run them through the chat UI and inspect `logs/query_log.jsonl` for answer quality and citation accuracy.
