# FedRAMP Readiness Assistant

A Streamlit RAG application that answers FedRAMP authorization questions grounded exclusively in a local document corpus.

---

## Prerequisites

- Python 3.11
- OpenAI API key (always required — used for embeddings)
- Anthropic **or** OpenAI API key for the generator LLM (configurable)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

### 3. Add documents

Place FedRAMP PDF files in the `docs/` directory.
The directory is gitignored — PDFs will never be committed.

### 4. Ingest documents

```bash
python scripts/ingest.py
```

This reads every PDF in `docs/`, splits it into chunks, embeds with OpenAI,
and persists the vector store to `data/chroma_db/`.

**Re-run this every time you add or replace documents.** Each run does a full
rebuild, so the index always reflects what is in `docs/`.

### 5. Run the app

```bash
streamlit run app/main.py
```

---

## Project Structure

```
fedramp-rag/
├── app/
│   ├── main.py               # Streamlit UI and chat loop
│   └── rag.py                # RAG engine: index loading, querying, citations
├── scripts/
│   └── ingest.py             # PDF ingestion and indexing pipeline
├── data/
│   └── chroma_db/            # Persisted vector store (gitignored, generated)
├── docs/                     # Place PDF documents here (gitignored)
├── eval/
│   └── test_questions.json   # 10 evaluation questions
├── logs/
│   └── query_log.jsonl       # Per-query log: question, answer, citations, chunks
├── .env.example              # Template for environment variables
├── requirements.txt
└── README.md
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name matching the provider |
| `OPENAI_API_KEY` | — | Required for embeddings (always) |
| `ANTHROPIC_API_KEY` | — | Required only if `LLM_PROVIDER=anthropic` |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model |

To use Anthropic as the generator:
```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-20241022
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...   # still needed for embeddings
```

---

## How It Works

1. **Ingest** — PDFs are loaded with page-level metadata preserved, split into
   512-token chunks with 64-token overlap, embedded with `text-embedding-3-small`,
   and stored in a local ChromaDB collection.

2. **Query** — The user's question is embedded and the top-5 most similar chunks
   are retrieved. The LLM generates an answer using only those chunks.

3. **Citations** — Every response includes inline citations in the form
   `[filename, p.X]` (or `[filename, chunk-ID]` when page info is unavailable).

4. **Refusal** — The LLM is instructed to respond with a fixed refusal message
   when the retrieved context does not support an answer.

5. **Logging** — Every query is appended to `logs/query_log.jsonl` with
   timestamp, question, answer, citations, and retrieved chunks.

---

## Evaluation

`eval/test_questions.json` contains 10 FedRAMP questions covering:

- Impact levels and baselines
- SSP content requirements
- Readiness Assessment Reports
- Continuous monitoring cadence
- POA&M requirements
- Authorization boundary
- Encryption standards (FIPS 140-2/3)
- JAB vs. Agency authorization paths
- Penetration testing requirements
- Multi-factor authentication

Run them manually through the chat UI and inspect `logs/query_log.jsonl`
to evaluate answer quality and citation accuracy.

---

## Notes

- The vector store is rebuilt from scratch on every `ingest.py` run.
- The Streamlit app caches the query engine in memory; restart the app after
  re-ingesting to pick up the new index.
- `logs/query_log.jsonl` grows indefinitely — delete or rotate it as needed.
