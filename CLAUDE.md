# FedRAMP RAG Assistant — AI Agent Context

> This file is read automatically by Claude Code, Cursor, and other AI coding assistants. It gives your AI agent full context about the project so it can help you effectively without needing to explore the codebase from scratch.

---

## Project Identity

- **Course:** BIT 5544, Virginia Tech Spring 2026
- **Purpose:** Proof-of-concept RAG (Retrieval-Augmented Generation) assistant for FedRAMP authorization artifacts
- **Stack:** Python 3.11 · LlamaIndex · ChromaDB · Streamlit · OpenAI embeddings · OpenAI or Anthropic LLM
- **Root:** The directory containing this file

---

## What This System Does

Users ask FedRAMP compliance questions through a chat interface. The system retrieves the most relevant passages from a local vector database of FedRAMP documents, then passes those passages to an LLM to compose a grounded, cited answer. The LLM is instructed to refuse if the retrieved context is insufficient.

Two document sources are indexed into a single ChromaDB collection:
1. **PDFs** — FedRAMP Rev 5 guidance documents (placed in `docs/`, not yet collected)
2. **FRMR JSON** — FedRAMP machine-readable requirements spec, auto-downloaded from GitHub

---

## Repository Layout

```
.
├── app/
│   ├── main.py              ← Streamlit chat UI
│   └── rag.py               ← RAG engine: index loading, querying, citations
├── scripts/
│   ├── ingest.py            ← PDF chunking and indexing pipeline
│   ├── ingest_json.py       ← FRMR JSON download, parsing, and indexing
│   └── ingest_all.py        ← Orchestrates both pipelines (PDFs first, JSON appends)
├── tests/
│   └── test_connections.py  ← API smoke tests (makes real calls, costs ~$0.00001)
├── eval/
│   └── test_questions.json  ← 10 evaluation questions
├── data/
│   ├── chroma_db/           ← Persisted vector index (gitignored, auto-created by ingest)
│   └── FRMR.documentation.json  ← Cached FRMR JSON (gitignored, auto-downloaded)
├── docs/                    ← Place PDF documents here (gitignored)
├── logs/
│   └── query_log.jsonl      ← Per-query log (gitignored, auto-created)
├── .env                     ← API keys (gitignored — never commit)
├── .env.example             ← Template for .env
├── CLAUDE.md                ← This file
├── PARTNER_SETUP.md         ← Setup instructions for new contributors
├── README.md                ← Project overview and quick-start
├── SYSTEM_OVERVIEW.md  ← Full technical documentation
├── pytest.ini               ← Test configuration
└── requirements.txt         ← Python dependencies
```

---

## Key Architecture Decisions

These decisions are intentional — don't change them without understanding the consequences:

| Decision | What it is | Why |
|----------|-----------|-----|
| Embeddings always use OpenAI | `text-embedding-3-small` via `OPENAI_API_KEY` | Consistency — changing the model invalidates the entire index |
| LLM is configurable | `LLM_PROVIDER=openai` or `anthropic` via `.env` | Allows testing different LLMs without re-ingesting |
| Single ChromaDB collection | `fedramp_docs` holds both PDFs and FRMR JSON | Single similarity search across all sources |
| Refusal via prompt | `QA_PROMPT` in `app/rag.py` instructs the LLM to refuse | No hard score threshold — LLM decides based on context |
| Citation metadata set at index time | `file_name` and `page_label` stored per chunk | Retrieved at query time to build `[FRMR-ADS, p.ADS-CSO-PUB]` labels |
| All queries logged | `logs/query_log.jsonl` (JSONL, one object per line) | Evaluation, debugging, and audit trail |
| Streamlit engine cached | `@st.cache_resource` on `build_query_engine()` | Index loads once per server session, not per query |

---

## Environment Variables

Defined in `.env` (copy from `.env.example`):

| Variable | Default | Required | Notes |
|----------|---------|----------|-------|
| `OPENAI_API_KEY` | — | Always | Used for embeddings regardless of LLM provider |
| `ANTHROPIC_API_KEY` | — | If `LLM_PROVIDER=anthropic` | |
| `LLM_PROVIDER` | `openai` | No | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` / `claude-haiku-4-5-20251001` | No | Must match provider |
| `EMBED_MODEL` | `text-embedding-3-small` | No | Changing requires full re-ingest |

---

## Hardcoded Constants (The Other Knobs)

These are module-level constants, not env vars. Change them in source if needed:

| File | Constant | Value | Effect |
|------|----------|-------|--------|
| `scripts/ingest.py` | `CHUNK_SIZE` | `512` | Tokens per PDF chunk |
| `scripts/ingest.py` | `CHUNK_OVERLAP` | `64` | Overlap between PDF chunks |
| `scripts/ingest_json.py` | `CHUNK_SIZE` | `384` | Tokens per JSON entry chunk |
| `scripts/ingest_json.py` | `CHUNK_OVERLAP` | `32` | Overlap between JSON chunks |
| `app/rag.py` | `TOP_K` | `5` | Chunks retrieved per query |
| `app/rag.py` | `COLLECTION_NAME` | `"fedramp_docs"` | ChromaDB collection name |
| `app/rag.py` | `QA_PROMPT` | *(full prompt)* | LLM system instructions + refusal rule |

After changing chunk sizes or `TOP_K`, re-run ingest and restart the app.

---

## Function Map

### `app/rag.py`

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `build_query_engine` | `() -> RetrieverQueryEngine` | Query engine | Raises `FileNotFoundError` if index missing |
| `query` | `(engine, question: str) -> dict` | `{answer, citations, chunks}` | Core query function |
| `_get_llm` | `() -> LLM` | LlamaIndex LLM | Reads `LLM_PROVIDER` from env |
| `_get_embed_model` | `() -> OpenAIEmbedding` | Embedding model | Always OpenAI |
| `_format_citation` | `(node: NodeWithScore) -> str` | `"[file, p.ID]"` | Reads node metadata |
| `_dedupe` | `(items: list) -> list` | Deduplicated list | Order-preserving |

### `app/main.py`

| Function | Notes |
|----------|-------|
| `get_engine()` | Streamlit-cached; calls `rag.build_query_engine()` once |
| `_render_message(msg)` | Renders one chat bubble with citations and context expander |
| `_log(question, result)` | Appends JSON record to `logs/query_log.jsonl` |

### `scripts/ingest.py`

| Function | Signature | Notes |
|----------|-----------|-------|
| `ingest_pdfs` | `(drop_existing=True) -> int` | Returns page count; `drop_existing=True` wipes collection first |

### `scripts/ingest_json.py`

| Function | Signature | Notes |
|----------|-----------|-------|
| `ingest_frmr` | `(drop_existing=True) -> int` | Returns document count |
| `_download_frmr` | `() -> dict` | Downloads or loads cached FRMR JSON |
| `_parse_frd` | `(frd_section) -> list[Document]` | Parses FedRAMP definitions |
| `_parse_frr` | `(frr_section) -> list[Document]` | Parses FedRAMP requirements |

### `scripts/ingest_all.py`

| Function | Notes |
|----------|-------|
| `main()` | Calls `ingest_pdfs(drop_existing=True)` then `ingest_frmr(drop_existing=False)` |

---

## Data Flow Summary

### Ingest (offline, run once)
```
docs/*.pdf → SimpleDirectoryReader → SentenceSplitter (512/64) → OpenAI embed → ChromaDB
FRMR GitHub → urllib download → _parse_frd/_parse_frr → SentenceSplitter (384/32) → OpenAI embed → ChromaDB (append)
```

### Query (every user message)
```
User question → OpenAI embed → ChromaDB cosine search (top 5) → QA_PROMPT + chunks → LLM → answer + citations → Streamlit UI → query_log.jsonl
```

---

## Citation System

Metadata is stored per chunk at index time and used at query time to build citation labels.

| Source | `file_name` metadata | `page_label` metadata | Citation format |
|--------|---------------------|----------------------|----------------|
| PDF | filename.pdf | page number | `[filename.pdf, p.12]` |
| FRMR definition | `"FRMR"` | definition ID | `[FRMR, p.FRD-ACV]` |
| FRMR requirement | `"FRMR-ADS"` etc. | requirement ID | `[FRMR-ADS, p.ADS-CSO-PUB]` |

---

## Current Corpus Status

| Source | Status | What it covers |
|--------|--------|---------------|
| FRMR JSON | Indexed | Program requirements: trust centers (ADS), persistent assessment (PVA), offering boundary (MAS), vulnerability disclosure (VDR), and all FedRAMP definitions (FRD) |
| Rev 5 PDFs | Not yet collected | NIST 800-53 controls, impact levels, SSP/RAR/POA&M guidance, MFA, encryption, pen testing |

**The 10 evaluation questions in `eval/test_questions.json` all require Rev 5 PDFs.** They will return polite refusals until PDFs are ingested.

**Good questions for the current corpus:**
- What is a trust center in FedRAMP?
- What does FedRAMP require for vulnerability disclosure?
- How do you define the boundary of a cloud service offering?
- What is a persistent vulnerability assessment?
- What does ADS stand for and what does it govern?

---

## Common Tasks

### Run the app
```bash
.venv\Scripts\activate     # Windows
streamlit run app/main.py
```

### Build or rebuild the index
```bash
python scripts/ingest_json.py    # FRMR JSON only (no PDFs needed)
python scripts/ingest_all.py     # Full corpus (PDFs + JSON)
```

### Run tests
```bash
pytest tests/test_connections.py -v    # Smoke tests (real API calls)
pytest -v                              # All tests
```

### Force fresh FRMR JSON download
```bash
del data\FRMR.documentation.json     # Windows
rm data/FRMR.documentation.json      # Mac/Linux
# then re-run ingest
```

### Wipe and rebuild the index from scratch
```bash
# Delete the vector store
rm -rf data/chroma_db/               # Mac/Linux
rd /s /q data\chroma_db              # Windows
# then re-run ingest
```

---

## Dependencies

See `requirements.txt` for pinned versions. Key packages:

| Package | Purpose |
|---------|---------|
| `llama-index-core` | RAG orchestration, chunking, retrieval, prompting |
| `llama-index-vector-stores-chroma` | ChromaDB integration for LlamaIndex |
| `llama-index-embeddings-openai` | OpenAI embedding model wrapper |
| `llama-index-llms-openai` | OpenAI LLM wrapper |
| `llama-index-llms-anthropic` | Anthropic Claude LLM wrapper |
| `chromadb` | Local vector database (persisted to `data/chroma_db/`) |
| `streamlit` | Web UI framework |
| `python-dotenv` | Loads `.env` file |

---

## Known Issues and Constraints

- The `.venv/` directory is local and not committed. Each contributor creates their own.
- `data/chroma_db/` is local and not committed. Each contributor runs ingest to build it.
- `docs/*.pdf` are gitignored. PDFs must be shared separately and placed in `docs/` manually.
- Changing `EMBED_MODEL` invalidates the existing index — you must wipe `data/chroma_db/` and re-ingest.
- The app must be restarted after re-ingesting to pick up the new index (Streamlit caches the engine at startup).
- Query log (`logs/query_log.jsonl`) grows indefinitely — rotate or archive manually.

---

## Further Reading

- `SYSTEM_OVERVIEW.md` — Deep-dive into every function, configuration option, data flow, citation system, and optimization opportunities
- `PARTNER_SETUP.md` — Step-by-step setup for new contributors
- `eval/test_questions.json` — 10 test questions for evaluating answer quality
- `logs/query_log.jsonl` — Runtime query log with similarity scores, useful for debugging retrieval quality
