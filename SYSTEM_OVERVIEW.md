# FedRAMP RAG Assistant — System Overview

> A technical guide to how the system works, what every piece does, how they connect, and where you can tune or improve things.

---

## Table of Contents

1. [What This System Does (Big Picture)](#1-what-this-system-does-big-picture)
2. [Repository Layout](#2-repository-layout)
3. [The Two-Phase Lifecycle: Ingest vs. Query](#3-the-two-phase-lifecycle-ingest-vs-query)
4. [Phase 1 — Ingest: Building the Knowledge Base](#4-phase-1--ingest-building-the-knowledge-base)
   - 4.1 [PDF Ingest (`scripts/ingest.py`)](#41-pdf-ingest-scriptsingestpy)
   - 4.2 [FRMR JSON Ingest (`scripts/ingest_json.py`)](#42-frmr-json-ingest-scriptsingest_jsonpy)
   - 4.3 [Master Orchestrator (`scripts/ingest_all.py`)](#43-master-orchestrator-scriptsingest_allpy)
5. [Phase 2 — Query: Answering Questions](#5-phase-2--query-answering-questions)
   - 5.1 [RAG Engine (`app/rag.py`)](#51-rag-engine-appragpy)
   - 5.2 [Streamlit UI (`app/main.py`)](#52-streamlit-ui-appmainpy)
6. [How the Pieces Connect: Full Data Flow](#6-how-the-pieces-connect-full-data-flow)
7. [The Citation System](#7-the-citation-system)
8. [Configuration Reference ("The Knobs")](#8-configuration-reference-the-knobs)
9. [The Query Log](#9-the-query-log)
10. [Tests and Smoke Checks](#10-tests-and-smoke-checks)
11. [What the Corpus Actually Contains](#11-what-the-corpus-actually-contains)
12. [Optimization and Improvement Opportunities](#12-optimization-and-improvement-opportunities)

---

## 1. What This System Does (Big Picture)

This is a **Retrieval-Augmented Generation (RAG)** assistant. RAG is a pattern where, instead of asking an LLM to answer from memory, you first search a local document database for the most relevant passages, then hand those passages to the LLM as context, and ask it to answer using only what it was given.

The flow in one sentence: **User asks a question → system searches indexed FedRAMP documents → top 5 passages go to an LLM → LLM writes an answer with inline citations → answer appears in a chat UI.**

The key benefit is that answers are grounded in actual FedRAMP documents rather than the LLM's general training data, and every claim can be traced back to a source.

---

## 2. Repository Layout

```
FedRamp RAG Project/
│
├── app/
│   ├── main.py              ← Streamlit chat UI (what you see in the browser)
│   └── rag.py               ← RAG engine (search + LLM logic)
│
├── scripts/
│   ├── ingest.py            ← PDF loading and indexing pipeline
│   ├── ingest_json.py       ← FRMR machine-readable JSON pipeline
│   └── ingest_all.py        ← Runs both pipelines in the right order
│
├── tests/
│   └── test_connections.py  ← Smoke tests: are API keys working?
│
├── eval/
│   └── test_questions.json  ← 10 evaluation questions for testing quality
│
├── data/
│   ├── chroma_db/           ← The vector database (auto-created, gitignored)
│   └── FRMR.documentation.json ← Cached FRMR JSON download (gitignored)
│
├── docs/                    ← Place your PDF documents here (gitignored)
├── logs/
│   └── query_log.jsonl      ← Every query logged here (gitignored)
│
├── .env                     ← Your API keys (gitignored, never commit this)
├── .env.example             ← Template showing what .env needs
├── requirements.txt         ← Python dependencies
└── pytest.ini               ← Test runner configuration
```

---

## 3. The Two-Phase Lifecycle: Ingest vs. Query

The system has two completely separate phases that you run at different times.

### Phase 1: Ingest (run once, or when documents change)
```
python scripts/ingest_all.py
```
This reads your documents, converts them into numerical vectors (embeddings), and stores them in a local database. This is **offline work** — the LLM is not involved. You pay only for embedding API calls. After this runs, you have a persistent knowledge base on disk.

### Phase 2: Query (run every time you use the app)
```
streamlit run app/main.py
```
This starts the web server. When you ask a question, the system searches the database from Phase 1, fetches the best matching passages, and sends them to the LLM to compose an answer. The LLM is involved here, so every question costs a small amount in API tokens.

**You only need to re-run ingest if you add, remove, or update documents.**

---

## 4. Phase 1 — Ingest: Building the Knowledge Base

### 4.1 PDF Ingest (`scripts/ingest.py`)

**What it does:** Reads every `.pdf` file from the `docs/` folder, splits each into overlapping text chunks, generates a vector embedding for each chunk, and saves everything to the ChromaDB vector store.

#### Key function: `ingest_pdfs(drop_existing: bool = True) -> int`

| Step | What happens |
|------|-------------|
| 1 | Glob `docs/*.pdf` — if none found, stops early |
| 2 | Check `OPENAI_API_KEY` is set |
| 3 | Use `SimpleDirectoryReader` to load PDFs (preserves page numbers as metadata) |
| 4 | If `drop_existing=True`, delete the ChromaDB collection and rebuild from scratch |
| 5 | Split text using `SentenceSplitter` — chunks of **512 tokens, 64-token overlap** |
| 6 | Call OpenAI's embedding API for each chunk — produces a 1536-dimensional vector |
| 7 | Store each (vector, text, metadata) tuple in ChromaDB |
| 8 | Return the number of document pages loaded |

**What gets stored in metadata per chunk:**
- `file_name` — The PDF filename (used for citations)
- `page_label` — The page number (used for citations)

**Why 512 tokens with 64-token overlap?**
512 tokens is roughly a half-page of text — enough context to be meaningful but short enough that you can retrieve specifically relevant sections. The 64-token overlap prevents a sentence from being cut in half and lost between two chunks.

---

### 4.2 FRMR JSON Ingest (`scripts/ingest_json.py`)

**What it does:** Downloads the FedRAMP machine-readable requirements document from GitHub, parses it into structured text entries, embeds each entry, and adds them to ChromaDB alongside the PDFs.

The FRMR JSON is structured data (not a PDF), so this pipeline has its own parser.

#### Key function: `ingest_frmr(drop_existing: bool = True) -> int`

| Step | What happens |
|------|-------------|
| 1 | Download or load cached `data/FRMR.documentation.json` |
| 2 | Parse the `FRD` section (definitions) via `_parse_frd()` |
| 3 | Parse the `FRR` section (requirements) via `_parse_frr()` |
| 4 | Embed all documents using OpenAI |
| 5 | Store in ChromaDB (appending, not replacing) |

#### Helper: `_download_frmr() -> dict`
Downloads from `https://raw.githubusercontent.com/FedRAMP/docs/main/FRMR.documentation.json` and caches it at `data/FRMR.documentation.json`. On subsequent runs, loads from cache. Delete the cache file to force a fresh download.

#### Helper: `_parse_frd(frd_section) -> list[Document]`
Parses the **FedRAMP Definitions** section. Each term becomes one LlamaIndex `Document`. The text combines: term name, definition, any notes, references, and alternate names.

Metadata assigned:
- `file_name = "FRMR"` (so citations say `[FRMR, p.FRD-ACV]`)
- `page_label = frd_id` (the definition ID, e.g., "FRD-ACV")
- `type = "definition"`

#### Helper: `_parse_frr(frr_section) -> list[Document]`
Parses the **FedRAMP Requirements and Recommendations** section. The JSON is deeply nested:
- Top level: document names (ADS, PVA, MAS, VDR, etc.)
- Within each document: frameworks (20x, rev5, both)
- Within each framework: actor groups (cso_pub, provider, etc.)
- Within each actor: individual requirement entries

Each requirement becomes one Document. The text combines: name, primary keyword, requirement statement, what it affects, and any notes. If the requirement varies by impact level, those variations are included.

Metadata assigned:
- `file_name = "FRMR-{doc}"` (e.g., "FRMR-ADS", so citations say `[FRMR-ADS, p.ADS-CSO-PUB]`)
- `page_label = req_id` (the requirement ID, e.g., "ADS-CSO-PUB")
- `type = "requirement"`

**Why smaller chunks (384/32) for JSON vs PDFs (512/64)?**
FRMR JSON entries are already pre-scoped to a single definition or requirement — they rarely need to be split further. The smaller chunk size means each entry stays whole rather than being artificially split.

---

### 4.3 Master Orchestrator (`scripts/ingest_all.py`)

**What it does:** Runs both pipelines in the correct order.

```
ingest_pdfs(drop_existing=True)    ← Phase 1: wipes and rebuilds
     ↓
ingest_frmr(drop_existing=False)   ← Phase 2: appends JSON to existing
```

**Why this order?**
- PDFs run first with `drop_existing=True`, which wipes the collection and starts clean
- FRMR JSON runs second with `drop_existing=False`, which appends into the same collection

This means both document types end up in a single unified ChromaDB collection (`fedramp_docs`), so a single query searches all sources simultaneously.

---

## 5. Phase 2 — Query: Answering Questions

### 5.1 RAG Engine (`app/rag.py`)

This is the core intelligence layer. It loads the index, handles searches, orchestrates LLM calls, and formats results.

#### Public function: `build_query_engine() -> RetrieverQueryEngine`

Called once when the app starts (cached by Streamlit). It:
1. Checks `data/chroma_db/` exists — raises `FileNotFoundError` if not (index not built yet)
2. Connects to the ChromaDB persistent client
3. Configures the **embedding model** (`_get_embed_model()`) — always OpenAI
4. Configures the **LLM** (`_get_llm()`) — OpenAI or Anthropic based on `.env`
5. Wraps the collection in a LlamaIndex `VectorStoreIndex`
6. Creates a `VectorIndexRetriever` set to retrieve **top 5** most similar chunks
7. Creates a response synthesizer that uses the `QA_PROMPT` system prompt
8. Returns a `RetrieverQueryEngine` combining the retriever and synthesizer

#### Public function: `query(engine, question) -> dict`

Called on every user message. Returns:
```python
{
    "answer":    "The LLM's full response text",
    "citations": ["[FRMR-ADS, p.ADS-CSO-PUB]", "[guide.pdf, p.12]"],  # deduplicated
    "chunks": [
        {
            "text":     "Full retrieved chunk text",
            "score":    0.7823,   # cosine similarity (0-1, higher = more relevant)
            "metadata": {...},    # all metadata stored at index time
            "citation": "[FRMR-ADS, p.ADS-CSO-PUB]"
        },
        # ... up to 5 chunks
    ]
}
```

Internally: calls `engine.query()` → LlamaIndex retrieves top-5 chunks → synthesizer passes them + the `QA_PROMPT` to the LLM → answer returned with source node metadata.

#### Private function: `_get_llm() -> LLM`

Reads `LLM_PROVIDER` from environment:
- `"anthropic"` → `llama_index.llms.anthropic.Anthropic(model=LLM_MODEL)`
- anything else (default `"openai"`) → `llama_index.llms.openai.OpenAI(model=LLM_MODEL)`

#### Private function: `_get_embed_model() -> OpenAIEmbedding`

Always returns OpenAI's `text-embedding-3-small`. The embedding model cannot be swapped via env vars because changing it would make the existing index incompatible (different dimensions). Any embedding model change requires a full re-ingest.

#### The `QA_PROMPT` System Prompt

This is the instruction given to the LLM on every query. It tells the LLM to:
- Answer **only** from the provided retrieved context
- Cite every claim inline as `[filename, p.X]`
- Refuse with a specific message if context is insufficient
- Never speculate or draw on outside knowledge

The refusal message: *"I cannot find sufficient information about this in the provided FedRAMP documents."*

**Important:** The refusal is enforced by the prompt, not by a similarity score threshold. Even if retrieval returns low-scoring chunks, those chunks get passed to the LLM, and the LLM is instructed to refuse if they don't actually answer the question.

#### Private function: `_format_citation(node) -> str`

Reads `node.node.metadata` and builds the citation label:

| Condition | Output format | Example |
|-----------|---------------|---------|
| Has `page_label` | `[file_name, p.PAGE]` | `[FRMR-ADS, p.ADS-CSO-PUB]` |
| No `page_label` | `[file_name, chunk-NODEID]` | `[guide.pdf, chunk-a1b2c3d4]` |

The `file_name` has its directory path stripped (basename only).

#### Private function: `_dedupe(items) -> list`

Removes duplicate citations while preserving their order of appearance.

---

### 5.2 Streamlit UI (`app/main.py`)

**What it does:** Provides the web-based chat interface. Handles user input, displays results, and logs queries.

#### `get_engine()` (Streamlit-cached with `@st.cache_resource`)

Calls `rag.build_query_engine()` once and caches the result for the lifetime of the Streamlit server. A loading spinner is shown the first time. Subsequent questions skip this step entirely.

#### `_render_message(msg: dict) -> None`

Renders a single message bubble:
- User messages: plain text
- Assistant messages: answer text + inline citation list + expandable "Retrieved context" section showing each chunk's citation label, similarity score, and text preview

#### `_log(question, result) -> None`

After every query, appends a structured JSON record to `logs/query_log.jsonl`:
```json
{
  "id": "uuid-here",
  "timestamp": "2026-03-10T14:37:00.123456+00:00",
  "question": "What is an authorization boundary?",
  "answer": "An authorization boundary is...",
  "citations": ["[FRMR-MAS, p.MAS-CSO-PUB-01]"],
  "chunks": [{"text": "...", "score": 0.8123, "citation": "..."}]
}
```

#### Main chat loop

```
1. User types in st.chat_input()
2. Message added to st.session_state.messages (chat history)
3. All previous messages re-rendered from session state
4. get_engine() called (cached, instant after first load)
5. rag.query(engine, question) called → returns answer dict
6. Answer, citations, and chunks rendered
7. Assistant message added to session state
8. _log() called to write to query_log.jsonl
```

The sidebar shows: project description, how RAG works, path to log file, and running query count.

---

## 6. How the Pieces Connect: Full Data Flow

### Ingest (run once)

```
docs/*.pdf
    │
    └─► scripts/ingest.py
            │  SimpleDirectoryReader (preserves page metadata)
            │  SentenceSplitter (512 tok / 64 overlap)
            │  OpenAI text-embedding-3-small → 1536-dim vector
            ▼
        data/chroma_db/  ◄── ChromaDB persistent store
            ▲
    └─► scripts/ingest_json.py
            │  urllib.request (download) → data/FRMR.documentation.json
            │  _parse_frd() → LlamaIndex Documents with metadata
            │  _parse_frr() → LlamaIndex Documents with metadata
            │  SentenceSplitter (384 tok / 32 overlap)
            │  OpenAI text-embedding-3-small → 1536-dim vector
            ▼
        (same collection, appended)
```

### Query (every user message)

```
Browser: User types question
    │
    ▼
app/main.py: st.chat_input() captures text
    │
    ▼
app/rag.py: build_query_engine() [first time only]
    │  chromadb.PersistentClient → loads data/chroma_db/
    │  Settings.embed_model = OpenAI text-embedding-3-small
    │  Settings.llm = OpenAI gpt-4o-mini  (or Anthropic claude-haiku)
    │  VectorIndexRetriever(similarity_top_k=5)
    │  RetrieverQueryEngine(retriever + synthesizer + QA_PROMPT)
    │
    ▼
app/rag.py: query(engine, question)
    │
    ├─► Step 1 — RETRIEVAL
    │       Question text → OpenAI embedding → 1536-dim query vector
    │       ChromaDB cosine similarity search → top-5 matching chunks
    │       Each chunk has: text, metadata, similarity score
    │
    ├─► Step 2 — AUGMENTED GENERATION
    │       QA_PROMPT + question + top-5 chunk texts → LLM API call
    │       LLM writes answer with inline citations
    │       Returns: answer text + source node references
    │
    ├─► Step 3 — CITATION FORMATTING
    │       _format_citation() reads metadata from each source node
    │       Builds citation labels, deduplicates
    │
    └─► Returns {answer, citations, chunks}
    │
    ▼
app/main.py: _render_message() → displayed in browser
    │
    ▼
app/main.py: _log() → appended to logs/query_log.jsonl
```

---

## 7. The Citation System

Citations are built from metadata that is attached to each document chunk **at index time** and retrieved **at query time**.

### How metadata gets set

**For PDFs:**
LlamaIndex's `SimpleDirectoryReader` automatically sets:
- `file_name` — the PDF's filename
- `page_label` — the page number from the PDF

**For FRMR JSON:**
Set manually in `_parse_frd()` and `_parse_frr()`:
- `file_name` — e.g., `"FRMR"` for definitions, `"FRMR-ADS"` for ADS requirements
- `page_label` — the requirement or definition ID (e.g., `"ADS-CSO-PUB"`)

### Citation formats

| Source | Example citation |
|--------|-----------------|
| PDF page | `[FedRAMP_High_Baseline.pdf, p.42]` |
| PDF (no page) | `[guide.pdf, chunk-a1b2c3d4]` |
| FRMR definition | `[FRMR, p.FRD-ACV]` |
| FRMR requirement | `[FRMR-ADS, p.ADS-CSO-PUB]` |

### Where citations appear
1. **Inline in the LLM's answer** — the LLM is instructed in the system prompt to cite every claim
2. **Below the answer** — the deduplicated list of cited sources
3. **In the "Retrieved context" expander** — paired with similarity scores and text previews
4. **In the query log** — stored permanently for later analysis

---

## 8. Configuration Reference ("The Knobs")

These are all the values you can change to alter how the system behaves.

### Environment variables (`.env` file)

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI credentials — needed for embeddings always |
| `ANTHROPIC_API_KEY` | *(required if using Anthropic)* | Anthropic credentials |
| `LLM_PROVIDER` | `openai` | Which LLM to use: `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` (OpenAI) or `claude-haiku-4-5-20251001` (Anthropic) | Specific model name |
| `EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model (changing this requires full re-ingest) |

**Examples:**

Switch to GPT-4o:
```
LLM_MODEL=gpt-4o
```

Switch to Claude Sonnet:
```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...
```

Switch to larger embedding model (requires re-ingest):
```
EMBED_MODEL=text-embedding-3-large
```

### Hardcoded constants (require code changes)

These are defined as module-level constants in the source files. They are not exposed as environment variables but are easy to find and change.

| File | Constant | Current value | Effect |
|------|----------|---------------|--------|
| `scripts/ingest.py` | `CHUNK_SIZE` | `512` | Tokens per PDF chunk — larger = more context per chunk, fewer chunks |
| `scripts/ingest.py` | `CHUNK_OVERLAP` | `64` | Token overlap between PDF chunks — larger = less chance of losing boundary text |
| `scripts/ingest_json.py` | `CHUNK_SIZE` | `384` | Tokens per JSON chunk |
| `scripts/ingest_json.py` | `CHUNK_OVERLAP` | `32` | Token overlap between JSON chunks |
| `app/rag.py` | `TOP_K` | `5` | Number of chunks retrieved per query — more = richer context, higher LLM token cost |
| `app/rag.py` | `COLLECTION_NAME` | `"fedramp_docs"` | ChromaDB collection name |
| `scripts/ingest_json.py` | `FRMR_URL` | GitHub raw URL | Where to download the FRMR JSON from |
| `app/rag.py` | `QA_PROMPT` | *(full prompt)* | The system instructions given to the LLM |

### Paths (hardcoded, relative to project root)

| What | Path | Notes |
|------|------|-------|
| Vector database | `data/chroma_db/` | Auto-created; delete to wipe index |
| PDF source | `docs/` | Place your PDFs here |
| FRMR JSON cache | `data/FRMR.documentation.json` | Delete to force re-download |
| Query log | `logs/query_log.jsonl` | Auto-created; grows indefinitely |

---

## 9. The Query Log

Every question and answer pair is logged to `logs/query_log.jsonl`. Each line is a self-contained JSON object:

```json
{
  "id": "3f7a2bc1-...",
  "timestamp": "2026-03-10T14:37:00.123456+00:00",
  "question": "What is an authorization boundary?",
  "answer": "An authorization boundary defines...",
  "citations": ["[FRMR-MAS, p.MAS-CSO-PUB-01]"],
  "chunks": [
    {
      "text": "The authorization boundary...",
      "score": 0.8234,
      "citation": "[FRMR-MAS, p.MAS-CSO-PUB-01]"
    }
  ]
}
```

**Useful for:**
- Reviewing what questions were asked and what answers were given
- Checking similarity scores to understand retrieval quality
- Building an offline evaluation dataset
- Debugging why a question got a poor answer (look at which chunks were retrieved)

**Note:** The log is gitignored and grows indefinitely. Rotate or archive it manually if it gets large.

---

## 10. Tests and Smoke Checks

### `tests/test_connections.py`

These are **smoke tests** — they verify the environment is wired up correctly, not that the RAG logic is correct. They make real API calls.

| Test | What it checks | Skipped when |
|------|---------------|--------------|
| `test_openai_key_is_set` | Key starts with "sk-" | Never |
| `test_openai_embedding` | Can call embeddings API, returns 1536 dims | No OpenAI key |
| `test_openai_chat` | Can call chat completions API | No OpenAI key |
| `test_anthropic_key_is_set` | Key starts with "sk-ant-" | Not using Anthropic |
| `test_anthropic_chat` | Can call Anthropic API | Not using Anthropic |
| `test_chromadb_local` | In-memory ChromaDB works | Never |

Run with:
```powershell
pytest tests/test_connections.py -v
```

### Evaluation questions (`eval/test_questions.json`)

Ten manually written questions covering the FedRAMP authorization process (impact levels, SSP, RAR, ConMon, POA&M, encryption, MFA, penetration testing, etc.). These are all questions that **require the Rev 5 PDF documents to answer** — they cannot be answered from the FRMR JSON alone.

These are not automated tests — they're a manual evaluation checklist. Once PDFs are ingested, ask each question and check whether the answer is correct and well-cited.

---

## 11. What the Corpus Actually Contains

This is important context for understanding why some questions get good answers and others get refusals.

### FRMR JSON (currently indexed)

The FRMR 20x JSON covers **FedRAMP program-specific requirements**, not NIST 800-53 security controls. It covers:

| Section | What's in it |
|---------|-------------|
| FRD (definitions) | FedRAMP terms and definitions (e.g., "authorization boundary", "cloud service offering") |
| FRR-ADS | Authorization Data Sharing — trust centers, continuous monitoring data sharing |
| FRR-PVA | Persistent Vulnerability Assessment — ongoing scanning requirements |
| FRR-MAS | Marking an Offering's Boundary — defining the scope of a cloud service |
| FRR-VDR | Vulnerability Disclosure — how to handle and report vulnerabilities |

**Good questions for the current corpus:**
- What is a trust center?
- What does FedRAMP require for vulnerability disclosure?
- How do you define the boundary of a cloud service offering?
- What is a persistent assessment?
- What does ADS mean in FedRAMP?

**Questions that will get refusals (require PDFs not yet loaded):**
- What are the three FedRAMP impact levels?
- What sections does an SSP need?
- What are the MFA requirements?
- What encryption standards are required?
- How does a JAB P-ATO differ from an Agency ATO?

### PDFs (not yet loaded)

The `docs/` folder is empty. The Rev 5 baselines, SSP template, RAR guidance, POA&M guidance, and other FedRAMP documents need to be placed here and ingested before the evaluation questions can be answered.

---

## 12. Optimization and Improvement Opportunities

### Retrieval Quality

**TOP_K (currently 5)**
The number of chunks retrieved per query. Increasing this gives the LLM more context but increases token costs and can dilute relevance by including weaker matches. 5 is a reasonable default. Consider trying 8-10 once PDFs are loaded to see if answer quality improves.
- Location: `app/rag.py`, line with `TOP_K = 5`

**Chunk size and overlap**
The current sizes (512/64 for PDFs, 384/32 for JSON) are reasonable defaults. If answers are getting cut off mid-sentence or missing context from the surrounding page, increase chunk size. If retrieval is returning chunks that are too generic, decrease chunk size.
- Location: `scripts/ingest.py` and `scripts/ingest_json.py`

**Hybrid search**
Currently uses pure vector (semantic) similarity. Adding keyword (BM25) search alongside vector search and combining their scores can improve retrieval for queries with specific terms (requirement IDs, control names). LlamaIndex supports this with `QdrantVectorStore` or custom retrievers.

**Reranking**
After retrieving the top-K chunks, pass them through a cross-encoder reranker (e.g., Cohere Rerank, or a local BGE reranker) which scores each chunk against the specific question more accurately than cosine similarity. This can meaningfully improve answer quality with no index changes.

### LLM Quality

**Model upgrade**
The default `gpt-4o-mini` is fast and cheap but less capable than `gpt-4o` or Claude Sonnet for complex multi-document synthesis. For a proof-of-concept, the mini model is fine. For better answers on complex questions, swap in:
```
LLM_MODEL=gpt-4o
```
or:
```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
```

**System prompt tuning**
The `QA_PROMPT` in `app/rag.py` controls how the LLM behaves. Current instructions are minimal. You could add:
- More specific citation format instructions
- Instructions to structure the answer with headings for multi-part questions
- Instructions to explicitly state when only partial information is available
- Role context ("You are a FedRAMP compliance specialist...")

**Response mode**
Currently uses `response_mode="compact"` which asks the LLM to synthesize all retrieved chunks into one concise answer. Other options in LlamaIndex:
- `"refine"` — Iteratively refines the answer by processing chunks one at a time (more thorough, slower, more expensive)
- `"tree_summarize"` — Hierarchical summarization (better for very long retrieved context)
- `"no_text"` — Return only the source chunks without LLM synthesis (useful for debugging retrieval)

### Embedding Model

**Upgrade to `text-embedding-3-large`**
The larger OpenAI embedding model has 3072 dimensions vs. 1536 and generally retrieves more relevant results. Costs approximately 2x more per token. Requires full re-ingest.
```
EMBED_MODEL=text-embedding-3-large
```

**Switch to local embeddings**
If cost is a concern or you want to avoid external API calls during ingest, LlamaIndex supports local embedding models (e.g., via `llama-index-embeddings-huggingface`). This makes ingest free and offline. Quality varies by model.

### Architecture

**Add a similarity score threshold**
Currently the refusal is purely prompt-based — the LLM decides if the context is relevant. You could add a hard cutoff in `app/rag.py`: if the highest-scoring chunk is below a threshold (e.g., 0.35), skip the LLM call entirely and return a canned refusal. This saves LLM token costs for completely out-of-scope questions.

**Metadata filtering**
When querying, you can pre-filter the ChromaDB collection by metadata fields before doing similarity search. For example, searching only within `type="requirement"` entries, or only within documents from a specific FedRAMP section. This is useful once the corpus is large.

**Separate collections by document type**
Instead of one `fedramp_docs` collection, consider separate collections (e.g., `fedramp_pdfs`, `fedramp_frmr`). This allows routing queries to the appropriate source, or running parallel queries against both and merging results.

**Streaming responses**
The current implementation waits for the full LLM response before displaying anything. LlamaIndex and Streamlit both support streaming, which makes the UI feel more responsive for long answers.

**Async query handling**
For a multi-user deployment, queries should be async so one slow LLM call doesn't block other users. Streamlit 1.32+ supports async functions with `async def` and `await`.

### Evaluation and Monitoring

**Automated evaluation**
The `eval/test_questions.json` file has 10 questions but no automated scoring. Adding a script that runs each question, captures the answer, and compares to expected answers (manually or with an LLM judge) would let you measure the impact of any configuration change.

**Query log analysis**
The `logs/query_log.jsonl` file grows with every query. A simple analysis script could:
- Plot similarity score distributions to find the retrieval quality baseline
- Flag queries that triggered refusals (LLM answer contains the refusal phrase)
- Show which source documents are cited most/least frequently
- Find repeated questions to build a FAQ

**Tracing**
LlamaIndex integrates with Arize Phoenix, LangSmith, and other LLM observability tools. Adding a tracer lets you see the full retrieval + generation trace for each query, which is very useful for debugging why an answer is wrong.

### Corpus Management

**Incremental PDF ingest**
Currently `ingest_pdfs()` drops and rebuilds the whole collection. For a large corpus, this is wasteful. A better approach: track which files have already been indexed (e.g., store file hashes in a JSON), and only re-index changed or new PDFs.

**FRMR JSON version tracking**
The FRMR spec is updated over time. The current implementation caches the downloaded file forever. Consider storing the downloaded version string and checking it against the current GitHub version periodically, then auto-refreshing when a new version is available.

**Document metadata enrichment**
PDFs could be tagged with additional metadata (document type, revision, date) to improve citation quality and enable metadata filtering. This would require a small mapping file (e.g., `docs/metadata.json`) that maps filenames to their metadata.

---

*Document generated March 2026. Reflects the codebase state after initial RAG scaffold (commits 3621a0d and f429222).*
