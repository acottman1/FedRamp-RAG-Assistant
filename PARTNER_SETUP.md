# Partner Setup Guide — FedRAMP RAG Assistant

This guide walks you through getting the project running on your own machine from scratch. You need Python and an OpenAI API key. Everything else is handled by the project itself.

---

## What You'll Need

- **Python 3.11** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com/)
- **An OpenAI API key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - This is required even if we switch the LLM to Anthropic, because OpenAI is always used for embeddings
  - Building the index costs roughly $0.01 — essentially free
  - Running queries costs fractions of a cent per question
- **An Anthropic API key** (optional) — only needed if you want to use Claude as the LLM instead of GPT-4o-mini

---

## Step 1: Clone the Repository

```bash
git clone <repo-url>
cd "FedRamp RAG Project"
```

Replace `<repo-url>` with the GitHub URL we shared.

---

## Step 2: Create a Virtual Environment

This keeps the project's Python packages isolated from anything else on your machine.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your terminal prompt when the environment is active. Always activate it before working on this project.

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs everything: LlamaIndex, ChromaDB, Streamlit, OpenAI SDK, Anthropic SDK, pytest. It takes a minute or two.

---

## Step 4: Configure Your API Keys

The project reads credentials from a `.env` file that is never committed to git.

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` in any text editor and fill it in:

```env
# Required — used for embeddings on every query
OPENAI_API_KEY=sk-...your-key-here...

# Which LLM to use for generating answers
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Only needed if LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=

# Embedding model — do not change this without re-running ingest
EMBED_MODEL=text-embedding-3-small
```

Save the file. The `.env` file is gitignored — it will never be accidentally committed.

---

## Step 5: Verify the Setup

Run the smoke tests to confirm your API keys work and ChromaDB is functional:

```bash
pytest tests/test_connections.py -v
```

Expected output (if using OpenAI only):
```
tests/test_connections.py::test_openai_key_is_set    PASSED
tests/test_connections.py::test_openai_embedding     PASSED
tests/test_connections.py::test_openai_chat          PASSED
tests/test_connections.py::test_anthropic_key_is_set SKIPPED (LLM_PROVIDER is not anthropic)
tests/test_connections.py::test_anthropic_chat       SKIPPED (LLM_PROVIDER is not anthropic)
tests/test_connections.py::test_chromadb_local       PASSED
```

If any test fails with an import error, re-run `pip install -r requirements.txt`.

---

## Step 6: Build the Knowledge Index

This downloads the FedRAMP machine-readable JSON from GitHub, parses it, and builds the local vector database. It does not require any PDFs.

```bash
python scripts/ingest_json.py
```

You'll see progress output. The FRMR JSON downloads automatically and is cached at `data/FRMR.documentation.json`. The vector index is saved to `data/chroma_db/`.

**Total time:** about 1–2 minutes.
**API cost:** roughly $0.01 in OpenAI embedding calls.

> **If we have PDFs to add:** place them in the `docs/` folder and run `python scripts/ingest_all.py` instead. This wipes and rebuilds the index with both PDFs and JSON combined. Re-run any time documents change.

---

## Step 7: Start the Application

```bash
streamlit run app/main.py
```

Your browser will open automatically to `http://localhost:8501`. If it doesn't, open that URL manually.

You'll see the FedRAMP Readiness Assistant chat interface. Type a question and hit Enter.

---

## Everyday Workflow

After the first setup, your daily routine is:

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# 2. Start the app
streamlit run app/main.py
```

That's it.

---

## Adding or Updating Documents

If you receive new PDFs:

1. Place them in the `docs/` folder
2. Run `python scripts/ingest_all.py`
3. Restart the Streamlit app (Ctrl+C, then re-run `streamlit run app/main.py`)

The app caches the index at startup — restarting is required to pick up a newly built index.

---

## Switching to Claude (Anthropic) as the LLM

Edit `.env`:

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

Restart the app. No re-ingest needed — embeddings always use OpenAI regardless of LLM provider.

---

## Where Things Live

| Path | What it is |
|------|-----------|
| `app/main.py` | Streamlit chat UI |
| `app/rag.py` | RAG engine (search + LLM logic) |
| `scripts/ingest_json.py` | FRMR JSON download and indexing |
| `scripts/ingest.py` | PDF indexing |
| `scripts/ingest_all.py` | Runs both ingest pipelines |
| `data/chroma_db/` | Vector database (auto-created, gitignored) |
| `data/FRMR.documentation.json` | Cached FRMR spec (auto-downloaded, gitignored) |
| `docs/` | Place PDFs here (gitignored) |
| `logs/query_log.jsonl` | Every query logged here (auto-created, gitignored) |
| `eval/test_questions.json` | 10 evaluation questions |
| `SYSTEM_OVERVIEW.md` | Full technical documentation of how everything works |
| `CLAUDE.md` | AI agent context file (read by Cursor, Claude Code, etc.) |

---

## Understanding What the System Can and Cannot Answer

The FRMR JSON covers **FedRAMP-specific program requirements** — things like trust centers, vulnerability disclosure, persistent assessments, and defining the cloud service offering boundary. It does NOT cover NIST 800-53 controls (encryption, MFA, pen testing, impact levels) — those live in the Rev 5 PDF guidance documents.

**Good questions right now (FRMR-only corpus):**
- What is a trust center?
- What does FedRAMP require for vulnerability disclosure?
- How do you define the boundary of a cloud service offering?
- What is a persistent vulnerability assessment?

**Questions that need PDFs (will get a polite refusal):**
- What are the three FedRAMP impact levels?
- What MFA requirements does FedRAMP mandate?
- What sections does an SSP need?
- What encryption standards are required?

Once PDFs are added and ingested, all of these will work.

---

## Working with Feature Branches

The `main` branch is the stable baseline.  Two feature branches extend the retrieval pipeline — you can check either out and run the app the same way.

| Branch | What it adds |
|--------|-------------|
| `feature/multi-query-expansion` | LLM generates query variants; RRF merges results |
| `feature/hybrid-search` | Adds BM25 keyword search on top of multi-query |

### Switching branches

```bash
# Multi-query only
git checkout feature/multi-query-expansion

# Hybrid search (BM25 + vector + multi-query)
git checkout feature/hybrid-search
```

### Extra dependency on `feature/hybrid-search`

The hybrid branch requires `bm25s`, a pure-Python BM25 library.  It is **not** in `requirements.txt` yet because it came with a broken C-extension dependency (`pystemmer`) that fails to compile on Python 3.14 / Windows.  Install it separately after activating your venv:

```bash
# Make sure the venv is active first — you should see (.venv) in your prompt
pip install llama-index-retrievers-bm25 --no-deps
pip install bm25s
```

Verify it worked:

```bash
python -c "import bm25s; print('bm25s OK:', bm25s.__version__)"
# Expected: bm25s OK: 0.3.2
```

You will see a harmless warning on import:
```
resource module not available on Windows
```
This just means `pystemmer` (the C stemmer) isn't installed — which is expected and intentional.  The retriever uses plain tokenization instead, which is fine for FedRAMP regulatory terminology.

After installing, run the app normally:

```bash
streamlit run app/main.py
```

The first query will show BM25 indexing progress bars in the terminal as it loads all 1,300+ chunks into memory.  This takes about half a second and is cached for the rest of the session.

---

## Troubleshooting

**`FileNotFoundError: Vector index not found`**
You haven't run ingest yet. Run `python scripts/ingest_json.py`.

**`openai.AuthenticationError`**
Your `OPENAI_API_KEY` in `.env` is wrong or missing. Check it at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

**`ModuleNotFoundError: No module named 'bm25s'`**
You're on the `feature/hybrid-search` branch and `bm25s` isn't installed.  Run:
```bash
pip install llama-index-retrievers-bm25 --no-deps
pip install bm25s
```

**`ModuleNotFoundError`** (any other module)
Your virtual environment isn't activated or `pip install -r requirements.txt` wasn't run. Activate it first, then reinstall.

**App gives refusals to everything**
The question is probably outside the FRMR corpus scope. See "Understanding What the System Can and Cannot Answer" above. Check `logs/query_log.jsonl` to see the similarity scores for the retrieved chunks.

**Streamlit port already in use**
```bash
streamlit run app/main.py --server.port 8502
```

---

## For More Context

- `SYSTEM_OVERVIEW.md` — Complete technical walkthrough: every function, every configuration option, how the pieces connect, and where to improve things
- `CLAUDE.md` — Project-level AI agent context (auto-loaded by Cursor, Claude Code, and other AI tools)
- `eval/test_questions.json` — 10 test questions to evaluate answer quality
- `logs/query_log.jsonl` — Every query logged with full metadata, useful for debugging
