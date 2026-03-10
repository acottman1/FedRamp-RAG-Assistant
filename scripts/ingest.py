"""Ingest pipeline: reads PDFs from docs/, chunks, embeds, and writes to ChromaDB.

Usage (from project root):
    python scripts/ingest.py

Re-run this whenever you add or replace documents. Each run does a full
re-ingest, dropping and rebuilding the vector store collection.
"""

import os
import sys
from pathlib import Path

# Resolve project root regardless of where the script is invoked from.
PROJECT_ROOT = Path(__file__).parent.parent

# Load .env before importing LlamaIndex (which reads env vars at import time).
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

DOCS_PATH = PROJECT_ROOT / "docs"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "fedramp_docs"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def main() -> None:
    pdf_files = list(DOCS_PATH.glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] No PDFs found in {DOCS_PATH}/")
        print("         Place FedRAMP PDF documents there and re-run.")
        sys.exit(1)

    print(f"[ingest] Found {len(pdf_files)} PDF(s):")
    for f in sorted(pdf_files):
        print(f"         • {f.name}")

    # ── Validate API key ──────────────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        print("[ingest] ERROR: OPENAI_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    # ── Imports (after env is loaded) ─────────────────────────────────────────
    import chromadb
    from llama_index.core import (
        Settings,
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    # ── Configure embeddings (no LLM needed for ingestion) ────────────────────
    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.llm = None

    # ── Load documents ────────────────────────────────────────────────────────
    print("[ingest] Loading documents…")
    reader = SimpleDirectoryReader(
        input_dir=str(DOCS_PATH),
        required_exts=[".pdf"],
        filename_as_id=True,
    )
    documents = reader.load_data()
    print(f"[ingest] Loaded {len(documents)} document page(s) across all PDFs.")

    # ── Set up ChromaDB ───────────────────────────────────────────────────────
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Drop existing collection so this is always a fresh, consistent build.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("[ingest] Dropped existing collection — rebuilding from scratch.")
    except Exception:
        pass  # Collection did not exist yet

    collection = client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Chunk, embed, and index ───────────────────────────────────────────────
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(
        f"[ingest] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) "
        "and embedding — this may take a few minutes…"
    )
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[parser],
        show_progress=True,
    )

    print(f"\n[ingest] Done. Index written to {CHROMA_PATH}/")
    print("[ingest] Start the app with:  streamlit run app/main.py")


if __name__ == "__main__":
    main()
