"""Ingest pipeline: reads PDFs from docs/, chunks, embeds, and writes to ChromaDB.

Usage (from project root):
    python scripts/ingest.py

Re-run this whenever you add or replace documents. Each run does a full
re-ingest, dropping and rebuilding the vector store collection.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

DOCS_PATH = PROJECT_ROOT / "docs"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "fedramp_docs"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def ingest_pdfs(drop_existing: bool = True) -> int:
    """Chunk and embed all PDFs in docs/ into ChromaDB.

    Args:
        drop_existing: If True, drop and recreate the collection first.
                       Set to False when appending after another ingest.

    Returns:
        Number of document pages loaded.
    """
    pdf_files = list(DOCS_PATH.glob("*.pdf"))
    if not pdf_files:
        print(f"[pdf] No PDFs found in {DOCS_PATH}/")
        print("      Place FedRAMP PDF documents there and re-run.")
        return 0

    print(f"[pdf] Found {len(pdf_files)} PDF(s):")
    for f in sorted(pdf_files):
        print(f"      • {f.name}")

    if not os.getenv("OPENAI_API_KEY"):
        print("[pdf] ERROR: OPENAI_API_KEY is not set. Check your .env file.")
        sys.exit(1)

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

    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.llm = None

    print("[pdf] Loading documents…")
    reader = SimpleDirectoryReader(
        input_dir=str(DOCS_PATH),
        required_exts=[".pdf"],
        filename_as_id=True,
    )
    documents = reader.load_data()
    print(f"[pdf] Loaded {len(documents)} page(s) across all PDFs.")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if drop_existing:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[pdf] Dropped existing collection — rebuilding from scratch.")
        except Exception:
            pass

    try:
        collection = client.create_collection(COLLECTION_NAME)
    except Exception:
        collection = client.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(f"[pdf] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) and embedding…")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[parser],
        show_progress=True,
    )

    print(f"[pdf] Done — {len(documents)} pages indexed.")
    return len(documents)


def main() -> None:
    ingest_pdfs(drop_existing=True)
    print(f"\n[pdf] Index written to {CHROMA_PATH}/")
    print("[pdf] Start the app with:  streamlit run app/main.py")


if __name__ == "__main__":
    main()
