"""Ingest pipeline: downloads FRMR.documentation.json from FedRAMP GitHub,
parses FRD (definitions) and FRR (requirements), and indexes into ChromaDB.

The JSON is cached locally at data/FRMR.documentation.json so subsequent
runs do not re-download it. Delete that file to force a fresh download.

Usage (from project root):
    python scripts/ingest_json.py

Or import ingest_frmr() from ingest_all.py to append to an existing collection.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

FRMR_URL = "https://raw.githubusercontent.com/FedRAMP/docs/main/FRMR.documentation.json"
FRMR_CACHE = PROJECT_ROOT / "data" / "FRMR.documentation.json"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "fedramp_docs"


# ── Download / load ───────────────────────────────────────────────────────────

def _download_frmr() -> dict:
    """Return parsed FRMR JSON, using local cache when available."""
    if FRMR_CACHE.exists():
        print(f"[json] Using cached FRMR JSON at {FRMR_CACHE.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[json] Downloading FRMR.documentation.json from FedRAMP GitHub…")
        FRMR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(FRMR_URL, FRMR_CACHE)
            print(f"[json] Saved to {FRMR_CACHE.relative_to(PROJECT_ROOT)}")
        except Exception as exc:
            print(f"[json] ERROR: Could not download FRMR JSON — {exc}")
            sys.exit(1)

    with open(FRMR_CACHE, encoding="utf-8") as f:
        return json.load(f)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_frd(frd_section: dict) -> list:
    """Parse FRD (FedRAMP Definitions) into LlamaIndex Documents.

    Citations will look like: [FRMR, p.FRD-ACV]
    """
    from llama_index.core import Document

    docs = []
    data = frd_section.get("data", {})

    for scope, entries in data.items():  # scope = "both", "20x", "rev5"
        if not isinstance(entries, dict):
            continue
        for frd_id, entry in entries.items():
            if not isinstance(entry, dict):
                continue

            term = entry.get("term", "")
            definition = entry.get("definition", "")
            if not definition:
                continue

            text_parts = [f"FedRAMP Term: {term}", f"Definition: {definition}"]

            note = entry.get("note", "")
            if note:
                text_parts.append(f"Note: {note}")

            reference = entry.get("reference", "")
            if reference:
                text_parts.append(f"Reference: {reference}")

            alts = entry.get("alts", [])
            if alts:
                text_parts.append(f"Also known as: {', '.join(alts)}")

            docs.append(
                Document(
                    text="\n\n".join(text_parts),
                    metadata={
                        "file_name": "FRMR",
                        "page_label": frd_id,          # → citation [FRMR, p.FRD-ACV]
                        "type": "definition",
                        "id": frd_id,
                        "term": term,
                        "scope": scope,
                        "source": "FRMR.documentation.json",
                    },
                )
            )

    return docs


def _parse_frr(frr_section: dict) -> list:
    """Parse FRR (FedRAMP Requirements & Recommendations) into LlamaIndex Documents.

    Citations will look like: [FRMR-ADS, p.ADS-CSO-PUB]
    """
    from llama_index.core import Document

    docs = []

    for doc_short, doc_data in frr_section.items():
        if not isinstance(doc_data, dict):
            continue

        doc_name = doc_data.get("info", {}).get("name", doc_short)
        data = doc_data.get("data", {})

        for framework, actor_groups in data.items():  # "20x", "rev5", "both"
            if not isinstance(actor_groups, dict):
                continue
            for actor, requirements in actor_groups.items():
                if not isinstance(requirements, dict):
                    continue
                for req_id, entry in requirements.items():
                    if not isinstance(entry, dict):
                        continue

                    name = entry.get("name", "")
                    keyword = entry.get("primary_key_word", "")
                    statement = entry.get("statement", "")
                    affects = entry.get("affects", [])

                    # Some entries vary by impact level instead of a flat statement
                    if not statement and "varies_by_level" in entry:
                        level_parts = []
                        for level, level_data in entry["varies_by_level"].items():
                            if isinstance(level_data, dict):
                                lvl_kw = level_data.get("primary_key_word", keyword)
                                lvl_stmt = level_data.get("statement", "")
                                if lvl_stmt:
                                    level_parts.append(
                                        f"{level.capitalize()}: {lvl_kw} {lvl_stmt}"
                                    )
                        statement = "\n".join(level_parts)
                        keyword = "varies by impact level"

                    if not statement:
                        continue

                    text_parts = []
                    if name:
                        text_parts.append(f"Requirement: {name}")
                    text_parts.append(f"{keyword} {statement}".strip())
                    if affects:
                        text_parts.append(f"Applies to: {', '.join(affects)}")

                    notes = entry.get("notes", []) or (
                        [entry["note"]] if entry.get("note") else []
                    )
                    for n in notes:
                        text_parts.append(f"Note: {n}")

                    docs.append(
                        Document(
                            text="\n\n".join(text_parts),
                            metadata={
                                "file_name": f"FRMR-{doc_short}",   # → [FRMR-ADS, p.ADS-CSO-PUB]
                                "page_label": req_id,
                                "type": "requirement",
                                "id": req_id,
                                "name": name,
                                "document": doc_name,
                                "framework": framework,
                                "keyword": keyword,
                                "source": "FRMR.documentation.json",
                            },
                        )
                    )

    return docs


# ── Indexer ───────────────────────────────────────────────────────────────────

def ingest_frmr(drop_existing: bool = True) -> int:
    """Download, parse, and index FRMR FRD + FRR sections into ChromaDB.

    Args:
        drop_existing: If True, drop and recreate the collection first.
                       Set to False when appending after ingest_pdfs().

    Returns:
        Number of documents indexed.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("[json] ERROR: OPENAI_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    import chromadb
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.llm = None

    # ── Download / load ───────────────────────────────────────────────────────
    data = _download_frmr()
    version = data.get("info", {}).get("version", "unknown")
    print(f"[json] FRMR version: {version}")

    # ── Parse ─────────────────────────────────────────────────────────────────
    frd_docs = _parse_frd(data.get("FRD", {}))
    frr_docs = _parse_frr(data.get("FRR", {}))
    all_docs = frd_docs + frr_docs
    print(f"[json] Parsed {len(frd_docs)} definitions + {len(frr_docs)} requirements = {len(all_docs)} total.")

    if not all_docs:
        print("[json] WARNING: No documents parsed — check FRMR JSON structure.")
        return 0

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if drop_existing:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[json] Dropped existing collection — rebuilding from scratch.")
        except Exception:
            pass

    try:
        collection = client.create_collection(COLLECTION_NAME)
    except Exception:
        collection = client.get_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # JSON entries are already well-scoped; use smaller chunks than PDFs.
    parser = SentenceSplitter(chunk_size=384, chunk_overlap=32)

    print("[json] Embedding and indexing… (this may take several minutes)")
    VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_context,
        transformations=[parser],
        show_progress=True,
    )

    print(f"[json] Done — {len(all_docs)} FRMR entries indexed.")
    return len(all_docs)


def main() -> None:
    ingest_frmr(drop_existing=True)
    print(f"\n[json] Index written to {CHROMA_PATH}/")
    print("[json] Start the app with:  streamlit run app/main.py")


if __name__ == "__main__":
    main()
