"""RAG engine: loads the persisted ChromaDB index and answers queries with citations."""

import os
from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "fedramp_docs"
TOP_K = 5

# ── Prompt ────────────────────────────────────────────────────────────────────
# {context_str} and {query_str} are the two variables LlamaIndex injects.
QA_PROMPT = PromptTemplate(
    "You are a FedRAMP compliance expert assistant.\n"
    "Answer questions using ONLY the retrieved context provided below.\n\n"
    "Rules:\n"
    "1. Every factual claim must be supported by the context.\n"
    "2. Cite every claim inline using the format [filename, p.X] or [filename, chunk-ID].\n"
    "3. If the context does not contain enough information to answer the question, respond with:\n"
    '   "I cannot find sufficient information about this in the provided FedRAMP documents."\n'
    "4. Do not speculate or add knowledge beyond what appears in the context.\n\n"
    "Retrieved context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Answer (with inline citations):"
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic  # lazy import
        return Anthropic(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    from llama_index.llms.openai import OpenAI  # lazy import
    return OpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _get_embed_model():
    return OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _format_citation(node) -> str:
    """Return a citation label for a single NodeWithScore."""
    meta = node.node.metadata
    filename = Path(meta.get("file_name", "unknown")).name
    page = meta.get("page_label", None)
    chunk_id = node.node.node_id[:8]
    return f"[{filename}, p.{page}]" if page else f"[{filename}, chunk-{chunk_id}]"


def _dedupe(items: list) -> list:
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


# ── Public API ────────────────────────────────────────────────────────────────

def build_query_engine() -> RetrieverQueryEngine:
    """Load the persisted ChromaDB index and return a ready query engine.

    Raises FileNotFoundError if the index has not been built yet.
    """
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {CHROMA_PATH}. "
            "Run  python scripts/ingest.py  first to build the index."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    Settings.embed_model = _get_embed_model()
    Settings.llm = _get_llm()

    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    synthesizer = get_response_synthesizer(
        text_qa_template=QA_PROMPT,
        response_mode="compact",
    )
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )


def query(engine: RetrieverQueryEngine, question: str) -> dict:
    """Run a question through the RAG engine.

    Returns a dict with keys:
      - answer     : str
      - citations  : list[str]  (deduplicated source labels)
      - chunks     : list[dict] (retrieved nodes with text, score, metadata, citation)
    """
    response = engine.query(question)
    answer = str(response).strip()
    source_nodes = response.source_nodes

    citations = _dedupe([_format_citation(n) for n in source_nodes])

    chunks = [
        {
            "text": node.node.text,
            "score": round(node.score, 4) if node.score is not None else None,
            "metadata": node.node.metadata,
            "citation": _format_citation(node),
        }
        for node in source_nodes
    ]

    return {"answer": answer, "citations": citations, "chunks": chunks}
