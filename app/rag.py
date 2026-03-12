"""RAG engine: loads the persisted ChromaDB index and answers queries with citations.

Retrieval strategy: multi-query expansion via QueryFusionRetriever.

HOW MULTI-QUERY EXPANSION WORKS
--------------------------------
A standard RAG system embeds the user's question once and does a single cosine
similarity search.  That works, but it has a blind spot: if the user's phrasing
doesn't closely match the phrasing in the source documents, the right chunks may
never surface.

Multi-query expansion adds a step before retrieval:
  1. An LLM generates N-1 paraphrase variants of the original question.
  2. All N queries (original + variants) are embedded and searched independently.
  3. The result sets are merged using Reciprocal Rank Fusion (RRF).
  4. The top K deduplicated chunks are sent to the answer LLM as context.

The key insight is that different phrasings cast a wider semantic net.  A chunk
about "multi-factor authentication" might rank #1 under the variant
"two-factor login requirements" but never appear under the original "MFA rules."

RECIPROCAL RANK FUSION (RRF)
------------------------------
RRF is a simple, robust algorithm for merging ranked lists without needing score
calibration between lists.  Each chunk's RRF score is:

    score(chunk) = Σ  1 / (k + rank_in_query_i)
                  i

where k=60 is a smoothing constant and rank_in_query_i is its position in the
i-th query's result list (1-indexed).  A chunk that ranks highly in multiple
queries gets a higher combined score.  A chunk that appears in only one list but
ranks first still contributes, so nothing is discarded outright.

RRF was chosen over simple score averaging because cosine similarity scores from
different queries are not directly comparable (they depend on the query vector),
whereas rank positions are always comparable.
"""

import os
from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "fedramp_docs"

# ── Retrieval tuning ──────────────────────────────────────────────────────────
#
# TOP_K — chunks the LLM actually sees.
#   Keep this at 5-10.  LLMs degrade with very long contexts because the
#   relevant information gets buried.  Every chunk also costs tokens, which
#   means higher latency and API cost.  5 is a good starting point; raise it
#   to 8 once you have PDFs in the corpus and need broader coverage.
TOP_K = 5

# NUM_QUERIES — total queries run, including the user's original.
#   4 means the LLM generates 3 paraphrase variants and we run all 4 through
#   ChromaDB.  The right number depends on how ambiguous or narrow the typical
#   question is.  For regulatory Q&A (FedRAMP), 3-4 works well.  Going much
#   higher than 5 yields diminishing returns and increases cost.
NUM_QUERIES = 4

# FUSION_TOP_K — how many chunks each individual query fetches before fusion.
#   With NUM_QUERIES=4, we run 4 searches each returning FUSION_TOP_K chunks,
#   giving RRF a pool of up to 4×8=32 candidates to choose TOP_K=5 from.
#   This "wider funnel, narrow output" pattern is the heart of why multi-query
#   improves recall without flooding the LLM with irrelevant context.
#   Rule of thumb: FUSION_TOP_K ≈ TOP_K × 1.5 to 2.
FUSION_TOP_K = 8

# ── Prompt ────────────────────────────────────────────────────────────────────
# {context_str} and {query_str} are the two variables LlamaIndex injects.
# The prompt is intentionally strict: the LLM must cite or refuse.  This is
# important for a compliance use case where hallucinated policy guidance could
# lead someone to make a wrong authorization decision.
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
    """Return the configured LLM instance.

    LLM_PROVIDER controls which backend is used.  The LLM serves two roles here:
      1. Generating query variants (inside QueryFusionRetriever)
      2. Synthesizing the final answer from retrieved chunks

    Using the same LLM for both keeps configuration simple.  In production you
    might use a cheap/fast model for query generation and a more capable model
    for answer synthesis — but that optimization is premature for a PoC.
    """
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
    """Return the embedding model.

    Embeddings MUST use the same model that was used during ingest.  If you
    change this model, every stored vector in ChromaDB becomes meaningless
    because the embedding spaces are incompatible.  You would need to wipe
    data/chroma_db/ and re-run ingest.  text-embedding-3-small is locked in
    as the default to prevent accidental mismatch.
    """
    return OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _format_citation(node) -> str:
    """Return a citation label for a single NodeWithScore.

    NodeWithScore is LlamaIndex's wrapper around a retrieved chunk.  It holds:
      - node.node.text       : the raw chunk text
      - node.node.metadata   : dict of fields stored at ingest time (file_name, page_label)
      - node.score           : the RRF score (not the raw cosine score — see note below)

    Note on scores after multi-query: scores are now RRF scores (typically 0.01–0.1),
    not raw cosine similarities (typically 0.3–0.9).  They are still useful for
    relative ranking but shouldn't be compared to pre-fusion baselines.
    """
    meta = node.node.metadata
    filename = Path(meta.get("file_name", "unknown")).name
    page = meta.get("page_label", None)
    chunk_id = node.node.node_id[:8]
    return f"[{filename}, p.{page}]" if page else f"[{filename}, chunk-{chunk_id}]"


def _dedupe(items: list) -> list:
    """Order-preserving deduplication.

    Used to collapse identical citation labels when the same source chunk
    appears multiple times across query variants.
    """
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


# ── Public API ────────────────────────────────────────────────────────────────

def build_query_engine() -> RetrieverQueryEngine:
    """Load the persisted ChromaDB index and return a multi-query query engine.

    Assembly order matters here — this is the dependency graph:

        ChromaDB client
            └── ChromaVectorStore
                    └── VectorStoreIndex          (the searchable index)
                            └── VectorIndexRetriever   (single-query vector search)
                                    └── QueryFusionRetriever   (wraps it, adds variants)
                                            └── RetrieverQueryEngine   (ties retriever to LLM)

    Each layer adds one responsibility:
      - VectorStoreIndex: knows how to read/write to ChromaDB
      - VectorIndexRetriever: knows how to do a single similarity search
      - QueryFusionRetriever: knows how to generate variants and fuse results
      - RetrieverQueryEngine: knows how to format context and call the LLM

    Raises FileNotFoundError if the index has not been built yet.
    """
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {CHROMA_PATH}. "
            "Run  python scripts/ingest_json.py  first to build the index."
        )

    # Connect to the local ChromaDB database.
    # PersistentClient means it reads from disk every time — no in-memory
    # state that could go stale between Streamlit reruns.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Register models globally so every LlamaIndex component can find them.
    # Settings is a global singleton — think of it like a config object that
    # all LlamaIndex classes read from automatically.
    Settings.embed_model = _get_embed_model()
    Settings.llm = _get_llm()

    # Build the index object from the existing vector store.
    # from_vector_store() does NOT re-embed anything — it just wraps the
    # ChromaDB collection so LlamaIndex can query it.
    index = VectorStoreIndex.from_vector_store(vector_store)

    # ── Layer 1: single-query vector retriever ────────────────────────────────
    # This is the same retriever we had before.  It takes one query string,
    # embeds it, and returns the FUSION_TOP_K most similar chunks.
    # We set similarity_top_k to FUSION_TOP_K (not TOP_K) here because we want
    # each of the NUM_QUERIES queries to bring back a wider set of candidates.
    # The fusion step will trim the merged pool down to TOP_K for the LLM.
    base_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=FUSION_TOP_K,
    )

    # ── Layer 2: multi-query fusion retriever ─────────────────────────────────
    # QueryFusionRetriever wraps base_retriever and adds the expansion logic.
    #
    # Parameters explained:
    #   retrievers         — list of underlying retrievers to run each query through.
    #                        We only have one (vector search), but the API accepts
    #                        multiple — which is how you'd add BM25 later (hybrid search).
    #
    #   similarity_top_k   — final number of chunks returned after fusion.
    #                        This is what the LLM will see.
    #
    #   num_queries        — total queries including the original.  The retriever
    #                        calls the LLM to generate (num_queries - 1) variants.
    #
    #   mode               — fusion algorithm.
    #                        "reciprocal_rerank" → RRF (described in the module docstring)
    #                        "simple"            → just concatenate and dedupe (no reranking)
    #
    #   use_async          — whether to run the NUM_QUERIES searches concurrently.
    #                        False here because Streamlit manages its own event loop
    #                        and mixing async contexts causes errors.  In a non-Streamlit
    #                        context (e.g., a FastAPI server), set this to True for speed.
    fusion_retriever = QueryFusionRetriever(
        retrievers=[base_retriever],
        similarity_top_k=TOP_K,
        num_queries=NUM_QUERIES,
        mode="reciprocal_rerank",
        use_async=False,
    )

    # ── Layer 3: response synthesizer ────────────────────────────────────────
    # The synthesizer receives the fused chunks and the question, formats them
    # into the QA_PROMPT, and calls the LLM to get an answer.
    # "compact" mode fits all chunks into a single LLM call.  The alternative
    # "tree_summarize" makes multiple LLM calls and summarizes hierarchically —
    # useful for very long contexts but overkill here.
    synthesizer = get_response_synthesizer(
        text_qa_template=QA_PROMPT,
        response_mode="compact",
    )

    # ── Layer 4: query engine ────────────────────────────────────────────────
    # RetrieverQueryEngine ties the retriever to the synthesizer.
    # When you call engine.query(question), it:
    #   1. Passes question → fusion_retriever → fused chunks
    #   2. Passes question + fused chunks → synthesizer → answer
    return RetrieverQueryEngine(
        retriever=fusion_retriever,
        response_synthesizer=synthesizer,
    )


def query(engine: RetrieverQueryEngine, question: str) -> dict:
    """Run a question through the multi-query RAG engine.

    Returns a dict with keys:
      - answer          : str
      - citations       : list[str]   deduplicated source labels
      - chunks          : list[dict]  retrieved nodes with text, score, metadata, citation
      - retrieval_mode  : str         always "multi-query" so the UI can label it

    The 'chunks' list now reflects the fused result set.  Because multiple query
    variants contributed, you may see more diverse sources here than you would
    from a single-pass retrieval — that diversity is the whole point.

    Note on scores: node.score is now an RRF score (~0.01–0.1 range), not a
    raw cosine similarity (~0.3–0.9 range).  Higher is still better, but don't
    compare these numbers to scores logged before this change was introduced.
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

    return {
        "answer": answer,
        "citations": citations,
        "chunks": chunks,
        "retrieval_mode": "multi-query",
    }
