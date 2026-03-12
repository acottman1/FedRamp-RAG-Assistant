"""RAG engine: loads the persisted ChromaDB index and answers queries with citations.

Retrieval strategy: hybrid search (dense vector + BM25 keyword) with multi-query expansion.

HOW THIS PIPELINE WORKS
------------------------
Each user question goes through three layers before reaching the LLM:

  Layer 1 — Query expansion:
    The LLM generates NUM_QUERIES-1 paraphrase variants of the question.
    Different phrasings cast a wider semantic net over the corpus.

  Layer 2 — Dual retrieval (hybrid search):
    For each of the NUM_QUERIES queries, TWO retrievers run in parallel:
      a) VectorIndexRetriever  — dense semantic search via OpenAI embeddings + ChromaDB cosine similarity
      b) SimpleBM25Retriever   — sparse keyword search (BM25) over all chunk texts in memory

    Vector search is good at *semantic* similarity ("authentication" finds "login").
    BM25 is good at *exact* matches (FRR-ADS-CSO-PUB, FIPS 199, POA&M).
    They fail in different ways, so combining them is more robust than either alone.

  Layer 3 — Reciprocal Rank Fusion (RRF):
    All result sets (NUM_QUERIES queries × 2 retrievers) are merged by RRF.
    RRF scores each chunk based on its rank across lists:

        score(chunk) = Σ  1 / (60 + rank_in_list_i)
                      i

    A chunk that ranks #2 in the vector results AND #1 in the BM25 results
    beats a chunk that only appeared in one list.  60 is a smoothing constant
    that prevents a single #1 ranking from dominating everything else.

    RRF is used instead of score averaging because cosine similarities and BM25
    scores live on completely different scales — you cannot meaningfully average them.
    Ranks are always comparable regardless of the underlying scoring function.
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

from bm25_retriever import SimpleBM25Retriever

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
    """Load the persisted ChromaDB index and return a hybrid multi-query engine.

    Assembly order — the dependency graph:

        ChromaDB client
            ├── ChromaVectorStore
            │       └── VectorStoreIndex
            │               └── VectorIndexRetriever  (dense semantic search)
            │                       └─┐
            │                         ├── QueryFusionRetriever  (multi-query + RRF)
            │                         │       └── RetrieverQueryEngine
            └── collection.get()      │
                    └── SimpleBM25Retriever (keyword search)
                            └─────────┘

    Key point: QueryFusionRetriever accepts a LIST of retrievers.
    For every query variant it generates, it runs ALL retrievers and merges
    the results with RRF.  Adding BM25 as a second retriever is literally
    one extra argument — the fusion machinery is already there from the
    multi-query branch.

    Raises FileNotFoundError if the index has not been built yet.
    """
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {CHROMA_PATH}. "
            "Run  python scripts/ingest_json.py  first to build the index."
        )

    # Connect to the local ChromaDB database.
    # PersistentClient means it reads from disk — no in-memory state that
    # could go stale between Streamlit reruns.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Register models globally so every LlamaIndex component can find them.
    # Settings is a global singleton — all LlamaIndex classes read from it.
    Settings.embed_model = _get_embed_model()
    Settings.llm = _get_llm()

    # Build the LlamaIndex wrapper around the ChromaDB vector store.
    # from_vector_store() does NOT re-embed anything — it just wraps the
    # collection so LlamaIndex can query it.
    index = VectorStoreIndex.from_vector_store(vector_store)

    # ── Retriever A: dense vector search ──────────────────────────────────────
    # Embeds each query and finds chunks with similar embedding vectors.
    # Good at semantic similarity: "authentication" → "login", "key management".
    # similarity_top_k=FUSION_TOP_K because we want a wide candidate pool per
    # query before RRF trims it down to TOP_K for the LLM.
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=FUSION_TOP_K,
    )

    # ── Retriever B: BM25 keyword search ──────────────────────────────────────
    # Loads all 1,300+ chunk texts from ChromaDB and builds an in-memory
    # inverted index.  No embeddings needed — pure term-frequency math.
    # Good at exact matches: FRR-ADS-CSO-PUB, FIPS 199, POA&M, specific IDs.
    #
    # This is the one-time expensive step (~0.5s on first load).
    # Streamlit's @st.cache_resource caches build_query_engine() across reruns,
    # so this only happens once per server session, not per query.
    bm25_retriever = SimpleBM25Retriever.from_chromadb(
        collection=collection,
        similarity_top_k=FUSION_TOP_K,
    )

    # ── Fusion retriever: multi-query + RRF over both retrievers ──────────────
    # QueryFusionRetriever runs every query variant through EVERY retriever in
    # the list, then merges all result sets with Reciprocal Rank Fusion.
    #
    # With NUM_QUERIES=4 and 2 retrievers:
    #   4 query variants × 2 retrievers = 8 retrieval calls per user question
    #   Each returns up to FUSION_TOP_K=8 chunks
    #   RRF pool: up to 64 candidates → final TOP_K=5 sent to the LLM
    #
    # use_async=False: Streamlit has its own event loop; nested async crashes.
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
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
        "retrieval_mode": "hybrid (vector + BM25, multi-query)",
    }
