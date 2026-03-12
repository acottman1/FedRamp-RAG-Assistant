"""SimpleBM25Retriever: pure-Python BM25 retriever for LlamaIndex.

WHY THIS EXISTS
---------------
LlamaIndex ships a BM25Retriever in the `llama-index-retrievers-bm25` package,
but it hardcodes a dependency on `pystemmer`, a C extension that wraps the
Snowball stemming library.  pystemmer does not compile on Python 3.14 on Windows
because the VS 2019 build tools ship an SDK version missing `stdlib.h`.

This file re-implements the same retriever using only:
  - `bm25s`   (pure Python BM25 engine, already a dependency of llama-index-retrievers-bm25)
  - `numpy`   (already required by llama-index-core)
  - LlamaIndex core classes

WHAT BM25 IS
------------
BM25 (Best Match 25) is a *keyword-based* ranking function developed in the 1990s
and still used at the core of Elasticsearch and many search engines today.

Where vector search asks "which chunks are semantically *close* to this question
in embedding space?", BM25 asks "which chunks contain the most *relevant keywords*
from this question?"

BM25 scores a chunk `d` for a query `q` roughly like this:

    score(q, d) = Σ  IDF(word) × TF(word, d) × saturation factor
                 word in q

  - IDF (Inverse Document Frequency): rare words score higher.
    "FedRAMP" appears in every chunk → low IDF.
    "FIPS-199" appears in 3 chunks → high IDF.
  - TF (Term Frequency): more occurrences of the word in the chunk → higher score.
    But TF is *saturated* — going from 1 to 2 occurrences matters a lot;
    going from 10 to 11 barely matters.
  - Document length normalization: a 50-word chunk with the word once scores
    higher than a 500-word chunk with it once, all else equal.

WHY BM25 COMPLEMENTS VECTOR SEARCH
-----------------------------------
Vector search can miss chunks that use *exact regulatory terminology*.
For example, the question "What is required under FRR-ADS-CSO-PUB?" embeds
to a dense vector, but that vector might be similar to generic compliance text.
BM25 will find chunks that literally contain "FRR-ADS-CSO-PUB" regardless of
the semantic neighborhood.

The two methods fail in different ways, so combining them (hybrid retrieval)
is consistently more robust than either alone.

STEMMING: WHY WE SKIP IT
--------------------------
Stemming reduces words to their root: "requirements" → "requir", "requiring" → "requir".
This improves recall for natural-language queries (a question about "requiring" finds
chunks about "requirements").

For FedRAMP documents, most important terms are acronyms or IDs (FRR, ADS, FIPS,
POA&M, CSO) that are already minimal — stemming adds no value and can
incorrectly conflate terms.  Skipping it is the right call here.
"""

from typing import List, Optional

import bm25s
import numpy as np
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)


class SimpleBM25Retriever(BaseRetriever):
    """BM25 retriever backed by bm25s, with no C-extension dependencies.

    Build it once at startup (expensive — indexes all chunks), then query it
    cheaply (pure in-memory keyword lookup, no network calls, no embeddings).

    Usage:
        retriever = SimpleBM25Retriever.from_chromadb(collection, top_k=8)
        nodes = retriever.retrieve("What is a trust center?")
    """

    def __init__(
        self,
        nodes: List[BaseNode],
        similarity_top_k: int = 5,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """Index `nodes` into an in-memory BM25 corpus.

        This is the expensive step — it tokenizes every chunk and builds the
        inverted index.  For 1,300 chunks it takes ~0.1 seconds.  Run once
        at startup and cache the result (Streamlit's @st.cache_resource handles this).

        Args:
            nodes: All document chunks to index.  BM25 needs the full corpus
                   upfront to compute IDF scores — it cannot add documents
                   incrementally without rebuilding.
            similarity_top_k: Number of chunks to return per query.
            callback_manager: LlamaIndex callback hook (optional).
        """
        self.similarity_top_k = similarity_top_k

        # Store the serialized node metadata so we can reconstruct full
        # NodeWithScore objects after BM25 returns array indexes.
        # node_to_metadata_dict() captures text, metadata, and node_id.
        self.corpus = [
            node_to_metadata_dict(node) | {"node_id": node.node_id}
            for node in nodes
        ]

        # Tokenize all chunk texts.
        # stemmer=None → plain lowercase word tokenization, no stemming.
        # stopwords="en" → removes common English words (the, is, a...) that
        #   carry no discriminating information and would dilute IDF scores.
        corpus_tokens = bm25s.tokenize(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            stopwords="en",
            stemmer=None,
        )

        # Build the BM25 index.
        # bm25s uses the Okapi BM25 variant (k1=1.5, b=0.75 by default).
        # These hyperparameters control TF saturation and length normalization;
        # the defaults are well-validated across many benchmarks.
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)

        # bm25s requires k ≤ num_docs.  Clamp similarity_top_k if the corpus
        # is smaller than requested (e.g., in tests with toy data).
        num_docs = int(self.bm25.scores.get("num_docs", len(nodes)))
        if num_docs < self.similarity_top_k:
            self.similarity_top_k = max(1, num_docs)

        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_chromadb(
        cls,
        collection,
        similarity_top_k: int = 5,
    ) -> "SimpleBM25Retriever":
        """Build a BM25 index by reading all documents from a ChromaDB collection.

        Why read from ChromaDB directly instead of the LlamaIndex docstore?
        When VectorStoreIndex is loaded with from_vector_store(), only the
        vector store reference is populated — the in-memory docstore is empty
        because LlamaIndex assumes you'll always query via embedding similarity.
        BM25 needs the raw text, so we go straight to ChromaDB.

        ChromaDB stores each chunk as:
          - documents[i]: the raw chunk text
          - metadatas[i]: a dict including '_node_content' (full LlamaIndex node
            serialized as JSON) and top-level fields like 'page_label', 'file_name'

        We reconstruct TextNode objects from those fields so they match the
        schema the rest of the RAG pipeline expects.

        Args:
            collection: A chromadb.Collection object (already connected).
            similarity_top_k: Number of BM25 results per query.
        """
        # Fetch ALL documents from ChromaDB.
        # include= controls what fields come back; we need text + metadata.
        # This is a full table scan — fine for thousands of chunks, would need
        # pagination for millions.
        all_docs = collection.get(include=["documents", "metadatas"])

        nodes: List[BaseNode] = []
        for text, meta, node_id in zip(
            all_docs["documents"],
            all_docs["metadatas"],
            all_docs["ids"],
        ):
            # Rebuild a TextNode with the stored metadata.
            # We only need text + metadata for BM25 — the embedding is not used.
            node = TextNode(
                text=text or "",
                metadata={
                    k: v
                    for k, v in meta.items()
                    # Strip LlamaIndex's internal bookkeeping keys.
                    # These are implementation details that bloat metadata and
                    # are never needed in BM25 results or citations.
                    if not k.startswith("_")
                },
                id_=node_id,
            )
            nodes.append(node)

        return cls(nodes=nodes, similarity_top_k=similarity_top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Run a BM25 keyword search and return top-k NodeWithScore objects.

        This method is called by LlamaIndex whenever the retriever is queried,
        either directly or from inside QueryFusionRetriever.

        BM25 scores are *not* cosine similarities — they are unbounded positive
        floats where higher means more relevant.  Typical range is 0–20 for
        well-matched chunks.  After RRF fusion these raw scores are discarded
        and replaced by RRF scores anyway.
        """
        tokenized_query = bm25s.tokenize(
            query_bundle.query_str,
            stemmer=None,
        )

        indexes, scores = self.bm25.retrieve(
            tokenized_query,
            k=self.similarity_top_k,
        )

        # bm25s returns results as 2D arrays (batch dimension first).
        # We only have one query per call, so take index 0.
        indexes = indexes[0]
        scores = scores[0]

        results: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            node_dict = self.corpus[int(idx)]
            # metadata_dict_to_node() deserializes back to a TextNode,
            # restoring all metadata fields including page_label and file_name.
            node = metadata_dict_to_node(node_dict)
            results.append(NodeWithScore(node=node, score=float(score)))

        return results
