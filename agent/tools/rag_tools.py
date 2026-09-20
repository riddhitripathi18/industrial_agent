"""
RAG Tools — Industrial Knowledge Base Query
============================================
Provides semantic search over industrial engineering documents using
ChromaDB + sentence-transformers (all-MiniLM-L6-v2).

Knowledge base covers:
  - TEMA Standards (fouling resistance limits, cleaning methods, U-values)
  - ISO 10816 Vibration Standards (severity zones, fault frequencies)
  - Bearing Failure Catalog (ISO 15243 damage types, IMS dataset specifics)
  - Maintenance SOPs (cleaning procedures, bearing replacement)

Usage:
    # First time: build the index
    python scripts/build_rag_index.py

    # Query the knowledge base
    from agent.tools.rag_tools import query_knowledge_base
    result = query_knowledge_base("What is the TEMA fouling resistance limit for crude oil?")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from agent.config import DATA_DIR, REPO_ROOT

# ChromaDB index and docs live in the REPO (not in the external data dir with CSVs)
RAG_INDEX_DIR: Path = REPO_ROOT / "data" / "rag_index"
DOCS_DIR:      Path = REPO_ROOT / "data" / "docs"
COLLECTION_NAME = "industrial_knowledge"


# ---------------------------------------------------------------------------
# Internal — lazy-loaded client & collection
# ---------------------------------------------------------------------------

_client     = None
_collection = None


def _get_collection():
    """Lazily initialize ChromaDB client and return the collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        raise ImportError(
            "ChromaDB not installed. Run: pip install chromadb sentence-transformers"
        )

    index_path = str(RAG_INDEX_DIR)

    # Check index exists
    if not RAG_INDEX_DIR.is_dir() or not any(RAG_INDEX_DIR.iterdir()):
        raise FileNotFoundError(
            f"RAG index not found at: {index_path}\n"
            "Build it first by running: python scripts/build_rag_index.py"
        )

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    _client = chromadb.PersistentClient(path=index_path)
    _collection = _client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_knowledge_base(
    query: str,
    n_results: int = 5,
    source_filter: Optional[str] = None,
) -> dict:
    """
    Search the industrial engineering knowledge base using semantic similarity.

    Searches across TEMA standards, ISO 10816 vibration standards,
    bearing failure catalog, and maintenance SOPs.

    Args:
        query:         Natural language question or keyword phrase.
                       Examples:
                         "What is the TEMA fouling resistance limit for crude oil?"
                         "How do I clean a heat exchanger with asphaltene deposits?"
                         "What does ISO Zone C mean for a bearing?"
                         "What are the symptoms of outer race spalling?"
        n_results:     Number of most relevant passages to return (default 5).
        source_filter: Optional. Filter by document source filename.
                       Options: 'TEMA_standards', 'ISO_10816_vibration_standard',
                                'bearing_failure_catalog', 'maintenance_sop'.

    Returns:
        dict with keys:
            query, n_results, source_filter,
            results: list of dicts, each with:
                rank, text (passage), source, section, distance, relevance_score.
            top_answer: most relevant passage as a plain string.

    Example:
        >>> result = query_knowledge_base("TEMA fouling limit crude oil")
        >>> result["top_answer"]
        "Standard design value for crude oil: Rf = 0.00050 m²·K/W..."
    """
    collection = _get_collection()

    # Build where clause for source filtering
    where = None
    if source_filter:
        where = {"source": {"$eq": source_filter}}

    try:
        raw = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where=where,
        )
    except Exception as e:
        return {"error": f"Query failed: {e}", "query": query}

    results = []
    ids        = raw["ids"][0]
    docs       = raw["documents"][0]
    metas      = raw["metadatas"][0]
    distances  = raw["distances"][0]

    for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
        # Convert L2 distance to 0-1 relevance score (lower distance = more relevant)
        relevance = round(max(0.0, 1.0 - dist / 2.0), 4)
        results.append({
            "rank":            rank + 1,
            "text":            doc,
            "source":          meta.get("source", "unknown"),
            "section":         meta.get("section", ""),
            "chunk_id":        doc_id,
            "distance":        round(float(dist), 6),
            "relevance_score": relevance,
        })

    top_answer = results[0]["text"] if results else "No relevant passages found."

    return {
        "query":         query,
        "n_results":     len(results),
        "source_filter": source_filter,
        "top_answer":    top_answer,
        "results":       results,
    }


def query_tema_standards(topic: str, n_results: int = 3) -> dict:
    """
    Query TEMA heat exchanger standards specifically.

    Shortcut for query_knowledge_base() filtered to TEMA_standards source.

    Args:
        topic:     What to look up (e.g. 'fouling resistance crude', 'cleaning methods').
        n_results: Number of passages to return.

    Returns:
        Same structure as query_knowledge_base().
    """
    return query_knowledge_base(topic, n_results=n_results, source_filter="TEMA_standards")


def query_vibration_standards(topic: str, n_results: int = 3) -> dict:
    """
    Query ISO 10816 vibration standards specifically.

    Args:
        topic:     What to look up (e.g. 'Zone D action', 'BPFO frequency', 'lubrication').
        n_results: Number of passages to return.

    Returns:
        Same structure as query_knowledge_base().
    """
    return query_knowledge_base(topic, n_results=n_results, source_filter="ISO_10816_vibration_standard")


def query_bearing_failures(topic: str, n_results: int = 3) -> dict:
    """
    Query the bearing failure catalog specifically.

    Args:
        topic:     What to look up (e.g. 'outer race spalling symptoms', 'IMS Test 2 failure').
        n_results: Number of passages to return.

    Returns:
        Same structure as query_knowledge_base().
    """
    return query_knowledge_base(topic, n_results=n_results, source_filter="bearing_failure_catalog")


def query_maintenance_sop(topic: str, n_results: int = 3) -> dict:
    """
    Query maintenance standard operating procedures specifically.

    Args:
        topic:     What to look up (e.g. 'bearing replacement steps', 'chemical cleaning procedure').
        n_results: Number of passages to return.

    Returns:
        Same structure as query_knowledge_base().
    """
    return query_knowledge_base(topic, n_results=n_results, source_filter="maintenance_sop")


def get_knowledge_base_status() -> dict:
    """
    Return status of the RAG knowledge base index.

    Returns:
        dict with: index_path, is_ready, n_chunks, sources (list of unique source docs).
    """
    if not RAG_INDEX_DIR.is_dir():
        return {
            "is_ready":  False,
            "index_path": str(RAG_INDEX_DIR),
            "message":    "Index not built. Run: python scripts/build_rag_index.py",
        }
    try:
        collection = _get_collection()
        count      = collection.count()
        # Get unique sources
        if count > 0:
            all_meta = collection.get(include=["metadatas"])["metadatas"]
            sources  = list({m.get("source", "unknown") for m in all_meta})
        else:
            sources = []
        return {
            "is_ready":   True,
            "index_path": str(RAG_INDEX_DIR),
            "n_chunks":   count,
            "sources":    sorted(sources),
        }
    except Exception as e:
        return {
            "is_ready":  False,
            "index_path": str(RAG_INDEX_DIR),
            "error":     str(e),
        }
