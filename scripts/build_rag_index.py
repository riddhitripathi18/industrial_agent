"""
Build RAG Index
===============
Chunks industrial knowledge base documents and embeds them into a
ChromaDB vector store using sentence-transformers (all-MiniLM-L6-v2).

Run once before using rag_tools:
    python scripts/build_rag_index.py

The index is saved to data/rag_index/ (gitignored).
Re-run any time documents in data/docs/ are updated.
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import DATA_DIR, REPO_ROOT

# Docs and RAG index live in the repo, not the external CSV data directory
DOCS_DIR        = REPO_ROOT / "data" / "docs"
RAG_INDEX_DIR   = REPO_ROOT / "data" / "rag_index"
COLLECTION_NAME = "industrial_knowledge"
CHUNK_SIZE      = 400   # characters per chunk
CHUNK_OVERLAP   = 80    # character overlap between consecutive chunks


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a document into overlapping chunks, preserving section headers as metadata.
    Each chunk gets: id, text, source, section.
    """
    lines   = text.split("\n")
    chunks  = []
    current_section = "General"
    buffer  = []
    buf_len = 0
    chunk_n = 0

    for line in lines:
        # Track section from Markdown headers
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped.lstrip("#").strip()
        elif stripped.startswith("### "):
            current_section = stripped.lstrip("#").strip()

        buffer.append(line)
        buf_len += len(line) + 1

        if buf_len >= chunk_size:
            chunk_text_val = "\n".join(buffer).strip()
            if chunk_text_val:
                chunks.append({
                    "id":      f"{source}_{chunk_n:04d}",
                    "text":    chunk_text_val,
                    "source":  source,
                    "section": current_section,
                })
                chunk_n += 1
            # Overlap: keep last few lines for context
            overlap_lines = []
            overlap_chars = 0
            for l in reversed(buffer):
                overlap_chars += len(l) + 1
                overlap_lines.insert(0, l)
                if overlap_chars >= overlap:
                    break
            buffer  = overlap_lines
            buf_len = overlap_chars

    # Final chunk
    remainder = "\n".join(buffer).strip()
    if remainder:
        chunks.append({
            "id":      f"{source}_{chunk_n:04d}",
            "text":    remainder,
            "source":  source,
            "section": current_section,
        })

    return chunks


def build_index():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb sentence-transformers")
        sys.exit(1)

    # Collect all markdown documents
    if not DOCS_DIR.is_dir():
        print(f"ERROR: Docs directory not found: {DOCS_DIR}")
        sys.exit(1)

    doc_files = list(DOCS_DIR.glob("*.md"))
    if not doc_files:
        print(f"ERROR: No .md files found in {DOCS_DIR}")
        sys.exit(1)

    print(f"Found {len(doc_files)} document(s) in {DOCS_DIR}")

    # Build all chunks
    all_ids   = []
    all_texts = []
    all_metas = []

    for doc_path in sorted(doc_files):
        source = doc_path.stem
        text   = doc_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, source)
        print(f"  {source}: {len(chunks)} chunks")
        for c in chunks:
            all_ids.append(c["id"])
            all_texts.append(c["text"])
            all_metas.append({"source": c["source"], "section": c["section"]})

    print(f"\nTotal chunks: {len(all_ids)}")

    # Create/overwrite ChromaDB collection
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(RAG_INDEX_DIR))

    # Delete existing collection if present
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing index.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "l2"},
    )

    # Embed in batches of 100
    BATCH = 100
    print(f"\nEmbedding {len(all_ids)} chunks using all-MiniLM-L6-v2...")
    for i in range(0, len(all_ids), BATCH):
        batch_ids   = all_ids[i:i + BATCH]
        batch_texts = all_texts[i:i + BATCH]
        batch_metas = all_metas[i:i + BATCH]
        collection.add(ids=batch_ids, documents=batch_texts, metadatas=batch_metas)
        pct = min(i + BATCH, len(all_ids)) / len(all_ids) * 100
        print(f"  {min(i + BATCH, len(all_ids))}/{len(all_ids)} ({pct:.0f}%)")

    print(f"\nIndex built successfully at: {RAG_INDEX_DIR}")
    print(f"Total documents embedded: {collection.count()}")
    print("\nSources indexed:")
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources  = sorted({m["source"] for m in all_meta})
    for s in sources:
        n = sum(1 for m in all_meta if m["source"] == s)
        print(f"  {s}: {n} chunks")


if __name__ == "__main__":
    build_index()
