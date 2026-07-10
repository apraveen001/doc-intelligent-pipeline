"""
DocMind — Vector store utility
------------------------------
ChromaDB Cloud client and collection helpers.

Single collection strategy (free tier compatible):
    - One collection for all sessions
    - Every chunk tagged with session_id in metadata
    - Queries and cleanup filter by session_id

Config via .env:
    CHROMA_API_KEY
    CHROMA_TENANT
    CHROMA_DATABASE
"""

import os

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
    raise EnvironmentError(
        "Missing ChromaDB config. Ensure CHROMA_API_KEY, CHROMA_TENANT, "
        "and CHROMA_DATABASE are set in your .env file."
    )


# ── Client setup ──────────────────────────────────────────────────────────────

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)


# ── Collection ────────────────────────────────────────────────────────────────

COLLECTION_NAME = "docmind-research"

def get_collection():
    """
    Get or create the single shared ChromaDB collection.
    All sessions share this collection — chunks are scoped by session_id metadata.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Store ─────────────────────────────────────────────────────────────────────

def store_embeddings(embedded_chunks: list[dict], filename: str) -> int:
    """
    Store embedded chunks into ChromaDB.
    Kept for backwards compatibility — mcp/tools.py handles ingestion directly.

    Expects each chunk to have:
        { "chunk_index": int, "text": str, "word_count": int, "embedding": list[float] }

    Returns the number of chunks stored.
    """
    collection = get_collection()

    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for chunk in embedded_chunks:
        chunk_id = f"{filename}::chunk_{chunk['chunk_index']}"
        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "filename":    filename,
            "chunk_index": chunk["chunk_index"],
            "word_count":  chunk["word_count"],
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)


# ── Query ─────────────────────────────────────────────────────────────────────

def query_collection(
    query_embedding: list[float],
    n_results: int = 5,
    session_id: str | None = None,
) -> dict:
    """
    Query ChromaDB with a vector, optionally filtered by session_id.
    Returns raw ChromaDB results dict (documents, metadatas, distances, ids).

    Note: the QA node calls collection.query() directly for session-scoped
    retrieval. This function is available for any direct query needs.
    """
    collection = get_collection()

    where = {"session_id": session_id} if session_id else None

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )