import os
from dotenv import load_dotenv
import chromadb

# ── Load environment variables ────────────────────────────────────────────────

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

def get_collection(name: str = "doc-intelligent-pipeline"):
    """Get or create a ChromaDB collection."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Store ─────────────────────────────────────────────────────────────────────

def store_embeddings(embedded_chunks: list[dict], filename: str) -> int:
    """
    Store embedded chunks into ChromaDB Cloud.

    Expects each chunk to have:
        {
            "chunk_index": int,
            "text": str,
            "word_count": int,
            "embedding": list[float],
        }

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
    filename: str | None = None,
) -> list[dict]:
    """
    Query ChromaDB with a vector and return the top n_results chunks.

    Optionally filter by filename to search within a single document.
    """
    collection = get_collection()

    where = {"filename": filename} if filename else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "chunk_id": results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return output