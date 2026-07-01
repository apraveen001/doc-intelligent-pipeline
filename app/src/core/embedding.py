import httpx


# ── Constants ─────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text


# ── Single chunk embedding ────────────────────────────────────────────────────

async def embed_text(text: str, client: httpx.AsyncClient) -> list[float]:
    """
    Send a single text string to Ollama and return its embedding vector.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embedding"]


# ── Batch embedding ───────────────────────────────────────────────────────────

async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks produced by extraction.parse_and_chunk.

    Expects each chunk to have at least:
        { "chunk_index": int, "text": str, "word_count": int }

    Returns the same list with an "embedding" key added to each chunk:
        { "chunk_index": int, "text": str, "word_count": int, "embedding": list[float] }
    """
    embedded = []

    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            vector = await embed_text(chunk["text"], client)
            embedded.append({**chunk, "embedding": vector})
            print(f"  [embedding] chunk {chunk['chunk_index']} — {len(vector)}d vector")

    return embedded


# ── Health check ──────────────────────────────────────────────────────────────

async def check_ollama() -> bool:
    """Return True if Ollama is reachable and the embedding model is available."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return any(EMBEDDING_MODEL in m for m in models)
    except Exception:
        return False