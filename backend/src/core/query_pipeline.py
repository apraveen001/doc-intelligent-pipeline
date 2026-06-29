import httpx
from core.embedding import embed_text
from core.vector_store import query_collection
from llm.llm import generate_response

DEFAULT_N_RESULTS = 5

async def run_query(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    filename: str | None = None,
) -> dict:
    if not query.strip():
        return {"query": query, "answer": "", "sources": []}

    # Step 1 — embed the query
    async with httpx.AsyncClient() as client:
        query_vector = await embed_text(query, client)

    # Step 2 — retrieve similar chunks from ChromaDB
    raw_results = query_collection(
        query_embedding=query_vector,
        n_results=n_results,
        filename=filename,
    )

    chunks = [
        {
            "rank":     i + 1,
            "chunk_id": r["chunk_id"],
            "text":     r["text"],
            "metadata": r["metadata"],
            "score":    round(1 - r["distance"], 4),
        }
        for i, r in enumerate(raw_results)
    ]

    # Step 3 — pass chunks to LLM and generate answer
    answer = await generate_response(query, chunks)

    return {
        "query":   query,
        "answer":  answer,
        "sources": chunks,
    }