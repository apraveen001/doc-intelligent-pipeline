import httpx
from core.embedding import embed_text
from core.vector_store import query_collection
from llm.llm import generate_response

DEFAULT_N_RESULTS = 3

async def run_query(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    filename: str | None = None,
) -> dict:
    if not query.strip():
        return {"query": query, "answer": "", "sources": []}

    async with httpx.AsyncClient() as client:
        query_vector = await embed_text(query, client)

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

    answer = await generate_response(query, chunks)

    return {
        "query":     query,
        "answer":    answer,
        "sources":   chunks,
        "avg_score": round(sum(c["score"] for c in chunks) / len(chunks), 4) if chunks else 0,
        "top_score": chunks[0]["score"] if chunks else 0,
    }
