import os
import httpx
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://api.ollama.com")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent document assistant.
Your job is to answer the user's questions based strictly on the provided document context.

Rules:
- Only use information from the context provided.
- If the answer is not in the context, say "I couldn't find relevant information in the uploaded documents."
- Be concise, clear, and factual.
- Do not make up information or use outside knowledge.
- If quoting directly from the document, make it clear.
"""

# ── LLM call ──────────────────────────────────────────────────────────────────

async def generate_response(query: str, chunks: list[dict]) -> str:
    """
    Takes the user query and retrieved chunks, sends them to the Ollama LLM,
    and returns the generated answer as a string.
    """
    context = "\n\n---\n\n".join(
        f"[Source {c['rank']} | Relevance: {c['score']}]\n{c['text']}"
        for c in chunks
    )

    user_message = f"""Context from documents:
{context}

User question: {query}"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()