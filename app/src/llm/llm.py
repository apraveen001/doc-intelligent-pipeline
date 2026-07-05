"""
DocMind — LLM utility
---------------------
Standalone Gemini wrapper for any LLM call outside the LangGraph nodes.
The graph nodes call Gemini directly — this module is a convenience utility
for simple one-off generation calls (e.g. future features, tests, CLI tools).

Config via .env:
    GEMINI_API_KEY   — required
    LLM_MODEL        — optional, defaults to gemini-2.0-flash
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL      = os.getenv("LLM_MODEL", "gemini-2.0-flash")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocMind, a precise research paper assistant.
Answer the user's questions based strictly on the provided context from research papers.

Rules:
- Only use information from the context provided.
- If the answer is not in the context, say "I couldn't find relevant information in the ingested papers."
- Cite sources inline using [source: <paper title>].
- Be concise, clear, and technically accurate.
- Do not hallucinate facts not present in the context.
"""

# ── LLM call ──────────────────────────────────────────────────────────────────

async def generate_response(query: str, chunks: list[dict]) -> str:
    """
    Standalone Gemini call — takes a query and retrieved chunks,
    returns a grounded answer string.

    Args:
        query:  User question.
        chunks: List of dicts with keys: text, source, score (optional).

    Returns:
        Generated answer as a string.
    """
    context = "\n\n---\n\n".join(
        f"[source: {c.get('source', 'unknown')}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"Context from papers:\n{context}\n\nQuestion: {query}"

    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    response = await model.generate_content_async(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 1024},
    )
    return response.text.strip()