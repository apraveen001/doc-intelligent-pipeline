"""
DocMind — LLM utility
Gemini via Vertex AI + Application Default Credentials (ADC).
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.getenv("GCP_PROJECT"),
    location=os.getenv("GCP_LOCATION", "us-central1"),
)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = """You are DocMind, an intelligent research paper assistant with a focused and scholarly persona.

Your sole purpose is to help users understand ingested research papers. Answer ONLY from the provided context chunks. Never use outside knowledge or general facts. If a question is out of scope, respond:
"That question is outside the scope of the ingested research papers. I can only answer questions based on the papers in this session."

Always cite sources inline using [source: <paper title>]. Be precise, clear, and academically grounded."""


async def generate_response(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[source: {c.get('source', 'unknown')}]\n{c['text']}"
        for c in chunks
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = await client.aio.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()