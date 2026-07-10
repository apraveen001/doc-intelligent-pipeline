"""
DocMind — Embedding utility
Google text-embedding-004 via Vertex AI + ADC.
768-dimensional vectors.
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
EMBEDDING_MODEL = "text-embedding-004"


async def embed_text(text: str) -> list[float]:
    """Embed a single string. Returns a 768-dimensional float vector."""
    result = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values


async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed all chunks. Adds 'embedding' key to each chunk dict."""
    embedded = []
    for chunk in chunks:
        vector = await embed_text(chunk["text"])
        embedded.append({**chunk, "embedding": vector})
        print(f"  [embedding] chunk {chunk['chunk_index']} — {len(vector)}d vector")
    return embedded