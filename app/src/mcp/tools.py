"""
DocMind MCP Tools
-----------------
Three tools controlled by LangGraph:

    1. search_papers      — query ArXiv API, return candidate papers
    2. fetch_and_ingest   — download PDF, chunk, embed, store in ChromaDB
    3. clear_session      — delete all chunks for a session from ChromaDB

These are plain async functions — LangGraph calls them directly.
No HTTP transport needed since MCP is internal to this project.

ChromaDB strategy:
    - Single collection (free tier compatible)
    - Every chunk tagged with session_id in metadata
    - Queries and cleanup filter by session_id
"""

from __future__ import annotations

import asyncio
import io
import logging
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from typing import Any

import httpx
import pdfplumber

from app.src.core.embedding import embed_text
from app.src.core.vector_store import get_collection

logger = logging.getLogger(__name__)

# ── ArXiv API config ──────────────────────────────────────────────────────────
ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_NAMESPACE = "{http://www.w3.org/2005/Atom}"
MAX_CANDIDATE_RESULTS = 10   # fetch this many, Gemini picks best 3
MAX_PAPERS_TO_INGEST = 3

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — search_papers
# ─────────────────────────────────────────────────────────────────────────────

async def search_papers(
    query: str,
    max_results: int = MAX_CANDIDATE_RESULTS,
) -> list[dict[str, str]]:
    """
    Search ArXiv for papers matching the query.

    Args:
        query:       ArXiv search query (already rephrased by Planner/Gemini).
        max_results: How many candidates to return for Gemini to rank.

    Returns:
        List of dicts:
        {
            "arxiv_id":   str,   # e.g. "2401.12345"
            "title":      str,
            "authors":    str,   # comma-joined
            "abstract":   str,
            "pdf_url":    str,
            "published":  str,   # ISO date
        }
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_BASE}?{urllib.parse.urlencode(params)}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    root = ET.fromstring(response.text)
    papers: list[dict[str, str]] = []

    for entry in root.findall(f"{ARXIV_NAMESPACE}entry"):
        arxiv_id_raw = entry.findtext(f"{ARXIV_NAMESPACE}id", "")
        # Extract clean ID from URL like http://arxiv.org/abs/2401.12345v1
        arxiv_id = arxiv_id_raw.split("/abs/")[-1].split("v")[0]

        title = (entry.findtext(f"{ARXIV_NAMESPACE}title") or "").strip().replace("\n", " ")
        abstract = (entry.findtext(f"{ARXIV_NAMESPACE}summary") or "").strip().replace("\n", " ")
        published = (entry.findtext(f"{ARXIV_NAMESPACE}published") or "")[:10]

        authors = ", ".join(
            (author.findtext(f"{ARXIV_NAMESPACE}name") or "")
            for author in entry.findall(f"{ARXIV_NAMESPACE}author")
        )

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "published": published,
        })

    logger.info("ArXiv search for '%s' returned %d candidates.", query, len(papers))
    return papers


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — fetch_and_ingest
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    500 words per chunk, 50-word overlap — consistent with existing pipeline.
    """
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    return chunks


async def fetch_and_ingest(
    paper: dict[str, str],
    session_id: str,
) -> dict[str, Any]:
    """
    Download a paper PDF, chunk it, embed each chunk, store in ChromaDB.

    Args:
        paper:      Dict from search_papers (must have pdf_url, arxiv_id, title).
        session_id: UUID string scoping this session in ChromaDB.

    Returns:
        {
            "arxiv_id":     str,
            "title":        str,
            "chunks_stored":int,
            "status":       "success" | "error",
            "error":        str | None,
        }
    """
    arxiv_id = paper["arxiv_id"]
    title = paper["title"]
    pdf_url = paper["pdf_url"]

    logger.info("Fetching PDF for '%s' (%s)", title, arxiv_id)

    # ── Download PDF ──────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
    except Exception as exc:
        logger.error("Failed to download PDF for %s: %s", arxiv_id, exc)
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "chunks_stored": 0,
            "status": "error",
            "error": f"Download failed: {exc}",
        }

    # ── Extract text ──────────────────────────────────────────────────────────
    try:
        full_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    except Exception as exc:
        logger.error("Failed to extract text from PDF %s: %s", arxiv_id, exc)
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "chunks_stored": 0,
            "status": "error",
            "error": f"PDF extraction failed: {exc}",
        }

    if not full_text.strip():
        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "chunks_stored": 0,
            "status": "error",
            "error": "No extractable text found in PDF.",
        }

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = _chunk_text(full_text)
    logger.info("Paper '%s' split into %d chunks.", title, len(chunks))

    # ── Embed + store ─────────────────────────────────────────────────────────
    collection = get_collection()
    stored = 0

    for idx, chunk_text in enumerate(chunks):
        try:
            embedding = await embed_text(chunk_text)
            chunk_id = f"{session_id}_{arxiv_id}_{idx}"

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[{
                    "session_id": session_id,
                    "arxiv_id":   arxiv_id,
                    "title":      title,
                    "authors":    paper.get("authors", ""),
                    "chunk_idx":  idx,
                    "source":     f"{title} (arxiv:{arxiv_id})",
                }],
            )
            stored += 1

        except Exception as exc:
            logger.warning("Failed to embed/store chunk %d of %s: %s", idx, arxiv_id, exc)
            continue

    logger.info("Stored %d/%d chunks for '%s'.", stored, len(chunks), title)
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "chunks_stored": stored,
        "status": "success",
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — clear_session
# ─────────────────────────────────────────────────────────────────────────────

async def clear_session(session_id: str) -> dict[str, Any]:
    """
    Delete all ChromaDB chunks belonging to this session.

    Args:
        session_id: The session UUID to wipe.

    Returns:
        {
            "session_id":   str,
            "deleted":      bool,
            "error":        str | None,
        }
    """
    logger.info("Clearing session %s from ChromaDB.", session_id)

    try:
        collection = get_collection()

        # Fetch all IDs for this session using metadata filter
        results = collection.get(
            where={"session_id": session_id},
            include=[],  # IDs only, no need to fetch embeddings/documents
        )

        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted %d chunks for session %s.", len(ids_to_delete), session_id
            )
        else:
            logger.info("No chunks found for session %s — nothing to delete.", session_id)

        return {
            "session_id": session_id,
            "deleted": True,
            "chunks_deleted": len(ids_to_delete),
            "error": None,
        }

    except Exception as exc:
        logger.error("Failed to clear session %s: %s", session_id, exc)
        return {
            "session_id": session_id,
            "deleted": False,
            "chunks_deleted": 0,
            "error": str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Session ID generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_session_id() -> str:
    """Generate a unique session ID. Called by FastAPI when a session starts."""
    return str(uuid.uuid4())