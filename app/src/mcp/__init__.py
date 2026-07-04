"""
DocMind MCP Tools package.

Public API:
    search_papers       — query ArXiv, return candidate papers
    fetch_and_ingest    — download, chunk, embed, store in ChromaDB
    clear_session       — wipe session chunks from ChromaDB
    generate_session_id — create a new UUID session ID
"""

from app.src.mcp.tools import (
    search_papers,
    fetch_and_ingest,
    clear_session,
    generate_session_id,
)

__all__ = [
    "search_papers",
    "fetch_and_ingest",
    "clear_session",
    "generate_session_id",
]