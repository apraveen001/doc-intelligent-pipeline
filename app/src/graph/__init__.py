"""
DocMind LangGraph agent graph package.

Public API:
    build_graph()   → compiled StateGraph
    run_graph()     → async, returns final GraphState
    stream_graph()  → async generator of SSE-ready dicts
    GraphState      → Pydantic state model
"""

from app.src.graph.graph import build_graph, run_graph, stream_graph
from app.src.graph.state import GraphState, RetrievedChunk

__all__ = [
    "build_graph",
    "run_graph",
    "stream_graph",
    "GraphState",
    "RetrievedChunk",
]