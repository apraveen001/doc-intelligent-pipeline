"""
DocMind LangGraph Agent Graph
------------------------------
Wires the four nodes into a compiled StateGraph:

    [START]
      │
      ▼
   Planner          — rephrase the query
      │
      ▼
   Retrieval        — embed + fetch top-k chunks
      │
      ▼
   Synthesis        — generate answer from chunks
      │
      ▼
   Validator ──────► END   (if grounded OR max retries reached)
      │
      └──────────────► Synthesis  (retry if not grounded, retries remaining)

Usage
-----
    from app.src.graph.graph import build_graph, run_graph

    graph = build_graph()

    # Standard (blocking) run
    result: GraphState = await run_graph(graph, user_query="What is RAG?")

    # Streaming run (yields SSE-ready dicts)
    async for event in stream_graph(graph, user_query="What is RAG?"):
        ...
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from langgraph.graph import END, START, StateGraph

from app.src.graph.nodes import (
    planner_node,
    retrieval_node,
    synthesis_node,
    validator_node,
)
from app.src.graph.state import GraphState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge: should we retry Synthesis or terminate?
# ─────────────────────────────────────────────────────────────────────────────

def _should_retry(state: GraphState) -> str:
    """
    After Validator runs, decide the next node.

    Returns "synthesis" to retry, or END to finish.
    """
    if state.error:
        logger.warning("Graph terminating early due to error: %s", state.error)
        return END  # type: ignore[return-value]

    if not state.is_grounded and state.retry_count < state.max_retries:
        logger.info(
            "Answer not grounded — retrying synthesis (attempt %d/%d).",
            state.retry_count,
            state.max_retries,
        )
        return "synthesis"

    return END  # type: ignore[return-value]


def _should_continue_after_retrieval(state: GraphState) -> str:
    """Abort early if retrieval encountered a fatal error."""
    if state.error:
        return END  # type: ignore[return-value]
    return "synthesis"


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the DocMind agent graph.

    Returns a compiled LangGraph StateGraph ready for invocation.
    """
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("planner", planner_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("validator", validator_node)

    # Linear edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retrieval")

    # Conditional: abort on retrieval error, else synthesise
    builder.add_conditional_edges(
        "retrieval",
        _should_continue_after_retrieval,
        {"synthesis": "synthesis", END: END},
    )

    builder.add_edge("synthesis", "validator")

    # Conditional: retry synthesis or end
    builder.add_conditional_edges(
        "validator",
        _should_retry,
        {"synthesis": "synthesis", END: END},
    )

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_graph(graph: StateGraph, user_query: str) -> GraphState:
    """
    Run the full graph and return the final state.

    Args:
        graph:      A compiled graph returned by build_graph().
        user_query: Raw user question.

    Returns:
        Final GraphState after all nodes have executed.
    """
    initial_state = GraphState(
        original_query=user_query,
        max_retries=2,
    )
    result = await graph.ainvoke(initial_state)
    # ainvoke returns a dict; coerce back to GraphState for typed access
    if isinstance(result, dict):
        return GraphState(**result)
    return result


async def stream_graph(
    graph: StateGraph,
    user_query: str,
) -> AsyncGenerator[dict, None]:
    """
    Stream node-transition events as they happen.

    Yields dicts compatible with FastAPI's SSE StreamingResponse:
        {"node": str, "status": str, "detail": str, "timestamp": str}

    The final event carries the complete result:
        {"node": "done", "status": "completed", "answer": str,
         "sources": list[str], "is_grounded": bool}
    """
    initial_state = GraphState(
        original_query=user_query,
        max_retries=2,
    )

    seen_events: set[int] = set()  # track by index to avoid re-emitting

    async for chunk in graph.astream(initial_state, stream_mode="values"):
        # chunk is the state dict after each node completes
        state = GraphState(**chunk) if isinstance(chunk, dict) else chunk

        # Yield any new node_events appended since last chunk
        for idx, event in enumerate(state.node_events):
            if idx not in seen_events:
                seen_events.add(idx)
                yield event

        # If the graph has terminated (answer is populated), emit a final event
        if state.answer and state.is_grounded or (
            state.retry_count >= state.max_retries
        ):
            # Will be yielded once on the last chunk — guard with a flag
            pass

    # After streaming ends, yield the final summary event
    # Re-invoke to get the complete final state (stream_mode="values" gives
    # intermediate states; we need the terminal one)
    final = await run_graph(graph, user_query)
    yield {
        "node": "done",
        "status": "completed",
        "answer": final.answer,
        "sources": final.answer_sources,
        "is_grounded": final.is_grounded,
        "validation_reasoning": final.validation_reasoning,
        "retry_count": final.retry_count,
        "error": final.error,
    }