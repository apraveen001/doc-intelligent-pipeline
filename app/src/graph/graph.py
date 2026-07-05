"""
DocMind Research Paper Assistant — LangGraph Agent Graph
---------------------------------------------------------
Two separate graphs for the two phases of the app:

  PHASE 1 — Research graph (run once per session)
  ─────────────────────────────────────────────────
    [START]
       │
       ▼
    Planner        — rephrase topic into ArXiv query
       │
       ▼
    Search         — hit ArXiv API, get 10 candidates
       │
       ▼
    Selector       — Gemini picks best 3 papers
       │
       ▼
    Ingestor       — download, chunk, embed, store in ChromaDB
       │
       ▼
     [END]         — frontend shows paper list, chat unlocked

  PHASE 2 — QA graph (run per user question)
  ───────────────────────────────────────────
    [START]
       │
       ▼
      QA            — retrieve chunks (session-scoped), generate answer
       │
       ▼
    Validator ─────► END   (grounded OR max retries reached)
       │
       └───────────► QA    (retry if not grounded, retries < max)

Usage
-----
    from app.src.graph.graph import (
        build_research_graph,
        build_qa_graph,
        run_research_phase,
        run_qa_phase,
        stream_research_phase,
        stream_qa_phase,
    )
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from langgraph.graph import END, START, StateGraph

from app.src.graph.nodes import (
    ingestor_node,
    planner_node,
    qa_node,
    search_node,
    selector_node,
    validator_node,
)
from app.src.graph.state import GraphState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edges
# ─────────────────────────────────────────────────────────────────────────────

def _abort_on_error(state: GraphState) -> str:
    """Generic early-exit edge — used after Search and Selector."""
    if state.error:
        logger.warning("Graph aborting early: %s", state.error)
        return END  # type: ignore[return-value]
    return "continue"


def _after_search(state: GraphState) -> str:
    if state.error or not state.candidate_papers:
        return END  # type: ignore[return-value]
    return "selector"


def _after_selector(state: GraphState) -> str:
    if state.error or not state.selected_papers:
        return END  # type: ignore[return-value]
    return "ingestor"


def _after_ingestor(state: GraphState) -> str:
    if state.error or not state.ingestion_complete:
        return END  # type: ignore[return-value]
    return END  # type: ignore[return-value]  # research phase always ends here


def _after_validator(state: GraphState) -> str:
    """Retry QA or end the QA phase."""
    if state.error:
        return END  # type: ignore[return-value]
    if not state.is_grounded and state.retry_count < state.max_retries:
        logger.info(
            "Answer not grounded — retrying QA (attempt %d/%d).",
            state.retry_count,
            state.max_retries,
        )
        return "qa"
    return END  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Graph factories
# ─────────────────────────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    """
    Build and compile Phase 1: Planner → Search → Selector → Ingestor.
    Run once when the user submits a topic.
    """
    builder = StateGraph(GraphState)

    builder.add_node("planner",  planner_node)
    builder.add_node("search",   search_node)
    builder.add_node("selector", selector_node)
    builder.add_node("ingestor", ingestor_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "search")

    builder.add_conditional_edges(
        "search",
        _after_search,
        {"selector": "selector", END: END},
    )
    builder.add_conditional_edges(
        "selector",
        _after_selector,
        {"ingestor": "ingestor", END: END},
    )
    builder.add_edge("ingestor", END)

    return builder.compile()


def build_qa_graph() -> StateGraph:
    """
    Build and compile Phase 2: QA → Validator (with retry loop).
    Run once per user question during the chat session.
    """
    builder = StateGraph(GraphState)

    builder.add_node("qa",        qa_node)
    builder.add_node("validator", validator_node)

    builder.add_edge(START, "qa")
    builder.add_edge("qa", "validator")

    builder.add_conditional_edges(
        "validator",
        _after_validator,
        {"qa": "qa", END: END},
    )

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Runners — Phase 1 (research)
# ─────────────────────────────────────────────────────────────────────────────

async def run_research_phase(
    graph: StateGraph,
    session_id: str,
    user_topic: str,
) -> GraphState:
    """
    Run the research phase and return the final state.

    Args:
        graph:      Compiled research graph from build_research_graph().
        session_id: UUID for this session (scopes ChromaDB chunks).
        user_topic: Raw topic the user typed or clicked.

    Returns:
        Final GraphState with ingested_papers and selected_papers populated.
    """
    initial = GraphState(
        session_id=session_id,
        user_topic=user_topic,
        max_retries=2,
    )
    result = await graph.ainvoke(initial)
    if isinstance(result, dict):
        return GraphState(**result)
    return result


async def stream_research_phase(
    graph: StateGraph,
    session_id: str,
    user_topic: str,
) -> AsyncGenerator[dict, None]:
    """
    Stream research phase node events as SSE-ready dicts.

    Yields:
        {"type": "node_event", "node": str, "status": str, "detail": str, "timestamp": str}
        {"type": "done", "papers": list[dict], "error": str | None}  ← final event
    """
    initial = GraphState(
        session_id=session_id,
        user_topic=user_topic,
        max_retries=2,
    )

    seen: set[int] = set()

    async for chunk in graph.astream(initial, stream_mode="values"):
        state = GraphState(**chunk) if isinstance(chunk, dict) else chunk
        for idx, event in enumerate(state.node_events):
            if idx not in seen:
                seen.add(idx)
                yield {"type": "node_event", **event}

    # Final event — paper list for the frontend to render
    final = await run_research_phase(graph, session_id, user_topic)
    yield {
        "type": "done",
        "papers": [
            {
                "arxiv_id":     p.arxiv_id,
                "title":        p.title,
                "authors":      p.authors,
                "abstract":     p.abstract,
                "published":    p.published,
                "chunks_stored": p.chunks_stored,
                "status":       p.status,
            }
            for p in final.ingested_papers
        ],
        "error": final.error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Runners — Phase 2 (QA)
# ─────────────────────────────────────────────────────────────────────────────

async def run_qa_phase(
    graph: StateGraph,
    current_state: GraphState,
    user_question: str,
) -> GraphState:
    """
    Run one QA turn and return the updated state.

    Args:
        graph:          Compiled QA graph from build_qa_graph().
        current_state:  The live GraphState carrying session_id,
                        conversation_history, etc.
        user_question:  The user's current question.

    Returns:
        Updated GraphState with new answer, sources, and conversation_history.
    """
    # Inject the new question and reset per-turn validator fields
    turn_state = current_state.model_copy(update={
        "user_question": user_question,
        "retrieved_chunks": [],
        "answer": "",
        "answer_sources": [],
        "is_grounded": False,
        "validation_reasoning": "",
        "retry_count": 0,
        "error": None,
    })

    result = await graph.ainvoke(turn_state)
    if isinstance(result, dict):
        return GraphState(**result)
    return result


async def stream_qa_phase(
    graph: StateGraph,
    current_state: GraphState,
    user_question: str,
) -> AsyncGenerator[dict, None]:
    """
    Stream one QA turn as SSE-ready dicts.

    Yields:
        {"type": "node_event", "node": str, "status": str, "detail": str, "timestamp": str}
        {"type": "done", "answer": str, "sources": list[str],
         "is_grounded": bool, "error": str | None}  ← final event
    """
    turn_state = current_state.model_copy(update={
        "user_question": user_question,
        "retrieved_chunks": [],
        "answer": "",
        "answer_sources": [],
        "is_grounded": False,
        "validation_reasoning": "",
        "retry_count": 0,
        "error": None,
    })

    seen: set[int] = set()
    base_event_count = len(current_state.node_events)

    async for chunk in graph.astream(turn_state, stream_mode="values"):
        state = GraphState(**chunk) if isinstance(chunk, dict) else chunk
        for idx, event in enumerate(state.node_events):
            # Only yield events added during this QA turn
            if idx >= base_event_count and idx not in seen:
                seen.add(idx)
                yield {"type": "node_event", **event}

    # Final event — answer for the frontend
    final = await run_qa_phase(graph, current_state, user_question)
    yield {
        "type": "done",
        "answer":       final.answer,
        "sources":      final.answer_sources,
        "is_grounded":  final.is_grounded,
        "error":        final.error,
    }