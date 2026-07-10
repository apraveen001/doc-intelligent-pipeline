"""
DocMind Research Paper Assistant — LangGraph State Schema
---------------------------------------------------------
Pydantic-typed state shared across all graph nodes.

Flow:
    Planner → Search → Selector → Ingestor → QA → Validator
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class PaperMeta(BaseModel):
    """Metadata for a single ArXiv paper returned by search."""
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    pdf_url: str
    published: str


class IngestedPaper(BaseModel):
    """Result of fetch_and_ingest for a single paper."""
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    published: str
    chunks_stored: int
    status: str          # "success" | "error"
    error: str | None = None


class RetrievedChunk(BaseModel):
    """A single chunk returned from ChromaDB during QA."""
    text: str
    source: str
    arxiv_id: str
    chunk_id: str
    distance: float


class ConversationTurn(BaseModel):
    """A single turn in the chat conversation."""
    role: str            # "user" | "assistant"
    content: str
    sources: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main graph state
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(BaseModel):
    """
    Full state passed between every LangGraph node.

    Each node returns a dict of only the fields it changes.
    LangGraph merges updates; node_events and conversation_history
    use append reducers so they are never overwritten.
    """

    # ── Session ───────────────────────────────────────────────────────────────
    session_id: str = Field(
        default="",
        description="UUID for this research session. Scopes ChromaDB chunks.",
    )

    # ── User input ────────────────────────────────────────────────────────────
    user_topic: str = Field(
        default="",
        description="Raw topic the user typed or clicked (e.g. 'LLM reasoning').",
    )
    user_question: str = Field(
        default="",
        description="Current question in the QA loop.",
    )

    # ── Planner output ────────────────────────────────────────────────────────
    arxiv_query: str = Field(
        default="",
        description="ArXiv-optimised search query produced by Planner from user_topic.",
    )
    planning_reasoning: str = Field(
        default="",
        description="Planner's one-line reasoning for the query rewrite.",
    )

    # ── Search output ─────────────────────────────────────────────────────────
    candidate_papers: list[PaperMeta] = Field(
        default_factory=list,
        description="Up to 10 candidate papers returned by ArXiv search.",
    )

    # ── Selector output ───────────────────────────────────────────────────────
    selected_papers: list[PaperMeta] = Field(
        default_factory=list,
        description="Best 3 papers chosen by Gemini from candidates.",
    )
    selection_reasoning: str = Field(
        default="",
        description="Gemini's reasoning for the paper selection.",
    )

    # ── Ingestor output ───────────────────────────────────────────────────────
    ingested_papers: list[IngestedPaper] = Field(
        default_factory=list,
        description="Results of fetch_and_ingest for each selected paper.",
    )
    ingestion_complete: bool = Field(
        default=False,
        description="True once all selected papers are chunked and stored.",
    )

    # ── QA output ─────────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks retrieved from ChromaDB for the current question.",
    )
    answer: str = Field(
        default="",
        description="Gemini's answer to the current question.",
    )
    answer_sources: list[str] = Field(
        default_factory=list,
        description="Source strings cited in the answer.",
    )

    # ── Conversation history (append-only) ────────────────────────────────────
    conversation_history: Annotated[list[ConversationTurn], operator.add] = Field(
        default_factory=list,
        description="Full chat history for PDF export and multi-turn context.",
    )

    # ── Validator state ───────────────────────────────────────────────────────
    is_grounded: bool = Field(
        default=False,
        description="True when Validator confirms answer is grounded in chunks.",
    )
    validation_reasoning: str = Field(
        default="",
        description="Validator's one-line grounding check reasoning.",
    )
    retry_count: int = Field(
        default=0,
        description="Number of QA → Validator retry cycles for current question.",
    )
    max_retries: int = Field(
        default=2,
        description="Max retries before accepting answer as-is.",
    )

    # ── SSE events (append-only) ──────────────────────────────────────────────
    node_events: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description=(
            "Append-only log of node transition events streamed to frontend via SSE. "
            "Each entry: {node, status, detail, timestamp}."
        ),
    )

    # ── Error handling ────────────────────────────────────────────────────────
    error: str | None = Field(
        default=None,
        description="Set by any node on unrecoverable error. Graph terminates early.",
    )

    class Config:
        arbitrary_types_allowed = True