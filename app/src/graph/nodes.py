"""
DocMind Research Paper Assistant — LangGraph Nodes
---------------------------------------------------
Six nodes wired into the agent graph:

    Planner → Search → Selector → Ingestor → QA → Validator
                                                ↑________|  (retry, max 2)

Each node is a plain async function:
    async def node_name(state: GraphState) -> dict
Returns only the fields it mutates; LangGraph merges them into state.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import google.generativeai as genai

from app.src.graph.state import (
    ConversationTurn,
    GraphState,
    IngestedPaper,
    PaperMeta,
    RetrievedChunk,
)
from app.src.mcp.tools import fetch_and_ingest, search_papers

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.0-flash"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(node: str, status: str, detail: str = "") -> dict:
    """Build a node-transition event dict for SSE streaming."""
    return {"node": node, "status": status, "detail": detail, "timestamp": _now_iso()}


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────────────────────────────────────────
# 1. PLANNER NODE
# Rephrases the user's topic into a clean ArXiv search query
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are an expert at constructing ArXiv search queries.

The user will give you a research topic in plain language.
Your job is to rewrite it into a precise, effective ArXiv search query.

Rules:
- Use specific technical terminology
- Focus on the core concept, not filler words
- Keep it concise — ArXiv search works best with 3-6 key terms
- Do NOT use boolean operators like AND/OR

Respond ONLY with a JSON object (no markdown fences):
{
  "arxiv_query": "<your optimised search query>",
  "reasoning": "<one sentence explaining your choices>"
}"""


async def planner_node(state: GraphState) -> dict:
    """
    Rephrase user_topic into an ArXiv-optimised search query.
    Returns: arxiv_query, planning_reasoning, node_events
    """
    events = [_event("planner", "started", f"Optimising query for: '{state.user_topic}'")]

    try:
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            system_instruction=_PLANNER_SYSTEM,
        )
        response = await model.generate_content_async(
            state.user_topic,
            generation_config={"temperature": 0.2, "max_output_tokens": 256},
        )

        parsed = _parse_json_response(response.text)
        arxiv_query = parsed.get("arxiv_query", state.user_topic).strip()
        reasoning = parsed.get("reasoning", "")

        events.append(_event("planner", "completed", f"ArXiv query: '{arxiv_query}'"))
        return {
            "arxiv_query": arxiv_query,
            "planning_reasoning": reasoning,
            "node_events": events,
        }

    except Exception as exc:
        logger.warning("Planner failed (%s) — using raw topic as query.", exc)
        events.append(_event("planner", "error", str(exc)))
        return {
            "arxiv_query": state.user_topic,
            "planning_reasoning": f"Planner error — using raw topic: {exc}",
            "node_events": events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEARCH NODE
# Calls MCP search_papers → gets up to 10 ArXiv candidates
# ─────────────────────────────────────────────────────────────────────────────

async def search_node(state: GraphState) -> dict:
    """
    Hit ArXiv API with the optimised query, return candidate papers.
    Returns: candidate_papers, node_events
    """
    events = [_event("search", "started", f"Searching ArXiv for: '{state.arxiv_query}'")]

    try:
        raw_papers = await search_papers(query=state.arxiv_query, max_results=10)

        candidates = [
            PaperMeta(
                arxiv_id=p["arxiv_id"],
                title=p["title"],
                authors=p["authors"],
                abstract=p["abstract"],
                pdf_url=p["pdf_url"],
                published=p["published"],
            )
            for p in raw_papers
        ]

        events.append(
            _event("search", "completed", f"Found {len(candidates)} candidate papers.")
        )
        return {
            "candidate_papers": candidates,
            "node_events": events,
        }

    except Exception as exc:
        logger.error("Search node failed: %s", exc)
        events.append(_event("search", "error", str(exc)))
        return {
            "candidate_papers": [],
            "error": f"ArXiv search failed: {exc}",
            "node_events": events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECTOR NODE
# Gemini picks the best 3 papers from the candidates
# ─────────────────────────────────────────────────────────────────────────────

_SELECTOR_SYSTEM = """You are a research paper curator.

You will receive a user's research topic and a list of candidate papers from ArXiv.
Your job is to select the 3 most relevant, high-quality, and recent papers.

Selection criteria (in order of priority):
1. Relevance to the user's topic
2. Recency (prefer newer papers)
3. Likely impact (prefer papers with clear contributions in the abstract)

Respond ONLY with a JSON object (no markdown fences):
{
  "selected_ids": ["<arxiv_id1>", "<arxiv_id2>", "<arxiv_id3>"],
  "reasoning": "<two sentences explaining your selection>"
}"""


async def selector_node(state: GraphState) -> dict:
    """
    Use Gemini to pick the best 3 papers from candidates.
    Returns: selected_papers, selection_reasoning, node_events
    """
    events = [_event("selector", "started", "Selecting best 3 papers from candidates.")]

    if not state.candidate_papers:
        events.append(_event("selector", "error", "No candidates to select from."))
        return {
            "selected_papers": [],
            "selection_reasoning": "No candidates available.",
            "error": "No papers found for this topic. Try a different search term.",
            "node_events": events,
        }

    # Build a readable paper list for Gemini
    paper_list = "\n\n".join(
        f"ID: {p.arxiv_id}\n"
        f"Title: {p.title}\n"
        f"Published: {p.published}\n"
        f"Authors: {p.authors}\n"
        f"Abstract: {p.abstract[:400]}..."
        for p in state.candidate_papers
    )

    prompt = f"User topic: {state.user_topic}\n\nCandidate papers:\n{paper_list}"

    try:
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            system_instruction=_SELECTOR_SYSTEM,
        )
        response = await model.generate_content_async(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 512},
        )

        parsed = _parse_json_response(response.text)
        selected_ids: list[str] = parsed.get("selected_ids", [])[:3]
        reasoning: str = parsed.get("reasoning", "")

        # Map IDs back to full PaperMeta objects
        id_to_paper = {p.arxiv_id: p for p in state.candidate_papers}
        selected = [id_to_paper[sid] for sid in selected_ids if sid in id_to_paper]

        # Fallback: if Gemini returned bad IDs, just take the top 3
        if not selected:
            selected = state.candidate_papers[:3]
            reasoning = "Fallback to top 3 by recency."

        events.append(
            _event(
                "selector",
                "completed",
                f"Selected: {', '.join(p.title[:40] for p in selected)}",
            )
        )
        return {
            "selected_papers": selected,
            "selection_reasoning": reasoning,
            "node_events": events,
        }

    except Exception as exc:
        logger.warning("Selector failed (%s) — falling back to top 3.", exc)
        fallback = state.candidate_papers[:3]
        events.append(_event("selector", "error", f"{exc} — using top 3 fallback."))
        return {
            "selected_papers": fallback,
            "selection_reasoning": f"Selector error — top 3 by recency: {exc}",
            "node_events": events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. INGESTOR NODE
# Calls MCP fetch_and_ingest for each selected paper concurrently
# ─────────────────────────────────────────────────────────────────────────────

async def ingestor_node(state: GraphState) -> dict:
    """
    Download, chunk, embed, and store all selected papers concurrently.
    Returns: ingested_papers, ingestion_complete, node_events
    """
    events = [
        _event(
            "ingestor",
            "started",
            f"Ingesting {len(state.selected_papers)} papers into knowledge base.",
        )
    ]

    if not state.selected_papers:
        events.append(_event("ingestor", "error", "No papers to ingest."))
        return {
            "ingested_papers": [],
            "ingestion_complete": False,
            "error": "No papers selected for ingestion.",
            "node_events": events,
        }

    # Ingest all papers concurrently
    tasks = [
        fetch_and_ingest(
            paper={
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors,
                "abstract": p.abstract,
                "pdf_url": p.pdf_url,
                "published": p.published,
            },
            session_id=state.session_id,
        )
        for p in state.selected_papers
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    ingested: list[IngestedPaper] = []
    for paper, result in zip(state.selected_papers, results):
        if isinstance(result, Exception):
            logger.error("Ingest failed for %s: %s", paper.arxiv_id, result)
            ingested.append(
                IngestedPaper(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    published=paper.published,
                    chunks_stored=0,
                    status="error",
                    error=str(result),
                )
            )
        else:
            ingested.append(
                IngestedPaper(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    published=paper.published,
                    chunks_stored=result.get("chunks_stored", 0),
                    status=result.get("status", "error"),
                    error=result.get("error"),
                )
            )
            events.append(
                _event(
                    "ingestor",
                    "progress",
                    f"✓ '{paper.title[:50]}' — {result.get('chunks_stored', 0)} chunks stored.",
                )
            )

    successful = [p for p in ingested if p.status == "success"]
    events.append(
        _event(
            "ingestor",
            "completed",
            f"{len(successful)}/{len(ingested)} papers ingested successfully.",
        )
    )

    return {
        "ingested_papers": ingested,
        "ingestion_complete": len(successful) > 0,
        "node_events": events,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. QA NODE
# Retrieves relevant chunks for user_question and generates a grounded answer
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM = """You are DocMind, a precise research paper assistant.

Answer the user's question using ONLY the provided context chunks from the ingested papers.
- Cite your sources inline using [source: <paper title>].
- If the context does not contain enough information, say so clearly.
- Be technically accurate and well-structured.
- Do not hallucinate facts not present in the context.

After your answer, on a new line output a JSON object (no markdown fences):
{"sources": ["<title1>", "<title2>", ...]}"""


async def qa_node(state: GraphState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB (filtered by session_id)
    and generate a grounded answer using Gemini.
    Returns: retrieved_chunks, answer, answer_sources, conversation_history, node_events
    """
    from app.src.core.embedding import embed_text
    from app.src.core.vector_store import get_collection

    question = state.user_question
    events = [_event("qa", "started", f"Answering: '{question}'")]

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        embedding = await embed_text(question)
        collection = get_collection()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=5,
            where={"session_id": state.session_id},
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    arxiv_id=meta.get("arxiv_id", ""),
                    chunk_id=meta.get("chunk_id", ""),
                    distance=float(dist),
                )
            )

        events.append(_event("qa", "retrieved", f"Retrieved {len(chunks)} relevant chunks."))

    except Exception as exc:
        logger.error("QA retrieval failed: %s", exc)
        events.append(_event("qa", "error", str(exc)))
        return {
            "retrieved_chunks": [],
            "answer": "Failed to retrieve context from the knowledge base.",
            "answer_sources": [],
            "error": f"QA retrieval failed: {exc}",
            "node_events": events,
        }

    # ── Generate answer ───────────────────────────────────────────────────────
    if not chunks:
        answer = "I couldn't find relevant information in the ingested papers for your question."
        events.append(_event("qa", "completed", "No relevant chunks found."))
        return {
            "retrieved_chunks": [],
            "answer": answer,
            "answer_sources": [],
            "conversation_history": [
                ConversationTurn(role="user", content=question),
                ConversationTurn(role="assistant", content=answer, sources=[]),
            ],
            "node_events": events,
        }

    # Build context block
    context_str = "\n\n---\n\n".join(
        f"[Chunk {i} | source: {c.source}]\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )

    # Include recent conversation history for multi-turn context
    history_str = ""
    if state.conversation_history:
        recent = state.conversation_history[-6:]  # last 3 turns
        history_str = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in recent
        )
        history_str = f"\nConversation so far:\n{history_str}\n"

    prompt = (
        f"{history_str}"
        f"Question: {question}\n\n"
        f"Context from papers:\n{context_str}"
    )

    try:
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            system_instruction=_QA_SYSTEM,
        )
        response = await model.generate_content_async(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 1024},
        )

        full_text = response.text.strip()

        # Split answer body from trailing sources JSON
        answer_body = full_text
        sources: list[str] = []
        if '{"sources":' in full_text:
            parts = full_text.rsplit("\n", 1)
            if len(parts) == 2:
                try:
                    sources_data = json.loads(parts[1].strip())
                    sources = sources_data.get("sources", [])
                    answer_body = parts[0].strip()
                except json.JSONDecodeError:
                    pass

        events.append(
            _event("qa", "completed", f"Answer generated ({len(answer_body)} chars).")
        )
        return {
            "retrieved_chunks": chunks,
            "answer": answer_body,
            "answer_sources": sources,
            "conversation_history": [
                ConversationTurn(role="user", content=question),
                ConversationTurn(role="assistant", content=answer_body, sources=sources),
            ],
            "node_events": events,
        }

    except Exception as exc:
        logger.error("QA synthesis failed: %s", exc)
        events.append(_event("qa", "error", str(exc)))
        return {
            "retrieved_chunks": chunks,
            "answer": f"Error generating answer: {exc}",
            "answer_sources": [],
            "error": f"QA synthesis failed: {exc}",
            "node_events": events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. VALIDATOR NODE
# Checks the answer is grounded in retrieved chunks; retries QA if not
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR_SYSTEM = """You are a grounding validator for a research paper RAG system.

Given an answer and the source chunks used to produce it, determine whether
the answer is fully grounded in those chunks (no hallucinated facts).

Respond ONLY with a JSON object (no markdown fences):
{
  "is_grounded": true | false,
  "reasoning": "<one-sentence explanation>"
}"""


async def validator_node(state: GraphState) -> dict:
    """
    Validate that the answer is grounded in retrieved chunks.
    Signals graph to retry QA if not grounded and retries remain.
    Returns: is_grounded, validation_reasoning, retry_count, node_events
    """
    events = [_event("validator", "started", "Checking answer grounding.")]

    if not state.answer or not state.retrieved_chunks:
        events.append(_event("validator", "completed", "Nothing to validate."))
        return {
            "is_grounded": True,
            "validation_reasoning": "No answer or chunks to validate.",
            "node_events": events,
        }

    context_str = "\n\n---\n\n".join(
        f"[source: {c.source}]\n{c.text}" for c in state.retrieved_chunks
    )
    prompt = f"Answer:\n{state.answer}\n\nSource chunks:\n{context_str}"

    try:
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            system_instruction=_VALIDATOR_SYSTEM,
        )
        response = await model.generate_content_async(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 256},
        )

        parsed = _parse_json_response(response.text)
        is_grounded: bool = parsed.get("is_grounded", True)
        reasoning: str = parsed.get("reasoning", "")

        new_retry_count = state.retry_count
        if not is_grounded and state.retry_count < state.max_retries:
            new_retry_count = state.retry_count + 1
            events.append(
                _event(
                    "validator",
                    "retry",
                    f"Not grounded — retry {new_retry_count}/{state.max_retries}.",
                )
            )
        else:
            events.append(
                _event(
                    "validator",
                    "completed",
                    "Grounded ✓" if is_grounded else "Max retries reached — accepting answer.",
                )
            )

        return {
            "is_grounded": is_grounded,
            "validation_reasoning": reasoning,
            "retry_count": new_retry_count,
            "node_events": events,
        }

    except Exception as exc:
        logger.warning("Validator failed (%s) — accepting answer as-is.", exc)
        events.append(_event("validator", "error", str(exc)))
        return {
            "is_grounded": True,
            "validation_reasoning": f"Validator error — accepting: {exc}",
            "node_events": events,
        }