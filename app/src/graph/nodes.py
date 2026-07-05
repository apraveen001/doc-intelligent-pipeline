"""
DocMind Research Paper Assistant — LangGraph Nodes
---------------------------------------------------
Six nodes using Vertex AI via Application Default Credentials (ADC).
No API key needed — authenticated via gcloud auth application-default login.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.src.graph.state import (
    ConversationTurn,
    GraphState,
    IngestedPaper,
    PaperMeta,
    RetrievedChunk,
)
from app.src.mcp.tools import fetch_and_ingest, search_papers

load_dotenv()
logger = logging.getLogger(__name__)

client = genai.Client(
    vertexai=True,
    project=os.getenv("GCP_PROJECT"),
    location=os.getenv("GCP_LOCATION", "us-central1"),
)
_GEMINI_MODEL = "gemini-2.0-flash"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(node: str, status: str, detail: str = "") -> dict:
    return {"node": node, "status": status, "detail": detail, "timestamp": _now_iso()}


def _parse_json_response(raw: str) -> dict:
    """Parse JSON from model output robustly."""
    import re
    raw = raw.strip()
    # Extract content between markdown fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    # Find the first { and last } to extract just the JSON object
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    # Normalize Python bool literals to JSON
    raw = raw.replace("True", "true").replace("False", "false").replace("None", "null")
    return json.loads(raw.strip())


async def _generate(system: str, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    response = await client.aio.models.generate_content(
        model=_GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. PLANNER NODE
# ─────────────────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are an expert at constructing ArXiv search queries.
Rewrite the user's research topic into a precise, effective ArXiv search query.
Rules:
- Use specific technical terminology
- Keep it concise — 3-6 key terms
- Do NOT use boolean operators

Respond ONLY with a JSON object (no markdown fences):
{
  "arxiv_query": "<your optimised search query>",
  "reasoning": "<one sentence explaining your choices>"
}"""


async def planner_node(state: GraphState) -> dict:
    events = [_event("planner", "started", f"Optimising query for: '{state.user_topic}'")]
    try:
        raw     = await _generate(_PLANNER_SYSTEM, state.user_topic, max_tokens=256)
        parsed  = _parse_json_response(raw)
        arxiv_query = parsed.get("arxiv_query", state.user_topic).strip()
        reasoning   = parsed.get("reasoning", "")
        events.append(_event("planner", "completed", f"ArXiv query: '{arxiv_query}'"))
        return {"arxiv_query": arxiv_query, "planning_reasoning": reasoning, "node_events": events}
    except Exception as exc:
        logger.warning("Planner failed (%s) — using raw topic.", exc)
        events.append(_event("planner", "error", str(exc)))
        return {"arxiv_query": state.user_topic, "planning_reasoning": str(exc), "node_events": events}


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEARCH NODE
# ─────────────────────────────────────────────────────────────────────────────

async def search_node(state: GraphState) -> dict:
    events = [_event("search", "started", f"Searching ArXiv for: '{state.arxiv_query}'")]
    try:
        raw_papers = await search_papers(query=state.arxiv_query, max_results=10)
        candidates = [PaperMeta(**p) for p in raw_papers]
        events.append(_event("search", "completed", f"Found {len(candidates)} candidates."))
        return {"candidate_papers": candidates, "node_events": events}
    except Exception as exc:
        logger.error("Search node failed: %s", exc)
        events.append(_event("search", "error", str(exc)))
        return {"candidate_papers": [], "error": f"ArXiv search failed: {exc}", "node_events": events}


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECTOR NODE
# ─────────────────────────────────────────────────────────────────────────────

_SELECTOR_SYSTEM = """You are a research paper curator.
Select the 3 most relevant, high-quality, and recent papers from the list.

Criteria (in order):
1. Relevance to the user's topic
2. Recency
3. Likely impact based on abstract

Respond ONLY with a JSON object (no markdown fences):
{
  "selected_ids": ["<arxiv_id1>", "<arxiv_id2>", "<arxiv_id3>"],
  "reasoning": "<two sentences explaining your selection>"
}"""


async def selector_node(state: GraphState) -> dict:
    events = [_event("selector", "started", "Selecting best 3 papers.")]
    if not state.candidate_papers:
        events.append(_event("selector", "error", "No candidates."))
        return {"selected_papers": [], "error": "No papers found.", "node_events": events}

    paper_list = "\n\n".join(
        f"ID: {p.arxiv_id}\nTitle: {p.title}\nPublished: {p.published}\nAbstract: {p.abstract[:400]}..."
        for p in state.candidate_papers
    )
    prompt = f"User topic: {state.user_topic}\n\nCandidates:\n{paper_list}"

    try:
        raw    = await _generate(_SELECTOR_SYSTEM, prompt, max_tokens=512, temperature=0.1)
        parsed = _parse_json_response(raw)
        selected_ids = parsed.get("selected_ids", [])[:3]
        reasoning    = parsed.get("reasoning", "")

        id_to_paper = {p.arxiv_id: p for p in state.candidate_papers}
        selected    = [id_to_paper[sid] for sid in selected_ids if sid in id_to_paper]
        if not selected:
            selected  = state.candidate_papers[:3]
            reasoning = "Fallback to top 3 by recency."

        events.append(_event("selector", "completed", f"Selected {len(selected)} papers."))
        return {"selected_papers": selected, "selection_reasoning": reasoning, "node_events": events}

    except Exception as exc:
        logger.warning("Selector failed (%s) — top 3 fallback.", exc)
        events.append(_event("selector", "error", str(exc)))
        return {"selected_papers": state.candidate_papers[:3], "selection_reasoning": str(exc), "node_events": events}


# ─────────────────────────────────────────────────────────────────────────────
# 4. INGESTOR NODE
# ─────────────────────────────────────────────────────────────────────────────

async def ingestor_node(state: GraphState) -> dict:
    events = [_event("ingestor", "started", f"Ingesting {len(state.selected_papers)} papers.")]
    if not state.selected_papers:
        events.append(_event("ingestor", "error", "No papers to ingest."))
        return {"ingested_papers": [], "ingestion_complete": False, "error": "No papers selected.", "node_events": events}

    tasks = [
        fetch_and_ingest(
            paper={
                "arxiv_id": p.arxiv_id, "title": p.title, "authors": p.authors,
                "abstract": p.abstract, "pdf_url": p.pdf_url, "published": p.published,
            },
            session_id=state.session_id,
        )
        for p in state.selected_papers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ingested: list[IngestedPaper] = []
    for paper, result in zip(state.selected_papers, results):
        if isinstance(result, Exception):
            ingested.append(IngestedPaper(
                arxiv_id=paper.arxiv_id, title=paper.title, authors=paper.authors,
                abstract=paper.abstract, published=paper.published,
                chunks_stored=0, status="error", error=str(result),
            ))
        else:
            ingested.append(IngestedPaper(
                arxiv_id=paper.arxiv_id, title=paper.title, authors=paper.authors,
                abstract=paper.abstract, published=paper.published,
                chunks_stored=result.get("chunks_stored", 0),
                status=result.get("status", "error"),
                error=result.get("error"),
            ))
            events.append(_event("ingestor", "progress",
                f"✓ '{paper.title[:50]}' — {result.get('chunks_stored', 0)} chunks stored."))

    successful = [p for p in ingested if p.status == "success"]
    events.append(_event("ingestor", "completed", f"{len(successful)}/{len(ingested)} papers ingested."))
    return {"ingested_papers": ingested, "ingestion_complete": len(successful) > 0, "node_events": events}


# ─────────────────────────────────────────────────────────────────────────────
# 5. QA NODE
# ─────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM = """You are DocMind, an intelligent research paper assistant with a focused and scholarly persona.

Your sole purpose is to help users understand the research papers that have been ingested into the current session. You have deep expertise in reading, interpreting, and explaining academic papers clearly.

STRICT SCOPE RULES — you must follow these without exception:
1. Answer ONLY from the provided context chunks from the ingested papers.
2. NEVER use outside knowledge, general facts, or internet information.
3. If a question is outside the scope of the ingested papers (e.g. current events, general knowledge, personal advice, anything not in the context), respond with exactly:
   "That question is outside the scope of the ingested research papers. I can only answer questions based on the papers in this session. Please ask something related to the research content."
4. NEVER mention real-world facts, news, or people not referenced in the papers.
5. NEVER search the web or use knowledge beyond the provided chunks.

RESPONSE STYLE:
- Be precise, clear, and academically grounded.
- Cite sources inline using [source: <paper title>] after every claim.
- Structure longer answers with short paragraphs — no walls of text.
- If the context partially answers the question, say what you found and clearly state what is missing.
- Keep answers focused and avoid padding.

SCOPE CHECK — before answering, ask yourself:
"Is the answer to this question contained in the provided context chunks?"
- YES → answer using only those chunks with citations.
- NO → return the out-of-scope response above. Do not attempt to answer.

After your answer, on a new line output (no markdown fences):
{"sources": ["<title1>", "<title2>", ...]}"""


async def qa_node(state: GraphState) -> dict:
    from app.src.core.embedding import embed_text
    from app.src.core.vector_store import get_collection

    question = state.user_question
    events   = [_event("qa", "started", f"Answering: '{question}'")]

    try:
        embedding  = await embed_text(question)
        collection = get_collection()
        results    = collection.query(
            query_embeddings=[embedding],
            n_results=5,
            where={"session_id": state.session_id},
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results.get("documents", [[]])[0],
            results.get("metadatas",  [[]])[0],
            results.get("distances",  [[]])[0],
        ):
            chunks.append(RetrievedChunk(
                text=doc, source=meta.get("source", "unknown"),
                arxiv_id=meta.get("arxiv_id", ""),
                chunk_id=meta.get("chunk_id", ""),
                distance=float(dist),
            ))
        events.append(_event("qa", "retrieved", f"Retrieved {len(chunks)} chunks."))

    except Exception as exc:
        logger.error("QA retrieval failed: %s", exc)
        events.append(_event("qa", "error", str(exc)))
        return {"retrieved_chunks": [], "answer": "Retrieval failed.", "answer_sources": [],
                "error": str(exc), "node_events": events}

    if not chunks:
        answer = "I couldn't find relevant information in the ingested papers."
        events.append(_event("qa", "completed", "No chunks found."))
        return {
            "retrieved_chunks": [], "answer": answer, "answer_sources": [],
            "conversation_history": [
                ConversationTurn(role="user", content=question),
                ConversationTurn(role="assistant", content=answer),
            ],
            "node_events": events,
        }

    context_str = "\n\n---\n\n".join(f"[Chunk {i} | source: {c.source}]\n{c.text}" for i, c in enumerate(chunks, 1))
    history_str = ""
    if state.conversation_history:
        recent      = state.conversation_history[-6:]
        history_str = "\n".join(f"{t.role.upper()}: {t.content}" for t in recent) + "\n\n"

    prompt = f"{history_str}Question: {question}\n\nContext:\n{context_str}"

    try:
        full_text   = await _generate(_QA_SYSTEM, prompt, max_tokens=1024, temperature=0.3)
        answer_body = full_text
        sources: list[str] = []

        if '{"sources":' in full_text:
            parts = full_text.rsplit("\n", 1)
            if len(parts) == 2:
                try:
                    sources     = json.loads(parts[1].strip()).get("sources", [])
                    answer_body = parts[0].strip()
                except json.JSONDecodeError:
                    pass

        events.append(_event("qa", "completed", f"Answer ready ({len(answer_body)} chars)."))
        return {
            "retrieved_chunks": chunks, "answer": answer_body, "answer_sources": sources,
            "conversation_history": [
                ConversationTurn(role="user", content=question),
                ConversationTurn(role="assistant", content=answer_body, sources=sources),
            ],
            "node_events": events,
        }

    except Exception as exc:
        logger.error("QA synthesis failed: %s", exc)
        events.append(_event("qa", "error", str(exc)))
        return {"retrieved_chunks": chunks, "answer": str(exc), "answer_sources": [],
                "error": str(exc), "node_events": events}


# ─────────────────────────────────────────────────────────────────────────────
# 6. VALIDATOR NODE
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR_SYSTEM = """You are a grounding validator for a RAG system.
Check if the answer is fully grounded in the provided source chunks.

Respond ONLY with a JSON object (no markdown fences):
{
  "is_grounded": true | false,
  "reasoning": "<one-sentence explanation>"
}"""


async def validator_node(state: GraphState) -> dict:
    events = [_event("validator", "started", "Checking grounding.")]
    if not state.answer or not state.retrieved_chunks:
        events.append(_event("validator", "completed", "Nothing to validate."))
        return {"is_grounded": True, "validation_reasoning": "Nothing to validate.", "node_events": events}

    context_str = "\n\n---\n\n".join(f"[source: {c.source}]\n{c.text}" for c in state.retrieved_chunks)
    prompt      = f"Answer:\n{state.answer}\n\nSource chunks:\n{context_str}"

    try:
        raw    = await _generate(_VALIDATOR_SYSTEM, prompt, max_tokens=256, temperature=0.1)
        parsed = _parse_json_response(raw)
        is_grounded = parsed.get("is_grounded", True)
        reasoning   = parsed.get("reasoning", "")

        new_retry_count = state.retry_count
        if not is_grounded and state.retry_count < state.max_retries:
            new_retry_count += 1
            events.append(_event("validator", "retry", f"Retry {new_retry_count}/{state.max_retries}."))
        else:
            events.append(_event("validator", "completed", "Grounded ✓" if is_grounded else "Max retries reached."))

        return {"is_grounded": is_grounded, "validation_reasoning": reasoning,
                "retry_count": new_retry_count, "node_events": events}

    except Exception as exc:
        logger.warning("Validator failed (%s) — accepting.", exc)
        events.append(_event("validator", "error", str(exc)))
        return {"is_grounded": True, "validation_reasoning": str(exc), "node_events": events}