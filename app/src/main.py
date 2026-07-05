"""
DocMind Research Paper Assistant — FastAPI Backend
--------------------------------------------------
Endpoints:

  GET  /                        — serve frontend
  GET  /health                  — health check

  POST /session/start           — create session, return session_id
  DELETE /session/{session_id}  — clear session chunks from ChromaDB

  GET  /research/stream         — SSE: run research phase (topic → papers)
  GET  /qa/stream               — SSE: run one QA turn (question → answer)

  GET  /session/{session_id}/papers   — list ingested papers for a session
  GET  /session/{session_id}/export   — export conversation as PDF
"""

import asyncio
import json
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

# Resolve static directory relative to this file, not the cwd
STATIC_DIR = Path(__file__).parent / "static"

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.src.graph.graph import (
    build_qa_graph,
    build_research_graph,
    run_qa_phase,
    stream_qa_phase,
    stream_research_phase,
)
from app.src.graph.state import GraphState
from app.src.mcp.tools import clear_session, generate_session_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle — build graphs once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building LangGraph graphs...")
    app.state.research_graph = build_research_graph()
    app.state.qa_graph       = build_qa_graph()

    # In-memory session store: session_id → GraphState
    # Holds live state between QA turns for the same session
    app.state.sessions: dict[str, GraphState] = {}

    logger.info("DocMind ready.")
    yield
    logger.info("DocMind shutting down.")


app = FastAPI(title="DocMind Research Paper Assistant", lifespan=lifespan)
# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────────

# @app.get("/")
# async def serve_frontend():
#     return FileResponse(str(STATIC_DIR / "index.html"))


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(app.state.sessions)}


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/session/start")
async def start_session():
    """
    Create a new research session.
    Returns a session_id the frontend must include in all subsequent requests.
    """
    session_id = generate_session_id()
    app.state.sessions[session_id] = GraphState(
        session_id=session_id,
        max_retries=2,
    )
    logger.info("Session started: %s", session_id)
    return JSONResponse(content={"session_id": session_id})


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    Clear all ChromaDB chunks for this session and remove it from memory.
    Called automatically when the user downloads the PDF.
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    result = await clear_session(session_id)
    app.state.sessions.pop(session_id, None)

    logger.info("Session ended: %s — %d chunks deleted.", session_id, result.get("chunks_deleted", 0))
    return JSONResponse(content={
        "message": "Session cleared.",
        "session_id": session_id,
        "chunks_deleted": result.get("chunks_deleted", 0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Research stream
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/research/stream")
async def research_stream(
    session_id: str = Query(..., description="Session ID from /session/start"),
    topic: str      = Query(..., description="Research topic e.g. 'LLM reasoning'"),
):
    """
    SSE stream: run the research phase for a topic.

    Events streamed to the frontend:
        {"type": "node_event", "node": str, "status": str, "detail": str, "timestamp": str}
        {"type": "done", "papers": list[dict], "error": str | None}

    Frontend connects like:
        const es = new EventSource(`/research/stream?session_id=...&topic=...`);
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/start first.")

    async def event_generator():
        try:
            final_state = await run_research_phase(
                app.state.research_graph,
                session_id=session_id,
                user_topic=topic,
            )

            # Stream node events from the final state
            for event in final_state.node_events:
                yield _sse({"type": "node_event", **event})
                await asyncio.sleep(0)

            # Save the full final state so /papers endpoint works
            app.state.sessions[session_id] = final_state

            # Emit done event with paper list
            yield _sse({
                "type": "done",
                "papers": [
                    {
                        "arxiv_id":      p.arxiv_id,
                        "title":         p.title,
                        "authors":       p.authors,
                        "abstract":      p.abstract,
                        "published":     p.published,
                        "chunks_stored": p.chunks_stored,
                        "status":        p.status,
                    }
                    for p in final_state.ingested_papers
                ],
                "error": final_state.error,
            })

        except Exception as exc:
            logger.error("Research stream error: %s", exc)
            traceback.print_exc()
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — QA stream
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/qa/stream")
async def qa_stream(
    session_id: str = Query(..., description="Session ID"),
    question: str   = Query(..., description="User question about the papers"),
):
    """
    SSE stream: run one QA turn against the ingested papers.

    Events streamed:
        {"type": "node_event", "node": str, "status": str, "detail": str, "timestamp": str}
        {"type": "done", "answer": str, "sources": list[str],
         "is_grounded": bool, "error": str | None}

    The session state is updated after each turn so conversation_history
    carries forward into the next question.
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    current_state = app.state.sessions[session_id]

    async def event_generator():
        nonlocal current_state
        try:
            async for event in stream_qa_phase(
                app.state.qa_graph,
                current_state=current_state,
                user_question=question,
            ):
                yield _sse(event)
                await asyncio.sleep(0)

            # After streaming, run once more to get the updated state
            # (stream_qa_phase streams events but doesn't return final state)
            updated = await run_qa_phase(
                app.state.qa_graph,
                current_state=current_state,
                user_question=question,
            )
            # Persist updated state (includes new conversation_history turn)
            app.state.sessions[session_id] = updated

        except Exception as exc:
            logger.error("QA stream error: %s", exc)
            traceback.print_exc()
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/session/{session_id}/papers")
async def list_session_papers(session_id: str):
    """
    Return the list of papers ingested for this session.
    Used by the frontend to render the paper cards.
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = app.state.sessions[session_id]
    papers = [
        {
            "arxiv_id":     p.arxiv_id,
            "title":        p.title,
            "authors":      p.authors,
            "abstract":     p.abstract,
            "published":    p.published,
            "chunks_stored": p.chunks_stored,
            "status":       p.status,
        }
        for p in state.ingested_papers
    ]
    return JSONResponse(content={"session_id": session_id, "papers": papers})


# ─────────────────────────────────────────────────────────────────────────────
# PDF export
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/session/{session_id}/export")
async def export_conversation(session_id: str):
    """
    Export the full conversation as a PDF and end the session.

    Flow:
        1. Build PDF from conversation_history
        2. Stream PDF to client
        3. Clear session from ChromaDB and memory

    After this endpoint is called, the session is gone.
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = app.state.sessions[session_id]

    if not state.conversation_history:
        raise HTTPException(status_code=400, detail="No conversation to export yet.")

    try:
        pdf_bytes = _build_pdf(state)
    except Exception as exc:
        logger.error("PDF build failed: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}")

    # Clear session after building PDF
    await clear_session(session_id)
    app.state.sessions.pop(session_id, None)
    logger.info("Session %s exported and cleared.", session_id)

    topic_slug = state.user_topic.replace(" ", "_")[:40] if state.user_topic else "research"
    filename   = f"docmind_{topic_slug}.pdf"

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _sanitize(text: str) -> str:
    """Replace non-latin-1 characters for fpdf2 compatibility."""
    return (text
        .replace("\u2014", "-")   # em dash
        .replace("\u2013", "-")   # en dash
        .replace("\u2018", "'")   # left single quote
        .replace("\u2019", "'")   # right single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u2022", "*")   # bullet
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def _build_pdf(state: GraphState) -> bytes:
    """Build a PDF from the conversation history using fpdf2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    topic = _sanitize(state.user_topic or "Research Session")
    pdf.cell(0, 10, f"DocMind - {topic}", ln=True)
    pdf.ln(4)

    # Papers section
    if state.ingested_papers:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Papers Analysed", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for p in state.ingested_papers:
            if p.status == "success":
                pdf.multi_cell(0, 6, _sanitize(f"* {p.title} ({p.published})"))
        pdf.ln(6)

    # Conversation
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Conversation", ln=True)
    pdf.ln(2)

    for turn in state.conversation_history:
        if turn.role == "user":
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, _sanitize(f"You: {turn.content}"))
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, _sanitize(f"DocMind: {turn.content}"))
            if turn.sources:
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 5, _sanitize("Sources: " + ", ".join(turn.sources)))
        pdf.ln(3)

    return bytes(pdf.output())