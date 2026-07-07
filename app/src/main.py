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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.src.graph.graph import (
    build_qa_graph,
    build_research_graph,
    run_qa_phase,
    run_research_phase,
    stream_qa_phase,
    stream_research_phase,
)
from app.src.graph.state import GraphState
from app.src.mcp.tools import clear_session, generate_session_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building LangGraph graphs...")
    app.state.research_graph = build_research_graph()
    app.state.qa_graph       = build_qa_graph()
    app.state.sessions: dict[str, GraphState] = {}
    logger.info("DocMind ready.")
    yield
    logger.info("DocMind shutting down.")


#app = FastAPI(title="DocMind Research Paper Assistant", lifespan=lifespan)
app = FastAPI(
    title="DocMind Research Paper Assistant",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://project-73517f81-3740-4c-77491.web.app",
    "https://project-73517f81-3740-4c-77491.firebaseapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(app.state.sessions)}


@app.post("/session/start")
async def start_session():
    session_id = generate_session_id()
    app.state.sessions[session_id] = GraphState(
        session_id=session_id,
        max_retries=2,
    )
    logger.info("Session started: %s", session_id)
    return JSONResponse(content={"session_id": session_id})


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
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


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/research/stream")
async def research_stream(
    session_id: str = Query(..., description="Session ID from /session/start"),
    topic: str      = Query(..., description="Research topic e.g. 'LLM reasoning'"),
):
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/start first.")

    async def event_generator():
        try:
            initial = GraphState(
                session_id=session_id,
                user_topic=topic,
                max_retries=2,
            )

            seen: set[int] = set()
            final_state = None

            # Stream node events progressively as each node completes
            async for chunk in app.state.research_graph.astream(
                initial, stream_mode="values"
            ):
                state = GraphState(**chunk) if isinstance(chunk, dict) else chunk
                final_state = state

                # Yield any new events added since last chunk
                for idx, event in enumerate(state.node_events):
                    if idx not in seen:
                        seen.add(idx)
                        yield _sse({"type": "node_event", **event})
                        await asyncio.sleep(0)

            # Save final state
            if final_state:
                app.state.sessions[session_id] = final_state
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
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/qa/stream")
async def qa_stream(
    session_id: str = Query(..., description="Session ID"),
    question: str   = Query(..., description="User question about the papers"),
):
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    current_state = app.state.sessions[session_id]

    async def event_generator():
        try:
            # Build the turn state — reset per-question fields
            turn_state = current_state.model_copy(update={
                "user_question": question,
                "retrieved_chunks": [],
                "answer": "",
                "answer_sources": [],
                "is_grounded": False,
                "validation_reasoning": "",
                "retry_count": 0,
                "error": None,
            })

            seen: set[int] = set()
            base_count = len(current_state.node_events)
            final_state = None

            # Stream node events progressively as each node completes
            async for chunk in app.state.qa_graph.astream(
                turn_state, stream_mode="values"
            ):
                state = GraphState(**chunk) if isinstance(chunk, dict) else chunk
                final_state = state

                for idx, event in enumerate(state.node_events):
                    if idx >= base_count and idx not in seen:
                        seen.add(idx)
                        yield _sse({"type": "node_event", **event})
                        await asyncio.sleep(0)

            # Persist final state so conversation_history accumulates
            if final_state:
                app.state.sessions[session_id] = final_state
                yield _sse({
                    "type":        "done",
                    "answer":      final_state.answer,
                    "sources":     final_state.answer_sources,
                    "is_grounded": final_state.is_grounded,
                    "error":       final_state.error,
                })

        except Exception as exc:
            logger.error("QA stream error: %s", exc)
            traceback.print_exc()
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/session/{session_id}/papers")
async def list_session_papers(session_id: str):
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    state = app.state.sessions[session_id]
    papers = [
        {
            "arxiv_id":      p.arxiv_id,
            "title":         p.title,
            "authors":       p.authors,
            "abstract":      p.abstract,
            "published":     p.published,
            "chunks_stored": p.chunks_stored,
            "status":        p.status,
        }
        for p in state.ingested_papers
    ]
    return JSONResponse(content={"session_id": session_id, "papers": papers})


@app.get("/session/{session_id}/export")
async def export_conversation(session_id: str):
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
    return (text
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2022", "*")
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def _build_pdf(state: GraphState) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    W = pdf.w - 30  # usable width: page width minus left+right margins

    pdf.set_font("Helvetica", "B", 16)
    topic = _sanitize(state.user_topic or "Research Session")
    pdf.cell(W, 10, f"DocMind - {topic}", ln=True)
    pdf.ln(4)

    if state.ingested_papers:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(W, 8, "Papers Analysed", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for p in state.ingested_papers:
            if p.status == "success":
                pdf.multi_cell(W, 6, _sanitize(f"* {p.title} ({p.published})"))
        pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(W, 8, "Conversation", ln=True)
    pdf.ln(2)

    for turn in state.conversation_history:
        if turn.role == "user":
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(W, 6, _sanitize(f"You: {turn.content}"))
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(W, 6, _sanitize(f"DocMind: {turn.content}"))
            if turn.sources:
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(W, 5, _sanitize("Sources: " + ", ".join(turn.sources)))
        pdf.ln(3)

    return bytes(pdf.output())