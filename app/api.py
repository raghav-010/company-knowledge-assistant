"""
api.py — FastAPI backend for Company Knowledge Assistant
Raghav Balaji V | github.com/raghav-010

Endpoints:
  GET  /              → serve the frontend UI
  POST /ingest        → kick off document ingestion (background task)
  GET  /ingest/status → poll ingest progress
  POST /ask           → ask a question, get a grounded answer

Design notes:
  - asyncio.Lock() prevents two ingest jobs from running simultaneously.
  - /ingest runs as a background asyncio.Task so the HTTP response
    returns immediately (ingest can take 30–60 seconds for large corpora).
  - category=None in /ask searches ALL document categories.
    Set to a specific category (e.g. "policies") to scope retrieval.
"""
from __future__ import annotations
import asyncio, time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .rag import answer_with_docs_async
from .ingest import run_ingest_async

app = FastAPI(title="Company Knowledge Assistant")

# Serve the frontend from app/static/
static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Ingest state ──────────────────────────────────────────────────────────────
_ingest_lock = asyncio.Lock()
_ingest_task: asyncio.Task | None = None
_ingest_last = {
    "status": "idle",       # idle | running | succeeded | failed
    "started_at": None,
    "finished_at": None,
    "stats": None,          # {"documents": N, "chunks": N}
    "error": None,
}


class Ask(BaseModel):
    question: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root_page():
    return FileResponse(static_dir / "index.html")


async def _ingest_job():
    _ingest_last.update({
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "stats": None,
        "error": None,
    })
    try:
        stats = await run_ingest_async()
        _ingest_last.update({
            "status": "succeeded",
            "finished_at": time.time(),
            "stats": stats,
        })
    except Exception as e:
        _ingest_last.update({
            "status": "failed",
            "finished_at": time.time(),
            "error": str(e),
        })


@app.post("/ingest")
async def kick_off_ingest():
    global _ingest_task
    async with _ingest_lock:
        if _ingest_task and not _ingest_task.done():
            return JSONResponse(
                {"ok": False, "message": "Ingestion already running"},
                status_code=409,
            )
        _ingest_task = asyncio.create_task(_ingest_job())
    return {"ok": True, "message": "Ingestion started"}


@app.get("/ingest/status")
async def ingest_status():
    return {"ok": True, **_ingest_last}


@app.post("/ask")
async def ask(q: Ask):
    start = time.perf_counter()

    # category=None → search across ALL document categories
    # Change to e.g. "policies" to scope to the policies/ folder only
    category = None

    answer, sources, contexts = await answer_with_docs_async(q.question, category)

    elapsed = time.perf_counter() - start
    print(f"⏱️  /ask took {elapsed:.2f}s")

    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
    }
