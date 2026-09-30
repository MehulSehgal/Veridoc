"""
FastAPI backend for the HTML/CSS/JS "investigation desk" frontend. This is a
thin HTTP layer over the same agents/retrieval modules the CLI and the
Streamlit UI use -- no logic is duplicated here.

Run:
    uvicorn backend.server:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aiml.agents.react_loop import run_react_loop
from aiml.agents.retriever_agent import RetrieverAgent
from aiml.retrieval.image_store import ImageStore
from aiml.retrieval.vector_store import VectorStore

app = FastAPI(title="Veridoc API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent / "static"

# process-local "open case file" state -- fine for a single-user demo;
# swap for a session/user-keyed dict if this ever needs to serve more than
# one person at a time.
_loaded_index_dir: str | None = None
_store: VectorStore | None = None
_image_store: ImageStore | None = None


class OpenCaseRequest(BaseModel):
    index_dir: str = "data/index"


class InvestigateRequest(BaseModel):
    question: str
    strategy: str = "cot"


def _require_case_open() -> None:
    if _store is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No case file is open yet. Build an index first with "
                "`python -m aiml.ingestion.build_index --pdf_dir data/papers --out_dir data/index`, "
                "then open that folder from the sidebar."
            ),
        )


@app.post("/api/open-case")
def open_case(req: OpenCaseRequest):
    global _loaded_index_dir, _store, _image_store

    index_dir = Path(req.index_dir)
    if not index_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Index directory '{req.index_dir}' does not exist yet. Build it with:\n"
                f"python -m aiml.ingestion.build_index --pdf_dir data/papers --out_dir {req.index_dir}"
            ),
        )

    try:
        _store = VectorStore.load(index_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    _image_store = ImageStore.load(index_dir) if ImageStore.exists(index_dir) else None
    _loaded_index_dir = str(index_dir)

    return {
        "status": "ok",
        "index_dir": _loaded_index_dir,
        "n_chunks": len(_store.chunks),
        "n_figures": len(_image_store.records) if _image_store else 0,
    }


@app.get("/api/status")
def status():
    return {
        "loaded": _store is not None,
        "index_dir": _loaded_index_dir,
        "n_chunks": len(_store.chunks) if _store else 0,
        "n_figures": len(_image_store.records) if _image_store else 0,
    }


@app.post("/api/investigate")
def investigate(req: InvestigateRequest):
    _require_case_open()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retriever = RetrieverAgent(_store, image_store=_image_store)
    result = run_react_loop(req.question, retriever, strategy=req.strategy)

    return {
        "final_answer": result.final_answer,
        "confidence": result.final_confidence,
        "iterations": result.iterations_used,
        "trace": [{"label": s.label, "content": s.content} for s in result.trace],
        "figures": [
            {
                "label": f.label,
                "source": f.source,
                "score": f.score,
                "url": f"/api/figure?path={f.crop_path}",
            }
            for f in result.figures
        ],
    }


@app.get("/api/figure")
def get_figure(path: str):
    _require_case_open()
    full_path = Path(_loaded_index_dir) / path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure crop not found on disk: {full_path}")
    return FileResponse(full_path)


# Serve the frontend itself. Mounted last/at the root so the more specific
# /api/* routes above always match first.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
