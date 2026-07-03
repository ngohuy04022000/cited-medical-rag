"""FastAPI service exposing the Cited Medical RAG pipeline.

Run locally:
    uvicorn src.api:app --reload
Then open http://localhost:8000/docs for interactive Swagger UI.

Endpoints:
    GET  /        — single-page web UI (chat-style, no build step)
    GET  /health  — liveness + index/config info (no API key required)
    POST /query   — answer a question with citations, confidence, and latency
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import get_settings, setup_logging
from src.generator import GenerationError
from src.rag import RAGPipeline

setup_logging()
logger = logging.getLogger("cmrag.api")
settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Module-level singleton: the index is built once and reused per request."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(settings=settings)
    return _pipeline


def _api_key_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key)


def _generation_ready() -> bool:
    """The offline 'local' and on-device 'hf' backends need no key; only the
    Anthropic backend requires one."""
    return settings.backend in ("local", "hf") or _api_key_configured()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the retriever at startup so the first request isn't slow.
    get_pipeline()
    logger.info("Cited Medical RAG API ready (model=%s)", settings.model)
    yield


app = FastAPI(
    title="Cited Medical RAG API",
    version="1.0.0",
    description="Citation-aware retinal-disease knowledge assistant.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort safety net: any exception not already turned into an
    HTTPException (e.g. an IndexError from a malformed LLM response, or an
    AnthropicError raised outside generate_answer's own guards) is logged
    with a full traceback server-side and mapped to a clean, non-leaking
    JSON error instead of propagating a raw stack trace to the client.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    include_chunks: bool = Field(
        False, description="Include the raw retrieved chunks in the response"
    )


class RetrievedChunk(BaseModel):
    source: str
    section: Optional[str] = None
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float
    refused: bool
    latency_ms: float
    retrieved_chunks: Optional[List[RetrievedChunk]] = None


@app.get("/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    return {
        "status": "ok",
        "backend": settings.backend,
        "model": settings.model,
        "num_chunks": pipeline.num_chunks,
        "api_key_configured": _api_key_configured(),
        "generation_ready": _generation_ready(),
    }


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    if not _generation_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "ANTHROPIC_API_KEY is not configured. Set it, or start the server "
                "with RAG_BACKEND=local to run offline without a key."
            ),
        )

    try:
        result = pipeline.query(request.question)
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    chunks = result["retrieved_chunks"] if request.include_chunks else None
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        refused=result["refused"],
        latency_ms=result["latency_ms"],
        retrieved_chunks=chunks,
    )
