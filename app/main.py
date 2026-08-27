"""FastAPI application — triage & TAM endpoints with SSE streaming."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.config import Settings, get_settings
from app.llm_client import LLMClient, get_llm_client
from app.retrieval import HybridRetriever, build_runtime
from app.schemas import TriageInput, TriageOutput
from app.triage_agent import triage_ticket, triage_ticket_stream

logger = logging.getLogger(__name__)

# ── Module-level singletons, populated during startup ────────────────────
_retriever: HybridRetriever | None = None
_llm: LLMClient | None = None


def _get_retriever() -> HybridRetriever:
    if _retriever is None:
        raise RuntimeError("Retriever not initialised — server still starting?")
    return _retriever


def _get_llm() -> LLMClient:
    if _llm is None:
        raise RuntimeError("LLM client not initialised — server still starting?")
    return _llm


# ── Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _retriever, _llm
    settings = get_settings()
    logger.info("Loading corpus and building indexes …")
    _retriever = await build_runtime(settings)
    _llm = get_llm_client(settings)
    corpus = _retriever.corpus
    logger.info(
        "Ready — %d tickets, %d accounts, %d KB chunks, provider=%s",
        len(corpus.tickets),
        len(corpus.accounts),
        len(corpus.kb_chunks),
        settings.llm_provider,
    )
    yield
    # Cleanup
    if hasattr(_llm, "aclose"):
        await _llm.aclose()  # type: ignore[union-attr]
    _llm = None
    _retriever = None


app = FastAPI(
    title="Zycus AI Support Suite",
    version="0.1.0",
    description="Intelligent Triage & TAM Account Synthesis",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    retriever = _get_retriever()
    return {
        "status": "ok",
        "tickets": len(retriever.corpus.tickets),
        "accounts": len(retriever.corpus.accounts),
        "kb_chunks": len(retriever.corpus.kb_chunks),
    }


# ── Task 1: Triage ──────────────────────────────────────────────────────

@app.post("/triage", response_model=TriageOutput)
async def triage_endpoint(payload: TriageInput) -> TriageOutput:
    """Classify a support ticket and return structured JSON."""
    retriever = _get_retriever()
    llm = _get_llm()
    settings = get_settings()
    result = await triage_ticket(
        payload.subject,
        payload.body,
        retriever=retriever,
        llm=llm,
        top_k=settings.retrieval_top_k,
    )
    return result


@app.post("/triage/stream")
async def triage_stream_endpoint(payload: TriageInput) -> EventSourceResponse:
    """Stream triage tokens as Server-Sent Events."""
    retriever = _get_retriever()
    llm = _get_llm()
    settings = get_settings()

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        async for token in triage_ticket_stream(
            payload.subject,
            payload.body,
            retriever=retriever,
            llm=llm,
            top_k=settings.retrieval_top_k,
        ):
            yield {"data": token}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())


# ── Task 2: TAM (placeholder wired in Phase 4) ──────────────────────────

@app.get("/account/{account_id}/brief")
async def tam_brief_endpoint(account_id: str) -> JSONResponse:
    """Generate a TAM account brief (Phase 4 implementation)."""
    retriever = _get_retriever()
    llm = _get_llm()
    account = retriever.corpus.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    # Import here to avoid circular imports during Phase 3 build
    from app.tam_summarizer import generate_tam_brief

    result = await generate_tam_brief(
        account_id=account_id,
        corpus=retriever.corpus,
        llm=llm,
    )
    return JSONResponse(content=result.model_dump())


@app.get("/account/{account_id}/brief/stream")
async def tam_brief_stream_endpoint(account_id: str) -> EventSourceResponse:
    """Stream TAM brief tokens as Server-Sent Events."""
    retriever = _get_retriever()
    llm = _get_llm()
    account = retriever.corpus.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

    from app.tam_summarizer import generate_tam_brief_stream

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        async for token in generate_tam_brief_stream(
            account_id=account_id,
            corpus=retriever.corpus,
            llm=llm,
        ):
            yield {"data": token}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())


# ── Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
