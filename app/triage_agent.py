"""Task 1 — Intelligent Triage pipeline.

Sanitises PII, retrieves relevant KB docs via hybrid search, then asks the
LLM to classify the ticket into product area / category / urgency / team and
draft an initial customer response.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.llm_client import LLMClient
from app.pii_sanitizer import sanitize_ticket_fields
from app.prompts import load_prompt
from app.retrieval import HybridRetriever, SearchHit
from app.schemas import TriageOutput

logger = logging.getLogger(__name__)

_TRIAGE_PROMPT_FILE = "triage_v1.txt"


def _format_kb_listing(hits: list[SearchHit]) -> str:
    """Build a human-readable KB document listing for the prompt."""
    if not hits:
        return "(no knowledge-base documents available)"
    lines: list[str] = []
    for hit in hits:
        lines.append(f"- {hit.doc_id}  [{hit.heading_path}]")
    return "\n".join(lines)


def _build_triage_prompt(
    subject: str,
    body: str,
    kb_hits: list[SearchHit],
) -> str:
    """Render the triage prompt template with ticket and KB context."""
    template = load_prompt(_TRIAGE_PROMPT_FILE)
    result = template
    for key, value in [
        ("kb_documents", _format_kb_listing(kb_hits)),
        ("subject", subject),
        ("body", body),
    ]:
        result = result.replace("{" + key + "}", value)
    return result


async def triage_ticket(
    subject: str,
    body: str,
    *,
    retriever: HybridRetriever,
    llm: LLMClient,
    top_k: int = 5,
) -> TriageOutput:
    """Run the full triage pipeline and return a validated TriageOutput."""
    safe_subject, safe_body = sanitize_ticket_fields(subject, body)
    query = f"{safe_subject} {safe_body}"
    kb_hits = retriever.search(query, top_k=top_k)
    prompt = _build_triage_prompt(safe_subject, safe_body, kb_hits)

    system = (
        "You are a JSON-only triage classifier. "
        "Output a single JSON object matching the TriageOutput schema. "
        "Do not include any other text."
    )

    result = await llm.generate_model(
        prompt,
        TriageOutput,
        system=system,
        temperature=0.0,
    )
    return result  # type: ignore[return-value]


async def triage_ticket_stream(
    subject: str,
    body: str,
    *,
    retriever: HybridRetriever,
    llm: LLMClient,
    top_k: int = 5,
) -> AsyncIterator[str]:
    """Stream triage tokens via SSE-compatible async iterator."""
    safe_subject, safe_body = sanitize_ticket_fields(subject, body)
    query = f"{safe_subject} {safe_body}"
    kb_hits = retriever.search(query, top_k=top_k)
    prompt = _build_triage_prompt(safe_subject, safe_body, kb_hits)

    system = (
        "You are a JSON-only triage classifier. "
        "Output a single JSON object matching the TriageOutput schema. "
        "Do not include any other text."
    )

    async for token in llm.stream(
        prompt,
        system=system,
        json_mode=True,
        response_model=TriageOutput,
        temperature=0.0,
    ):
        yield token
