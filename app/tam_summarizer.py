"""Task 2 — TAM Account Summarizer pipeline.

Two-stage prompt chaining with code-level substring verification:
  Stage 1 (Extraction): LLM extracts risk signals with verbatim quotes.
  Code Validation: Quotes are substring-checked against raw ticket bodies.
  Stage 2 (Synthesis): Only verified quotes feed the executive brief prompt.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from app.llm_client import LLMClient
from app.prompts import load_prompt
from app.retrieval import SupportCorpus
from app.schemas import RiskItem, TAMExtractOutput, TAMOutput

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT_FILE = "tam_extract_v1.txt"
_SYNTH_PROMPT_FILE = "tam_synth_v1.txt"


def _safe_format(template: str, **kwargs: Any) -> str:
    """Replace {name} placeholders without crashing on stray braces in
    interpolated values or JSON examples embedded in the template."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result





def _format_tickets_block(tickets: list[dict[str, Any]]) -> str:
    """Render ticket summaries for the extraction prompt."""
    if not tickets:
        return "(no tickets in 90-day window)"
    lines: list[str] = []
    for t in tickets:
        lines.append(
            f"### {t['ticket_id']} — {t.get('subject', 'No subject')}\n"
            f"Product: {t.get('product', 'N/A')} | Area: {t.get('product_area', 'N/A')} | "
            f"Category: {t.get('category', 'N/A')} | Urgency: {t.get('urgency', 'N/A')} | "
            f"Status: {t.get('status', 'N/A')}\n"
            f"Created: {t.get('created_at', 'N/A')}\n"
            f"Body:\n\"{t.get('body', '')}\"\n"
        )
    return "\n".join(lines)


def _format_verified_risks(risks: list[RiskItem]) -> str:
    """Render verified risk signals for the synthesis prompt."""
    if not risks:
        return "(no verified risk signals)"
    lines: list[str] = []
    for i, risk in enumerate(risks, 1):
        lines.append(
            f"{i}. **{risk.issue}**\n"
            f"   Quote: \"{risk.quote}\"\n"
            f"   Ticket: {risk.ticket_id}"
        )
    return "\n".join(lines)


def _build_extract_prompt(
    account: dict[str, Any],
    tickets: list[dict[str, Any]],
) -> str:
    """Render the Stage 1 extraction prompt."""
    template = load_prompt(_EXTRACT_PROMPT_FILE)
    contact = account.get("primary_contact", {})
    return _safe_format(template,
        account_id=account["account_id"],
        company=account.get("company", "Unknown"),
        plan_tier=account.get("plan_tier", "N/A"),
        arr_usd=account.get("arr_usd", 0),
        seats_licensed=account.get("seats_licensed", 0),
        seats_active=account.get("seats_active", 0),
        health_status=account.get("health_status", "N/A"),
        usage_trend=account.get("usage_trend", "N/A"),
        renewal_date=account.get("renewal_date", "N/A"),
        primary_contact_name=contact.get("name", "N/A"),
        primary_contact_title=contact.get("title", "N/A"),
        products=", ".join(account.get("products", [])),
        escalation_notes="; ".join(account.get("escalation_notes", [])) or "None",
        tickets_block=_format_tickets_block(tickets),
    )


def _build_synth_prompt(
    account: dict[str, Any],
    verified_risks: list[RiskItem],
    ticket_count: int,
) -> str:
    """Render the Stage 2 synthesis prompt."""
    template = load_prompt(_SYNTH_PROMPT_FILE)
    contact = account.get("primary_contact", {})
    return _safe_format(template,
        account_id=account["account_id"],
        company=account.get("company", "Unknown"),
        plan_tier=account.get("plan_tier", "N/A"),
        arr_usd=account.get("arr_usd", 0),
        seats_licensed=account.get("seats_licensed", 0),
        seats_active=account.get("seats_active", 0),
        health_status=account.get("health_status", "N/A"),
        usage_trend=account.get("usage_trend", "N/A"),
        renewal_date=account.get("renewal_date", "N/A"),
        primary_contact_name=contact.get("name", "N/A"),
        primary_contact_title=contact.get("title", "N/A"),
        products=", ".join(account.get("products", [])),
        ticket_count=ticket_count,
        verified_risks_block=_format_verified_risks(verified_risks),
    )


def _verify_quotes(
    risks: list[RiskItem],
    tickets: list[dict[str, Any]],
) -> list[RiskItem]:
    """Code-level substring validation: only keep risks whose quote is an
    exact substring of the referenced ticket's body."""
    tickets_by_id = {t["ticket_id"]: t for t in tickets}
    verified: list[RiskItem] = []
    for risk in risks:
        ticket = tickets_by_id.get(risk.ticket_id)
        if ticket is None:
            logger.warning(
                "Quote verification: ticket %s not found in window — dropping risk",
                risk.ticket_id,
            )
            continue
        body = ticket.get("body", "")
        if risk.quote in body:
            verified.append(risk)
            logger.debug(
                "Quote verified: %r in %s", risk.quote[:60], risk.ticket_id
            )
        else:
            logger.warning(
                "Quote verification FAILED for %s: %r not found in ticket body — dropping",
                risk.ticket_id,
                risk.quote[:80],
            )
    return verified


async def generate_tam_brief(
    *,
    account_id: str,
    corpus: SupportCorpus,
    llm: LLMClient,
) -> TAMOutput:
    """Full two-stage TAM pipeline with substring verification."""
    account = corpus.get_account(account_id)
    if account is None:
        raise ValueError(f"Account {account_id} not found")

    tickets = corpus.tickets_for_account(account_id)

    # ── Stage 1: Extract risk signals ────────────────────────────────
    extract_prompt = _build_extract_prompt(account, tickets)
    extract_system = (
        "You are a JSON-only risk extraction engine. "
        "Output a single JSON object matching the TAMExtractOutput schema. "
        "Do not include any other text."
    )
    extract_result = await llm.generate_model(
        extract_prompt,
        TAMExtractOutput,
        system=extract_system,
        temperature=0.0,
    )
    raw_risks: list[RiskItem] = extract_result.risks  # type: ignore[attr-defined]
    logger.info(
        "Stage 1 extracted %d risk signals for %s", len(raw_risks), account_id
    )

    # ── Code-level verification ──────────────────────────────────────
    verified_risks = _verify_quotes(raw_risks, tickets)
    logger.info(
        "Verified %d / %d risk signals for %s",
        len(verified_risks),
        len(raw_risks),
        account_id,
    )

    # ── Stage 2: Synthesise executive brief ──────────────────────────
    synth_prompt = _build_synth_prompt(account, verified_risks, len(tickets))
    synth_system = (
        "You are a JSON-only executive brief generator. "
        "Output a single JSON object matching the TAMOutput schema. "
        "Do not include any other text."
    )
    brief = await llm.generate_model(
        synth_prompt,
        TAMOutput,
        system=synth_system,
        temperature=0.0,
    )
    return brief  # type: ignore[return-value]


async def generate_tam_brief_stream(
    *,
    account_id: str,
    corpus: SupportCorpus,
    llm: LLMClient,
) -> AsyncIterator[str]:
    """Stream the Stage 2 synthesis tokens (Stage 1 runs eagerly first)."""
    account = corpus.get_account(account_id)
    if account is None:
        raise ValueError(f"Account {account_id} not found")

    tickets = corpus.tickets_for_account(account_id)

    # Stage 1 runs eagerly
    extract_prompt = _build_extract_prompt(account, tickets)
    extract_system = (
        "You are a JSON-only risk extraction engine. "
        "Output a single JSON object matching the TAMExtractOutput schema. "
        "Do not include any other text."
    )
    extract_result = await llm.generate_model(
        extract_prompt,
        TAMExtractOutput,
        system=extract_system,
        temperature=0.0,
    )
    raw_risks: list[RiskItem] = extract_result.risks  # type: ignore[attr-defined]
    verified_risks = _verify_quotes(raw_risks, tickets)

    # Stage 2 streams
    synth_prompt = _build_synth_prompt(account, verified_risks, len(tickets))
    synth_system = (
        "You are a JSON-only executive brief generator. "
        "Output a single JSON object matching the TAMOutput schema. "
        "Do not include any other text."
    )
    async for token in llm.stream(
        synth_prompt,
        system=synth_system,
        json_mode=True,
        response_model=TAMOutput,
        temperature=0.0,
    ):
        yield token
