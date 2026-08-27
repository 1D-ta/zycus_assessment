"""RAGAS-style evaluation metrics using LLM-as-a-judge and programmatic checks."""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field
from app.llm_client import LLMClient
from app.schemas import RiskItem

logger = logging.getLogger(__name__)


class JudgeScore(BaseModel):
    reasoning: str = Field(..., description="Explanation of how the score was determined.")
    score: float = Field(..., description="A score between 0.0 (worst) and 1.0 (best).")


async def compute_faithfulness(
    generated_text: str,
    context: str,
    llm: LLMClient,
) -> JudgeScore:
    """Evaluate if all facts in the generated text are fully grounded in the context."""
    if not generated_text.strip():
        return JudgeScore(reasoning="Generated text is empty.", score=1.0)
    if not context.strip():
        return JudgeScore(reasoning="Source context is empty.", score=0.0)

    prompt = (
        f"You are an objective AI quality judge.\n"
        f"Your task is to evaluate the Faithfulness (groundedness) of a generated text against a source context.\n\n"
        f"--- SOURCE CONTEXT ---\n{context}\n\n"
        f"--- GENERATED TEXT ---\n{generated_text}\n\n"
        f"Evaluate whether every statement in the GENERATED TEXT is directly supported by the SOURCE CONTEXT.\n"
        f"Look for any claims, numbers, or facts in the generated text that do not appear in or cannot be logically derived from the source context.\n"
        f"Provide your step-by-step reasoning and a final score from 0.0 (entirely unsupported or hallucinated) to 1.0 (completely grounded with zero hallucinations)."
    )

    system = (
        "You are a strict evaluation judge. Output a single JSON object matching the JudgeScore schema. "
        "Do not include any other text."
    )

    try:
        score_obj = await llm.generate_model(
            prompt,
            JudgeScore,
            system=system,
            temperature=0.0,
        )
        return score_obj  # type: ignore[return-value]
    except Exception as exc:
        logger.error("Failed to compute faithfulness: %s", exc)
        return JudgeScore(reasoning=f"Error during LLM evaluation: {exc}", score=0.0)


async def compute_relevance(
    generated_text: str,
    query_or_profile: str,
    llm: LLMClient,
) -> JudgeScore:
    """Evaluate if the generated text is highly relevant and directly addresses the query or profile."""
    if not generated_text.strip():
        return JudgeScore(reasoning="Generated text is empty.", score=0.0)

    prompt = (
        f"You are an objective AI quality judge.\n"
        f"Your task is to evaluate the Relevance of a generated response/summary to the user input/profile.\n\n"
        f"--- USER INPUT/PROFILE ---\n{query_or_profile}\n\n"
        f"--- GENERATED TEXT ---\n{generated_text}\n\n"
        f"Assess how relevant, focused, and complete the GENERATED TEXT is with respect to the USER INPUT/PROFILE.\n"
        f"Identify any redundant, off-topic, or unhelpful remarks.\n"
        f"Provide your step-by-step reasoning and a final score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant, direct, and complete)."
    )

    system = (
        "You are a strict evaluation judge. Output a single JSON object matching the JudgeScore schema. "
        "Do not include any other text."
    )

    try:
        score_obj = await llm.generate_model(
            prompt,
            JudgeScore,
            system=system,
            temperature=0.0,
        )
        return score_obj  # type: ignore[return-value]
    except Exception as exc:
        logger.error("Failed to compute relevance: %s", exc)
        return JudgeScore(reasoning=f"Error during LLM evaluation: {exc}", score=0.0)


def verify_quotes_programmatic(
    open_risks: list[RiskItem],
    tickets: list[dict],
) -> float:
    """Compute exact quote verification rate (0.0 to 1.0).
    
    Returns 1.0 if there are no risks extracted (valid if no risks present).
    Otherwise, returns the fraction of risks whose quote exists verbatim in the
    corresponding ticket body.
    """
    if not open_risks:
        return 1.0

    tickets_by_id = {t["ticket_id"]: t for t in tickets}
    valid_count = 0

    for risk in open_risks:
        tkt = tickets_by_id.get(risk.ticket_id)
        if not tkt:
            continue
        if risk.quote in tkt.get("body", ""):
            valid_count += 1

    return valid_count / len(open_risks)

