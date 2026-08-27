"""Evaluation harness runner for Task 3.

Executes triage and TAM test cases, runs metrics, handles rate limiting delays,
and generates eval_report.json.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.llm_client import get_llm_client
from app.retrieval import build_runtime
from app.triage_agent import triage_ticket
from app.tam_summarizer import generate_tam_brief, _format_tickets_block
from eval.metrics import compute_faithfulness, compute_relevance, verify_quotes_programmatic

# Setup logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_harness")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
REPORT_PATH = PROJECT_ROOT / "eval_report.json"


async def evaluate_triage(
    retriever: Any,
    llm: Any,
    settings: Any,
) -> list[dict[str, Any]]:
    test_cases_path = EVAL_DIR / "test_cases_triage.json"
    if not test_cases_path.is_file():
        logger.error("Triage test cases file not found at %s", test_cases_path)
        return []

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    logger.info("Starting Triage evaluation on %d test cases...", len(test_cases))

    for tc in test_cases:
        logger.info("Evaluating Triage case %s: %s", tc["id"], tc["description"])
        start_time = time.perf_counter()
        
        # Run classification
        output = await triage_ticket(
            subject=tc["subject"],
            body=tc["body"],
            retriever=retriever,
            llm=llm,
            top_k=settings.retrieval_top_k,
        )
        latency = time.perf_counter() - start_time
        if settings.llm_provider != "mock":
            await asyncio.sleep(4.0)

        # Match accuracy
        expected = tc["expected"]
        matches = {
            "product_area": output.product_area == expected["product_area"],
            "category": output.category == expected["category"],
            "urgency_tier": output.urgency_tier == expected["urgency_tier"],
            "routed_team": output.routed_team == expected["routed_team"],
        }
        exact_match = all(matches.values())

        # Retrieve KB context for faithfulness evaluation
        query = f"{tc['subject']} {tc['body']}"
        kb_hits = retriever.search(query, top_k=settings.retrieval_top_k)
        context = "\n\n".join([hit.text for hit in kb_hits])

        # LLM-as-judge scoring
        faithfulness_res = await compute_faithfulness(output.draft_response, context, llm)
        if settings.llm_provider != "mock":
            await asyncio.sleep(4.0)
        relevance_res = await compute_relevance(output.draft_response, query, llm)
        if settings.llm_provider != "mock":
            await asyncio.sleep(4.0)

        # Triage pass/fail: passes if exact match is true and faithfulness/relevance scores are >= 0.7.
        is_pass = exact_match and (faithfulness_res.score >= 0.7) and (relevance_res.score >= 0.7)
        pass_fail = "PASS" if is_pass else "FAIL"

        # Triage quality score: average of exact match (0 or 1), faithfulness, and relevance.
        quality_score = float((1.0 if exact_match else 0.0) + faithfulness_res.score + relevance_res.score) / 3.0

        res_record = {
            "id": tc["id"],
            "description": tc["description"],
            "input": {"subject": tc["subject"], "body": tc["body"]},
            "expected": expected,
            "predicted": output.model_dump(),
            "latency_seconds": latency,
            "accuracy": {
                "field_matches": matches,
                "exact_match": exact_match,
            },
            "pass_fail": pass_fail,
            "quality_score": quality_score,
            "metrics": {
                "faithfulness": {
                    "score": faithfulness_res.score,
                    "reasoning": faithfulness_res.reasoning,
                },
                "relevance": {
                    "score": relevance_res.score,
                    "reasoning": relevance_res.reasoning,
                },
            },
        }
        results.append(res_record)
        logger.info(
            "Finished Triage case %s | Exact Match: %s | Faithfulness: %.2f | Relevance: %.2f | Pass/Fail: %s | Quality: %.2f",
            tc["id"], exact_match, faithfulness_res.score, relevance_res.score, pass_fail, quality_score
        )

    return results


async def evaluate_tam(
    retriever: Any,
    llm: Any,
    settings: Any,
) -> list[dict[str, Any]]:
    test_cases_path = EVAL_DIR / "test_cases_tam.json"
    if not test_cases_path.is_file():
        logger.error("TAM test cases file not found at %s", test_cases_path)
        return []

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    logger.info("Starting TAM evaluation on %d test cases...", len(test_cases))

    corpus = retriever.corpus

    for tc in test_cases:
        logger.info("Evaluating TAM case %s: %s", tc["id"], tc["description"])
        account_id = tc["account_id"]
        
        start_time = time.perf_counter()
        failed = False
        error_msg = ""
        output_data = {}
        latency = 0.0

        try:
            output = await generate_tam_brief(
                account_id=account_id,
                corpus=corpus,
                llm=llm,
            )
            latency = time.perf_counter() - start_time
            output_data = output.model_dump()
            if settings.llm_provider != "mock":
                await asyncio.sleep(4.0)
        except Exception as exc:
            latency = time.perf_counter() - start_time
            failed = True
            error_msg = str(exc)

        # Verify status code expectations
        expected_status = tc["expected_status"]
        actual_status = 404 if failed and "not found" in error_msg.lower() else (500 if failed else 200)
        status_match = actual_status == expected_status

        # Evaluate metrics if the brief was successfully generated
        quote_rate = 1.0
        faithfulness_score = 1.0
        faithfulness_reason = "No output to judge."
        relevance_score = 1.0
        relevance_reason = "No output to judge."

        if not failed:
            account = corpus.get_account(account_id) or {}
            tickets = corpus.tickets_for_account(account_id)
            
            # Programmatic quote validation check
            quote_rate = verify_quotes_programmatic(output.open_risks, tickets)

            # Format account profile and ticket history context for judge
            context = f"Account Metadata: {json.dumps(account)}\n\nTicket History:\n{_format_tickets_block(tickets)}"
            
            # LLM-as-judge scoring
            faithfulness_res = await compute_faithfulness(output.executive_summary, context, llm)
            if settings.llm_provider != "mock":
                await asyncio.sleep(4.0)
            relevance_res = await compute_relevance(
                output.executive_summary + "\nTalking points: " + ", ".join(output.talking_points),
                f"Account: {account.get('company', 'Unknown')}. Health: {account.get('health_status', 'Unknown')}. Trend: {account.get('usage_trend', 'Unknown')}.",
                llm
            )
            if settings.llm_provider != "mock":
                await asyncio.sleep(4.0)
            
            faithfulness_score = faithfulness_res.score
            faithfulness_reason = faithfulness_res.reasoning
            relevance_score = relevance_res.score
            relevance_reason = relevance_res.reasoning

        # TAM pass/fail: passes if status match is true and quote verification rate >= 0.9.
        is_pass = status_match and (quote_rate >= 0.9)
        pass_fail = "PASS" if is_pass else "FAIL"

        # TAM quality score: average of status match, quote verification rate, faithfulness, and relevance.
        quality_score = float((1.0 if status_match else 0.0) + quote_rate + faithfulness_score + relevance_score) / 4.0

        res_record = {
            "id": tc["id"],
            "description": tc["description"],
            "account_id": account_id,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "status_match": status_match,
            "latency_seconds": latency,
            "error": error_msg if failed else None,
            "predicted": output_data,
            "pass_fail": pass_fail,
            "quality_score": quality_score,
            "metrics": {
                "quote_verification_rate": quote_rate,
                "faithfulness": {
                    "score": faithfulness_score,
                    "reasoning": faithfulness_reason,
                },
                "relevance": {
                    "score": relevance_score,
                    "reasoning": relevance_reason,
                },
            },
        }
        results.append(res_record)
        logger.info(
            "Finished TAM case %s | Status Match: %s | Quote Verification: %.2f | Faithfulness: %.2f | Pass/Fail: %s | Quality: %.2f",
            tc["id"], status_match, quote_rate, faithfulness_score, pass_fail, quality_score
        )

    return results


async def run_evaluation() -> None:
    settings = get_settings()
    logger.info("Building retriever and LLM client runtime...")
    retriever = await build_runtime(settings)
    llm = get_llm_client(settings)

    start_eval_time = time.time()
    triage_results = await evaluate_triage(retriever, llm, settings)
    tam_results = await evaluate_tam(retriever, llm, settings)
    total_eval_duration = time.time() - start_eval_time

    # Triage Aggregates
    triage_count = len(triage_results)
    avg_triage_latency = sum(r["latency_seconds"] for r in triage_results) / triage_count if triage_count > 0 else 0
    triage_exact_matches = sum(1 for r in triage_results if r["accuracy"]["exact_match"])
    triage_accuracy = triage_exact_matches / triage_count if triage_count > 0 else 0
    avg_triage_faithfulness = sum(r["metrics"]["faithfulness"]["score"] for r in triage_results) / triage_count if triage_count > 0 else 0
    avg_triage_relevance = sum(r["metrics"]["relevance"]["score"] for r in triage_results) / triage_count if triage_count > 0 else 0
    triage_pass_rate = sum(1 for r in triage_results if r["pass_fail"] == "PASS") / triage_count if triage_count > 0 else 0
    avg_triage_quality = sum(r["quality_score"] for r in triage_results) / triage_count if triage_count > 0 else 0

    # TAM Aggregates
    tam_count = len(tam_results)
    avg_tam_latency = sum(r["latency_seconds"] for r in tam_results) / tam_count if tam_count > 0 else 0
    tam_status_matches = sum(1 for r in tam_results if r["status_match"])
    tam_status_accuracy = tam_status_matches / tam_count if tam_count > 0 else 0
    
    successful_tams = [r for r in tam_results if r["actual_status"] == 200]
    success_tam_count = len(successful_tams)
    avg_tam_quote_rate = sum(r["metrics"]["quote_verification_rate"] for r in successful_tams) / success_tam_count if success_tam_count > 0 else 0
    avg_tam_faithfulness = sum(r["metrics"]["faithfulness"]["score"] for r in successful_tams) / success_tam_count if success_tam_count > 0 else 0
    avg_tam_relevance = sum(r["metrics"]["relevance"]["score"] for r in successful_tams) / success_tam_count if success_tam_count > 0 else 0
    tam_pass_rate = sum(1 for r in tam_results if r["pass_fail"] == "PASS") / tam_count if tam_count > 0 else 0
    avg_tam_quality = sum(r["quality_score"] for r in tam_results) / tam_count if tam_count > 0 else 0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runner_metadata": {
            "llm_provider": settings.llm_provider,
            "model_name": settings.gemini_model,
            "duration_seconds": total_eval_duration,
        },
        "aggregate_statistics": {
            "triage": {
                "total_cases": triage_count,
                "exact_match_accuracy": triage_accuracy,
                "average_faithfulness": avg_triage_faithfulness,
                "average_relevance": avg_triage_relevance,
                "pass_rate": triage_pass_rate,
                "average_quality_score": avg_triage_quality,
                "average_latency_seconds": avg_triage_latency,
            },
            "tam": {
                "total_cases": tam_count,
                "status_accuracy": tam_status_accuracy,
                "average_quote_verification_rate": avg_tam_quote_rate,
                "average_faithfulness": avg_tam_faithfulness,
                "average_relevance": avg_tam_relevance,
                "pass_rate": tam_pass_rate,
                "average_quality_score": avg_tam_quality,
                "average_latency_seconds": avg_tam_latency,
            }
        },
        "detailed_results": {
            "triage": triage_results,
            "tam": tam_results,
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report successfully written to %s", REPORT_PATH)


if __name__ == "__main__":
    asyncio.run(run_evaluation())

