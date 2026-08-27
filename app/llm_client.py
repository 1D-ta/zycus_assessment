"""LLM provider abstraction. The live path calls the Gemini Developer API. This client defaults to the configured Gemini model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.schemas import TAMExtractOutput, TAMOutput, TriageOutput

logger = logging.getLogger(__name__)

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
)

_TICKET_ID_RE = re.compile(r"TKT-\d+")
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the provider cannot complete a generation request."""


class RateLimitError(LLMError):
    """Raised on HTTP 429 so callers and the eval harness can back off."""


class LLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        raise NotImplementedError
        if False:  # pragma: no cover
            yield ""

    async def generate_model(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> BaseModel:
        raw = await self.generate(
            prompt,
            system=system,
            json_mode=True,
            response_model=response_model,
            temperature=temperature,
        )
        payload = _extract_json_object(raw)
        return response_model.model_validate(payload)


class DeterministicMockLLM(LLMClient):
    """Offline provider: identical prompts always yield identical completions."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        del temperature
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        if json_mode or response_model is not None:
            return json.dumps(_mock_payload(prompt, system, response_model), ensure_ascii=True)
        digest = hashlib.sha256(f"{system or ''}\n{prompt}".encode("utf-8")).hexdigest()[:12]
        return f"Deterministic mock completion [{digest}]."

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        text = await self.generate(
            prompt,
            system=system,
            json_mode=json_mode,
            response_model=response_model,
            temperature=temperature,
        )
        for token in _chunk_tokens(text):
            yield token


class GeminiLLM(LLMClient):
    """Async Gemini generateContent / streamGenerateContent client with backoff."""

    def __init__(self, settings: Settings | None = None) -> None:
        import httpx

        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        self._httpx = httpx
        self._client = httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> str:
        body = _gemini_request_body(
            prompt,
            system=system,
            json_mode=json_mode or response_model is not None,
            response_model=response_model,
            temperature=temperature,
        )
        url = GEMINI_GENERATE_URL.format(model=self.settings.gemini_model)
        payload = await self._request_with_backoff("POST", url, body, stream=False)
        text = _first_candidate_text(payload)
        if not text:
            raise LLMError("Gemini returned an empty candidate")
        return text

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        body = _gemini_request_body(
            prompt,
            system=system,
            json_mode=json_mode or response_model is not None,
            response_model=response_model,
            temperature=temperature,
        )
        url = GEMINI_STREAM_URL.format(model=self.settings.gemini_model)
        async for token in self._stream_with_backoff(url, body):
            yield token

    async def _request_with_backoff(
        self,
        method: str,
        url: str,
        body: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        del stream
        response = await self._send_with_retries(method, url, body, stream=False)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError("Gemini returned non-JSON body") from exc
        return payload

    async def _stream_with_backoff(
        self, url: str, body: dict[str, Any]
    ) -> AsyncIterator[str]:
        response = await self._send_with_retries(
            "POST",
            url,
            body,
            stream=True,
            params={"alt": "sse"},
        )
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                text = _first_candidate_text(chunk)
                if text:
                    yield text
        finally:
            await response.aclose()

    async def _send_with_retries(
        self,
        method: str,
        url: str,
        body: dict[str, Any],
        *,
        stream: bool,
        params: dict[str, str] | None = None,
    ) -> Any:
        delay = self.settings.llm_backoff_base_seconds
        last_error: Exception | None = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                request = self._client.build_request(
                    method,
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.settings.gemini_api_key,
                    },
                    json=body,
                    params=params,
                )
                response = await self._client.send(request, stream=stream)
                if response.status_code == 429:
                    await response.aclose()
                    raise RateLimitError("Gemini rate limited the request (HTTP 429)")
                if response.status_code in _RETRYABLE_STATUS:
                    await response.aclose()
                    raise LLMError(f"Gemini transient error HTTP {response.status_code}")
                if response.status_code >= 400:
                    detail = await _safe_response_text(response)
                    await response.aclose()
                    raise LLMError(
                        f"Gemini request failed HTTP {response.status_code}: {detail[:500]}"
                    )
                return response
            except (
                RateLimitError,
                LLMError,
                self._httpx.HTTPError,
                self._httpx.TimeoutException,
            ) as exc:
                last_error = exc
                retryable = isinstance(
                    exc,
                    (RateLimitError, self._httpx.HTTPError, self._httpx.TimeoutException),
                ) or (isinstance(exc, LLMError) and "transient" in str(exc).lower())
                if not retryable or attempt == self.settings.llm_max_retries:
                    raise
                jitter = random.uniform(0, delay * 0.25)
                sleep_for = delay + jitter
                logger.warning(
                    "Gemini call failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    self.settings.llm_max_retries,
                    exc,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                delay *= 2
        raise LLMError("Gemini request failed after retries") from last_error


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    cfg = settings or get_settings()
    provider = cfg.llm_provider.strip().lower()
    if provider == "mock":
        return DeterministicMockLLM()
    if provider == "gemini":
        return GeminiLLM(cfg)
    raise LLMError(f"Unsupported LLM_PROVIDER={cfg.llm_provider!r}; use mock or gemini")


def _gemini_request_body(
    prompt: str,
    *,
    system: str | None,
    json_mode: bool,
    response_model: type[BaseModel] | None,
    temperature: float,
) -> dict[str, Any]:
    generation: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        generation["responseMimeType"] = "application/json"
    if response_model is not None:
        generation["responseMimeType"] = "application/json"
        generation["responseSchema"] = _gemini_schema(response_model)

    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def _dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    import copy
    schema_copy = copy.deepcopy(schema)

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                parts = ref_path.lstrip("#/").split("/")
                current = schema_copy
                for part in parts:
                    current = current[part]
                return resolve(current)
            else:
                return {k: resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(schema_copy)
    if isinstance(resolved, dict) and "$defs" in resolved:
        del resolved["$defs"]
    return resolved


def _gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    dereferenced = _dereference_schema(schema)
    return _strip_unsupported_schema_keys(dereferenced)


def _strip_unsupported_schema_keys(node: Any) -> Any:
    if isinstance(node, dict):
        blocked = {"title", "default", "$defs"}
        cleaned = {
            key: _strip_unsupported_schema_keys(value)
            for key, value in node.items()
            if key not in blocked
        }
        defs = node.get("$defs")
        if defs and "$ref" in node:
            return cleaned
        if defs:
            cleaned["$defs"] = _strip_unsupported_schema_keys(defs)
        return cleaned
    if isinstance(node, list):
        return [_strip_unsupported_schema_keys(item) for item in node]
    return node


def _first_candidate_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    return "".join(texts)


async def _safe_response_text(response: Any) -> str:
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if response.is_stream_consumed:
            return ""
        return (await response.aread()).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise LLMError("Model did not return JSON") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise LLMError("Model JSON must be an object")
    return parsed


def _chunk_tokens(text: str) -> list[str]:
    parts = re.findall(r"\S+\s*", text)
    return parts or [text]


def _mock_payload(
    prompt: str,
    system: str | None,
    response_model: type[BaseModel] | None,
) -> dict[str, Any]:
    if response_model is not None and response_model.__name__ == "JudgeScore":
        return {"reasoning": "Mock evaluation reasoning.", "score": 1.0}
    if response_model is TAMExtractOutput:
        return _mock_tam_extract(prompt)
    if response_model is TAMOutput:
        return _mock_tam_synth(prompt)
    if response_model is TriageOutput:
        return _mock_triage(prompt)

    blob = f"{system or ''}\n{prompt}".lower()
    if "verbatim" in blob or "risk signals" in blob:
        return _mock_tam_extract(prompt)
    if "executive_summary" in blob or "talking_points" in blob:
        return _mock_tam_synth(prompt)
    return _mock_triage(prompt)



def _mock_triage(prompt: str) -> dict[str, Any]:
    lowered = prompt.lower()
    if "bill" in lowered or "invoice" in lowered or "seat" in lowered:
        category = "Billing"
        product_area = "Billing"
        routed_team = "Billing Operations"
        kb = "billing/billing-and-plans"
        urgency = "P3"
    elif "feature" in lowered or "request:" in lowered or "bulk" in lowered:
        category = "Feature Request"
        product_area = "Platform"
        routed_team = "Product Management"
        kb = "onboarding/onboarding-guide"
        urgency = "P4"
    elif "timeout" in lowered or "error" in lowered or "bug" in lowered:
        category = "Bug"
        product_area = "Connectors"
        routed_team = "Technical Support L2"
        kb = "products/databridge-pro"
        urgency = "P2"
    else:
        category = "How-To"
        product_area = "General"
        routed_team = "Technical Support L1"
        kb = "troubleshooting/performance-and-integrations"
        urgency = "P3"

    if "p1" in lowered or "critical" in lowered or "down" in lowered:
        urgency = "P1"
        routed_team = "Incident Response"

    return {
        "product_area": product_area,
        "category": category,
        "urgency_tier": urgency,
        "urgency_reasoning": f"Urgency is {urgency} because the ticket indicates a {category.lower()} in {product_area.lower()}.",
        "relevant_kb_doc": kb,
        "routed_team": routed_team,
        "draft_response": (
            "Thanks for reaching out. We have classified this ticket and attached the "
            "most relevant knowledge-base article while an agent reviews the details."
        ),
    }


def _mock_tam_extract(prompt: str) -> dict[str, Any]:
    ticket_ids = _TICKET_ID_RE.findall(prompt)
    quote_match = re.search(r'"([^"]{12,280})"', prompt)
    quote = quote_match.group(1) if quote_match else "We are evaluating other vendors"
    risks = []
    if ticket_ids:
        risks.append(
            {
                "issue": "Customer reported instability or churn intent",
                "quote": quote,
                "ticket_id": ticket_ids[0],
            }
        )
    return {"risks": risks}


def _mock_tam_synth(prompt: str) -> dict[str, Any]:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    ticket_ids = _TICKET_ID_RE.findall(prompt)
    risks = []
    if ticket_ids:
        risks.append(
            {
                "issue": "Open delivery risk in the 90-day window",
                "quote": "evaluating other vendors",
                "ticket_id": ticket_ids[0],
            }
        )
    return {
        "executive_summary": (
            f"Account brief {digest}: the last 90 days show concentrated support load "
            "and at least one explicit retention signal. Review open P1/P2 items before "
            "the next QBR and confirm a named champion is in place."
        ),
        "open_risks": risks,
        "talking_points": [
            "Confirm remaining open P1/P2 tickets and owners.",
            "Walk through usage trend and licensed vs active seats.",
            "Align on a 30-day recovery plan before renewal discussions.",
        ],
    }


if __name__ == "__main__":
    async def _main() -> None:
        client = DeterministicMockLLM()
        first = await client.generate(
            "Unable to connect DataBridge Pro timeout",
            json_mode=True,
            response_model=TriageOutput,
        )
        second = await client.generate(
            "Unable to connect DataBridge Pro timeout",
            json_mode=True,
            response_model=TriageOutput,
        )
        assert first == second, "mock provider is not deterministic"
        parsed = TriageOutput.model_validate_json(first)
        tokens = [token async for token in client.stream("invoice seat overage", json_mode=True)]
        assert "".join(tokens)
        print(json.dumps({"deterministic": True, "triage": parsed.model_dump()}, indent=2))

    asyncio.run(_main())
