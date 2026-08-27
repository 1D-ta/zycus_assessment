"""In-memory corpus plus hybrid BM25 + ONNX dense retrieval with RRF fusion.

Indexes are built once at startup. Request handlers must look up these
structures — they must never parse tickets.json / accounts.json inline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_HR_SPLIT_RE = re.compile(r"\n---+\n")


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def parse_reference_now(raw: str | None) -> datetime:
    if raw and raw.strip():
        return _parse_iso_datetime(raw.strip())
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    heading_path: str
    text: str
    tokenized: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    heading_path: str
    text: str
    score: float
    bm25_rank: int | None
    dense_rank: int | None


@dataclass
class SupportCorpus:
    """Async-loaded tickets, accounts, and knowledge-base chunks."""

    tickets: list[dict[str, Any]]
    accounts: list[dict[str, Any]]
    tickets_by_id: dict[str, dict[str, Any]]
    accounts_by_id: dict[str, dict[str, Any]]
    tickets_by_account: dict[str, list[dict[str, Any]]]
    kb_chunks: list[KnowledgeChunk]
    reference_now: datetime
    window_days: int

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        return self.accounts_by_id.get(account_id)

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        return self.tickets_by_id.get(ticket_id)

    def tickets_for_account(
        self,
        account_id: str,
        *,
        window_days: int | None = None,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return tickets for an account inside the deterministic time window."""
        days = self.window_days if window_days is None else window_days
        cutoff = (as_of or self.reference_now) - timedelta(days=days)
        indexed = self.tickets_by_account.get(account_id, [])
        return [
            ticket
            for ticket in indexed
            if _parse_iso_datetime(ticket["created_at"]) > cutoff
        ]


@dataclass
class HybridRetriever:
    """Reciprocal Rank Fusion over BM25Okapi and MiniLM dense cosine ranks."""

    corpus: SupportCorpus
    embeddings: np.ndarray
    bm25: BM25Okapi
    model: SentenceTransformer
    rrf_k: int = 60
    _chunk_by_id: dict[str, KnowledgeChunk] = field(init=False)

    def __post_init__(self) -> None:
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in self.corpus.kb_chunks}

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip() or not self.corpus.kb_chunks:
            return []

        tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_order = list(np.argsort(bm25_scores)[::-1])

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        dense_scores = self.embeddings @ query_vec
        dense_order = list(np.argsort(dense_scores)[::-1])

        fused = _reciprocal_rank_fusion(
            [bm25_order, dense_order],
            k=self.rrf_k,
        )

        bm25_rank_of = {idx: rank for rank, idx in enumerate(bm25_order, start=1)}
        dense_rank_of = {idx: rank for rank, idx in enumerate(dense_order, start=1)}

        hits: list[SearchHit] = []
        for chunk_idx, score in fused[:top_k]:
            chunk = self.corpus.kb_chunks[chunk_idx]
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_path=chunk.source_path,
                    title=chunk.title,
                    heading_path=chunk.heading_path,
                    text=chunk.text,
                    score=score,
                    bm25_rank=bm25_rank_of.get(chunk_idx),
                    dense_rank=dense_rank_of.get(chunk_idx),
                )
            )
        return hits

    def top_document_id(self, query: str, top_k: int = 5) -> str | None:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return None
        return hits[0].doc_id


def _reciprocal_rank_fusion(
    rank_lists: Iterable[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_kb_file(path: Path, kb_root: Path) -> list[KnowledgeChunk]:
    raw = path.read_text(encoding="utf-8")
    relative = path.relative_to(kb_root).as_posix()
    doc_id = relative.rsplit(".", 1)[0]
    sections = [part.strip() for part in _HR_SPLIT_RE.split(raw) if part.strip()]
    if not sections:
        sections = [raw.strip()] if raw.strip() else []

    chunks: list[KnowledgeChunk] = []
    running_headings: list[str] = []
    for index, section in enumerate(sections):
        headings = [
            match.group(2).strip() for match in _HEADING_RE.finditer(section)
        ]
        if headings:
            running_headings = headings
        title = headings[0] if headings else (running_headings[0] if running_headings else doc_id)
        heading_path = " > ".join(headings or running_headings or [title])
        chunk_id = f"{doc_id}#chunk-{index}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_path=relative,
                title=title,
                heading_path=heading_path,
                text=section,
                tokenized=tuple(tokenize(section)),
            )
        )
    return chunks


def load_knowledge_chunks(kb_dir: Path) -> list[KnowledgeChunk]:
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    markdown_files = sorted(kb_dir.rglob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No Markdown documents under {kb_dir}")

    chunks: list[KnowledgeChunk] = []
    for path in markdown_files:
        chunks.extend(_split_kb_file(path, kb_dir))
    logger.info("Loaded %s KB chunks from %s files", len(chunks), len(markdown_files))
    return chunks


def _index_tickets_accounts(
    tickets: list[dict[str, Any]],
    accounts: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tickets_by_id = {row["ticket_id"]: row for row in tickets}
    accounts_by_id = {row["account_id"]: row for row in accounts}
    tickets_by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ticket in tickets:
        tickets_by_account[ticket["account_id"]].append(ticket)
    for account_id in tickets_by_account:
        tickets_by_account[account_id].sort(key=lambda row: row["created_at"])
    return tickets_by_id, accounts_by_id, dict(tickets_by_account)


def load_embedding_model(settings: Settings) -> SentenceTransformer:
    backend = settings.embedding_backend.strip().lower()
    model_kwargs: dict[str, Any] = {}
    if backend == "onnx":
        model_kwargs["provider"] = "CPUExecutionProvider"

    try:
        model = SentenceTransformer(
            settings.embedding_model,
            backend=backend,
            model_kwargs=model_kwargs if backend == "onnx" else {},
        )
        logger.info(
            "Loaded embedding model %s with backend=%s",
            settings.embedding_model,
            backend,
        )
        return model
    except Exception:
        if backend == "onnx":
            logger.exception(
                "ONNX backend failed for %s; falling back to PyTorch",
                settings.embedding_model,
            )
            return SentenceTransformer(settings.embedding_model)
        raise


async def load_corpus(settings: Settings | None = None) -> SupportCorpus:
    cfg = settings or get_settings()
    tickets_path = cfg.tickets_path
    accounts_path = cfg.accounts_path
    if not tickets_path.is_file():
        raise FileNotFoundError(f"Missing tickets dataset: {tickets_path}")
    if not accounts_path.is_file():
        raise FileNotFoundError(f"Missing accounts dataset: {accounts_path}")

    tickets, accounts, kb_chunks = await asyncio.gather(
        asyncio.to_thread(_read_json, tickets_path),
        asyncio.to_thread(_read_json, accounts_path),
        asyncio.to_thread(load_knowledge_chunks, cfg.kb_dir),
    )

    tickets_by_id, accounts_by_id, tickets_by_account = _index_tickets_accounts(
        tickets, accounts
    )
    return SupportCorpus(
        tickets=tickets,
        accounts=accounts,
        tickets_by_id=tickets_by_id,
        accounts_by_id=accounts_by_id,
        tickets_by_account=tickets_by_account,
        kb_chunks=kb_chunks,
        reference_now=parse_reference_now(cfg.tam_reference_now),
        window_days=cfg.tam_window_days,
    )


def build_retriever(
    corpus: SupportCorpus,
    settings: Settings | None = None,
    model: SentenceTransformer | None = None,
) -> HybridRetriever:
    cfg = settings or get_settings()
    encoder = model or load_embedding_model(cfg)
    texts = [chunk.text for chunk in corpus.kb_chunks]
    matrix = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )
    tokenized_corpus = [list(chunk.tokenized) for chunk in corpus.kb_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return HybridRetriever(
        corpus=corpus,
        embeddings=np.asarray(matrix, dtype=np.float32),
        bm25=bm25,
        model=encoder,
        rrf_k=cfg.rrf_k,
    )


async def build_runtime(settings: Settings | None = None) -> HybridRetriever:
    cfg = settings or get_settings()
    corpus = await load_corpus(cfg)
    return await asyncio.to_thread(build_retriever, corpus, cfg)


async def _validate() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    retriever = await build_runtime()
    corpus = retriever.corpus
    assert corpus.tickets, "tickets.json produced an empty corpus"
    assert corpus.accounts, "accounts.json produced an empty corpus"
    assert corpus.kb_chunks, "knowledge-base produced no chunks"
    assert retriever.embeddings.shape[0] == len(corpus.kb_chunks)
    assert retriever.embeddings.shape[1] > 0

    billing_hits = retriever.search("invoice seat overage billing dispute", top_k=3)
    assert billing_hits, "hybrid search returned no billing hits"
    assert any("billing" in hit.doc_id.lower() or "billing" in hit.source_path.lower() for hit in billing_hits)

    timeout_hits = retriever.search("ERR_CONNECTION_TIMEOUT DataBridge connectors", top_k=3)
    assert timeout_hits, "hybrid search returned no timeout hits"

    sample_account_id = corpus.accounts[0]["account_id"]
    windowed = corpus.tickets_for_account(sample_account_id)
    assert isinstance(windowed, list)

    print(
        json.dumps(
            {
                "tickets": len(corpus.tickets),
                "accounts": len(corpus.accounts),
                "kb_chunks": len(corpus.kb_chunks),
                "embedding_dim": int(retriever.embeddings.shape[1]),
                "billing_top_doc": billing_hits[0].doc_id,
                "timeout_top_doc": timeout_hits[0].doc_id,
                "sample_account_window_tickets": len(windowed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_validate())
