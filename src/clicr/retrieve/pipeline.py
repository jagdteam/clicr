"""Staged retrieval pipeline.

rewrite, hyde, embed_query, vector, keyword, fuse, rerank, select, graph, abstain. The
advanced stages (hyde, rerank, graph, abstain) are switches in SearchConfig; a stage that is
switched off does not run and gets no timing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from clicr.index.store import IndexStore
from clicr.models import Chunk
from clicr.providers.base import ChatModel, Embedder
from clicr.retrieve import graph
from clicr.retrieve.hyde import DRAFT_CHARS, draft_snippet
from clicr.retrieve.rerank import CrossEncoderReranker, LLMReranker, Reranker
from clicr.retrieve.rewrite import rewrite_query
from clicr.retrieve.search import Fused, SearchConfig, rrf_fuse
from clicr.retrieve.types import Candidate, RetrievalResult, Turn

log = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Mutable data handed from stage to stage."""

    question: str
    history: list[Turn]
    query: str
    hyde_draft: str | None = None
    qvec: np.ndarray | None = None
    vector_hits: list[tuple[int, float]] = field(default_factory=list)
    keyword_hits: list[tuple[int, float]] = field(default_factory=list)
    fused: list[Fused] = field(default_factory=list)
    fused_rank: dict[int, int] = field(default_factory=dict)  # rowid -> 1-based, pre-rerank
    rerank: dict[int, tuple[float, int]] = field(default_factory=dict)  # rowid -> (score, rank)
    chunks: dict[int, Chunk] = field(default_factory=dict)  # fetched chunks by rowid
    pool: list[Candidate] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    abstained: bool = False
    abstain_signal: float | None = None


Stage = Callable[[PipelineState], None]


class Retriever:
    def __init__(
        self,
        store: IndexStore,
        embedder: Embedder,
        chat: ChatModel | None = None,
        cfg: SearchConfig | None = None,
        rewrite: bool = True,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chat = chat
        self.cfg = cfg or SearchConfig()
        self.rewrite = rewrite
        self.reranker = reranker if reranker is not None else self._default_reranker()

    def _default_reranker(self) -> Reranker | None:
        if self.cfg.rerank == "cross":
            return CrossEncoderReranker(self.cfg.rerank_model)
        if self.cfg.rerank == "llm" and self.chat is not None:
            return LLMReranker(self.chat)
        return None

    def stages(self) -> list[tuple[str, Stage]]:
        """Ordered (name, callable) stages."""
        return [
            ("rewrite", self._rewrite),
            ("hyde", self._hyde),
            ("embed_query", self._embed_query),
            ("vector", self._vector),
            ("keyword", self._keyword),
            ("fuse", self._fuse),
            ("rerank", self._rerank),
            ("select", self._select),
            ("graph", self._graph),
            ("abstain", self._abstain),
        ]

    def _enabled(self, name: str, s: PipelineState) -> bool:
        cfg = self.cfg
        if name == "rewrite":
            return bool(s.history and self.chat and self.rewrite)
        if name == "hyde":
            return cfg.hyde and self.chat is not None and cfg.mode != "keyword"
        if name == "rerank":
            return cfg.rerank != "none" and self.reranker is not None
        if name == "graph":
            return cfg.graph and cfg.graph_budget > 0
        if name == "abstain":
            return cfg.abstain_threshold is not None
        return True

    def retrieve(self, question: str, history: list[Turn] | None = None) -> RetrievalResult:
        state = PipelineState(question, list(history or []), question)
        timings: dict[str, float] = {}
        for name, stage in self.stages():
            if not self._enabled(name, state):
                continue
            start = time.perf_counter()
            stage(state)
            timings[name] = (time.perf_counter() - start) * 1000.0
        return RetrievalResult(
            question,
            state.query,
            state.candidates,
            timings,
            pool=state.pool,
            hyde_draft=state.hyde_draft[:DRAFT_CHARS] if state.hyde_draft else None,
            abstained=state.abstained,
            abstain_signal=state.abstain_signal,
        )

    # -- helpers -------------------------------------------------------------
    def _fetch(self, s: PipelineState, rowids: list[int]) -> None:
        missing = [rid for rid in rowids if rid not in s.chunks]
        s.chunks.update(self.store.get_chunks(missing))

    def _pool_size(self) -> int:
        return self.cfg.rerank_pool if self.cfg.rerank != "none" else self.cfg.k_candidates

    # -- stages ------------------------------------------------------------
    def _rewrite(self, s: PipelineState) -> None:
        assert self.chat is not None
        s.query = rewrite_query(self.chat, s.history, s.question)

    def _hyde(self, s: PipelineState) -> None:
        assert self.chat is not None
        s.hyde_draft = draft_snippet(self.chat, s.query) or None

    def _embed_query(self, s: PipelineState) -> None:
        if self.cfg.mode == "keyword":
            return
        if s.hyde_draft:
            # HyDE: the draft is code, so embed it the way chunks were embedded
            s.qvec = self.embedder.embed_documents([s.hyde_draft])[0]
        else:
            s.qvec = self.embedder.embed_query(s.query)

    def _vector(self, s: PipelineState) -> None:
        if self.cfg.mode != "keyword" and s.qvec is not None:
            s.vector_hits = self.store.vector_search(s.qvec, self.cfg.k_candidates)

    def _keyword(self, s: PipelineState) -> None:
        if self.cfg.mode != "vector":
            s.keyword_hits = self.store.keyword_search(s.query, self.cfg.k_candidates)

    def _fuse(self, s: PipelineState) -> None:
        s.fused = rrf_fuse(s.vector_hits, s.keyword_hits, self.cfg.rrf_k)
        s.fused_rank = {row[0]: rank for rank, row in enumerate(s.fused, start=1)}

    def _rerank(self, s: PipelineState) -> None:
        assert self.reranker is not None
        head = s.fused[: self.cfg.rerank_pool]
        self._fetch(s, [row[0] for row in head])
        head = [row for row in head if row[0] in s.chunks]
        if not head:
            return
        texts = [s.chunks[row[0]].embed_text(header=True) for row in head]
        scores = self.reranker.score(s.query, texts)
        # sort is stable, so equal scores keep fused order
        order = sorted(range(len(head)), key=lambda i: -scores[i])
        for rank, i in enumerate(order, start=1):
            s.rerank[head[i][0]] = (float(scores[i]), rank)
        reranked = {row[0] for row in head}
        s.fused = [head[i] for i in order] + [r for r in s.fused if r[0] not in reranked]

    def _select(self, s: PipelineState) -> None:
        limit = max(self._pool_size(), self.cfg.top_k)
        self._fetch(s, [row[0] for row in s.fused[:limit]])
        pool: list[Candidate] = []
        for rid, fused, vrank, vscore, krank, kscore in s.fused[:limit]:
            chunk = s.chunks.get(rid)
            if chunk is None:
                continue
            cand = Candidate(rid, chunk, vrank, vscore, krank, kscore, fused)
            cand.trace["fuse"] = {"fused_score": fused, "fused_rank": s.fused_rank[rid]}
            if rid in s.rerank:
                score, rank = s.rerank[rid]
                cand.trace["rerank"] = {
                    "rerank_score": score,
                    "rerank_rank": rank,
                    "method": self.cfg.rerank,
                }
            pool.append(cand)
        # select keeps C3 semantics: the top_k fused rows, minus any without a chunk
        top = {row[0] for row in s.fused[: self.cfg.top_k]}
        out: list[Candidate] = []
        for cand in pool:
            if cand.rowid in top:
                cand.trace["select"] = {"rank": len(out) + 1, "mode": self.cfg.mode}
                out.append(cand)
        s.pool = pool[: self._pool_size()]
        s.candidates = out

    def _graph(self, s: PipelineState) -> None:
        known = {c.rowid: c for c in s.pool}
        fused = {row[0]: row for row in s.fused}
        s.candidates = s.candidates + graph.expand(
            self.store, s.candidates, self.cfg.graph_budget, known, fused
        )

    def _abstain(self, s: PipelineState) -> None:
        threshold = self.cfg.abstain_threshold
        assert threshold is not None
        # only the cross-encoder's scores are comparable across queries; the LLM
        # reranker's top score is always 1.0, so it falls back to the vector cosine
        if s.rerank and self.cfg.rerank == "cross":
            signal: float | None = max(score for score, _ in s.rerank.values())
        elif s.vector_hits:
            signal = s.vector_hits[0][1]
        else:
            signal = None  # nothing to judge (keyword-only): never abstain
        s.abstain_signal = signal
        s.abstained = signal is not None and signal < threshold
        for cand in s.candidates:
            cand.trace["abstain"] = {
                "signal": signal,
                "threshold": threshold,
                "abstained": s.abstained,
            }
