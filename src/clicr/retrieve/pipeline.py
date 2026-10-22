"""Staged retrieval pipeline: rewrite, embed, vector, keyword, fuse, select."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from clicr.index.store import IndexStore
from clicr.providers.base import ChatModel, Embedder
from clicr.retrieve.rewrite import rewrite_query
from clicr.retrieve.search import Fused, SearchConfig, rrf_fuse
from clicr.retrieve.types import Candidate, RetrievalResult, Turn


@dataclass
class PipelineState:
    """Mutable data handed from stage to stage."""

    question: str
    history: list[Turn]
    query: str
    qvec: np.ndarray | None = None
    vector_hits: list[tuple[int, float]] = field(default_factory=list)
    keyword_hits: list[tuple[int, float]] = field(default_factory=list)
    fused: list[Fused] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)


Stage = Callable[[PipelineState], None]


class Retriever:
    def __init__(
        self,
        store: IndexStore,
        embedder: Embedder,
        chat: ChatModel | None = None,
        cfg: SearchConfig | None = None,
        rewrite: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chat = chat
        self.cfg = cfg or SearchConfig()
        self.rewrite = rewrite

    def stages(self) -> list[tuple[str, Stage]]:
        """Ordered (name, callable) stages; later chunks insert more here."""
        return [
            ("rewrite", self._rewrite),
            ("embed_query", self._embed_query),
            ("vector", self._vector),
            ("keyword", self._keyword),
            ("fuse", self._fuse),
            ("select", self._select),
        ]

    def retrieve(self, question: str, history: list[Turn] | None = None) -> RetrievalResult:
        state = PipelineState(question, list(history or []), question)
        timings: dict[str, float] = {}
        for name, stage in self.stages():
            if name == "rewrite" and not (state.history and self.chat and self.rewrite):
                continue
            start = time.perf_counter()
            stage(state)
            timings[name] = (time.perf_counter() - start) * 1000.0
        return RetrievalResult(question, state.query, state.candidates, timings)

    # -- stages ------------------------------------------------------------
    def _rewrite(self, s: PipelineState) -> None:
        assert self.chat is not None
        s.query = rewrite_query(self.chat, s.history, s.question)

    def _embed_query(self, s: PipelineState) -> None:
        if self.cfg.mode != "keyword":
            s.qvec = self.embedder.embed_query(s.query)

    def _vector(self, s: PipelineState) -> None:
        if self.cfg.mode != "keyword" and s.qvec is not None:
            s.vector_hits = self.store.vector_search(s.qvec, self.cfg.k_candidates)

    def _keyword(self, s: PipelineState) -> None:
        if self.cfg.mode != "vector":
            s.keyword_hits = self.store.keyword_search(s.query, self.cfg.k_candidates)

    def _fuse(self, s: PipelineState) -> None:
        s.fused = rrf_fuse(s.vector_hits, s.keyword_hits, self.cfg.rrf_k)

    def _select(self, s: PipelineState) -> None:
        top = s.fused[: self.cfg.top_k]
        chunks = dict(self.store.get_chunks([row[0] for row in top]))
        out: list[Candidate] = []
        for rid, fused, vrank, vscore, krank, kscore in top:
            chunk = chunks.get(rid)
            if chunk is None:
                continue
            cand = Candidate(rid, chunk, vrank, vscore, krank, kscore, fused)
            cand.trace["fuse"] = {"fused_score": fused}
            cand.trace["select"] = {"rank": len(out) + 1, "mode": self.cfg.mode}
            out.append(cand)
        s.candidates = out
