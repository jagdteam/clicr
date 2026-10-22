"""Data types shared by the retrieval pipeline and the answer engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from clicr.models import Chunk


@dataclass
class Candidate:
    rowid: int
    chunk: Chunk
    vector_rank: int | None
    vector_score: float | None
    keyword_rank: int | None
    keyword_score: float | None
    fused_score: float
    trace: dict[str, object] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    question: str
    query_used: str
    candidates: list[Candidate]
    timings_ms: dict[str, float] = field(default_factory=dict)
    # every fused candidate considered (for --explain), in final order
    pool: list[Candidate] = field(default_factory=list)
    hyde_draft: str | None = None
    abstained: bool = False
    abstain_signal: float | None = None


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
