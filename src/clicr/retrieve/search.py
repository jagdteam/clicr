"""Search configuration and reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Mode = Literal["vector", "keyword", "hybrid"]
Fused = tuple[int, float, int | None, float | None, int | None, float | None]


@dataclass(frozen=True)
class SearchConfig:
    mode: Mode = "hybrid"
    k_candidates: int = 20
    rrf_k: int = 60
    top_k: int = 6


def rrf_fuse(
    vector_hits: list[tuple[int, float]],
    keyword_hits: list[tuple[int, float]],
    rrf_k: int,
) -> list[Fused]:
    """Fuse two ranked lists. Returns (rowid, fused, vrank, vscore, krank, kscore), best first.

    fused = sum of 1 / (rrf_k + rank) over the lists containing the rowid (ranks are 1-based).
    Ties break on the higher vector score, then the lower rowid.
    """
    vec = {rid: (rank, score) for rank, (rid, score) in enumerate(vector_hits, start=1)}
    kw = {rid: (rank, score) for rank, (rid, score) in enumerate(keyword_hits, start=1)}
    rows: list[Fused] = []
    for rid in vec.keys() | kw.keys():
        vrank, vscore = vec.get(rid, (None, None))
        krank, kscore = kw.get(rid, (None, None))
        fused = sum(1.0 / (rrf_k + r) for r in (vrank, krank) if r is not None)
        rows.append((rid, fused, vrank, vscore, krank, kscore))
    rows.sort(key=lambda r: (-r[1], -(r[3] if r[3] is not None else float("-inf")), r[0]))
    return rows
