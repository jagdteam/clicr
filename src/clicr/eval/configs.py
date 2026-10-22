"""The ablation matrix: ordered, cumulative rows, and the embedders they run with."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from clicr.retrieve.search import SearchConfig
from clicr.retrieve.types import Turn

EMBEDDERS = (
    "ollama:embeddinggemma",
    "ollama:qwen3-embedding:0.6b",
    "fastembed:BAAI/bge-small-en-v1.5",
    "fastembed:sentence-transformers/all-MiniLM-L6-v2",
)
TOP_K = 10  # final candidates per question, so Recall@10 sees ten chunks


@dataclass(frozen=True)
class Row:
    name: str
    chunker: str  # "line" (old 500-char windows) | "syntax"
    header: bool
    mode: str  # "vector" | "hybrid"
    rerank: str  # "none" | "cross"
    graph: bool
    hyde: bool
    rewrite: bool  # follow_up items only

    @property
    def needs_chat(self) -> bool:
        return self.hyde or self.rewrite

    @property
    def shape(self) -> tuple[str, bool]:
        """What the index depends on: rows with the same shape share an index."""
        return (self.chunker, self.header)

    def as_dict(self) -> dict:
        return asdict(self)


def _build() -> tuple[Row, ...]:
    current = dict(
        chunker="line",
        header=False,
        mode="vector",
        rerank="none",
        graph=False,
        hyde=False,
        rewrite=False,
    )
    steps = [
        ("baseline", {}),
        ("syntax", {"chunker": "syntax"}),
        ("+header", {"header": True}),
        ("+hybrid", {"mode": "hybrid"}),
        ("+rerank", {"rerank": "cross"}),
        ("+graph", {"graph": True}),
        ("+hyde", {"hyde": True}),
        ("+rewrite", {"rewrite": True}),
    ]
    rows = []
    for name, change in steps:
        current = {**current, **change}
        rows.append(Row(name, **current))
    return tuple(rows)


ROWS = _build()
ROW_NAMES = tuple(row.name for row in ROWS)


def get_rows(names: Sequence[str] | None) -> list[Row]:
    """The named rows in matrix order (all rows when `names` is None)."""
    if names is None:
        return list(ROWS)
    unknown = [n for n in names if n not in ROW_NAMES]
    if unknown:
        raise ValueError(f"unknown config(s) {', '.join(unknown)}; known: {', '.join(ROW_NAMES)}")
    return [row for row in ROWS if row.name in names]


def is_cpu_embedder(name: str) -> bool:
    return name.startswith("fastembed:")


def history_for(row: Row, item: dict) -> list[Turn]:
    """History for the rewrite stage: only the +rewrite row, only follow_up items.

    Every other row asks a follow-up as its raw question, with no history.
    """
    if not (row.rewrite and item.get("category") == "follow_up"):
        return []
    return [Turn(turn["q"], turn["a"]) for turn in item.get("history", [])]


def search_config(row: Row, top_k: int = TOP_K, abstain: float | None = None) -> SearchConfig:
    """The pipeline settings for a row.

    abstain=None records the abstain signal without ever abstaining (threshold -inf).
    """
    return SearchConfig(
        mode=row.mode,  # type: ignore[arg-type]
        top_k=top_k,
        k_candidates=max(20, top_k),
        rerank=row.rerank,  # type: ignore[arg-type]
        graph=row.graph,
        hyde=row.hyde,
        abstain_threshold=float("-inf") if abstain is None else abstain,
    )
