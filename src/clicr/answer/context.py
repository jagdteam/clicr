"""Pack retrieved chunks into prompt sources under a token budget.

Steps: drop chunks that overlap a better-ranked chunk of the same file, merge same-file
neighbours (touching or at most MERGE_GAP lines apart), keep the best-ranked sources that fit
the budget (always at least one), then group them by file: files by best rank, chunks by line.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from clicr.models import Chunk
from clicr.retrieve.types import Candidate

MERGE_GAP = 2  # lines allowed between two chunks that still merge into one source

LinesOf = Callable[[str], list[str] | None]  # path -> the file's current lines, if available


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


@dataclass
class _Group:
    rank: int  # best (lowest) rank among the members
    members: list[tuple[int, Candidate]]  # (rank, candidate), sorted by start line

    @property
    def start(self) -> int:
        return self.members[0][1].chunk.start_line

    @property
    def end(self) -> int:
        return max(c.chunk.end_line for _, c in self.members)


def _overlaps(a: Chunk, b: Chunk) -> bool:
    return a.path == b.path and a.start_line <= b.end_line and b.start_line <= a.end_line


def _dedupe(candidates: list[Candidate]) -> list[tuple[int, Candidate]]:
    kept: list[tuple[int, Candidate]] = []
    for rank, cand in enumerate(candidates):
        if not any(_overlaps(cand.chunk, other.chunk) for _, other in kept):
            kept.append((rank, cand))
    return kept


def _merge_groups(ranked: list[tuple[int, Candidate]]) -> list[_Group]:
    by_file: dict[str, list[tuple[int, Candidate]]] = {}
    for item in ranked:
        by_file.setdefault(item[1].chunk.path, []).append(item)
    groups: list[_Group] = []
    for items in by_file.values():
        items.sort(key=lambda item: item[1].chunk.start_line)
        current = _Group(items[0][0], [items[0]])
        for rank, cand in items[1:]:
            if cand.chunk.start_line - current.end - 1 <= MERGE_GAP:
                current.members.append((rank, cand))
                current.rank = min(current.rank, rank)
            else:
                groups.append(current)
                current = _Group(rank, [(rank, cand)])
        groups.append(current)
    return groups


def _original_text(group: _Group, lines_of: LinesOf | None) -> str | None:
    """The merged span read from the file, if the file still matches every member."""
    if lines_of is None:
        return None
    lines = lines_of(group.members[0][1].chunk.path)
    if lines is None or group.end > len(lines):
        return None
    for _, cand in group.members:
        c = cand.chunk
        if "\n".join(lines[c.start_line - 1 : c.end_line]) != c.text:
            return None  # file changed since indexing
    return "\n".join(lines[group.start - 1 : group.end])


def _to_source(group: _Group, lines_of: LinesOf | None) -> Candidate:
    if len(group.members) == 1:
        return group.members[0][1]
    best = min(group.members, key=lambda item: item[0])[1]
    text = _original_text(group, lines_of)
    if text is None:
        text = "\n".join(c.chunk.text for _, c in group.members)
    chunk = replace(best.chunk, start_line=group.start, end_line=group.end, text=text, refs=())
    trace = dict(best.trace)
    trace["merged_from"] = [c.rowid for _, c in sorted(group.members, key=lambda m: m[0])]
    return replace(best, chunk=chunk, trace=trace)


def pack(
    candidates: list[Candidate], budget_tokens: int, lines_of: LinesOf | None = None
) -> list[Candidate]:
    """Sources for the prompt, in the order they should be numbered."""
    groups = _merge_groups(_dedupe(candidates))
    sources = sorted(((g.rank, _to_source(g, lines_of)) for g in groups), key=lambda s: s[0])
    kept: list[tuple[int, Candidate]] = []
    used = 0
    for rank, source in sources:
        cost = estimate_tokens(source.chunk.text)
        if kept and used + cost > budget_tokens:
            break
        kept.append((rank, source))
        used += cost
    # group by file: files in order of their best kept rank, chunks by line inside a file
    file_rank: dict[str, int] = {}
    for rank, source in kept:
        file_rank.setdefault(source.chunk.path, rank)
    kept.sort(key=lambda s: (file_rank[s[1].chunk.path], s[1].chunk.start_line))
    return [source for _, source in kept]
