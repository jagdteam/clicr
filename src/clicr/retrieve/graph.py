"""Code-graph expansion: add the definitions of names the selected chunks call (1 hop)."""

from __future__ import annotations

from clicr.index.store import IndexStore
from clicr.retrieve.search import Fused
from clicr.retrieve.types import Candidate

MAX_DEFINITIONS = 3  # a name defined in more chunks than this is too ambiguous to follow


def expand(
    store: IndexStore,
    selected: list[Candidate],
    budget: int,
    known: dict[int, Candidate],
    fused: dict[int, Fused],
) -> list[Candidate]:
    """Definitions to add after `selected`, at most `budget` of them.

    Walks the selected candidates in rank order and their refs in call order. `known` maps
    rowids to candidates already built (the fused pool), so a pool entry is reused and its
    trace gains the graph step; `fused` supplies ranks for rowids outside the pool.
    """
    chosen = {c.rowid for c in selected}
    seen: set[str] = set()
    added: list[Candidate] = []
    for source in selected:
        for name in source.chunk.refs:
            if len(added) >= budget:
                return added
            if name in seen:
                continue
            seen.add(name)
            rowids = store.definitions(name)
            if len(rowids) > MAX_DEFINITIONS:
                continue
            for rid in rowids:
                if rid in chosen or len(added) >= budget:
                    continue
                cand = known.get(rid) or _build(store, rid, fused.get(rid))
                if cand is None:
                    continue
                cand.trace["graph"] = {"via": name, "from_rowid": source.rowid}
                chosen.add(rid)
                added.append(cand)
    return added


def _build(store: IndexStore, rowid: int, row: Fused | None) -> Candidate | None:
    found = store.get_chunks([rowid])
    if not found:
        return None
    chunk = found[0][1]
    if row is None:
        return Candidate(rowid, chunk, None, None, None, None, 0.0)
    _, fused, vrank, vscore, krank, kscore = row
    return Candidate(rowid, chunk, vrank, vscore, krank, kscore, fused)
