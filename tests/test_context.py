from __future__ import annotations

from pathlib import Path

import pytest

from clicr.answer.context import estimate_tokens, pack
from clicr.answer.engine import Answerer, DoneEvent, RetrievedEvent
from clicr.index.store import IndexStore
from clicr.index.updater import update_index
from clicr.ingest.chunkers import SyntaxChunker
from clicr.models import Chunk
from clicr.providers.fake import FakeEmbedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.search import SearchConfig
from clicr.retrieve.types import Candidate

EMB = FakeEmbedder(64)
FILE = [f"line {i}" for i in range(1, 101)]  # "line n" sits on line n


def cand(rowid: int, path: str, start: int, end: int, text: str | None = None) -> Candidate:
    body = text if text is not None else "\n".join(FILE[start - 1 : end])
    chunk = Chunk(path, start, end, body, "python", f"s{rowid}", "function")
    c = Candidate(rowid, chunk, rowid, 0.5, None, None, 1.0 / rowid)
    c.trace["select"] = {"rank": rowid}
    return c


def spans(packed: list[Candidate]) -> list[tuple[str, int, int]]:
    return [(c.chunk.path, c.chunk.start_line, c.chunk.end_line) for c in packed]


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_overlap_with_higher_rank_dropped() -> None:
    packed = pack([cand(1, "a.py", 10, 20), cand(2, "a.py", 15, 30)], 10_000)
    assert spans(packed) == [("a.py", 10, 20)]
    assert packed[0].rowid == 1


def test_neighbours_merged_with_correct_lines() -> None:
    # gap of 2 lines (6, 7) merges; gap of 3 lines (11-13) does not
    first, second, far = cand(2, "a.py", 1, 5), cand(1, "a.py", 8, 10), cand(3, "a.py", 14, 15)
    packed = pack([second, first, far], 10_000, lambda path: FILE)
    assert spans(packed) == [("a.py", 1, 10), ("a.py", 14, 15)]
    merged = packed[0]
    assert merged.chunk.text == "\n".join(FILE[0:10])
    assert merged.rowid == 1 and merged.chunk.symbol == "s1"
    assert merged.trace["select"] == {"rank": 1}
    assert merged.trace["merged_from"] == [1, 2]


def test_merge_without_original_lines_concatenates() -> None:
    packed = pack([cand(1, "a.py", 6, 8), cand(2, "a.py", 1, 5)], 10_000)
    assert spans(packed) == [("a.py", 1, 8)]
    assert packed[0].chunk.text == "\n".join(FILE[0:5]) + "\n" + "\n".join(FILE[5:8])


def test_merge_ignores_stale_file_lines() -> None:
    stale = ["changed"] * 100
    packed = pack([cand(1, "a.py", 1, 3), cand(2, "a.py", 5, 6)], 10_000, lambda p: stale)
    assert packed[0].chunk.text == "\n".join(FILE[0:3]) + "\n" + "\n".join(FILE[4:6])


def test_group_by_file_best_rank_first_then_lines() -> None:
    packed = pack(
        [cand(1, "b.py", 50, 60), cand(2, "a.py", 1, 3), cand(3, "b.py", 1, 5),
         cand(4, "c.py", 1, 2)],
        10_000,
    )  # fmt: skip
    assert spans(packed) == [("b.py", 1, 5), ("b.py", 50, 60), ("a.py", 1, 3), ("c.py", 1, 2)]


def test_budget_cut_keeps_best_ranked_and_at_least_one() -> None:
    big = "x" * 400  # 100 tokens each
    cands = [cand(1, "a.py", 50, 50, big), cand(2, "b.py", 1, 1, big), cand(3, "a.py", 1, 1, big)]
    assert spans(pack(cands, 250)) == [("a.py", 50, 50), ("b.py", 1, 1)]
    assert spans(pack(cands, 10)) == [("a.py", 50, 50)]
    assert spans(pack(cands, 0)) == [("a.py", 50, 50)]
    assert pack([], 100) == []


# -- engine integration -------------------------------------------------------------
class RecordingChat:
    name = "rec"

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def complete(self, messages: list[dict]) -> str:
        return "x"

    def stream(self, messages: list[dict]):
        self.messages = messages
        yield "See [1]."


@pytest.fixture
def store(tmp_path: Path, sample_repo: Path):
    chunker = SyntaxChunker()
    st = IndexStore.open(
        tmp_path / "idx", embedder_name=EMB.name, dim=EMB.dim, chunker_name=chunker.name
    )
    update_index(sample_repo, st, chunker, EMB)
    return st


def test_engine_numbers_packed_sources(store, sample_repo: Path) -> None:
    chat = RecordingChat()
    retriever = Retriever(store, EMB, cfg=SearchConfig(top_k=6, context_tokens=10_000))
    events = list(Answerer(retriever, chat, root=sample_repo).ask_stream("get user by id"))
    retrieved, done = events[0], events[-1]
    assert isinstance(retrieved, RetrievedEvent) and isinstance(done, DoneEvent)
    expected = pack(retrieved.result.candidates, 10_000)
    assert spans(retrieved.sources) == spans(done.sources) == spans(expected)
    prompt = chat.messages[-1]["content"]
    for n, c in enumerate(done.sources, start=1):
        assert f"[{n}] {c.chunk.path}:{c.chunk.start_line}-{c.chunk.end_line}" in prompt
    assert f"[{len(done.sources) + 1}] " not in prompt
    assert "pack" in done.timings_ms


def test_engine_context_tokens_zero_disables_packing(store) -> None:
    retriever = Retriever(store, EMB, cfg=SearchConfig(top_k=6, context_tokens=0))
    done = Answerer(retriever, RecordingChat()).ask("get user by id")
    assert [c.rowid for c in done.sources] == [c.rowid for c in done.result.candidates]
