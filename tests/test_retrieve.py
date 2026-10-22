from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from clicr.answer.citations import check_citations, parse_citations
from clicr.answer.engine import Answerer, DoneEvent, RetrievedEvent, TokenEvent
from clicr.answer.prompt import NOT_FOUND, SYSTEM_PROMPT, build_messages
from clicr.index.store import IndexStore
from clicr.index.updater import update_index
from clicr.ingest.chunkers import SyntaxChunker
from clicr.models import Chunk
from clicr.providers.fake import FakeChat, FakeEmbedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.rewrite import rewrite_query
from clicr.retrieve.search import SearchConfig, rrf_fuse
from clicr.retrieve.types import Candidate, Turn

EMB = FakeEmbedder(64)
REMINDER = (
    "\n\nAnswer using ONLY the sources above. Put the source number in square brackets "
    "after every claim, like [1]. "
    f'If the sources do not contain the answer, reply exactly "{NOT_FOUND}".'
)


class ScriptedChat:
    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply

    def stream(self, messages: list[dict]):
        yield self.reply


@pytest.fixture
def store(tmp_path: Path, sample_repo: Path):
    chunker = SyntaxChunker(max_lines=120)
    st = IndexStore.open(
        tmp_path / "idx",
        embedder_name=EMB.name,
        dim=EMB.dim,
        chunker_name=chunker.name,
        header=True,
    )
    update_index(sample_repo, st, chunker, EMB)
    return st


# -- RRF ---------------------------------------------------------------------
def test_rrf_math_single_list_item_and_order() -> None:
    vec = [(10, 0.9), (20, 0.8), (30, 0.7)]
    kw = [(20, 5.0), (40, 4.0)]
    rows = rrf_fuse(vec, kw, 60)
    by_id = {r[0]: r for r in rows}
    assert by_id[20][1] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[10][1] == pytest.approx(1 / 61)
    assert by_id[40][1] == pytest.approx(1 / 62)
    assert by_id[40][2:4] == (None, None)
    assert by_id[10][4:6] == (None, None)
    assert [r[0] for r in rows] == [20, 10, 40, 30]


def test_rrf_tie_breaks_on_vector_score_then_rowid() -> None:
    # equal fused (rank 1 in one list each): the one with a vector score wins
    rows = rrf_fuse([(7, 0.5)], [(3, 9.0)], 60)
    assert [r[0] for r in rows] == [7, 3]
    # equal fused and equal vector scores: lower rowid first
    both = rrf_fuse([(2, 0.5), (1, 0.5)], [(1, 3.0), (2, 2.0)], 60)
    assert both[0][1] == pytest.approx(both[1][1])
    assert [r[0] for r in both] == [1, 2]
    # equal fused, different vector scores: higher vector score first
    both = rrf_fuse([(2, 0.9), (1, 0.4)], [(1, 3.0), (2, 2.0)], 60)
    assert [r[0] for r in both] == [2, 1]


# -- pipeline ----------------------------------------------------------------
@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_modes(store, mode: str) -> None:
    r = Retriever(store, EMB, cfg=SearchConfig(mode=mode, top_k=3))
    res = r.retrieve("get user by id")
    assert 0 < len(res.candidates) <= 3
    top = res.candidates[0]
    assert "get_user_by_id" in top.chunk.text
    if mode == "vector":
        assert all(c.keyword_rank is None for c in res.candidates)
        assert top.fused_score == pytest.approx(1 / 61)
    if mode == "keyword":
        assert all(c.vector_rank is None for c in res.candidates)
        assert top.fused_score == pytest.approx(1 / 61)
    if mode == "hybrid":
        assert top.vector_rank == 1 and top.keyword_rank == 1


def test_stage_timings_and_trace(store) -> None:
    res = Retriever(store, EMB).retrieve("get user by id")
    assert list(res.timings_ms) == ["embed_query", "vector", "keyword", "fuse", "select"]
    assert all(v >= 0 for v in res.timings_ms.values())
    assert "select" in res.candidates[0].trace and "fuse" in res.candidates[0].trace


def test_rewrite_stage_runs_with_history(store) -> None:
    chat = ScriptedChat("get_user_by_id")
    r = Retriever(store, EMB, chat=chat)
    res = r.retrieve("where is it used?", [Turn("what is get_user_by_id?", "A function.")])
    assert res.query_used == "get_user_by_id"
    assert list(res.timings_ms)[0] == "rewrite"
    off = Retriever(store, EMB, chat=chat, rewrite=False).retrieve("x", [Turn("a", "b")])
    assert "rewrite" not in off.timings_ms and off.query_used == "x"


# -- rewrite -----------------------------------------------------------------
def test_rewrite_empty_history_passthrough() -> None:
    chat = ScriptedChat("ignored")
    assert rewrite_query(chat, [], "q?") == "q?"
    assert chat.calls == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('"find the parser"', "find the parser"),
        ("Query: find the parser", "find the parser"),
        ("query: 'find the parser'", "find the parser"),
        ("", "orig"),
        ("   ", "orig"),
        ("x" * 301, "orig"),
    ],
)
def test_rewrite_cleaning_and_fallback(reply: str, expected: str) -> None:
    out = rewrite_query(ScriptedChat(reply), [Turn("a", "b")], "orig")
    assert out == expected


def test_rewrite_uses_last_three_turns_and_truncates() -> None:
    chat = ScriptedChat("q")
    turns = [Turn(f"question{i}", "A" * 1000) for i in range(5)]
    rewrite_query(chat, turns, "next")
    user = chat.calls[0][-1]["content"]
    assert "question2" in user and "question1" not in user
    assert "A" * 601 not in user and "A" * 600 in user


# -- prompt ------------------------------------------------------------------
def _cand(path: str, symbol: str | None, lang: str | None = "python") -> Candidate:
    chunk = Chunk(path, 3, 9, "def f(): ...", lang, symbol, "function")
    return Candidate(1, chunk, 1, 0.5, None, None, 0.1)


def test_prompt_numbering_and_fence() -> None:
    msgs = build_messages("why?", [_cand("a.py", "f"), _cand("b.go", None, "go")], None)
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "I couldn't find this in the indexed code." in SYSTEM_PROMPT
    body = msgs[-1]["content"]
    assert "[1] a.py:3-9 (f)\n```python\ndef f(): ...\n```" in body
    assert "[2] b.go:3-9\n```go\n" in body
    assert "Question: why?" in body
    assert body.endswith(REMINDER)


def test_prompt_history_last_three() -> None:
    hist = [Turn(f"q{i}", f"a{i}") for i in range(5)]
    msgs = build_messages("now", [_cand("a.py", "f")], hist)
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert msgs[1]["content"] == "q2"


# -- citations ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See [1].", [1]),
        ("See [2, 1] and [3]", [2, 1, 3]),
        ("Both [1][2] done", [1, 2]),
        ("Range [1-3]", [1, 2, 3]),
        ("Not [x] or [link](http://a.b/1) or [text]", []),
        ("Code `a[1]` and\n```py\nx = y[2]\n```\nthen [3]", [3]),
        ("[2] then [1] then [2]", [2, 1]),
    ],
)
def test_parse_citations(text: str, expected: list[int]) -> None:
    assert parse_citations(text) == expected


def test_check_citations() -> None:
    rep = check_citations("A [1] B [4] C [1].", 3)
    assert rep.cited == [1, 4] and rep.invalid == [4]
    assert rep.uncited == [2, 3] and rep.has_citations
    none = check_citations("I couldn't find this in the indexed code.", 2)
    assert not none.has_citations and none.uncited == [1, 2]


# -- engine ------------------------------------------------------------------
def test_engine_event_order(store) -> None:
    chat = FakeChat()
    answerer = Answerer(Retriever(store, EMB, chat=chat), chat)
    events = list(answerer.ask_stream("get user by id"))
    assert isinstance(events[0], RetrievedEvent)
    assert isinstance(events[-1], DoneEvent)
    middle = events[1:-1]
    assert middle and all(isinstance(e, TokenEvent) for e in middle)
    done = events[-1]
    assert done.answer == "".join(e.text for e in middle) == "Answer based on [1]."
    assert done.citations.cited == [1] and not done.citations.invalid
    assert "generate" in done.timings_ms and "vector" in done.timings_ms
    assert answerer.ask("get user by id").answer == done.answer


def test_engine_no_candidates_skips_chat(store) -> None:
    class ExplodingChat:
        name = "boom"

        def complete(self, messages: list[dict]) -> str:
            raise AssertionError("chat must not be called")

        def stream(self, messages: list[dict]):
            raise AssertionError("chat must not be called")

    class EmptyRetriever:
        def retrieve(self, question, history=None):
            res = Retriever(store, EMB).retrieve(question, history)
            return dataclasses.replace(res, candidates=[])

    events = list(Answerer(EmptyRetriever(), ExplodingChat()).ask_stream("anything"))
    assert isinstance(events[0], RetrievedEvent)
    assert events[1] == TokenEvent(NOT_FOUND)
    done = events[-1]
    assert isinstance(done, DoneEvent) and len(events) == 3
    assert done.answer == NOT_FOUND and not done.citations.has_citations
    assert done.timings_ms["generate"] == 0.0
