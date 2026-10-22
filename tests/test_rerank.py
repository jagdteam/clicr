from __future__ import annotations

from pathlib import Path

import pytest

from clicr.index.store import IndexStore
from clicr.index.updater import update_index
from clicr.ingest.chunkers import SyntaxChunker
from clicr.providers.fake import FakeEmbedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.rerank import CrossEncoderReranker, LLMReranker
from clicr.retrieve.search import SearchConfig

EMB = FakeEmbedder(64)
QUESTION = "get user by id"


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


class ReverseReranker:
    """Scores rise with position, so reranking reverses the fused order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return [float(i) for i in range(len(texts))]


@pytest.fixture
def store(tmp_path: Path, sample_repo: Path):
    chunker = SyntaxChunker()
    st = IndexStore.open(
        tmp_path / "idx", embedder_name=EMB.name, dim=EMB.dim, chunker_name=chunker.name
    )
    update_index(sample_repo, st, chunker, EMB)
    return st


def test_rerank_reverses_order(store) -> None:
    base = Retriever(store, EMB, cfg=SearchConfig(top_k=5)).retrieve(QUESTION)
    assert len(base.candidates) == 5
    fake = ReverseReranker()
    cfg = SearchConfig(top_k=5, rerank="cross", rerank_pool=5)
    res = Retriever(store, EMB, cfg=cfg, reranker=fake).retrieve(QUESTION)
    assert [c.rowid for c in res.candidates] == [c.rowid for c in base.candidates][::-1]
    query, texts = fake.calls[0]
    assert query == QUESTION and len(texts) == 5
    assert all(t.startswith("# path: ") for t in texts)
    top = res.candidates[0]
    assert top.trace["rerank"]["rerank_rank"] == 1
    assert top.trace["rerank"]["rerank_score"] == pytest.approx(4.0)
    assert top.trace["fuse"]["fused_rank"] == 5
    names = list(res.timings_ms)
    assert names.index("fuse") < names.index("rerank") < names.index("select")


def test_rerank_pool_limits_scored_texts(store) -> None:
    fake = ReverseReranker()
    cfg = SearchConfig(top_k=3, rerank="cross", rerank_pool=4)
    res = Retriever(store, EMB, cfg=cfg, reranker=fake).retrieve(QUESTION)
    assert len(fake.calls[0][1]) == 4
    assert len(res.pool) == 4
    assert [c.trace["rerank"]["rerank_rank"] for c in res.candidates] == [1, 2, 3]


def test_rerank_off_is_noop(store) -> None:
    fake = ReverseReranker()
    res = Retriever(store, EMB, cfg=SearchConfig(top_k=5), reranker=fake).retrieve(QUESTION)
    assert fake.calls == []
    assert "rerank" not in res.timings_ms
    assert all("rerank" not in c.trace for c in res.candidates)


def test_llm_rerank_in_pipeline(store) -> None:
    chat = ScriptedChat("2, 1")
    base = Retriever(store, EMB, cfg=SearchConfig(top_k=3)).retrieve(QUESTION)
    cfg = SearchConfig(top_k=3, rerank="llm", rerank_pool=3)
    res = Retriever(store, EMB, chat=chat, cfg=cfg).retrieve(QUESTION)
    ids = [c.rowid for c in base.candidates]
    assert [c.rowid for c in res.candidates] == [ids[1], ids[0], ids[2]]
    assert len(chat.calls) == 1


# -- LLM listwise reranker -----------------------------------------------------
def test_llm_reranker_parses_order() -> None:
    scores = LLMReranker(ScriptedChat("3, 1, 2")).score("q", ["a", "b", "c"])
    assert scores == pytest.approx([2 / 3, 1 / 3, 1.0])


def test_llm_reranker_ignores_bad_numbers_and_appends_missing() -> None:
    scores = LLMReranker(ScriptedChat("Best: 5, 2, 2, 0, 1")).score("q", ["a", "b", "c"])
    # order 2, 1, then the missing 3 in fused order
    assert scores == pytest.approx([2 / 3, 1.0, 1 / 3])


@pytest.mark.parametrize("reply", ["no idea", "", "7 8 9"])
def test_llm_reranker_falls_back_to_fused_order(reply: str) -> None:
    scores = LLMReranker(ScriptedChat(reply)).score("q", ["a", "b", "c"])
    assert scores == pytest.approx([1.0, 2 / 3, 1 / 3])


def test_llm_reranker_prompt_numbers_and_truncates() -> None:
    chat = ScriptedChat("1")
    long = "\n".join(f"line{i}" for i in range(1, 101))
    LLMReranker(chat).score("where is x?", [long, "short"])
    user = chat.calls[0][-1]["content"]
    assert "where is x?" in user
    assert "[1]" in user and "[2]" in user and "short" in user
    assert "line40\n" in user and "line41" not in user


def test_llm_reranker_empty() -> None:
    chat = ScriptedChat("1")
    assert LLMReranker(chat).score("q", []) == []
    assert chat.calls == []


@pytest.mark.fastembed
def test_cross_encoder_real_model() -> None:
    reranker = CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    scores = reranker.score(
        "how do I open a database connection?",
        [
            "def connect_db(url):\n    # open a database connection\n    return pool.connect(url)",
            "def render(template):\n    return template.draw()",
            "The sky is blue on a clear day.",
        ],
    )
    assert len(scores) == 3
    assert scores[0] == max(scores)
