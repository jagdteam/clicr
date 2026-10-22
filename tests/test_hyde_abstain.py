from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clicr.answer.engine import Answerer, DoneEvent, RetrievedEvent, TokenEvent
from clicr.answer.prompt import NOT_FOUND
from clicr.index.store import IndexStore
from clicr.index.updater import update_index
from clicr.ingest.chunkers import SyntaxChunker
from clicr.providers.fake import FakeChat, FakeEmbedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.search import SearchConfig

EMB = FakeEmbedder(64)
DRAFT = "def get_user_by_id(user_id):\n    return db.users.get(user_id)"
QUESTION = "how are users looked up?"


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


class ExplodingChat:
    name = "boom"

    def complete(self, messages: list[dict]) -> str:
        raise AssertionError("chat must not be called")

    def stream(self, messages: list[dict]):
        raise AssertionError("chat must not be called")


class SpyEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(64)
        self.docs: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.docs.extend(texts)
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return super().embed_query(text)


@pytest.fixture
def store(tmp_path: Path, sample_repo: Path):
    chunker = SyntaxChunker()
    st = IndexStore.open(
        tmp_path / "idx", embedder_name=EMB.name, dim=EMB.dim, chunker_name=chunker.name
    )
    update_index(sample_repo, st, chunker, EMB)
    return st


@pytest.fixture
def spy_search(store, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    seen: dict[str, list] = {"qvec": [], "keyword": []}
    vector, keyword = store.vector_search, store.keyword_search

    def spy_vector(qvec, k):
        seen["qvec"].append(np.asarray(qvec))
        return vector(qvec, k)

    def spy_keyword(query, k):
        seen["keyword"].append(query)
        return keyword(query, k)

    monkeypatch.setattr(store, "vector_search", spy_vector)
    monkeypatch.setattr(store, "keyword_search", spy_keyword)
    return seen


# -- HyDE ------------------------------------------------------------------------
def test_hyde_embeds_draft_as_document(store, spy_search) -> None:
    spy = SpyEmbedder()
    chat = ScriptedChat(DRAFT)
    res = Retriever(store, spy, chat=chat, cfg=SearchConfig(hyde=True)).retrieve(QUESTION)
    assert spy.docs == [DRAFT] and spy.queries == []
    assert np.array_equal(spy_search["qvec"][0], EMB.embed_documents([DRAFT])[0])
    assert spy_search["keyword"] == [QUESTION]
    assert res.hyde_draft == DRAFT
    assert QUESTION in chat.calls[0][-1]["content"]
    names = list(res.timings_ms)
    assert names.index("hyde") < names.index("embed_query")
    assert "get_user_by_id" in res.candidates[0].chunk.text


def test_hyde_strips_fences_and_truncates_stored_draft(store) -> None:
    body = "x = 1\n" * 200
    spy = SpyEmbedder()
    chat = ScriptedChat(f"```python\n{body}```")
    res = Retriever(store, spy, chat=chat, cfg=SearchConfig(hyde=True)).retrieve(QUESTION)
    assert spy.docs == [body.strip()]
    assert res.hyde_draft == body.strip()[:500]


def test_hyde_without_chat_or_off_uses_query(store, spy_search) -> None:
    for retriever in (
        Retriever(store, SpyEmbedder(), cfg=SearchConfig(hyde=True)),
        Retriever(store, SpyEmbedder(), chat=ScriptedChat(DRAFT), cfg=SearchConfig()),
    ):
        res = retriever.retrieve(QUESTION)
        assert retriever.embedder.queries == [QUESTION] and retriever.embedder.docs == []
        assert "hyde" not in res.timings_ms and res.hyde_draft is None


# -- abstain ---------------------------------------------------------------------
def test_abstain_above_top_score_skips_model(store) -> None:
    cfg = SearchConfig(abstain_threshold=1.01)
    retriever = Retriever(store, EMB, cfg=cfg)
    res = retriever.retrieve("get user by id")
    assert res.abstained
    top_vector = max(c.vector_score for c in res.pool if c.vector_score is not None)
    assert res.abstain_signal == pytest.approx(top_vector)
    assert res.candidates[0].trace["abstain"]["abstained"] is True
    events = list(Answerer(retriever, ExplodingChat()).ask_stream("get user by id"))
    assert isinstance(events[0], RetrievedEvent)
    assert events[1] == TokenEvent(NOT_FOUND)
    done = events[-1]
    assert isinstance(done, DoneEvent) and len(events) == 3
    assert done.answer == NOT_FOUND and done.sources == []


def test_abstain_below_top_score_answers(store) -> None:
    retriever = Retriever(store, EMB, cfg=SearchConfig(abstain_threshold=-1.0))
    res = retriever.retrieve("get user by id")
    assert not res.abstained and res.abstain_signal is not None
    assert "abstain" in res.timings_ms
    done = Answerer(retriever, FakeChat()).ask("get user by id")
    assert done.answer == "Answer based on [1]."


def test_abstain_uses_top_rerank_score(store) -> None:
    class Fixed:
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [0.25] + [0.1] * (len(texts) - 1)

    def run(threshold: float):
        cfg = SearchConfig(rerank="cross", abstain_threshold=threshold)
        return Retriever(store, EMB, cfg=cfg, reranker=Fixed()).retrieve("get user by id")

    above = run(0.3)
    assert above.abstained and above.abstain_signal == pytest.approx(0.25)
    assert not run(0.2).abstained


def test_abstain_without_vector_hits_never_abstains(store) -> None:
    cfg = SearchConfig(mode="keyword", abstain_threshold=1.01)
    res = Retriever(store, EMB, cfg=cfg).retrieve("get user by id")
    assert not res.abstained and res.abstain_signal is None


def test_abstain_off_is_noop(store) -> None:
    res = Retriever(store, EMB).retrieve("get user by id")
    assert not res.abstained and "abstain" not in res.timings_ms


def test_abstain_llm_rerank_uses_vector_cosine(store) -> None:
    class TopOne:  # LLM-style scores: the best candidate is always 1.0
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [1.0] + [0.5] * (len(texts) - 1)

    cfg = SearchConfig(rerank="llm", abstain_threshold=1.01)
    res = Retriever(store, EMB, cfg=cfg, reranker=TopOne()).retrieve("get user by id")
    top_vector = max(c.vector_score for c in res.pool if c.vector_score is not None)
    assert res.abstain_signal == pytest.approx(top_vector) and res.abstain_signal < 1.0
    assert res.abstained
