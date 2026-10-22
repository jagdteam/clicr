from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from clicr.index.store import IndexStore
from clicr.index.textsplit import search_terms
from clicr.index.updater import update_index
from clicr.ingest.chunkers import LineChunker, SyntaxChunker
from clicr.models import Chunk
from clicr.providers.fake import FakeChat, FakeEmbedder

EMB = FakeEmbedder(64)


def open_store(
    d: Path,
    chunker_name: str = "syntax-120",
    embedder: FakeEmbedder = EMB,
    header: bool = True,
):
    return IndexStore.open(
        d, embedder_name=embedder.name, dim=embedder.dim, chunker_name=chunker_name, header=header
    )


def test_search_terms() -> None:
    assert search_terms("get_user_by_id") == "get_user_by_id get user by id"
    assert search_terms("parseHTTPResponse") == "parsehttpresponse parse http response"
    assert search_terms("Hello") == "hello"


def test_fake_embedder_deterministic_and_similar() -> None:
    a, b, c = EMB.embed_documents(["get user by id", "get_user_by_id", "render the template"])
    assert a.dtype == np.float32 and a.shape == (64,)
    assert np.allclose(np.linalg.norm(a), 1.0)
    assert float(a @ b) > float(a @ c)
    assert np.array_equal(EMB.embed_query("x y"), FakeEmbedder(64).embed_query("x y"))
    assert EMB.embed_documents([]).shape == (0, 64)


def test_fake_chat() -> None:
    chat = FakeChat()
    assert chat.complete([]) == "Answer based on [1]."
    assert "".join(chat.stream([])) == "Answer based on [1]."


def make_chunk(sym: str, text: str, path: str = "a.py") -> Chunk:
    return Chunk(path, 1, 3, text, "python", sym, "function")


def test_store_round_trip(tmp_path: Path) -> None:
    store = open_store(tmp_path / "idx")
    chunks = [
        make_chunk("get_user_by_id", "def get_user_by_id(i):\n    return db.fetch(i)"),
        make_chunk("render", "def render(t):\n    return template.draw(t)", "b.py"),
    ]
    store.upsert_file("a.py", "h1", 1.0, chunks[:1], EMB.embed_documents([chunks[0].text]))
    store.upsert_file("b.py", "h2", 1.0, chunks[1:], EMB.embed_documents([chunks[1].text]))
    assert store.count() == 2
    assert store.file_hashes() == {"a.py": "h1", "b.py": "h2"}
    rowids = [r for r, _ in store.keyword_search("user id", 5)]
    assert [c for _, c in store.get_chunks(rowids)][0] == chunks[0]
    # replacing a file's chunks keeps counts right
    store.upsert_file("a.py", "h3", 2.0, chunks[:1], EMB.embed_documents([chunks[0].text]))
    assert store.count() == 2
    store.delete_file("b.py")
    assert store.count() == 1 and "b.py" not in store.file_hashes()
    assert store.keyword_search("template", 5) == []


def test_keyword_search_special_characters(tmp_path: Path) -> None:
    store = open_store(tmp_path / "idx")
    c = make_chunk("foo", "foo AND bar")
    store.upsert_file("a.py", "h", 1.0, [c], EMB.embed_documents([c.text]))
    for q in ['"foo" AND (', "NEAR(", "*", '""', "a-b:c", ""]:
        assert isinstance(store.keyword_search(q, 3), list)
    assert store.keyword_search('"foo" AND (', 3)


def test_vector_search_ranks_near_duplicate_first(tmp_path: Path) -> None:
    store = open_store(tmp_path / "idx")
    texts = [
        "connect to the database pool",
        "render html template output",
        "parse command line args",
    ]
    chunks = [make_chunk(f"s{i}", t, f"f{i}.py") for i, t in enumerate(texts)]
    for i, c in enumerate(chunks):
        store.upsert_file(c.path, str(i), 1.0, [c], EMB.embed_documents([c.text]))
    top = store.vector_search(EMB.embed_query("connect to the database pool now"), 2)
    assert store.get_chunks([top[0][0]])[0][1].symbol == "s0"
    assert top[0][1] > top[1][1]
    rowids, matrix = store.vector_matrix()
    assert matrix.shape == (3, 64) and len(rowids) == 3
    assert store.vector_matrix()[1] is matrix  # cached
    store.delete_file("f0.py")
    assert store.vector_matrix()[1].shape == (2, 64)


def test_store_compat_and_reset(tmp_path: Path) -> None:
    d = tmp_path / "idx"
    store = open_store(d)
    assert store.compatible
    c = make_chunk("x", "x = 1")
    store.upsert_file("a.py", "h", 1.0, [c], EMB.embed_documents([c.text]))
    store.close()
    other = open_store(d, chunker_name="line-500-50")
    assert not other.compatible
    other.reset()
    assert other.compatible and other.count() == 0 and other.file_hashes() == {}
    other.close()
    assert open_store(d, chunker_name="line-500-50").compatible


def test_updater_add_change_delete(sample_repo: Path, tmp_path: Path) -> None:
    store = open_store(tmp_path / "idx")
    chunker = SyntaxChunker()
    seen: list[tuple[str, int, int]] = []
    s1 = update_index(sample_repo, store, chunker, EMB, progress=lambda *a: seen.append(a))
    assert s1.added == 6 and s1.changed == s1.deleted == s1.unchanged == 0
    assert s1.chunks_embedded == store.count() > 6
    assert seen[-1][1] == seen[-1][2] == 6

    s2 = update_index(sample_repo, store, chunker, EMB)
    assert (s2.added, s2.changed, s2.deleted, s2.unchanged, s2.chunks_embedded) == (0, 0, 0, 6, 0)

    (sample_repo / "util.js").write_text("function only() {}\n", encoding="utf-8")
    (sample_repo / "notes.md").unlink()
    (sample_repo / "new.py").write_text("def new():\n    pass\n", encoding="utf-8")
    s3 = update_index(sample_repo, store, chunker, EMB)
    assert (s3.added, s3.changed, s3.deleted, s3.unchanged) == (1, 1, 1, 4)
    assert "notes.md" not in store.file_hashes()
    assert "new.py" in store.file_hashes()


def test_updater_rebuilds_when_incompatible(sample_repo: Path, tmp_path: Path) -> None:
    d = tmp_path / "idx"
    store = open_store(d)
    update_index(sample_repo, store, SyntaxChunker(), EMB)
    store.close()
    store = open_store(d, chunker_name=LineChunker().name)
    assert not store.compatible
    stats = update_index(sample_repo, store, LineChunker(), EMB)
    assert stats.added == 6 and stats.unchanged == 0
    assert store.compatible


class SpyEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(64)
        self.seen: list[str] = []

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        return super().embed_documents(texts)


def test_updater_embeds_header_text(sample_repo: Path, tmp_path: Path) -> None:
    spy = SpyEmbedder()
    store = open_store(tmp_path / "idx", embedder=spy)
    update_index(sample_repo, store, SyntaxChunker(), spy, header=True)
    assert spy.seen
    assert all(t.startswith("# path: ") for t in spy.seen)
    assert any("# symbol: get_user_by_id\n" in t for t in spy.seen)


def test_header_toggle_forces_rebuild(sample_repo: Path, tmp_path: Path) -> None:
    d = tmp_path / "idx"
    store = open_store(d)
    update_index(sample_repo, store, SyntaxChunker(), EMB, header=True)
    store.close()
    store = open_store(d, header=False)
    assert not store.compatible
    stats = update_index(sample_repo, store, SyntaxChunker(), EMB, header=False)
    assert stats.added == 6
    assert stats.unchanged == 0
    assert stats.chunks_embedded > 0


def test_updater_skips_vanished_file(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Path.read_bytes

    def flaky(self: Path) -> bytes:
        if self.name == "util.js":
            raise FileNotFoundError(self)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    store = open_store(tmp_path / "idx")
    stats = update_index(sample_repo, store, SyntaxChunker(), EMB)
    assert stats.added == 5
    assert "util.js" not in store.file_hashes()


def _all_ids(store: IndexStore) -> set[str]:
    rowids = [int(r) for r in store.vector_matrix()[0]]
    return {c.id for _, c in store.get_chunks(rowids)}


def test_updater_crlf_same_chunks(sample_repo: Path, tmp_path: Path) -> None:
    lf_bytes = (sample_repo / "app.py").read_bytes().replace(b"\r\n", b"\n")
    (sample_repo / "app.py").write_bytes(lf_bytes)
    store_lf = open_store(tmp_path / "lf")
    update_index(sample_repo, store_lf, SyntaxChunker(), EMB)
    (sample_repo / "app.py").write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    store_crlf = open_store(tmp_path / "crlf")
    update_index(sample_repo, store_crlf, SyntaxChunker(), EMB)
    assert _all_ids(store_lf) == _all_ids(store_crlf)


def test_upsert_rolls_back_on_error(tmp_path: Path) -> None:
    store = open_store(tmp_path / "idx")
    good = make_chunk("keep", "keep this text")
    store.upsert_file("a.py", "h", 1.0, [good], EMB.embed_documents([good.text]))
    c1 = make_chunk("new1", "brand new one", "a.py")
    dup = Chunk("a.py", 5, 6, "x", "python", "new2", "function")
    # the duplicate id violates UNIQUE on the third insert, mid-transaction
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_file("a.py", "h2", 2.0, [c1, dup, dup], EMB.embed_documents(["a", "b", "c"]))
    assert store.count() == 1
    assert store.file_hashes() == {"a.py": "h"}
    assert store.get_chunks([1])[0][1].symbol == "keep"
    assert store.keyword_search("keep", 3)
    assert store.keyword_search("brand", 3) == []
