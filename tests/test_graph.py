from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from clicr.index.store import IndexStore
from clicr.index.updater import update_index
from clicr.ingest.chunkers import SyntaxChunker
from clicr.models import Chunk
from clicr.providers.fake import FakeEmbedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.search import SearchConfig

EMB = FakeEmbedder(64)
CHUNKER = SyntaxChunker()

HANDLER = """def handler(request):
    data = parse(request)
    common()
    return helper(data)
"""


@pytest.fixture
def graph_repo(tmp_path: Path) -> Path:
    root = tmp_path / "graph_repo"
    root.mkdir()
    (root / "a.py").write_text(HANDLER, encoding="utf-8")
    (root / "b.py").write_text("def helper(x):\n    return x * 2\n", encoding="utf-8")
    (root / "c.py").write_text("def parse(r):\n    return r\n", encoding="utf-8")
    for i in range(4):  # defined in 4 chunks: ambiguous
        (root / f"d{i}.py").write_text("def common():\n    return 1\n", encoding="utf-8")
    return root


def _open(d: Path) -> IndexStore:
    return IndexStore.open(d, embedder_name=EMB.name, dim=EMB.dim, chunker_name=CHUNKER.name)


@pytest.fixture
def store(tmp_path: Path, graph_repo: Path):
    st = _open(tmp_path / "idx")
    update_index(graph_repo, st, CHUNKER, EMB)
    return st


def _rows(tmp_path: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(tmp_path / "idx" / "index.sqlite")
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _rowid_of(store: IndexStore, path: str) -> int:
    rowids = [int(r) for r in store.vector_matrix()[0]]
    return next(rid for rid, c in store.get_chunks(rowids) if c.path == path)


# -- chunker refs --------------------------------------------------------------
@pytest.mark.parametrize(
    ("language", "source", "expected"),
    [
        ("python", "def f():\n    helper(1)\n    self.obj.meth(2)\n", ("helper", "meth")),
        ("javascript", "function f() {\n  helper(1);\n  this.obj.meth(2);\n}\n",
         ("helper", "meth")),
        ("typescript", "function f(): void {\n  helper(1);\n  a.b.meth(2);\n}\n",
         ("helper", "meth")),
        ("tsx", "function f() {\n  helper(1);\n  obj.meth(2);\n}\n", ("helper", "meth")),
        ("go", "package main\n\nfunc f() {\n\thelper(1)\n\ts.obj.Meth(2)\n}\n", ("helper", "Meth")),
        ("rust", "fn f() {\n    helper(1);\n    self.obj.meth(2);\n    Foo::new();\n}\n",
         ("helper", "meth", "new")),
        ("java", "class A {\n  void f() {\n    helper(1);\n    this.obj.meth(2);\n  }\n}\n",
         ("helper", "meth")),
        ("c", "void f() {\n  helper(1);\n  s->fn(2);\n}\n", ("helper", "fn")),
        ("cpp", "void f() {\n  helper(1);\n  p->m(3);\n  ns::fn(4);\n}\n", ("helper", "m", "fn")),
    ],
)  # fmt: skip
def test_chunker_records_call_refs(language: str, source: str, expected: tuple) -> None:
    chunks = CHUNKER.chunk(f"x.{language}", source, language)
    chunk = next(c for c in chunks if "helper(1)" in c.text)
    assert chunk.refs == expected


def test_refs_only_inside_chunk_span_and_deduped() -> None:
    src = "def a():\n    x()\n    x()\n\n\ndef b():\n    y()\n"
    chunks = {c.symbol: c for c in CHUNKER.chunk("m.py", src, "python")}
    assert chunks["a"].refs == ("x",)
    assert chunks["b"].refs == ("y",)


def test_refs_do_not_change_identity() -> None:
    plain = Chunk("a.py", 1, 2, "x()", "python", "f", "function")
    with_refs = Chunk("a.py", 1, 2, "x()", "python", "f", "function", refs=("x",))
    assert plain.id == with_refs.id
    assert plain == with_refs


# -- store tables ----------------------------------------------------------------
def test_symbols_and_refs_rows_after_indexing(store, tmp_path: Path) -> None:
    handler = _rowid_of(store, "a.py")
    helper = _rowid_of(store, "b.py")
    assert store.definitions("helper") == [helper]
    assert len(store.definitions("common")) == 4
    refs = _rows(tmp_path, f"SELECT name FROM refs WHERE rowid = {handler} ORDER BY _rowid_")
    assert [r[0] for r in refs] == ["parse", "common", "helper"]
    assert store.get_chunks([handler])[0][1].refs == ("parse", "common", "helper")


def test_short_symbol_name_for_methods(tmp_path: Path) -> None:
    st = _open(tmp_path / "idx")
    c = Chunk("k.py", 1, 2, "def m(self):\n    pass", "python", "Klass.m", "method")
    st.upsert_file("k.py", "h", 1.0, [c], EMB.embed_documents([c.text]))
    assert st.definitions("m") == [1]
    assert st.definitions("Klass.m") == []


def test_rows_removed_on_delete_and_change(store, graph_repo: Path, tmp_path: Path) -> None:
    (graph_repo / "b.py").write_text("def helper2(x):\n    return x\n", encoding="utf-8")
    (graph_repo / "a.py").unlink()
    update_index(graph_repo, store, CHUNKER, EMB)
    assert store.definitions("helper") == []
    assert store.definitions("helper2") == [_rowid_of(store, "b.py")]
    assert store.definitions("handler") == []
    assert _rows(tmp_path, "SELECT COUNT(*) FROM refs") == [(0,)]
    n_symbols = _rows(tmp_path, "SELECT COUNT(*) FROM symbols")[0][0]
    assert n_symbols == store.count()


def test_reset_clears_graph_tables(store, tmp_path: Path) -> None:
    store.reset()
    assert _rows(tmp_path, "SELECT COUNT(*) FROM refs") == [(0,)]
    assert _rows(tmp_path, "SELECT COUNT(*) FROM symbols") == [(0,)]


def test_index_without_schema_meta_is_rebuilt(store, graph_repo: Path, tmp_path: Path) -> None:
    store.close()
    conn = sqlite3.connect(tmp_path / "idx" / "index.sqlite")
    with conn:
        conn.execute("DELETE FROM meta WHERE key = 'schema'")
    conn.close()
    old = _open(tmp_path / "idx")
    assert not old.compatible
    stats = update_index(graph_repo, old, CHUNKER, EMB)
    assert stats.added == 7 and stats.unchanged == 0
    assert old.compatible
    assert IndexStore.read_meta(tmp_path / "idx")["schema"] == "2"


# -- expansion -------------------------------------------------------------------
def _retrieve(store: IndexStore, **cfg) -> list:
    config = SearchConfig(mode="keyword", top_k=1, **cfg)
    return Retriever(store, EMB, cfg=config).retrieve("handler")


def test_graph_adds_called_definitions(store) -> None:
    res = _retrieve(store, graph=True, graph_budget=3)
    paths = [c.chunk.path for c in res.candidates]
    assert paths == ["a.py", "c.py", "b.py"]  # parse, then helper; common is ambiguous
    first, parse, helper = res.candidates
    assert parse.trace["graph"] == {"via": "parse", "from_rowid": first.rowid}
    assert helper.trace["graph"]["via"] == "helper"
    assert helper.chunk.symbol == "helper"
    assert "graph" not in first.trace
    assert "graph" in res.timings_ms


def test_graph_budget_respected(store) -> None:
    res = _retrieve(store, graph=True, graph_budget=1)
    assert [c.chunk.path for c in res.candidates] == ["a.py", "c.py"]


def test_graph_skips_already_selected(store) -> None:
    res = Retriever(store, EMB, cfg=SearchConfig(mode="keyword", top_k=2, graph=True)).retrieve(
        "handler parse"
    )
    paths = [c.chunk.path for c in res.candidates]
    assert paths.count("c.py") == 1
    assert paths[-1] == "b.py"


def test_graph_off_is_noop(store) -> None:
    res = _retrieve(store)
    assert [c.chunk.path for c in res.candidates] == ["a.py"]
    assert "graph" not in res.timings_ms
