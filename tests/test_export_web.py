from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from clicr.answer.engine import Answerer
from clicr.cli import app
from clicr.export_web import (
    bm25_scores,
    build_bm25,
    export_web,
    fastembed_prefixes,
    fts_tokens,
)
from clicr.index.store import IndexStore, _fts_text
from clicr.index.textsplit import search_terms
from clicr.index.updater import update_index
from clicr.ingest.chunkers import SyntaxChunker
from clicr.paths import index_dir_for
from clicr.providers.fake import FakeChat, FakeEmbedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.search import SearchConfig
from clicr.ui.errors import UserError

EMB = FakeEmbedder(64)
QUERIES = [
    "get_user_by_id",
    "getUserById lookup",
    "where is the Greeter class hello method",
    "fmt Println server start",
    "inner pkg return",
]


@pytest.fixture
def indexed(tmp_path: Path, sample_repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLICR_HOME", str(tmp_path / "home"))
    chunker = SyntaxChunker(max_lines=120)
    store = IndexStore.open(
        index_dir_for(sample_repo),
        embedder_name=EMB.name,
        dim=EMB.dim,
        chunker_name=chunker.name,
        header=True,
    )
    update_index(sample_repo, store, chunker, EMB)
    yield sample_repo, store
    store.close()


def _export(root: Path, out: Path, **kwargs) -> dict:
    kwargs.setdefault("embedder_factory", lambda name: EMB)
    kwargs.setdefault("embedder_ok", lambda name: True)
    kwargs.setdefault("prefixes", lambda model: ("", ""))
    return export_web(root, out, **kwargs)


def _read(out: Path, name: str):
    return json.loads((out / name).read_text(encoding="utf-8"))


def _texts(store: IndexStore) -> list[str]:
    rowids, _ = store.vector_matrix()
    return [search_terms(_fts_text(c)) for _, c in store.get_chunks([int(r) for r in rowids])]


def test_files_json_holds_every_chunked_file(indexed, tmp_path: Path):
    root, _ = indexed
    out = tmp_path / "bundle"
    manifest = _export(root, out, source={"name": "demo", "include": "src/pkg"})
    files = _read(out, "files.json")
    chunks = _read(out, "chunks.json")
    assert set(files) == {c["path"] for c in chunks}
    for c in chunks:
        lines = files[c["path"]].split("\n")
        assert "\n".join(lines[c["start"] - 1 : c["end"]]) == c["text"], c["path"]
    assert manifest["files"]["files.json"] == (out / "files.json").stat().st_size


def test_files_json_normalises_crlf(indexed, tmp_path: Path):
    root, store = indexed
    (root / "crlf.py").write_bytes(b"def crlf():\r\n    return 1\r\n")
    update_index(root, store, SyntaxChunker(max_lines=120), EMB)
    out = tmp_path / "bundle"
    _export(root, out)
    files = _read(out, "files.json")
    assert files["crlf.py"] == "def crlf():\n    return 1\n"
    assert all("\r" not in text for text in files.values())


def test_file_set_and_manifest(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "bundle"
    manifest = _export(root, out)
    assert sorted(p.name for p in out.iterdir()) == [
        "bm25.json",
        "chunks.json",
        "files.json",
        "graph.json",
        "manifest.json",
        "vectors.bin",
    ]
    assert _read(out, "manifest.json") == manifest
    assert manifest["format"] == 1
    assert manifest["embed"] == {
        "fastembed_model": "fake-64",
        "dim": 64,
        "normalized": True,
        "query_prefix": "",
        "doc_prefix": "",
        "quant": "int8/127",
    }
    assert manifest["n_chunks"] == store.count()
    assert manifest["bm25"]["k1"] == 1.2 and manifest["bm25"]["b"] == 0.75
    assert manifest["bm25"]["n_docs"] == store.count()
    assert (manifest["rrf_k"], manifest["k_candidates"], manifest["top_k"]) == (60, 20, 6)
    assert manifest["header"] is True
    assert manifest["source"]["include"] == "." and manifest["source"]["name"] == "sample_repo"
    assert manifest["files"]["vectors.bin"] == (out / "vectors.bin").stat().st_size
    assert "manifest.json" not in manifest["files"]
    assert b": " not in (out / "manifest.json").read_bytes()  # compact separators


def test_vectors_are_int8_and_close_to_float(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "bundle"
    _export(root, out)
    _, matrix = store.vector_matrix()
    raw = np.frombuffer((out / "vectors.bin").read_bytes(), dtype=np.int8)
    assert raw.size == len(matrix) * 64
    quant = raw.reshape(len(matrix), 64).astype(np.float32) / 127.0
    assert np.abs(quant).max() <= 1.0
    norms = np.linalg.norm(quant, axis=1) * np.linalg.norm(matrix, axis=1)
    assert ((quant * matrix).sum(axis=1) / norms).min() >= 0.99


def test_chunks_in_rowid_order_with_include_prefix(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "bundle"
    _export(root, out, source={"name": "demo", "include": "src/pkg"})
    chunks = _read(out, "chunks.json")
    rowids, _ = store.vector_matrix()
    expected = store.get_chunks([int(r) for r in rowids])
    assert [c["i"] for c in chunks] == list(range(len(expected)))
    for entry, (_, chunk) in zip(chunks, expected, strict=True):
        assert entry["path"] == f"src/pkg/{chunk.path}"
        assert (entry["start"], entry["end"]) == (chunk.start_line, chunk.end_line)
        assert (entry["symbol"], entry["lang"], entry["text"]) == (
            chunk.symbol,
            chunk.language,
            chunk.text,
        )
    assert _read(out, "manifest.json")["source"]["include"] == "src/pkg"


def test_bm25_postings_match_a_direct_count(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "bundle"
    _export(root, out)
    bm25 = _read(out, "bm25.json")
    texts = _texts(store)
    assert [len(fts_tokens(t)) for t in texts] == bm25["doc_len"]
    postings = bm25["postings"]
    for word in ("get", "user", "fmt", "println", "greeter"):
        expected = [
            [i, fts_tokens(t).count(word)] for i, t in enumerate(texts) if word in fts_tokens(t)
        ]
        assert expected and postings[word] == expected
    # a key with spaces is a phrase: get_user_by_id and its adjacent parts
    assert "get user by id" in postings
    assert postings == build_bm25(texts)["postings"]


def test_graph_matches_store_tables(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "bundle"
    _export(root, out)
    graph = _read(out, "graph.json")
    rowids, _ = store.vector_matrix()
    position = {int(r): i for i, r in enumerate(rowids)}
    conn = sqlite3.connect(index_dir_for(root) / "index.sqlite")
    expected_refs: dict[str, list[str]] = {}
    for rid, name in conn.execute("SELECT rowid, name FROM refs ORDER BY _rowid_"):
        expected_refs.setdefault(str(position[rid]), []).append(name)
    expected_defs: dict[str, list[int]] = {}
    for name, rid in conn.execute("SELECT name, rowid FROM symbols ORDER BY rowid"):
        expected_defs.setdefault(name, []).append(position[rid])
    conn.close()
    assert expected_refs and graph["refs"] == expected_refs
    assert expected_defs and graph["defs"] == expected_defs
    assert store.definitions("get_user_by_id") == [
        int(rowids[i]) for i in graph["defs"]["get_user_by_id"]
    ]


def test_parity_file(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "bundle"
    items = [
        {"question": "where is get_user_by_id defined", "category": "lookup"},
        {"question": "what colour is the moon", "category": "unanswerable"},
        {"question": "fmt Println server", "category": "lookup"},
        {"question": "template draw", "category": "lookup"},
        {"question": "Greeter hello", "category": "lookup"},
    ]
    _export(root, out, parity_items=items, source={"name": "demo", "include": "lib"})
    parity = _read(out, "parity.json")
    asked = [i["question"] for i in items if i["category"] != "unanswerable"]
    assert [p["question"] for p in parity["items"]] == asked
    retriever = Retriever(store, EMB, None, SearchConfig(), rewrite=False)
    first = parity["items"][0]
    cands = retriever.retrieve(first["question"]).candidates[:6]
    assert first["python_top"] == [
        f"lib/{c.chunk.path}:{c.chunk.start_line}-{c.chunk.end_line}" for c in cands
    ]
    assert first["terms"] == search_terms(first["question"])
    assert len(parity["query_vectors"]) == 3
    assert parity["query_vectors"][0] == [round(float(x), 6) for x in EMB.embed_query(asked[0])]
    assert "parity.json" in _read(out, "manifest.json")["files"]


def test_questions_plain_and_recorded(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "plain"
    _export(root, out, questions=["How do I get a user?", {"question": "Q2", "answer": "A2"}])
    plain = _read(out, "questions.json")
    assert plain[0] == {
        "question": "How do I get a user?",
        "answer": None,
        "sources": [],
        "model": None,
        "recorded_at": None,
    }
    assert plain[1]["answer"] == "A2"

    chat = FakeChat()
    answerer = Answerer(Retriever(store, EMB, chat, SearchConfig()), chat, root=root)
    out2 = tmp_path / "recorded"
    _export(root, out2, questions=["How do I get a user?"], answerer=answerer)
    rec = _read(out2, "questions.json")[0]
    assert rec["answer"] == "Answer based on [1]." and rec["model"] == "fake-chat"
    datetime.fromisoformat(rec["recorded_at"])
    assert len(rec["sources"]) == 1
    assert set(rec["sources"][0]) == {"path", "start", "end"}


def test_size_cap_fails_and_names_files(indexed, tmp_path: Path):
    root, _ = indexed
    with pytest.raises(UserError, match=r"over the 0\.001 MB limit.*vectors\.bin"):
        _export(root, tmp_path / "bundle", max_mb=0.001)


def test_refuses_non_fastembed_index(indexed, tmp_path: Path):
    root, _ = indexed
    with pytest.raises(UserError, match="fastembed"):
        export_web(root, tmp_path / "b", embedder_factory=lambda name: EMB)
    assert not (tmp_path / "b").exists()
    args = ["export-web", "--path", str(root), "--out", str(tmp_path / "c")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "fastembed" in result.output
    assert not (tmp_path / "c").exists()


def test_missing_index_is_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLICR_HOME", str(tmp_path / "home"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(UserError, match="No index"):
        export_web(tmp_path / "empty", tmp_path / "b", embedder_factory=lambda name: EMB)


def test_bm25_reference_matches_fts5_top5(indexed, tmp_path: Path):
    root, store = indexed
    out = tmp_path / "bundle"
    _export(root, out)
    bm25 = _read(out, "bm25.json")
    rowids, _ = store.vector_matrix()
    position = {int(r): i for i, r in enumerate(rowids)}
    for query in QUERIES:
        expected = store.keyword_search(query, 5)
        got = bm25_scores(query, bm25)[:5]
        assert expected, query
        assert [i for i, _ in got] == [position[r] for r, _ in expected], query
        for (_, score), (_, want) in zip(got, expected, strict=True):
            assert score == pytest.approx(want, rel=1e-9)
    assert bm25_scores("zzzznotthere", bm25) == []


def test_fts_tokens_follow_unicode61():
    assert fts_tokens("get_user_by_id Café x2") == ["get", "user", "by", "id", "cafe", "x2"]
    assert fts_tokens("") == []


def test_fastembed_prefixes_none_for_plain_models_refused_for_prompts():
    assert fastembed_prefixes("BAAI/bge-small-en-v1.5") == ("", "")
    assert fastembed_prefixes("sentence-transformers/all-MiniLM-L6-v2") == ("", "")
    with pytest.raises(UserError, match="task prompt"):
        fastembed_prefixes("jinaai/jina-embeddings-v3")
    with pytest.raises(UserError, match="does not list"):
        fastembed_prefixes("nope/nothing")
