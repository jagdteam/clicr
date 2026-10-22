"""Ablation rows: cumulative, ordered, and rewrite only on follow-ups."""

from __future__ import annotations

import pytest

from clicr.eval import configs
from clicr.retrieve.types import Turn

ORDER = ["baseline", "syntax", "+header", "+hybrid", "+rerank", "+graph", "+hyde", "+rewrite"]
SWITCHES = ("chunker", "header", "mode", "rerank", "graph", "hyde", "rewrite")


def test_rows_are_in_order():
    assert [row.name for row in configs.ROWS] == ORDER


def test_baseline_is_the_old_design():
    base = configs.ROWS[0]
    assert (base.chunker, base.header, base.mode) == ("line", False, "vector")
    assert (base.rerank, base.graph, base.hyde, base.rewrite) == ("none", False, False, False)


def test_rows_are_cumulative_one_change_each():
    rows = [row.as_dict() for row in configs.ROWS]
    for before, after in zip(rows, rows[1:], strict=False):
        changed = [k for k in SWITCHES if before[k] != after[k]]
        assert len(changed) == 1, (before, after)
    last = configs.ROWS[-1]
    assert (last.chunker, last.header) == ("syntax", True)
    assert (last.mode, last.rerank) == ("hybrid", "cross")
    assert last.graph and last.hyde and last.rewrite


def test_only_hyde_and_rewrite_need_chat():
    needs = [row.name for row in configs.ROWS if row.needs_chat]
    assert needs == ["+hyde", "+rewrite"]


def test_get_rows_keeps_canonical_order_and_rejects_unknown():
    assert [r.name for r in configs.get_rows(["+graph", "baseline"])] == ["baseline", "+graph"]
    assert [r.name for r in configs.get_rows(None)] == ORDER
    with pytest.raises(ValueError, match="nope"):
        configs.get_rows(["nope"])


def test_rewrite_history_only_for_follow_up_items():
    follow = {"category": "follow_up", "history": [{"q": "what opens it?", "a": "bind()."}]}
    lookup = {"category": "lookup", "history": []}
    rewrite = configs.ROWS[-1]
    hyde = configs.ROWS[-2]
    assert configs.history_for(rewrite, follow) == [Turn("what opens it?", "bind().")]
    assert configs.history_for(rewrite, lookup) == []
    assert configs.history_for(hyde, follow) == []  # earlier rows use the raw question


def test_search_config_follows_the_row():
    cfg = configs.search_config(configs.ROWS[5], top_k=10)
    assert (cfg.mode, cfg.rerank, cfg.graph, cfg.hyde) == ("hybrid", "cross", True, False)
    assert cfg.top_k == 10 and cfg.k_candidates >= 10
    assert cfg.abstain_threshold == float("-inf")  # signal recorded, never abstains


def test_embedders_listed():
    assert configs.EMBEDDERS == (
        "ollama:embeddinggemma",
        "ollama:qwen3-embedding:0.6b",
        "fastembed:BAAI/bge-small-en-v1.5",
        "fastembed:sentence-transformers/all-MiniLM-L6-v2",
    )
    assert [e for e in configs.EMBEDDERS if configs.is_cpu_embedder(e)] == list(
        configs.EMBEDDERS[2:]
    )
