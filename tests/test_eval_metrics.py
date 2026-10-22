"""Evaluation metrics on hand-made cases."""

from __future__ import annotations

import math

import pytest

from clicr.eval import metrics

GOLD_A = ("src/a.py", 20, 30)
GOLD_B = ("src/b.py", 5, 9)


def test_overlap_edges():
    assert metrics.overlaps(("src/a.py", 10, 20), GOLD_A)  # touching on line 20 counts
    assert metrics.overlaps(("src/a.py", 30, 40), GOLD_A)  # touching on line 30 counts
    assert not metrics.overlaps(("src/a.py", 10, 19), GOLD_A)  # off by one before
    assert not metrics.overlaps(("src/a.py", 31, 40), GOLD_A)  # off by one after
    assert not metrics.overlaps(("src/other.py", 20, 30), GOLD_A)  # other file
    assert metrics.overlaps(("src/a.py", 22, 25), GOLD_A)  # inside


def test_gold_spans_from_items():
    gold = [{"path": "src/a.py", "start_line": 20, "end_line": 30, "symbol": None}]
    assert metrics.gold_spans(gold) == [GOLD_A]


def test_multi_hop_partial_recall_and_mrr():
    ranked = [("src/x.py", 1, 5)] * 2 + [("src/a.py", 25, 26)] + [("src/y.py", 1, 2)] * 5
    ranked.append(("src/b.py", 1, 5))  # rank 9: hits GOLD_B, beyond @5
    score = metrics.score_retrieval(ranked, [GOLD_A, GOLD_B])
    assert score["gold_ranks"] == [3, 9]
    assert score["recall@5"] == 0.5
    assert score["recall@10"] == 1.0
    assert score["recall@final"] == 1.0
    assert score["rr"] == pytest.approx(1 / 3)


def test_no_hit_scores_zero():
    score = metrics.score_retrieval([("src/x.py", 1, 5)], [GOLD_A])
    assert score["gold_ranks"] == [None]
    assert score["recall@5"] == score["recall@10"] == score["rr"] == 0.0


def test_recall_final_counts_past_ten():
    ranked = [("src/x.py", i, i) for i in range(1, 12)] + [GOLD_A]  # rank 12
    score = metrics.score_retrieval(ranked, [GOLD_A])
    assert score["recall@10"] == 0.0
    assert score["recall@final"] == 1.0


def test_aggregate_skips_unanswerable_and_groups_by_category():
    records = [
        {"category": "lookup", "gold_ranks": [1], "recall@5": 1.0, "recall@10": 1.0,
         "recall@final": 1.0, "rr": 1.0},
        {"category": "lookup", "gold_ranks": [None], "recall@5": 0.0, "recall@10": 0.0,
         "recall@final": 0.0, "rr": 0.0},
        {"category": "multi_hop", "gold_ranks": [2, None], "recall@5": 0.5, "recall@10": 0.5,
         "recall@final": 0.5, "rr": 0.5},
        {"category": "unanswerable", "gold_ranks": [], "recall@5": 0.0, "recall@10": 0.0,
         "recall@final": 0.0, "rr": 0.0},
    ]  # fmt: skip
    agg = metrics.aggregate(records)
    assert agg["overall"]["n"] == 3
    assert agg["overall"]["recall@5"] == pytest.approx(0.5)
    assert agg["overall"]["mrr"] == pytest.approx(0.5)
    assert set(agg["by_category"]) == {"lookup", "multi_hop"}
    assert agg["by_category"]["lookup"]["recall@10"] == pytest.approx(0.5)


def test_abstain_precision_recall_f1():
    predicted = [True, True, False, False, True]
    actual = [True, False, True, False, True]  # True = unanswerable (the positive class)
    m = metrics.prf(predicted, actual)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    empty = metrics.prf([False, False], [True, False])
    assert empty["precision"] == empty["recall"] == empty["f1"] == 0.0


def test_calibrate_picks_best_f1_and_lower_threshold_on_ties():
    # unanswerable items have low signals; one answerable item is lowest of all
    signals = [0.1, 0.2, 0.3, 0.8, 0.9, None]
    unanswerable = [False, True, True, False, False, False]
    best = metrics.calibrate_abstain(signals, unanswerable)
    assert best is not None
    # threshold 0.8 abstains 0.1, 0.2, 0.3 -> P 2/3, R 1, F1 0.8; that beats 0.3 (P 1/2, R 1/2)
    assert best["threshold"] == pytest.approx(0.8)
    assert best["f1"] == pytest.approx(0.8)
    # a tie: 0.4 (P 1, R 1/2) and "abstain on all" (P 1/2, R 1) both give F1 2/3
    tie = metrics.calibrate_abstain([0.1, 0.4, 0.6, 0.7], [True, False, False, True])
    assert tie is not None and tie["threshold"] == pytest.approx(0.4)
    assert tie["f1"] == pytest.approx(2 / 3)


def test_calibrate_can_abstain_on_everything():
    best = metrics.calibrate_abstain([0.4, 0.5], [True, True])
    assert best is not None and best["recall"] == 1.0
    assert best["threshold"] > 0.5 and math.isclose(best["threshold"], 0.5, rel_tol=1e-9)


def test_calibrate_without_signals_or_unanswerable():
    assert metrics.calibrate_abstain([None, None], [True, False]) is None
    assert metrics.calibrate_abstain([0.1, 0.2], [False, False]) is None


def test_apply_threshold_zeroes_abstained_items():
    records = [
        {"category": "lookup", "gold_ranks": [1], "recall@5": 1.0, "recall@10": 1.0,
         "recall@final": 1.0, "rr": 1.0, "signal": 0.1},
        {"category": "lookup", "gold_ranks": [2], "recall@5": 1.0, "recall@10": 1.0,
         "recall@final": 1.0, "rr": 0.5, "signal": 0.9},
    ]  # fmt: skip
    out = metrics.apply_threshold(records, 0.5)
    assert out[0]["recall@5"] == 0.0 and out[0]["rr"] == 0.0 and out[0]["gold_ranks"] == [None]
    assert out[1] == records[1]
    assert records[0]["rr"] == 1.0  # input untouched


def test_latency_mean_p95_and_total():
    timings = [{"vector": float(i), "fuse": 1.0} for i in range(1, 21)]
    lat = metrics.latency(timings)
    assert lat["vector"]["mean"] == pytest.approx(10.5)
    assert lat["vector"]["p95"] == pytest.approx(19.05)
    assert lat["fuse"]["p95"] == pytest.approx(1.0)
    assert lat["total"]["mean"] == pytest.approx(11.5)


def test_citation_accuracy_ignores_invalid_citations():
    sources = [("src/a.py", 18, 22), ("src/z.py", 1, 3), ("src/b.py", 9, 12)]
    m = metrics.citation_metrics(
        cited=[1, 2, 7], invalid=[7], source_spans=sources, gold=[GOLD_A, GOLD_B]
    )
    assert m["n_valid"] == 2 and m["n_invalid"] == 1
    assert m["citation_accuracy"] == pytest.approx(0.5)
    assert m["cited_any_gold"] is True
    none = metrics.citation_metrics(cited=[9], invalid=[9], source_spans=sources, gold=[GOLD_A])
    assert none["citation_accuracy"] is None and none["cited_any_gold"] is False


def test_not_found_detection():
    from clicr.answer.prompt import NOT_FOUND

    assert metrics.is_not_found(NOT_FOUND)
    assert metrics.is_not_found(f"  {NOT_FOUND.rstrip('.')}  ")
    assert not metrics.is_not_found("It is in app.py [1].")
