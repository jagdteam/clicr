"""Retrieval, abstain, latency and answer metrics for the evaluation.

A retrieved chunk HITS a gold span when it is in the same file and the inclusive line
ranges overlap (sharing one line counts). Retrieval metrics cover answerable questions
only; abstain metrics treat "unanswerable" as the positive class.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from clicr.answer.prompt import NOT_FOUND

Span = tuple[str, int, int]  # (repo-relative path, start_line, end_line), inclusive
KS = (5, 10)
RETRIEVAL_KEYS = ("recall@5", "recall@10", "recall@final", "mrr")


def gold_spans(gold: list[dict]) -> list[Span]:
    return [(g["path"], int(g["start_line"]), int(g["end_line"])) for g in gold]


def overlaps(a: Span, b: Span) -> bool:
    return a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2]


def gold_ranks(ranked: Sequence[Span], gold: Sequence[Span]) -> list[int | None]:
    """For each gold span, the 1-based rank of the first retrieved chunk hitting it."""
    ranks: list[int | None] = []
    for span in gold:
        hit = next((i for i, got in enumerate(ranked, start=1) if overlaps(got, span)), None)
        ranks.append(hit)
    return ranks


def score_retrieval(ranked: Sequence[Span], gold: Sequence[Span]) -> dict:
    """Per-question scores over the final ranked list (after rerank and graph expansion).

    recall@k: gold spans hit within the top k / gold spans (multi-hop counts each span).
    recall@final: the same over the whole final list, graph additions included.
    rr: 1 / rank of the first retrieved chunk that hits any gold span (0 if none).
    """
    ranks = gold_ranks(ranked, gold)
    found = [r for r in ranks if r is not None]
    n = len(ranks)
    out: dict = {"gold_ranks": ranks}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in found if r <= k) / n if n else 0.0
    out["recall@final"] = len(found) / n if n else 0.0
    out["rr"] = 1.0 / min(found) if found else 0.0
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize(records: Sequence[dict]) -> dict:
    out = {key: _mean([r[key] for r in records]) for key in RETRIEVAL_KEYS if key != "mrr"}
    out["mrr"] = _mean([r["rr"] for r in records])
    out["n"] = len(records)
    return out


def aggregate(records: Sequence[dict]) -> dict:
    """Overall and per-category retrieval metrics over the answerable records."""
    answerable = [r for r in records if r["gold_ranks"]]
    categories = sorted({r["category"] for r in answerable})
    return {
        "overall": summarize(answerable),
        "by_category": {
            cat: summarize([r for r in answerable if r["category"] == cat]) for cat in categories
        },
    }


def prf(predicted: Sequence[bool], actual: Sequence[bool]) -> dict:
    """Precision, recall and F1 with True as the positive class."""
    tp = sum(1 for p, a in zip(predicted, actual, strict=True) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual, strict=True) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual, strict=True) if not p and a)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def abstains(signal: float | None, threshold: float) -> bool:
    """The pipeline's rule: abstain below the threshold; no signal means no abstain."""
    return signal is not None and signal < threshold


def calibrate_abstain(signals: Sequence[float | None], unanswerable: Sequence[bool]) -> dict | None:
    """The threshold with the best abstain F1 (ties: the lower threshold), or None.

    Candidates are every distinct signal (abstain on the items strictly below it) plus a
    value just above the highest signal (abstain on everything that has a signal).
    """
    values = sorted({s for s in signals if s is not None})
    if not values or not any(unanswerable):
        return None
    candidates = [*values, math.nextafter(values[-1], math.inf)]
    best: dict | None = None
    for threshold in candidates:  # ascending, and only a strictly better F1 replaces
        scores = prf([abstains(s, threshold) for s in signals], unanswerable)
        if best is None or scores["f1"] > best["f1"]:
            best = {"threshold": threshold, **scores}
    return best


def apply_threshold(records: Sequence[dict], threshold: float) -> list[dict]:
    """Records as if the pipeline had abstained below `threshold`: abstained items find nothing."""
    out = []
    for rec in records:
        if abstains(rec.get("signal"), threshold):
            rec = {**rec, "gold_ranks": [None] * len(rec["gold_ranks"]), "rr": 0.0}
            rec.update({key: 0.0 for key in RETRIEVAL_KEYS if key != "mrr"})
        out.append(rec)
    return out


def latency(timings: Sequence[dict[str, float]]) -> dict:
    """Mean and p95 per stage (over the questions that ran it) and for the stage total."""
    stages = sorted({name for t in timings for name in t})
    out = {}
    for stage in stages:
        values = [t[stage] for t in timings if stage in t]
        out[stage] = {"mean": _mean(values), "p95": float(np.percentile(values, 95))}
    totals = [sum(t.values()) for t in timings]
    out["total"] = {
        "mean": _mean(totals),
        "p95": float(np.percentile(totals, 95)) if totals else 0.0,
    }
    return out


def citation_metrics(
    cited: Sequence[int], invalid: Sequence[int], source_spans: Sequence[Span], gold: Sequence[Span]
) -> dict:
    """citation_accuracy = valid citations whose source hits gold / valid citations (None if 0)."""
    bad = set(invalid)
    valid = [n for n in cited if n not in bad and 1 <= n <= len(source_spans)]
    good = [n for n in valid if any(overlaps(source_spans[n - 1], g) for g in gold)]
    return {
        "n_valid": len(valid),
        "n_invalid": len(cited) - len(valid),
        "citation_accuracy": len(good) / len(valid) if valid else None,
        "cited_any_gold": bool(good),
    }


def is_not_found(answer: str) -> bool:
    """The model gave the "not found" reply (punctuation and case aside)."""
    return NOT_FOUND.rstrip(".").lower() in answer.lower()


def summarize_answers(records: Sequence[dict]) -> dict:
    answerable = [r for r in records if r["category"] != "unanswerable"]
    unanswerable = [r for r in records if r["category"] == "unanswerable"]
    accuracies = [r["citation_accuracy"] for r in answerable if r["citation_accuracy"] is not None]
    judged = {
        key: [r[key] for r in answerable if r.get(key) is not None]
        for key in ("faithfulness", "correctness")
    }
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "citation_accuracy": _mean(accuracies) if accuracies else None,
        "with_valid_citations": _mean([1.0 if r["n_valid"] else 0.0 for r in answerable]),
        "cited_any_gold": _mean([1.0 if r["cited_any_gold"] else 0.0 for r in answerable]),
        "answerable_not_found": _mean([1.0 if r["not_found"] else 0.0 for r in answerable]),
        "unanswerable_not_found": (
            _mean([1.0 if r["not_found"] else 0.0 for r in unanswerable]) if unanswerable else None
        ),
        "faithfulness": _mean(judged["faithfulness"]) if judged["faithfulness"] else None,
        "correctness": _mean(judged["correctness"]) if judged["correctness"] else None,
        "n_judged": sum(1 for r in answerable if r.get("judged")),
        "n_judge_unparsed": sum(
            1
            for r in answerable
            if r.get("judged") and (r.get("faithfulness") is None or r.get("correctness") is None)
        ),
    }
