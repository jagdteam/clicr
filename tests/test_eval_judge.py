"""Judge score parsing: only a standalone integer 1-5 counts."""

from __future__ import annotations

from clicr.eval import judge, metrics


def test_good_reply_parses():
    assert judge.parse_scores("faithfulness: 4\ncorrectness: 3") == {
        "faithfulness": 4,
        "correctness": 3,
    }
    assert judge.parse_scores("Faithfulness = 5.\nCorrectness: 1") == {
        "faithfulness": 5,
        "correctness": 1,
    }


def test_out_of_range_decimal_and_range_echo_are_unparsed():
    assert judge.parse_scores("faithfulness: 10\ncorrectness: 3")["faithfulness"] is None
    assert judge.parse_scores("faithfulness: 4.5\ncorrectness: 3")["faithfulness"] is None
    echo = judge.parse_scores("faithfulness: 1-5\ncorrectness: 1-5")
    assert echo == {"faithfulness": None, "correctness": None}
    assert judge.parse_scores("no scores here") == {"faithfulness": None, "correctness": None}


def _answer(faithfulness, correctness, judged=True) -> dict:
    return {
        "category": "lookup",
        "citation_accuracy": None,
        "n_valid": 0,
        "cited_any_gold": False,
        "not_found": False,
        "judged": judged,
        "faithfulness": faithfulness,
        "correctness": correctness,
    }


def test_unparsed_judge_replies_are_counted():
    records = [_answer(4, 3), _answer(None, 2), _answer(None, None), _answer(None, None, False)]
    summary = metrics.summarize_answers(records)
    assert summary["n_judged"] == 3
    assert summary["n_judge_unparsed"] == 2
    assert summary["faithfulness"] == 4.0
    assert summary["correctness"] == 2.5


def test_parse_scores_tolerates_markdown_and_score_word() -> None:
    reply = "**Faithfulness:** **4**\ncorrectness score: 3."
    assert judge.parse_scores(reply) == {"faithfulness": 4, "correctness": 3}
