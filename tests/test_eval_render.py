"""README eval table: rendered from results JSON, idempotent, staleness check."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clicr.eval import report

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "render_eval_table.py"
DEFAULT = "ollama:embeddinggemma"
OTHER = "fastembed:BAAI/bge-small-en-v1.5"


def _row(config: str, embedder: str, r5: float, r10: float, mrr: float, p95: float) -> dict:
    final = r10 + 0.05
    overall = {"recall@5": r5, "recall@10": r10, "recall@final": final, "mrr": mrr, "n": 54}
    return {
        "config": config,
        "embedder": embedder,
        "metrics": {"overall": overall, "by_category": {}},
        "latency": {"total": {"mean": p95 / 2, "p95": p95}},
    }


def _fixture() -> dict:
    rows = [
        _row("baseline", DEFAULT, 0.3, 0.4, 0.25, 10.0),
        _row("syntax", DEFAULT, 0.5, 0.6, 0.4, 11.0),
        _row("+rerank", DEFAULT, 0.7, 0.8, 0.6, 90.5),
        _row("baseline", OTHER, 0.2, 0.3, 0.2, 5.0),
        _row("syntax", OTHER, 0.4, 0.5, 0.3, 6.0),
        _row("+rerank", OTHER, 0.6, 0.7, 0.5, 80.0),
    ]
    return {
        "suite": "flask",
        "commit": "22d924701a6ae2e4cd01e9a15bbaf3946094af65",
        "ref": "3.1.3",
        "models": {"embedders": [DEFAULT, OTHER]},
        "rows": rows,
        "best": {"config": "+rerank", "embedder": DEFAULT},
        "abstain": {
            "config": "+rerank",
            "embedder": DEFAULT,
            "signal": "cross-encoder score",
            "threshold": 0.12349999,
            "precision": 0.75,
            "recall": 0.5,
            "f1": 0.6,
            "n_unanswerable": 6,
        },
        "answers": {
            "chat_model": "ollama:qwen2.5-coder:7b",
            "judge_model": "ollama:qwen2.5-coder:7b",
            "config": "+rerank",
            "embedder": DEFAULT,
            "summary": {
                "citation_accuracy": 0.8,
                "cited_any_gold": 0.7,
                "unanswerable_not_found": 0.5,
                "faithfulness": 4.2,
                "correctness": 3.9,
            },
        },
    }


def test_render_section_tables():
    text = report.render_section([_fixture()], DEFAULT)
    assert "| Config | Recall@5 | Recall@10 | Context recall | MRR | p95 retrieval ms |" in text
    assert "| baseline | 0.300 | 0.400 | 0.450 | 0.250 | 10.0 |" in text
    assert "| +rerank | 0.700 | 0.800 | 0.850 | 0.600 | 90.5 |" in text
    assert "Context recall:" in text  # the one-line note under the table
    assert f"| {OTHER} | 0.600 | 0.700 | 0.500 | 80.0 |" in text  # embedder table on +rerank
    assert "threshold 0.1234" in text and "F1 0.600" in text
    assert "same local model family" in text


def test_render_without_answers_has_no_answer_line():
    data = _fixture()
    data["answers"] = None
    assert "Answer track" not in report.render_section([data], DEFAULT)


def test_replace_section_appends_markers_and_is_idempotent():
    readme = "# CliCR\n\nIntro.\n"
    once = report.replace_section(readme, "TABLE A")
    assert once.startswith(readme)
    assert once.endswith(f"{report.START}\nTABLE A\n{report.END}\n")
    assert report.replace_section(once, "TABLE A") == once
    swapped = report.replace_section(once, "TABLE B")
    assert "TABLE B" in swapped and "TABLE A" not in swapped
    with pytest.raises(ValueError):
        report.replace_section(f"# x\n{report.START}\n", "T")


def _script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_script_renders_idempotently_and_checks_staleness(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "flask.json").write_text(json.dumps(_fixture()), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# CliCR\n\nIntro.\n", encoding="utf-8")
    args = ("--results-dir", str(results), "--readme", str(readme))

    assert _script(*args, "--check").returncode == 1  # no markers yet: stale
    assert _script(*args).returncode == 0
    first = readme.read_bytes()
    assert b"| +rerank | 0.700 | 0.800 | 0.850 |" in first
    assert _script(*args).returncode == 0
    assert readme.read_bytes() == first
    assert _script(*args, "--check").returncode == 0

    data = _fixture()
    data["rows"][0]["metrics"]["overall"]["recall@5"] = 0.35
    (results / "flask.json").write_text(json.dumps(data), encoding="utf-8")
    stale = _script(*args, "--check")
    assert stale.returncode == 1 and "stale" in stale.stdout
    assert readme.read_bytes() == first  # --check never writes


@pytest.mark.parametrize(
    ("threshold", "shown"),
    [(0.12349999, "0.1234"), (0.5, "0.5000"), (-1.23456, "-1.2346"), (0.99999, "0.9999")],
)
def test_threshold_is_shown_rounded_down(threshold: float, shown: str):
    data = _fixture()
    data["abstain"]["threshold"] = threshold
    text = report.render_section([data], DEFAULT)
    assert f"threshold {shown} " in text
