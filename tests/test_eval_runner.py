"""Eval runner end to end on a tiny in-test suite (fake models, no clone, no network)."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from clicr.cli import app
from clicr.eval import command, configs, judge, runner
from clicr.eval.check import compare_results
from clicr.eval.suites import repo_path, source_root
from clicr.index.textsplit import search_terms
from clicr.providers.fake import FakeEmbedder
from clicr.retrieve import hyde, rewrite

USERS = '''"""User helpers."""


def get_user_by_id(user_id):
    """Look up a user by id."""
    return load_record("users", user_id)


def load_record(table, key):
    """Read one record from a table."""
    return {"table": table, "key": key}
'''

SERVER = '''"""HTTP server."""


class Server:
    def start(self, port):
        """Start serving on a port."""
        return bind_socket(port)


def bind_socket(port):
    """Open a listening socket."""
    return ("0.0.0.0", port)
'''

SUITE = {
    "repo": "local",
    "ref": "test",
    "commit": "abcdef1234567890abcdef1234567890abcdef12",
    "include": "src/pkg",
    "license": "MIT",
}

ITEMS = [
    {
        "id": "t-1",
        "question": "Where is get_user_by_id defined?",
        "category": "lookup",
        "gold": [{"path": "src/pkg/users.py", "start_line": 4, "end_line": 6, "symbol": None}],
        "history": [],
        "reference_answer": "In users.py.",
        "notes": "",
    },
    {
        "id": "t-2",
        "question": "How does Server.start open a socket, and how is a user record loaded?",
        "category": "multi_hop",
        "gold": [
            {"path": "src/pkg/server.py", "start_line": 5, "end_line": 7, "symbol": None},
            {"path": "src/pkg/users.py", "start_line": 9, "end_line": 11, "symbol": None},
        ],
        "history": [],
        "reference_answer": "bind_socket and load_record.",
        "notes": "",
    },
    {
        "id": "t-3",
        "question": "Where is it defined?",
        "category": "follow_up",
        "gold": [{"path": "src/pkg/server.py", "start_line": 10, "end_line": 12, "symbol": None}],
        "history": [{"q": "What opens the listening socket?", "a": "bind_socket does."}],
        "reference_answer": "In server.py.",
        "notes": "",
    },
    {
        "id": "t-4",
        "question": "Where are the database migrations?",
        "category": "unanswerable",
        "gold": [],
        "history": [],
        "reference_answer": "There are none.",
        "notes": "",
    },
]

MACHINE = {"cpu": "test-cpu", "gpu": None, "os": "test-os"}
EMBEDDER = "fastembed:fake-model"


class ScriptedChat:
    """Answers each internal prompt the way a cooperative model would; records the calls."""

    name = "scripted-chat"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, messages: list[dict]) -> str:
        system = messages[0]["content"]
        self.calls.append(system)
        if system == rewrite.SYSTEM:
            return "bind_socket listening socket"
        if system == hyde.SYSTEM:
            return "def bind_socket(port):\n    return port"
        if system == judge.SYSTEM:
            return "faithfulness: 4\ncorrectness: 3"
        return ""

    def stream(self, messages: list[dict]):
        yield "It is in [1]."


class OverlapReranker:
    """Scores by shared search terms: deterministic and offline."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        terms = set(search_terms(query).split())
        return [float(len(terms & set(search_terms(t).split()))) for t in texts]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "other").mkdir()
    (root / "src" / "pkg" / "users.py").write_text(USERS, encoding="utf-8")
    (root / "src" / "pkg" / "server.py").write_text(SERVER, encoding="utf-8")
    (root / "other" / "outside.py").write_text("def get_user_by_id():\n    pass\n", "utf-8")
    return root


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch):
    """Stage timings from a counter, so two runs produce the same numbers."""

    def reset() -> None:
        ticks = itertools.count()
        monkeypatch.setattr(time, "perf_counter", lambda: next(ticks) * 0.001)

    reset()
    return reset


def _run(checkout: Path, tmp_path: Path, **kwargs) -> dict:
    kwargs.setdefault("chat", ScriptedChat())
    kwargs.setdefault("rows", list(configs.ROWS))
    return runner.run_suite(
        "tiny",
        SUITE,
        ITEMS,
        source=checkout,
        embedders=[EMBEDDER],
        embedder_factory=lambda name: FakeEmbedder(64),
        reranker=OverlapReranker(),
        index_root=tmp_path / "eval-index",
        machine=MACHINE,
        **kwargs,
    )


def test_path_prefix_mapping(tmp_path: Path):
    assert repo_path("src/flask", "app.py") == "src/flask/app.py"
    assert repo_path("src/flask/", "sub/x.py") == "src/flask/sub/x.py"
    assert repo_path(".", "app.py") == "app.py"
    assert repo_path("", "app.py") == "app.py"
    assert source_root(tmp_path, "src/flask") == tmp_path / "src/flask"
    assert source_root(tmp_path, ".") == tmp_path


def test_results_schema(checkout: Path, tmp_path: Path, fake_clock):
    chat = ScriptedChat()
    results = _run(checkout, tmp_path, chat=chat)
    for key in ("suite", "commit", "created", "clicr_version", "machine", "models", "rows"):
        assert key in results
    assert results["abstain"] is not None and results["answers"] is None
    assert results["suite"] == "tiny" and results["commit"] == SUITE["commit"]
    assert results["machine"] == MACHINE
    assert [r["config"] for r in results["rows"]] == [r.name for r in configs.ROWS]
    for row in results["rows"]:
        assert row["embedder"] == EMBEDDER
        overall = row["metrics"]["overall"]
        assert set(overall) == {"recall@5", "recall@10", "recall@final", "mrr", "n"}
        assert overall["n"] == 3  # answerable only
        assert "unanswerable" not in row["metrics"]["by_category"]
        assert "p95" in row["latency"]["total"]
    assert len(results["items"]) == len(configs.ROWS) * len(ITEMS)
    ranked = [entry for rec in results["items"] for entry in rec["ranked"]]
    assert ranked and all(entry.startswith("src/pkg/") for entry in ranked)
    # mapped paths match gold: the tiny repo fits in top 10, so everything is found
    assert all(r["metrics"]["overall"]["recall@10"] == 1.0 for r in results["rows"])
    # rewrite runs only in the +rewrite row, only for the one follow-up item
    assert chat.calls.count(rewrite.SYSTEM) == 1
    assert chat.calls.count(hyde.SYSTEM) == 2 * len(ITEMS)
    abstain = results["abstain"]
    assert {"threshold", "precision", "recall", "f1", "metrics_at_threshold"} <= set(abstain)


def test_output_is_deterministic(checkout: Path, tmp_path: Path, fake_clock):
    first = _run(checkout, tmp_path)
    fake_clock()
    second = _run(checkout, tmp_path)  # reuses the indexes built by the first run
    assert first["created"] and second["created"]
    a = runner.dump_results({**first, "created": "X"})
    b = runner.dump_results({**second, "created": "X"})
    assert a == b
    out = tmp_path / "results" / "tiny.json"
    runner.write_results(out, first)
    data = out.read_bytes()
    assert b"\r\n" not in data and data.endswith(b"\n")
    loaded = json.loads(data)
    assert list(loaded) == sorted(loaded)  # keys sorted


def test_floats_are_rounded():
    text = runner.dump_results({"b": 1 / 3, "a": [2 / 3, None, float("nan")], "c": True})
    assert json.loads(text) == {"a": [0.6667, None, None], "b": 0.3333, "c": True}


def test_answer_track_with_judge(checkout: Path, tmp_path: Path, fake_clock):
    chat = ScriptedChat()
    results = _run(checkout, tmp_path, chat=chat, answers=True, judge=chat)
    answers = results["answers"]
    assert answers["chat_model"] == "scripted-chat" and answers["judge_model"] == "scripted-chat"
    assert answers["config"] == results["best"]["config"]
    assert [rec["id"] for rec in answers["items"]] == [item["id"] for item in ITEMS]
    judged = [rec for rec in answers["items"] if rec["category"] != "unanswerable"]
    assert all(rec["faithfulness"] == 4 and rec["correctness"] == 3 for rec in judged)
    summary = answers["summary"]
    assert summary["faithfulness"] == 4.0 and summary["correctness"] == 3.0
    for rec in judged:
        if rec["n_valid"]:
            assert rec["citation_accuracy"] is not None


def test_chat_rows_need_a_chat_model(checkout: Path, tmp_path: Path):
    with pytest.raises(ValueError, match=r"\+hyde"):
        _run(checkout, tmp_path, chat=None)


# -- check ------------------------------------------------------------------


def _row(metric: float) -> dict:
    return {
        "config": "baseline",
        "embedder": EMBEDDER,
        "metrics": {"overall": {"recall@5": metric, "mrr": 0.5, "n": 3}, "by_category": {}},
    }


def test_compare_results_tolerance():
    committed = {"commit": "c", "rows": [_row(0.50)]}
    assert all(d.ok for d in compare_results(committed, {"commit": "c", "rows": [_row(0.515)]}))
    misses = [
        d for d in compare_results(committed, {"commit": "c", "rows": [_row(0.55)]}) if not d.ok
    ]
    assert [m.metric for m in misses] == ["overall.recall@5"]
    missing_row = compare_results({"commit": "c", "rows": []}, {"commit": "c", "rows": [_row(0.5)]})
    assert not all(d.ok for d in missing_row)


def _write_suite(tmp_path: Path) -> tuple[Path, Path]:
    suites = tmp_path / "suites.toml"
    lines = [f'{k} = "{v}"' for k, v in SUITE.items()]
    suites.write_text("[suites.tiny]\n" + "\n".join(lines) + "\n", encoding="utf-8")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "tiny.jsonl").write_text(
        "\n".join(json.dumps(item) for item in ITEMS) + "\n", encoding="utf-8"
    )
    return suites, datasets


def test_check_passes_on_fresh_results_and_fails_on_perturbation(
    checkout: Path, tmp_path: Path, fake_clock
):
    suites, datasets = _write_suite(tmp_path)
    hooks = command.Hooks(
        checkout=lambda name, suite: checkout,
        embedder_factory=lambda name: FakeEmbedder(64),
        chat_factory=ScriptedChat,
        judge_factory=ScriptedChat,
        reranker=OverlapReranker(),
        index_root=tmp_path / "eval-index",
        machine=MACHINE,
    )
    console = Console(file=open(tmp_path / "console.txt", "w", encoding="utf-8"), width=200)  # noqa: SIM115
    out = tmp_path / "results" / "tiny.json"

    def run(**flags) -> int:
        args = dict(
            retrieval_only=False,
            embedders=f"{EMBEDDER},ollama:not-on-cpu",
            configs=None,
            answers=False,
            judge=False,
            check=False,
            out=str(out),
        )
        args.update(flags)
        return command.run_command(
            "tiny",
            **args,
            console=console,
            suites_file=suites,
            dataset_dir=datasets,
            hooks=hooks,
        )

    assert run(check=True) == 1  # no committed file yet
    assert run() == 0
    assert out.is_file()
    assert run(check=True) == 0  # only the CPU embedder and the no-chat rows are rerun

    data = json.loads(out.read_text(encoding="utf-8"))
    data["rows"][1]["metrics"]["overall"]["mrr"] -= 0.05
    out.write_text(json.dumps(data), encoding="utf-8")
    assert run(check=True) == 1
    console.file.close()
    log = (tmp_path / "console.txt").read_text(encoding="utf-8")
    assert "overall.mrr" in log


def test_cli_eval_rejects_unknown_suite():
    res = CliRunner().invoke(app, ["eval", "--suite", "no-such-suite"])
    assert res.exit_code == 1
    assert "no-such-suite" in res.output


def test_cli_eval_help_lists_options():
    res = CliRunner().invoke(app, ["eval", "--help"])
    assert res.exit_code == 0
    for flag in ("--suite", "--retrieval-only", "--embedders", "--configs", "--answers"):
        assert flag in res.output
    for flag in ("--judge", "--check", "--out"):
        assert flag in res.output


# -- review fixes ---------------------------------------------------------------


def test_index_dir_changes_with_include_and_fingerprint(tmp_path: Path):
    def key(include: str = "src/flask", fingerprint: str = "fp1") -> Path:
        return runner.index_dir(
            tmp_path, "s", "abcdef123", "e", "syntax-120", True, include, fingerprint
        )

    assert key() == key()
    assert key(include=".") != key()
    assert key(fingerprint="fp2") != key()


def test_code_fingerprint_follows_sources_and_versions(tmp_path: Path):
    source = tmp_path / "chunkers.py"
    source.write_bytes(b"A = 1\n")
    versions = {"tree-sitter": "0.25.2", "tree-sitter-language-pack": "1.22.1"}
    base = runner.code_fingerprint([source], versions)
    assert runner.code_fingerprint([source], versions) == base
    source.write_bytes(b"A = 1\r\n")  # line endings alone do not change it
    assert runner.code_fingerprint([source], versions) == base
    source.write_bytes(b"A = 2\n")
    changed_source = runner.code_fingerprint([source], versions)
    assert changed_source != base
    bumped = {**versions, "tree-sitter": "0.26.0"}
    assert runner.code_fingerprint([source], bumped) != changed_source
    default = runner.code_fingerprint()  # the real ingest code + installed versions
    assert default == runner.code_fingerprint() and len(default) == 40


def test_results_record_the_index_fingerprint(checkout: Path, tmp_path: Path, fake_clock):
    results = _run(checkout, tmp_path, rows=list(configs.ROWS[:1]))
    assert results["index_fingerprint"] == runner.code_fingerprint()


def test_abstain_threshold_is_not_rounded():
    text = runner.dump_results(
        {
            "abstain": {"threshold": 0.123456789, "f1": 0.123456789},
            "answers": {"abstain_threshold": -1.987654321},
        }
    )
    data = json.loads(text)
    assert data["abstain"]["threshold"] == 0.123456789
    assert data["answers"]["abstain_threshold"] == -1.987654321
    assert data["abstain"]["f1"] == 0.1235
