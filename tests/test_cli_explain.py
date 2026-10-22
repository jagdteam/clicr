from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clicr.answer.prompt import NOT_FOUND
from clicr.cli import app

runner = CliRunner()
FAKE = ["--embed-provider", "fake", "--chat-provider", "fake"]
HEADERS = [
    "path:lines", "symbol", "vector rank", "keyword rank", "fused rank", "rerank score",
    "graph via", "source",
]  # fmt: skip


@pytest.fixture(autouse=True)
def clicr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("CLICR_HOME", str(home))
    for name in ("CHAT_PROVIDER", "EMBED_PROVIDER", "CHUNKER", "RERANK", "GRAPH", "HYDE"):
        monkeypatch.delenv(f"CLICR_{name}", raising=False)
    monkeypatch.delenv("CLICR_ABSTAIN_THRESHOLD", raising=False)
    return home


@pytest.fixture
def indexed(sample_repo: Path) -> Path:
    res = runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"])
    assert res.exit_code == 0, res.output
    return sample_repo


def _cells(line: str) -> list[str]:
    """Table cells of one rendered row (Rich draws Unicode or ASCII borders)."""
    return [cell.strip() for cell in re.split(r"[│┃|]", line.strip().strip("│┃|"))]


def test_ask_explain_table(indexed: Path) -> None:
    res = runner.invoke(
        app,
        ["ask", "where is get_user_by_id defined?", "--path", str(indexed), "--explain", *FAKE],
        env={"COLUMNS": "200"},
    )
    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    header = next(line for line in lines if "path:lines" in line)
    assert _cells(header) == HEADERS
    row = next(line for line in lines if "app.py:7-9" in line)
    assert _cells(row) == ["app.py:7-9", "get_user_by_id", "1", "1", "1", "-", "-", "1"]
    # the table comes before the answer, which cites the same numbering as Sources
    assert res.output.index("path:lines") < res.output.index("Answer based on [1].")
    assert "[1] app.py:" in res.output


def test_ask_without_explain_has_no_table(indexed: Path) -> None:
    res = runner.invoke(app, ["ask", "get user by id", "--path", str(indexed), *FAKE])
    assert res.exit_code == 0, res.output
    assert "path:lines" not in res.output


def test_ask_advanced_flags(indexed: Path) -> None:
    res = runner.invoke(
        app,
        ["ask", "get user by id", "--path", str(indexed), "--rerank", "llm", "--graph",
         "--hyde", "--abstain-threshold", "2.0", "--explain", *FAKE],
        env={"COLUMNS": "200"},
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert NOT_FOUND in res.output  # LLM rerank scores top out at 1.0 < 2.0
    assert "path:lines" in res.output


def test_ask_bad_rerank_choice(indexed: Path) -> None:
    res = runner.invoke(app, ["ask", "hi", "--path", str(indexed), "--rerank", "magic", *FAKE])
    assert res.exit_code != 0
    assert "Traceback" not in res.output


def test_explain_escapes_markup_in_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pages" / "[id]").mkdir(parents=True)
    (repo / "pages" / "[id]" / "view.py").write_text(
        "def render_page(item):\n    return item\n", encoding="utf-8"
    )
    (repo / "x[").mkdir()
    (repo / "x[" / "y].py").write_text("def render_other(item):\n    return item\n", "utf-8")
    res = runner.invoke(app, ["index", str(repo), "--embed-provider", "fake"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app,
        ["ask", "render page", "--path", str(repo), "--explain", *FAKE],
        env={"COLUMNS": "200"},
    )
    assert res.exit_code == 0, res.output
    assert "MarkupError" not in res.output
    assert "pages/[id]/view.py" in res.output.replace("\\", "/")
