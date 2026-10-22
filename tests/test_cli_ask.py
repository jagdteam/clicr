from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clicr.cli import app
from clicr.paths import index_dir_for

runner = CliRunner()


@pytest.fixture(autouse=True)
def clicr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("CLICR_HOME", str(home))
    for name in ("CHAT_PROVIDER", "EMBED_PROVIDER", "CHUNKER"):
        monkeypatch.delenv(f"CLICR_{name}", raising=False)
    return home


def test_index_then_ask(sample_repo: Path, clicr_home: Path) -> None:
    res = runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"])
    assert res.exit_code == 0, res.output
    assert "Index:" in res.output and "Elapsed:" in res.output
    assert (clicr_home / "indexes").is_dir()

    res = runner.invoke(
        app,
        [
            "ask",
            "where is get_user_by_id defined?",
            "--path",
            str(sample_repo),
            "--embed-provider",
            "fake",
            "--chat-provider",
            "fake",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Answer based on [1]." in res.output
    assert "Sources:" in res.output
    assert "[1] app.py:" in res.output
    assert "citations: 1 valid, 0 invalid" in res.output


def test_index_rebuild_and_incremental(sample_repo: Path) -> None:
    args = ["index", str(sample_repo), "--embed-provider", "fake"]
    assert runner.invoke(app, args).exit_code == 0
    again = runner.invoke(app, args)
    assert again.exit_code == 0 and "0 added" in again.output
    rebuilt = runner.invoke(app, [*args, "--rebuild"])
    assert rebuilt.exit_code == 0 and "0 added" not in rebuilt.output


def test_ask_without_index(sample_repo: Path) -> None:
    res = runner.invoke(
        app,
        ["ask", "hi", "--path", str(sample_repo), "--embed-provider", "fake",
         "--chat-provider", "fake"],
    )  # fmt: skip
    assert res.exit_code == 1
    assert "No index for" in res.output and "Run: clicr index" in res.output
    assert "Traceback" not in res.output


def test_ask_uses_stored_index_settings(sample_repo: Path) -> None:
    idx = runner.invoke(
        app,
        ["index", str(sample_repo), "--chunker", "line", "--no-header", "--embed-provider", "fake"],
    )
    assert idx.exit_code == 0, idx.output
    res = runner.invoke(
        app,
        ["ask", "where is get_user_by_id defined?", "--path", str(sample_repo)],
        env={"CLICR_CHAT_PROVIDER": "fake"},
    )
    assert res.exit_code == 0, res.output
    assert "Sources:" in res.output and "citations: 1 valid" in res.output


def test_ask_with_missing_dim_in_meta_fails(sample_repo: Path) -> None:
    assert (
        runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"]).exit_code == 0
    )
    db = index_dir_for(sample_repo.resolve()) / "index.sqlite"
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("DELETE FROM meta WHERE key = 'dim'")
    conn.close()
    res = runner.invoke(
        app,
        ["ask", "hi", "--path", str(sample_repo)],
        env={"CLICR_CHAT_PROVIDER": "fake"},
    )
    assert res.exit_code == 1
    assert "is unreadable or inconsistent" in res.output and "--rebuild" in res.output
    assert "Traceback" not in res.output


def test_ask_with_different_explicit_embedder_fails(sample_repo: Path) -> None:
    assert (
        runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"]).exit_code == 0
    )
    res = runner.invoke(
        app,
        ["ask", "hi", "--path", str(sample_repo), "--embed-model", "other-model"],
        env={"CLICR_CHAT_PROVIDER": "fake"},
    )
    assert res.exit_code == 1
    assert "was built with embedder" in res.output and "clicr index" in res.output
    assert "Traceback" not in res.output
