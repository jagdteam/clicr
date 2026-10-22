from __future__ import annotations

import json
from pathlib import Path

import pytest

from clicr.index.updater import UpdateStats
from clicr.sessions import SessionTurn, append_turn, load_turns, session_path
from clicr.watch import handle_batch, run_batch, summary_line


@pytest.fixture(autouse=True)
def clicr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("CLICR_HOME", str(home))
    return home


def test_sessions_jsonl_round_trip(clicr_home: Path) -> None:
    first = SessionTurn(
        "q1",
        "rewritten q1",
        "a1 [1]",
        sources=["app.py:1-9"],
        citations={"cited": [1], "invalid": [], "uncited": []},
        timings_ms={"generate": 1.5},
    )
    second = SessionTurn("q2", "q2", "a2")
    append_turn("sid", first)
    append_turn("sid", second)
    path = session_path("sid")
    assert path.parent == clicr_home / "sessions"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[0]["query_used"] == "rewritten q1"
    assert rows[0]["sources"] == ["app.py:1-9"] and rows[0]["timestamp"]
    assert load_turns("sid") == [first, second]
    assert load_turns("missing") == []


def _batch(*paths: Path) -> list[str]:
    return [str(p) for p in paths]


def test_watch_batch_runs_update_once(sample_repo: Path) -> None:
    calls: list[int] = []

    def update() -> UpdateStats:
        calls.append(1)
        return UpdateStats(changed=1)

    stats = handle_batch(
        _batch(sample_repo / "app.py", sample_repo / "util.js"),
        sample_repo,
        sample_repo / ".idx",
        update,
        lambda: set(),
    )
    assert stats == UpdateStats(changed=1) and len(calls) == 1
    assert "1 changed" in summary_line(stats)


def test_watch_batch_skips_index_dir_and_ignored(sample_repo: Path) -> None:
    index_dir = sample_repo / "idx"
    index_dir.mkdir()
    calls: list[int] = []

    def update() -> UpdateStats:
        calls.append(1)
        return UpdateStats()

    noise = _batch(
        index_dir / "index.sqlite",
        sample_repo / "build" / "out.py",  # in .gitignore
        sample_repo / "debug.log",
    )
    assert handle_batch(noise, sample_repo, index_dir, update, lambda: set()) is None
    assert calls == []


def test_watch_batch_deleted_known_file_triggers_update(sample_repo: Path) -> None:
    gone = sample_repo / "pkg" / "inner.py"
    gone.unlink()
    calls: list[int] = []
    handle_batch(
        _batch(gone),
        sample_repo,
        sample_repo / "idx",
        lambda: calls.append(1) or UpdateStats(deleted=1),
        lambda: {"pkg/inner.py"},
    )
    assert calls == [1]


def test_watch_deleted_directory_triggers_update(sample_repo: Path) -> None:
    gone = sample_repo / "pkg"
    (gone / "inner.py").unlink()
    gone.rmdir()  # the event names only the folder
    calls: list[int] = []
    handle_batch(
        _batch(gone),
        sample_repo,
        sample_repo / "idx",
        lambda: calls.append(1) or UpdateStats(deleted=1),
        lambda: {"pkg/inner.py", "app.py"},
    )
    assert calls == [1]


def test_run_batch_survives_filesystem_errors(sample_repo: Path) -> None:
    def vanished() -> UpdateStats:
        raise FileNotFoundError("folder vanished")

    line = run_batch(
        _batch(sample_repo / "app.py"), sample_repo, sample_repo / "idx", vanished, lambda: set()
    )
    assert line is not None and "folder vanished" in line and "Still watching" in line
    ok = run_batch(
        _batch(sample_repo / "app.py"),
        sample_repo,
        sample_repo / "idx",
        lambda: UpdateStats(changed=1),
        lambda: set(),
    )
    assert ok is not None and "1 changed" in ok
    assert run_batch([], sample_repo, sample_repo / "idx", vanished, lambda: set()) is None
