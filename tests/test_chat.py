from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from clicr.chat import run_repl
from clicr.cli import _open_answerer, app
from clicr.providers.fake import FakeChat
from clicr.sessions import export_markdown, load_turns, new_session_id

runner = CliRunner()


class SpyChat(FakeChat):
    name = "spy-chat"

    def __init__(self) -> None:
        self.completes = 0

    def complete(self, messages: list[dict]) -> str:
        self.completes += 1  # only the follow-up rewrite calls complete() here
        return super().complete(messages)


@pytest.fixture(autouse=True)
def clicr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("CLICR_HOME", str(home))
    for name in ("CHAT_PROVIDER", "EMBED_PROVIDER", "CHUNKER"):
        monkeypatch.delenv(f"CLICR_{name}", raising=False)
    return home


def _reader(lines: list[str]):
    it = iter(lines)

    def read() -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read


def test_repl_session(sample_repo: Path, tmp_path: Path) -> None:
    assert (
        runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"]).exit_code == 0
    )
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, no_color=True, width=100)
    export = tmp_path / "session.md"
    spy = SpyChat()
    session_id = new_session_id()
    with _open_answerer(str(sample_repo), "hybrid", "fake", None, {"chat_provider": "fake"}) as (
        answerer,
        _settings,
        _root,
    ):
        answerer.chat = answerer.retriever.chat = spy
        state = run_repl(
            answerer,
            console,
            _reader(
                [
                    "where is get_user_by_id defined?",
                    "what does it return?",
                    "/sources",
                    "/model",
                    "/bogus",
                    f"/export {export}",
                    "/clear",
                    "/help",
                    "/exit",
                    "never reached",
                ]
            ),
            session_id,
        )
    text = out.getvalue()
    assert spy.completes == 1  # the first question had no history; the follow-up was rewritten
    assert len(state.turns) == 2 and state.turns[1].query_used != state.turns[1].question
    assert text.count("Answer based on [1].") == 2
    assert text.count("citations: 1 valid, 0 invalid") == 2
    assert text.count("[1] app.py:") >= 3  # two answers, then /sources
    assert "chat model: spy-chat" in text
    assert "Unknown command /bogus" in text and "/help" in text
    assert "Conversation cleared." in text and "/export [path]" in text
    assert state.history == []  # /clear
    saved = export.read_text(encoding="utf-8")
    assert "where is get_user_by_id defined?" in saved and "what does it return?" in saved
    assert len(load_turns(session_id)) == 2  # each turn was appended to the JSONL


def test_repl_ctrl_d_and_ctrl_c_at_prompt() -> None:
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, no_color=True, width=100)
    calls = iter([KeyboardInterrupt, EOFError])

    def read() -> str:
        raise next(calls)

    state = run_repl(None, console, read, "s")  # type: ignore[arg-type]
    assert state.turns == []


def test_repl_ctrl_c_cancels_answer_not_repl(sample_repo: Path) -> None:
    assert (
        runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"]).exit_code == 0
    )

    class Interrupting(FakeChat):
        def stream(self, messages):
            yield "partial "
            raise KeyboardInterrupt

    out = io.StringIO()
    console = Console(file=out, force_terminal=False, no_color=True, width=100)
    with _open_answerer(str(sample_repo), "hybrid", "fake", None, {"chat_provider": "fake"}) as (
        answerer,
        _settings,
        _root,
    ):
        answerer.chat = answerer.retriever.chat = Interrupting()
        state = run_repl(answerer, console, _reader(["where is get_user_by_id defined?"]), "s")
    assert "(answer cancelled)" in out.getvalue()
    assert state.turns == [] and state.history == []


def test_export_markdown_format() -> None:
    from clicr.sessions import SessionTurn

    turn = SessionTurn(
        "q one",
        "q one",
        "The answer [1].",
        sources=["a.py:1-5"],
        citations={"cited": [1], "invalid": [], "uncited": []},
    )
    md = export_markdown([turn])
    assert md.startswith("# clicr session\n")
    assert "## Q1. q one" in md and "The answer [1]." in md
    assert "- [1] a.py:1-5" in md and "_citations: 1 valid, 0 invalid_" in md


def _index(repo: Path) -> None:
    assert runner.invoke(app, ["index", str(repo), "--embed-provider", "fake"]).exit_code == 0


def _repl_once(sample_repo: Path, lines: list[str], overrides: dict | None = None):
    out = io.StringIO()
    console = Console(file=out, force_terminal=False, no_color=True, width=100)
    ov = {"chat_provider": "fake", **(overrides or {})}
    with _open_answerer(str(sample_repo), "hybrid", "fake", None, ov) as (answerer, _s, _r):
        state = run_repl(
            answerer, console, _reader(lines), "s", switch_model=lambda name: FakeChat()
        )
        return out.getvalue(), state, answerer


def test_export_errors_keep_the_repl_running(sample_repo: Path, tmp_path: Path) -> None:
    _index(sample_repo)
    missing = tmp_path / "no" / "such" / "dir" / "s.md"
    text, _, _ = _repl_once(sample_repo, [f"/export {missing}", f"/export {tmp_path}", "/help"])
    assert text.count("Could not save to") == 2
    assert "/export [path]" in text  # /help still ran afterwards


def test_export_expands_user(
    sample_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(sample_repo)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    _repl_once(sample_repo, ["/export ~/out.md"])
    assert (tmp_path / "out.md").is_file()


def test_model_switch_rebuilds_llm_reranker(sample_repo: Path) -> None:
    from clicr.retrieve.rerank import LLMReranker

    _index(sample_repo)
    _, _, answerer = _repl_once(sample_repo, ["/model other"], {"rerank": "llm"})
    assert isinstance(answerer.retriever.reranker, LLMReranker)
    assert answerer.retriever.reranker.chat is answerer.chat is answerer.retriever.chat


def test_chat_with_piped_stdin(sample_repo: Path) -> None:
    _index(sample_repo)
    res = runner.invoke(
        app,
        ["chat", "--path", str(sample_repo), "--embed-provider", "fake", "--chat-provider", "fake"],
        input="where is get_user_by_id defined?" + chr(10) + "/help" + chr(10),
    )
    assert res.exit_code == 0, res.output
    assert "Answer based on [1]." in res.output and "/export [path]" in res.output
    assert "Session saved:" in res.output


def test_chat_without_questions_saves_no_session(sample_repo: Path) -> None:
    _index(sample_repo)
    res = runner.invoke(
        app,
        ["chat", "--path", str(sample_repo), "--embed-provider", "fake", "--chat-provider", "fake"],
        input="",
    )
    assert res.exit_code == 0 and "Session saved" not in res.output
