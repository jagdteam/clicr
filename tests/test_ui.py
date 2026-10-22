from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from clicr.answer.citations import CitationReport
from clicr.cli import app
from clicr.config import ConfigError
from clicr.models import Chunk
from clicr.providers.base import ProviderError
from clicr.retrieve.types import Candidate
from clicr.ui.errors import UserError, friendly
from clicr.ui.render import source_panels, status_line

runner = CliRunner()
FAKE = ["--embed-provider", "fake", "--chat-provider", "fake"]


@pytest.fixture(autouse=True)
def clicr_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("CLICR_HOME", str(home))
    for name in ("CHAT_PROVIDER", "EMBED_PROVIDER", "CHUNKER"):
        monkeypatch.delenv(f"CLICR_{name}", raising=False)
    return home


def _candidate(n: int, lines: int = 3) -> Candidate:
    text = "\n".join(f"line_{n}_{i} = {i}" for i in range(lines))
    chunk = Chunk(f"mod{n}.py", 10, 10 + lines - 1, text, "python", f"sym{n}", "function")
    return Candidate(n, chunk, None, None, None, None, 0.0)


def _record(width: int = 100) -> Console:
    return Console(record=True, width=width, force_terminal=True, color_system="truecolor")


def test_plain_ask_snapshot(sample_repo: Path) -> None:
    assert (
        runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"]).exit_code == 0
    )
    res = runner.invoke(
        app, ["ask", "where is get_user_by_id defined?", "--path", str(sample_repo), *FAKE]
    )
    assert res.exit_code == 0, res.output
    assert "\x1b[" not in res.output
    assert "Answer based on [1]." in res.output
    assert "Sources:" in res.output and "[1] app.py:" in res.output
    assert "def get_user_by_id" in res.output  # the cited source's code, with line numbers
    assert "citations: 1 valid, 0 invalid" in res.output
    assert "retrieve" in res.output and "generate" in res.output


def test_plain_flag_matches_piped_output(sample_repo: Path) -> None:
    assert (
        runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"]).exit_code == 0
    )
    args = ["ask", "where is get_user_by_id defined?", "--path", str(sample_repo), *FAKE]
    res = runner.invoke(app, [*args, "--plain"])
    assert res.exit_code == 0 and "\x1b[" not in res.output
    assert "citations: 1 valid, 0 invalid" in res.output


def test_source_panels_rich() -> None:
    sources = [_candidate(1), _candidate(2), _candidate(3)]
    console = _record()
    source_panels(console, sources, CitationReport([2], [], [1, 3], True))
    text = console.export_text()
    assert "[2] mod2.py:10-12  sym2" in text
    assert "10" in text and "line_2_0 = 0" in text  # line-numbered code
    assert "line_1_0" not in text  # uncited sources get no panel
    assert "Also retrieved, not cited" in text and "[1] mod1.py:10-12" in text


def test_source_panels_cap_long_chunks() -> None:
    console = _record()
    source_panels(console, [_candidate(1, lines=45)], CitationReport([1], [], [], True))
    text = console.export_text()
    assert "line_1_39" in text and "line_1_40" not in text
    assert "… 5 more lines" in text


def test_status_line_invalid_is_red() -> None:
    console = _record()
    status_line(console, CitationReport([1, 5], [5], [], True), {"embed_query": 10, "generate": 5})
    text = console.export_text(clear=False)
    assert "citations: 1 valid, 1 invalid (invalid: 5)" in text
    assert "retrieve 10 ms, generate 5 ms" in text
    assert "\x1b[31m" in console.export_text(styles=True)


def test_friendly_maps_expected_errors() -> None:
    assert friendly(UserError("No index for x. Run: clicr index x")) == (
        "Error: No index for x. Run: clicr index x"
    )
    assert friendly(ProviderError("boom")).startswith("Error: boom\nHint:")
    assert friendly(ConfigError("bad value")).startswith("Error: bad value\nHint:")
    for exc in (ConnectionError("x"), httpx.ConnectError("x")):
        assert "Cannot reach Ollama" in (friendly(exc, ("fake", "ollama")) or "")
        generic = friendly(exc, ("claude", "fastembed")) or ""
        assert "Cannot reach the model provider (claude, fastembed)" in generic
        assert "Ollama" not in generic
        assert "Cannot reach the model provider." in (friendly(exc) or "")
    assert friendly(ValueError("unexpected")) is None


def test_ask_without_index_is_friendly(sample_repo: Path) -> None:
    res = runner.invoke(app, ["ask", "hi", "--path", str(sample_repo), *FAKE])
    assert res.exit_code == 1
    assert "No index for" in res.output and "Traceback" not in res.output


def test_ask_provider_error_is_friendly(sample_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        runner.invoke(app, ["index", str(sample_repo), "--embed-provider", "fake"]).exit_code == 0
    )

    def boom(*_a, **_k):
        raise ProviderError("model exploded")

    monkeypatch.setattr("clicr.providers.fake.FakeChat.stream", boom)
    res = runner.invoke(app, ["ask", "hi", "--path", str(sample_repo), *FAKE])
    assert res.exit_code == 1
    assert "model exploded" in res.output and "clicr doctor" in res.output
    assert "Traceback" not in res.output


def test_emoji_codes_stay_literal(capsys: pytest.CaptureFixture[str]) -> None:
    from clicr.answer.engine import TokenEvent
    from clicr.ui.console import make_console
    from clicr.ui.render import stream_answer

    cand = _candidate(1)
    cand.chunk = Chunk("m.py", 1, 1, "x = ':pizza:'", "python", None, "function")
    for plain in (True, False):
        console = make_console(plain)
        stream_answer(console, [TokenEvent("eat :pizza: now")], True)
        source_panels(console, [cand], CitationReport([1], [], [], True))
    out = capsys.readouterr().out
    assert out.count("eat :pizza: now") == 2 and out.count("x = ':pizza:'") == 2
    assert "🍕" not in out
