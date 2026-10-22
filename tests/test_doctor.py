import pytest
from typer.testing import CliRunner

import clicr.cli as cli
from clicr.config import Settings

runner = CliRunner()


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("CLICR_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_doctor_ok_exit_zero(monkeypatch):
    monkeypatch.setattr(
        cli, "run_checks", lambda s, r: [("Python", "OK", "3.12"), ("Key", "WARN", "unset")]
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "OK" in result.output and "WARN" in result.output


def test_doctor_fail_exit_one(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", lambda s, r: [("Ollama server", "FAIL", "down")])
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_doctor_fix_downloads(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "_fix_grammars", lambda: called.append(1))
    monkeypatch.setattr(cli, "run_checks", lambda s, r: [("Python", "OK", "x")])
    assert runner.invoke(cli.app, ["doctor", "--fix"]).exit_code == 0
    assert called


def test_run_checks_with_mocked_network(monkeypatch, tmp_path):
    def fake_json(url, timeout=3.0):
        if url.endswith("/api/version"):
            return {"version": "9.9"}
        return {"models": [{"name": "embeddinggemma:latest"}]}

    monkeypatch.setattr(cli, "_http_json", fake_json)
    monkeypatch.setattr(cli, "_missing_grammars", lambda: [])
    checks = {n: (s, d) for n, s, d in cli.run_checks(Settings(), tmp_path)}
    assert checks["Ollama server"][0] == "OK" and "9.9" in checks["Ollama server"][1]
    assert checks["Embedding model"][0] == "OK"
    assert checks["Chat model"][0] == "FAIL"
    assert "ollama pull qwen2.5-coder:7b" in checks["Chat model"][1]
    assert checks["ANTHROPIC_API_KEY"][0] == "WARN"
    assert checks["Index"][0] == "WARN"


def test_run_checks_unreachable(monkeypatch, tmp_path):
    def boom(url, timeout=3.0):
        raise OSError("refused")

    monkeypatch.setattr(cli, "_http_json", boom)
    monkeypatch.setattr(cli, "_missing_grammars", lambda: [])
    checks = {n: s for n, s, _ in cli.run_checks(Settings(), tmp_path)}
    assert checks["Ollama server"] == "FAIL"


def test_doctor_config_error_row_and_continues(monkeypatch):
    from clicr.config import ConfigError

    def bad(root, overrides):
        raise ConfigError("broken toml")

    seen = {}
    monkeypatch.setattr(cli, "load_settings", bad)
    monkeypatch.setattr(cli, "run_checks", lambda s, r: seen.update(s=s) or [("Python", "OK", "x")])
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "Config" in result.output and "broken toml" in result.output
    assert seen["s"] == Settings()


def test_doctor_fix_failure_is_a_row(monkeypatch):
    def boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(cli, "_fix_grammars", boom)
    monkeypatch.setattr(cli, "run_checks", lambda s, r: [("Python", "OK", "x")])
    result = runner.invoke(cli.app, ["doctor", "--fix"])
    assert result.exit_code == 1
    assert "FAIL" in result.output and "offline" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
