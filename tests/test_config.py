import logging

import pytest

from clicr.config import ConfigError, load_settings


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.setenv("CLICR_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in ("TOP_K", "HEADER", "MAX_LINES", "CHAT_PROVIDER", "EMBED_MODEL"):
        monkeypatch.delenv(f"CLICR_{name}", raising=False)
    (tmp_path / "home").mkdir()
    (tmp_path / "repo").mkdir()


def test_defaults():
    s = load_settings(None, {})
    assert (s.chat_provider, s.embed_model, s.top_k, s.header) == (
        "ollama",
        "embeddinggemma",
        6,
        True,
    )


def test_five_layer_precedence(tmp_path, monkeypatch):
    home, repo = tmp_path / "home", tmp_path / "repo"
    (home / "config.toml").write_text(
        "top_k = 1\nmax_lines = 10\nembed_model = 'a'\nchunker = 'line'\n"
    )
    (repo / "clicr.toml").write_text("top_k = 2\nmax_lines = 20\nembed_model = 'b'\n")
    monkeypatch.setenv("CLICR_TOP_K", "3")
    monkeypatch.setenv("CLICR_MAX_LINES", "30")
    s = load_settings(repo, {"top_k": 4, "max_lines": None})
    assert s.top_k == 4  # override beats env, repo file, home file
    assert s.max_lines == 30  # None override ignored; env beats repo file
    assert s.embed_model == "b"  # repo file beats home file
    assert s.chunker == "line"  # home file beats default


def test_env_typing(monkeypatch):
    monkeypatch.setenv("CLICR_HEADER", "false")
    monkeypatch.setenv("CLICR_TOP_K", "11")
    s = load_settings(None, {})
    assert s.header is False and s.top_k == 11
    monkeypatch.setenv("CLICR_TOP_K", "lots")
    with pytest.raises(ConfigError):
        load_settings(None, {})


def test_unknown_key_warns(tmp_path, caplog):
    (tmp_path / "repo" / "clicr.toml").write_text("bogus = 1\ntop_k = 7\n")
    with caplog.at_level(logging.WARNING):
        s = load_settings(tmp_path / "repo", {})
    assert s.top_k == 7
    assert "bogus" in caplog.text


def test_claude_without_key_errors():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_settings(None, {"chat_provider": "claude"})


def test_claude_with_key_ok(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert load_settings(None, {"chat_provider": "claude"}).chat_provider == "claude"


def test_bad_choice():
    with pytest.raises(ConfigError):
        load_settings(None, {"chunker": "nope"})


def test_new_model_fields_defaults():
    s = load_settings(None, {})
    assert s.claude_model == "claude-sonnet-5-5"
    assert s.fastembed_model == "BAAI/bge-small-en-v1.5"


def test_float_int_rules(tmp_path):
    (tmp_path / "repo" / "clicr.toml").write_text("top_k = 3.5\n")
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "repo", {})
    (tmp_path / "repo" / "clicr.toml").write_text("top_k = 3.0\n")
    assert load_settings(tmp_path / "repo", {}).top_k == 3


@pytest.mark.parametrize("field", ["top_k", "max_lines"])
def test_must_be_positive(field):
    with pytest.raises(ConfigError, match=field):
        load_settings(None, {field: 0})


def test_empty_env_is_unset(monkeypatch):
    monkeypatch.setenv("CLICR_TOP_K", "")
    monkeypatch.setenv("CLICR_HEADER", "")
    s = load_settings(None, {})
    assert s.top_k == 6 and s.header is True


def test_retrieval_switch_defaults():
    s = load_settings(None, {})
    assert (s.rerank, s.graph, s.hyde, s.abstain_threshold) == ("none", True, False, None)
    assert s.rerank_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert (s.rerank_pool, s.graph_budget, s.context_tokens) == (30, 3, 5000)


def test_retrieval_switches_from_env_and_toml(tmp_path, monkeypatch):
    (tmp_path / "repo" / "clicr.toml").write_text('rerank = "cross"\ngraph = true\n')
    monkeypatch.setenv("CLICR_ABSTAIN_THRESHOLD", "0.35")
    monkeypatch.setenv("CLICR_HYDE", "yes")
    s = load_settings(tmp_path / "repo", {})
    assert (s.rerank, s.graph, s.hyde) == ("cross", True, True)
    assert s.abstain_threshold == pytest.approx(0.35)
    assert load_settings(tmp_path / "repo", {"rerank": "llm"}).rerank == "llm"


def test_retrieval_switch_validation(monkeypatch):
    with pytest.raises(ConfigError, match="rerank"):
        load_settings(None, {"rerank": "magic"})
    with pytest.raises(ConfigError, match="abstain_threshold"):
        load_settings(None, {"abstain_threshold": "high"})
    with pytest.raises(ConfigError, match="rerank_pool"):
        load_settings(None, {"rerank_pool": 0})
    with pytest.raises(ConfigError, match="graph_budget"):
        load_settings(None, {"graph_budget": -1})
