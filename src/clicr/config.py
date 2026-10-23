"""Layered settings: defaults < ~/.clicr/config.toml < <repo>/clicr.toml < env < CLI flags."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from clicr.paths import clicr_home

log = logging.getLogger(__name__)


class ConfigError(ValueError):
    """The configuration is invalid or incomplete."""


@dataclass(frozen=True)
class Settings:
    """CliCR settings. Layers: defaults < config.toml < clicr.toml < CLICR_<FIELD> env < flags.

    Attributes:
        chat_provider: "ollama" (local) or "claude" (needs ANTHROPIC_API_KEY).
        chat_model: Ollama chat model name (used when chat_provider is "ollama").
        claude_model: Claude model id (used when chat_provider is "claude").
        embed_provider: "ollama", "fastembed" (CPU) or "fake" (tests only).
        embed_model: Ollama embedding model name (used when embed_provider is "ollama").
        fastembed_model: fastembed model name (used when embed_provider is "fastembed").
        ollama_host: URL of the Ollama server.
        chunker: "syntax" (tree-sitter) or "line" (fixed windows).
        max_lines: largest syntax chunk, in lines (>= 1).
        header: prepend a file/symbol header to each chunk before embedding.
        top_k: how many chunks to retrieve per question (>= 1).
        rerank: rerank the fused candidates: "none", "cross" (cross-encoder on CPU) or "llm".
        rerank_model: fastembed cross-encoder model (used when rerank is "cross").
        rerank_pool: how many fused candidates the reranker scores (>= 1).
        graph: add the definitions of functions the top chunks call (1 hop).
        graph_budget: most chunks the graph expansion may add (>= 0).
        hyde: embed a model-drafted hypothetical snippet instead of the question.
        abstain_threshold: answer "not found" when the best score is below this (unset = off).
        context_tokens: token budget for the sources in the prompt (0 = no packing).
    """

    chat_provider: str = "ollama"  # ollama | claude | fake (tests only)
    chat_model: str = "qwen2.5-coder:7b"
    claude_model: str = "claude-sonnet-5-5"
    embed_provider: str = "ollama"  # ollama | fastembed | fake
    embed_model: str = "embeddinggemma"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    ollama_host: str = "http://127.0.0.1:11434"
    chunker: str = "syntax"  # syntax | line
    max_lines: int = 120
    header: bool = True
    top_k: int = 6
    rerank: str = "none"  # none | cross | llm
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 30
    graph: bool = True
    graph_budget: int = 3
    hyde: bool = False
    abstain_threshold: float | None = None
    context_tokens: int = 5000


_CHOICES = {
    "chat_provider": ("ollama", "claude", "fake"),
    "embed_provider": ("ollama", "fastembed", "fake"),
    "chunker": ("syntax", "line"),
    "rerank": ("none", "cross", "llm"),
}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(name: str, value: Any, kind: type) -> Any:
    if kind is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigError(f"{name}: expected true/false, got {value!r}")
    if kind is int:
        if isinstance(value, bool):
            raise ConfigError(f"{name}: expected an integer, got {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{name}: expected an integer, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}: expected an integer, got {value!r}") from None
    if kind is float:
        if isinstance(value, bool):
            raise ConfigError(f"{name}: expected a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{name}: expected a number, got {value!r}") from None
    return str(value)


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML ({exc})") from exc


def load_settings(repo_root: Path | None = None, overrides: dict[str, Any] | None = None):
    known = {f.name: f.type for f in fields(Settings)}
    kinds = {"str": str, "int": int, "bool": bool, "float | None": float}
    values: dict[str, Any] = {}

    def apply(layer: dict[str, Any], source: str) -> None:
        for key, value in layer.items():
            if value is None:
                continue
            if key not in known:
                log.warning("Ignoring unknown setting %r from %s", key, source)
                continue
            values[key] = _coerce(key, value, kinds[str(known[key])])

    apply(_read_toml(clicr_home() / "config.toml"), "config.toml")
    if repo_root is not None:
        apply(_read_toml(Path(repo_root) / "clicr.toml"), "clicr.toml")
    env = {
        name: os.environ[f"CLICR_{name.upper()}"]
        for name in known
        if os.environ.get(f"CLICR_{name.upper()}", "") != ""
    }
    apply(env, "environment")
    apply(overrides or {}, "command line")

    for name, allowed in _CHOICES.items():
        if name in values and values[name] not in allowed:
            raise ConfigError(f"{name}: must be one of {', '.join(allowed)} (got {values[name]!r})")
    for name in ("top_k", "max_lines", "rerank_pool"):
        if name in values and values[name] < 1:
            raise ConfigError(f"{name}: must be at least 1 (got {values[name]})")
    for name in ("graph_budget", "context_tokens"):
        if name in values and values[name] < 0:
            raise ConfigError(f"{name}: must be at least 0 (got {values[name]})")
    settings = Settings(**values)
    if settings.chat_provider == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError(
            "chat_provider is 'claude' but ANTHROPIC_API_KEY is not set. "
            "Set the environment variable, or use chat_provider = 'ollama'."
        )
    return settings
