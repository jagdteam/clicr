"""Turn expected failures into one friendly line plus a hint."""

from __future__ import annotations

import httpx

from clicr.config import ConfigError
from clicr.providers.base import ProviderError


class UserError(Exception):
    """A problem the user can fix (such as a missing index); the message says how."""


def friendly(exc: BaseException, providers: tuple[str, ...] = ()) -> str | None:
    """An 'Error: ...' line plus a hint line; None when the exception is not an expected one.

    `providers` names the configured chat/embed providers: a dropped connection is only
    blamed on Ollama when Ollama is in use.
    """
    if isinstance(exc, UserError):
        return f"Error: {exc}"
    if isinstance(exc, ProviderError):
        return f"Error: {exc}\nHint: run `clicr doctor` to check the providers."
    if isinstance(exc, ConfigError):
        return f"Error: {exc}\nHint: check your clicr settings (CLICR_* variables or config file)."
    if isinstance(exc, ConnectionError | httpx.ConnectError):
        if "ollama" in providers:
            return "Error: Cannot reach Ollama.\nHint: start it with `ollama serve`, then retry."
        named = f" ({', '.join(dict.fromkeys(providers))})" if providers else ""
        return (
            f"Error: Cannot reach the model provider{named}.\n"
            "Hint: check your network and settings, or run `clicr doctor`."
        )
    return None
