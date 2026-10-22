"""The one place that decides between rich and plain terminal output."""

from __future__ import annotations

import sys

from rich.console import Console


def make_console(plain: bool | None = None) -> Console:
    """Plain when asked for (--plain) or when stdout is not a terminal (piped, redirected)."""
    if plain or not sys.stdout.isatty():
        return Console(force_terminal=False, no_color=True, highlight=False, emoji=False)
    return Console(highlight=False, emoji=False)


def is_plain(console: Console) -> bool:
    return not console.is_terminal
