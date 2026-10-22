"""Turn code text into search terms, splitting identifiers into words."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+")
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _parts(token: str) -> list[str]:
    return [p.lower() for p in _PART.findall(token)]


def search_terms(text: str) -> str:
    """Original lowercase tokens plus their identifier parts, space separated."""
    out: list[str] = []
    for token in _TOKEN.findall(text):
        out.append(token.lower())
        parts = _parts(token)
        if len(parts) > 1:
            out.extend(parts)
    return " ".join(out)
