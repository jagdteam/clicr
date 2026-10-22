"""Find and validate [n] citations in an answer."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCED = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INLINE = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_CITE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
_MAX_RANGE = 100


@dataclass(frozen=True)
class CitationReport:
    cited: list[int]
    invalid: list[int]
    uncited: list[int]
    has_citations: bool


def _expand(body: str) -> list[int]:
    nums: list[int] = []
    for part in body.split(","):
        bounds = [int(x) for x in re.split(r"\s*[-–]\s*", part.strip())]
        if len(bounds) == 2 and bounds[0] <= bounds[1] <= bounds[0] + _MAX_RANGE:
            nums.extend(range(bounds[0], bounds[1] + 1))
        else:
            nums.extend(bounds)
    return nums


def parse_citations(text: str) -> list[int]:
    """Cited source numbers in order of first appearance (code and links ignored)."""
    text = _FENCED.sub(" ", text)
    text = _INLINE.sub(" ", text)
    text = _LINK.sub(" ", text)
    seen: list[int] = []
    for match in _CITE.finditer(text):
        for n in _expand(match.group(1)):
            if n not in seen:
                seen.append(n)
    return seen


def check_citations(text: str, n_sources: int) -> CitationReport:
    cited = parse_citations(text)
    invalid = [n for n in cited if not 1 <= n <= n_sources]
    uncited = [n for n in range(1, n_sources + 1) if n not in cited]
    return CitationReport(cited, invalid, uncited, bool(cited))
