"""Core data model shared by ingest, index and retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A contiguous span of a source file."""

    path: str
    start_line: int
    end_line: int
    text: str
    language: str | None
    symbol: str | None
    kind: str
    # names this chunk calls (code graph); not part of identity or equality
    refs: tuple[str, ...] = field(default=(), compare=False)

    @property
    def id(self) -> str:
        text_hash = hashlib.sha1(self.text.encode("utf-8", "replace")).hexdigest()
        key = f"{self.path}:{self.start_line}:{self.end_line}:{text_hash}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    def embed_text(self, header: bool = True) -> str:
        if not header:
            return self.text
        head = f"# path: {self.path}\n"
        if self.symbol is not None:
            head += f"# symbol: {self.symbol}\n"
        return head + self.text
