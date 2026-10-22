"""Provider interfaces."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 vector."""
        ...


class ChatModel(Protocol):
    name: str

    def stream(self, messages: list[dict]) -> Iterator[str]: ...

    def complete(self, messages: list[dict]) -> str: ...
