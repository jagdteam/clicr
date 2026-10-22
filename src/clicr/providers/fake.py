"""Deterministic offline providers for tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import numpy as np

from clicr.index.textsplit import search_terms


class FakeEmbedder:
    """Hashing bag-of-words embedder: lexically similar texts get close vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"fake-{dim}"

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for term in search_terms(text).split():
            digest = hashlib.sha1(term.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vec[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class FakeChat:
    name = "fake-chat"
    _ANSWER = "Answer based on [1]."

    def complete(self, messages: list[dict]) -> str:
        return self._ANSWER

    def stream(self, messages: list[dict]) -> Iterator[str]:
        words = self._ANSWER.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
