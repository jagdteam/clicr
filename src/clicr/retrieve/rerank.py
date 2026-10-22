"""Rerankers: score (query, chunk) pairs to reorder the fused candidates."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

from clicr.paths import clicr_home
from clicr.providers.base import ChatModel, ProviderError

log = logging.getLogger(__name__)

SNIPPET_LINES = 40
SYSTEM = (
    "You rank code snippets by how well they help answer a question about a codebase. "
    "Reply with ONLY the snippet numbers, best first, separated by commas, like: 3, 1, 2"
)
_INT = re.compile(r"\d+")


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text (higher is better)."""
        ...


class CrossEncoderReranker:
    """fastembed cross-encoder on CPU; the model loads on first use."""

    def __init__(self, model: str, cache_dir: Path | None = None) -> None:
        self.model = model
        self.cache_dir = cache_dir if cache_dir is not None else clicr_home() / "models"
        self._encoder = None

    @property
    def _model(self):
        if self._encoder is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise ProviderError("fastembed is not installed. Fix: uv add fastembed") from exc
            self._encoder = TextCrossEncoder(model_name=self.model, cache_dir=str(self.cache_dir))
        return self._encoder

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, texts)]


def _order_scores(order: list[int]) -> list[float]:
    """Turn a best-first order of 0-based indexes into scores: best = 1.0, then evenly down."""
    n = len(order)
    scores = [0.0] * n
    for position, index in enumerate(order):
        scores[index] = (n - position) / n
    return scores


def parse_order(reply: str, n: int) -> list[int] | None:
    """0-based best-first order from a reply naming 1-based numbers, or None if unusable."""
    order: list[int] = []
    for match in _INT.finditer(reply):
        number = int(match.group())
        if 1 <= number <= n and number - 1 not in order:
            order.append(number - 1)
    if not order:
        return None
    return order + [i for i in range(n) if i not in order]


class LLMReranker:
    """Listwise rerank: the chat model orders numbered snippets."""

    def __init__(self, chat: ChatModel) -> None:
        self.chat = chat

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        snippets = "\n\n".join(
            f"[{i}]\n```\n" + "\n".join(text.split("\n")[:SNIPPET_LINES]) + "\n```"
            for i, text in enumerate(texts, start=1)
        )
        user = (
            f"Question: {query}\n\nSnippets:\n\n{snippets}\n\n"
            f"Order snippets 1-{len(texts)} from most to least useful for the question. "
            "Reply with only the numbers."
        )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        reply = self.chat.complete(messages)
        order = parse_order(reply, len(texts))
        if order is None:
            log.debug("LLM rerank reply unparseable (%r); keeping fused order", reply[:200])
            order = list(range(len(texts)))
        return _order_scores(order)
