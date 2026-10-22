"""HyDE: embed a model-drafted snippet that would answer the query, not the query itself."""

from __future__ import annotations

import re

from clicr.providers.base import ChatModel

SYSTEM = (
    "You write a short hypothetical code snippet (at most 15 lines) that would answer a "
    "question about a codebase. Guess plausible names. Reply with only the code."
)
DRAFT_CHARS = 500  # how much of the draft RetrievalResult keeps
_FENCE = re.compile(r"^\s*```[\w+-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    match = _FENCE.match(text)
    return (match.group(1) if match else text).strip()


def draft_snippet(chat: ChatModel, query: str) -> str:
    """The model's hypothetical answer as code ("" when the model returns nothing)."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    return _strip_fences(chat.complete(messages))
