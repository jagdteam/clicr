"""Turn a follow-up question into a standalone search query."""

from __future__ import annotations

import logging
import re

from clicr.providers.base import ChatModel
from clicr.retrieve.types import Turn

log = logging.getLogger(__name__)

SYSTEM = (
    "You rewrite a follow-up question about a codebase into ONE standalone search query. "
    "Use the conversation to resolve pronouns and references. "
    "Reply with only the query, no quotes and no explanation."
)
MAX_QUERY_CHARS = 300
_PREFIX = re.compile(r"^\s*(?:search\s+)?query\s*:\s*", re.IGNORECASE)


def _clean(text: str) -> str:
    text = _PREFIX.sub("", text.strip()).strip()
    return text.strip("\"'`“”‘’ \n")


def rewrite_query(chat: ChatModel, history: list[Turn], question: str) -> str:
    if not history:
        return question
    lines: list[str] = []
    for turn in history[-3:]:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer[:600]}")
    user = "Conversation:\n" + "\n".join(lines) + f"\n\nFollow-up question: {question}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    query = _clean(chat.complete(messages))
    if not query or len(query) > MAX_QUERY_CHARS:
        log.debug("rewrite rejected (len=%d); using original question", len(query))
        return question
    return query
