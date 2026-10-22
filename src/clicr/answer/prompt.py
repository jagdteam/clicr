"""Grounded prompt with numbered sources."""

from __future__ import annotations

from clicr.retrieve.types import Candidate, Turn

NOT_FOUND = "I couldn't find this in the indexed code."

SYSTEM_PROMPT = (
    "You answer questions about a codebase using ONLY the numbered sources provided.\n"
    "- Cite every claim with its source number in square brackets, like [1] or [2][3].\n"
    f'- If the sources do not contain the answer, reply exactly "{NOT_FOUND}" and nothing else.\n'
    "- Be concise.\n"
    "- Quote code in fenced code blocks."
)


def format_source(n: int, cand: Candidate) -> str:
    c = cand.chunk
    symbol = f" ({c.symbol})" if c.symbol else ""
    lang = c.language or ""
    return f"[{n}] {c.path}:{c.start_line}-{c.end_line}{symbol}\n```{lang}\n{c.text}\n```"


def build_messages(
    question: str, candidates: list[Candidate], history: list[Turn] | None
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-3:]:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": turn.answer})
    sources = "\n\n".join(format_source(i, c) for i, c in enumerate(candidates, start=1))
    # small local models ignore system rules, so repeat the key ones after the question
    user = (
        f"Sources:\n\n{sources}\n\nQuestion: {question}"
        "\n\nAnswer using ONLY the sources above. Put the source number in square brackets "
        "after every claim, like [1]. "
        f'If the sources do not contain the answer, reply exactly "{NOT_FOUND}".'
    )
    messages.append({"role": "user", "content": user})
    return messages
