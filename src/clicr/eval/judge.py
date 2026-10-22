"""LLM-as-judge for the answer track.

A fixed prompt scores faithfulness (are the answer's claims supported by the shown sources)
and correctness (does it agree with the reference answer), each 1-5. The caller runs the
judge model at temperature 0. The judge is the same local model family as the answerer,
so its scores are biased; the README says so.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from clicr.answer.prompt import format_source
from clicr.providers.base import ChatModel
from clicr.retrieve.types import Candidate

SYSTEM = (
    "You grade answers to questions about a codebase. You are strict and consistent.\n"
    "Score two things from 1 (worst) to 5 (best):\n"
    "- faithfulness: every claim in the answer is supported by the numbered sources shown. "
    "5 = all claims supported; 1 = mostly unsupported or contradicted. An answer that says "
    "it could not find the information makes no claims: score it 5.\n"
    "- correctness: the answer agrees with the reference answer on the facts that matter. "
    "5 = same facts; 1 = wrong or missing the point.\n"
    "Reply with exactly two lines and nothing else:\n"
    "faithfulness: <1-5>\n"
    "correctness: <1-5>"
)
# the label (optionally "score", markdown bold allowed), ":" or "=", then the whole next
# token (e.g. "4", "10", "4.5", "1-5")
_SCORE = {
    key: re.compile(rf"\b{key}\b(?:\s+score)?[\s*]*[:=]?[\s*]*([^\s,;]+)", re.IGNORECASE)
    for key in ("faithfulness", "correctness")
}
_VALID = re.compile(r"[1-5]")  # a standalone integer 1-5; anything else is unparsed


def judge_messages(
    question: str, sources: Sequence[Candidate], answer: str, reference: str
) -> list[dict]:
    shown = "\n\n".join(format_source(i, c) for i, c in enumerate(sources, start=1))
    user = (
        f"Question: {question}\n\n"
        f"Sources shown to the answerer:\n\n{shown or '(none)'}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        f"Reference answer:\n{reference}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_scores(reply: str) -> dict[str, int | None]:
    """Each score as an int 1-5, or None when missing or not a standalone 1-5 integer."""
    out: dict[str, int | None] = {}
    for key, pattern in _SCORE.items():
        match = pattern.search(reply)
        token = match.group(1).strip("*_").rstrip(".") if match else ""  # "**4**", "4." fine
        out[key] = int(token) if _VALID.fullmatch(token) else None
    return out


def judge_answer(
    chat: ChatModel, question: str, sources: Sequence[Candidate], answer: str, reference: str
) -> dict[str, int | None]:
    return parse_scores(chat.complete(judge_messages(question, sources, answer, reference)))
