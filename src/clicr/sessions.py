"""Chat sessions on disk: one JSONL line per turn, plus a markdown export."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from clicr.paths import clicr_home


@dataclass
class SessionTurn:
    question: str
    query_used: str
    answer: str
    sources: list[str] = field(default_factory=list)  # "path:start-end", in [n] order
    citations: dict[str, list[int]] = field(default_factory=dict)  # cited / invalid / uncited
    timings_ms: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().astimezone().isoformat(timespec="seconds")


def new_session_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def session_path(session_id: str) -> Path:
    return clicr_home() / "sessions" / f"{session_id}.jsonl"


def append_turn(session_id: str, turn: SessionTurn) -> Path:
    path = session_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
    return path


def load_turns(session_id: str) -> list[SessionTurn]:
    path = session_path(session_id)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SessionTurn(**json.loads(line)) for line in lines if line.strip()]


def export_markdown(turns: list[SessionTurn]) -> str:
    out = ["# clicr session", ""]
    for i, turn in enumerate(turns, start=1):
        out += [f"## Q{i}. {turn.question}", "", turn.answer, ""]
        if turn.sources:
            out.append("Sources:")
            out += [f"- [{n}] {src}" for n, src in enumerate(turn.sources, start=1)]
            out.append("")
        cits = turn.citations
        if cits:
            invalid = len(cits.get("invalid", []))
            valid = len(cits.get("cited", [])) - invalid
            out += [f"_citations: {valid} valid, {invalid} invalid_", ""]
    return "\n".join(out)
