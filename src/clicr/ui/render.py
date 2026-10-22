"""Render a streamed answer, its cited source panels and a status line."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from clicr.answer.citations import CitationReport
from clicr.answer.engine import DoneEvent, Event, RetrievedEvent, TokenEvent
from clicr.retrieve.types import Candidate
from clicr.ui.console import is_plain

MAX_CODE_LINES = 40


@dataclass
class Streamed:
    retrieved: RetrievedEvent | None = None
    done: DoneEvent | None = None


def stream_answer(
    console: Console,
    events: Iterable[Event],
    plain: bool,
    on_retrieved: Callable[[RetrievedEvent], None] | None = None,
) -> Streamed:
    """Show tokens as they arrive: live Markdown, or plain text when `plain`."""
    out = Streamed()
    text = ""
    live = None if plain else Live(console=console, auto_refresh=False, transient=False)
    if live is not None:
        live.start()
    try:
        for event in events:
            if isinstance(event, RetrievedEvent):
                out.retrieved = event
                if on_retrieved is not None:
                    on_retrieved(event)
            elif isinstance(event, TokenEvent):
                text += event.text
                if live is not None:
                    live.update(Markdown(text), refresh=True)
                else:
                    console.print(event.text, end="", markup=False, soft_wrap=True)
            elif isinstance(event, DoneEvent):
                out.done = event
    finally:
        if live is not None:
            live.stop()
    if plain:
        console.print("")
    return out


def _location(n: int, cand: Candidate) -> str:
    c = cand.chunk
    return f"[{n}] {c.path}:{c.start_line}-{c.end_line}  {c.symbol or ''}".rstrip()


def _shown_lines(cand: Candidate) -> tuple[str, int]:
    lines = cand.chunk.text.splitlines()
    return "\n".join(lines[:MAX_CODE_LINES]), max(0, len(lines) - MAX_CODE_LINES)


def source_panels(console: Console, sources: list[Candidate], citations: CitationReport) -> None:
    """One panel (plain: one code block) per cited source, then the uncited ones in a line."""
    plain = is_plain(console)
    cited = [n for n in citations.cited if 1 <= n <= len(sources)]
    console.print("")
    if not sources:
        console.print("Sources: none", markup=False, soft_wrap=True)
        return
    if plain:
        console.print("Sources:", markup=False, soft_wrap=True)
    for n in cited:
        cand = sources[n - 1]
        code, more = _shown_lines(cand)
        note = f"… {more} more lines" if more else ""
        if plain:
            console.print(_location(n, cand), markup=False, soft_wrap=True)
            start = cand.chunk.start_line
            width = len(str(start + code.count("\n")))
            for i, line in enumerate(code.split("\n")):
                console.print(f"  {start + i:>{width}} | {line}", markup=False, soft_wrap=True)
            if note:
                console.print(f"  {note}", markup=False, soft_wrap=True)
            continue
        syntax = Syntax(
            code,
            cand.chunk.language or "text",
            line_numbers=True,
            start_line=cand.chunk.start_line,
        )
        body = Group(syntax, Text(note, style="dim")) if note else syntax
        console.print(Panel(body, title=Text(_location(n, cand)), title_align="left"))
    rest = [n for n in range(1, len(sources) + 1) if n not in cited]
    if rest:
        listing = "; ".join(_location(n, sources[n - 1]) for n in rest)
        console.print(f"Also retrieved, not cited: {listing}", markup=False, soft_wrap=True)


def status_line(console: Console, citations: CitationReport, timings_ms: dict[str, float]) -> None:
    valid = len(citations.cited) - len(citations.invalid)
    line = Text(f"citations: {valid} valid, ")
    bad = f"{len(citations.invalid)} invalid"
    if citations.invalid:
        bad += f" (invalid: {', '.join(map(str, citations.invalid))})"
    line.append(bad, style="red" if citations.invalid else "")
    retrieve = sum(v for k, v in timings_ms.items() if k not in ("generate", "pack"))
    generate = timings_ms.get("generate", 0.0)
    line.append(f"  |  retrieve {retrieve:.0f} ms, generate {generate:.0f} ms")
    console.print(line, soft_wrap=True)
