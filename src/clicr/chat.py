"""The interactive `clicr chat` loop. Input is injected so tests need no terminal."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console

from clicr.answer.engine import Answerer, DoneEvent
from clicr.providers.base import ChatModel
from clicr.retrieve.rerank import LLMReranker
from clicr.retrieve.types import Candidate, Turn
from clicr.sessions import SessionTurn, append_turn, export_markdown
from clicr.ui.console import is_plain
from clicr.ui.errors import friendly
from clicr.ui.render import source_panels, status_line, stream_answer

HELP = """Commands:
  /help            show this help
  /sources         show the sources of the last answer again
  /export [path]   save the session as markdown (default: clicr-session-<time>.md)
  /clear           forget the conversation so far (follow-ups start fresh)
  /model [name]    show or switch the chat model for this session
  /exit            leave (Ctrl-D works too)"""


@dataclass
class ChatState:
    history: list[Turn] = field(default_factory=list)
    turns: list[SessionTurn] = field(default_factory=list)  # whole session, for /export
    last_sources: list[Candidate] = field(default_factory=list)
    last_done: DoneEvent | None = None


def _say(console: Console, text: str) -> None:
    console.print(text, markup=False, soft_wrap=True)


def _record(session_id: str, state: ChatState, question: str, done: DoneEvent) -> None:
    rep = done.citations
    turn = SessionTurn(
        question=question,
        query_used=done.result.query_used,
        answer=done.answer,
        sources=[f"{c.chunk.path}:{c.chunk.start_line}-{c.chunk.end_line}" for c in done.sources],
        citations={"cited": rep.cited, "invalid": rep.invalid, "uncited": rep.uncited},
        timings_ms={k: round(v, 1) for k, v in done.timings_ms.items()},
    )
    state.turns.append(turn)
    state.history.append(Turn(question, done.answer))
    state.last_sources, state.last_done = done.sources, done
    append_turn(session_id, turn)


def _ask(answerer: Answerer, console: Console, state: ChatState, session_id: str, q: str) -> None:
    plain = is_plain(console)
    try:
        streamed = stream_answer(console, answerer.ask_stream(q, state.history), plain)
    except KeyboardInterrupt:
        _say(console, "\n(answer cancelled)")
        return
    except Exception as exc:
        message = friendly(exc)
        if message is None:
            raise
        _say(console, message)
        return
    if streamed.done is None:
        return
    _record(session_id, state, q, streamed.done)
    source_panels(console, streamed.done.sources, streamed.done.citations)
    status_line(console, streamed.done.citations, streamed.done.timings_ms)


def _export(console: Console, state: ChatState, arg: str) -> None:
    default = f"clicr-session-{datetime.now():%Y%m%d-%H%M%S}.md"
    target = Path(arg).expanduser() if arg else Path.cwd() / default
    try:
        target.write_text(export_markdown(state.turns), encoding="utf-8")
    except OSError as exc:
        _say(console, f"Could not save to {target}: {exc.strerror or exc}")
        return
    _say(console, f"Saved {len(state.turns)} turn(s) to {target}")


def _model(
    console: Console,
    answerer: Answerer,
    switch_model: Callable[[str], ChatModel] | None,
    arg: str,
) -> None:
    if not arg:
        _say(console, f"chat model: {answerer.chat.name}")
        return
    if switch_model is None:
        _say(console, "Switching models is not available here.")
        return
    chat = switch_model(arg)
    answerer.chat = chat
    answerer.retriever.chat = chat
    if answerer.retriever.cfg.rerank == "llm":  # the reranker holds its own model reference
        answerer.retriever.reranker = LLMReranker(chat)
    _say(console, f"chat model: {chat.name}")


def stdin_reader() -> Callable[[], str]:
    """Line reader for piped input: end of input raises EOFError, which ends the chat."""

    def read() -> str:
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line

    return read


def run_repl(
    answerer: Answerer,
    console: Console,
    read_line: Callable[[], str],
    session_id: str,
    switch_model: Callable[[str], ChatModel] | None = None,
) -> ChatState:
    """Read lines until /exit or end of input (EOFError); returns the final state."""
    state = ChatState()
    _say(console, "clicr chat. Ask about the code; /help lists commands, /exit leaves.")
    while True:
        try:
            line = read_line().strip()
        except KeyboardInterrupt:  # Ctrl-C at the prompt only drops the line
            continue
        except EOFError:
            break
        if not line:
            continue
        if not line.startswith("/"):
            _ask(answerer, console, state, session_id, line)
            continue
        command, _, arg = line.partition(" ")
        arg = arg.strip()
        if command == "/exit":
            break
        elif command == "/help":
            _say(console, HELP)
        elif command == "/sources":
            if state.last_done is None:
                _say(console, "No answer yet.")
            else:
                source_panels(console, state.last_sources, state.last_done.citations)
        elif command == "/clear":
            state.history.clear()
            _say(console, "Conversation cleared.")
        elif command == "/export":
            _export(console, state, arg)
        elif command == "/model":
            try:
                _model(console, answerer, switch_model, arg)
            except Exception as exc:
                message = friendly(exc)
                if message is None:
                    raise
                _say(console, message)
        else:
            _say(console, f"Unknown command {command}. Type /help for the list.")
    return state
