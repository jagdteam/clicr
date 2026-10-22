"""Streaming answer engine: retrieve, build the prompt, stream the answer, check citations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from clicr.answer.citations import CitationReport, check_citations
from clicr.answer.context import pack
from clicr.answer.prompt import NOT_FOUND, build_messages
from clicr.ingest.chunkers import normalize_newlines
from clicr.providers.base import ChatModel
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.search import SearchConfig
from clicr.retrieve.types import Candidate, RetrievalResult, Turn


@dataclass(frozen=True)
class RetrievedEvent:
    result: RetrievalResult
    # the numbered prompt sources: [n] in the answer is sources[n - 1]
    sources: list[Candidate] = field(default_factory=list)


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    answer: str
    citations: CitationReport
    timings_ms: dict[str, float]
    result: RetrievalResult
    sources: list[Candidate] = field(default_factory=list)


Event = RetrievedEvent | TokenEvent | DoneEvent


class Answerer:
    def __init__(self, retriever: Retriever, chat: ChatModel, root: Path | None = None) -> None:
        """`root` is the indexed folder; with it, merged sources are re-read from the files."""
        self.retriever = retriever
        self.chat = chat
        self.root = root
        cfg = getattr(retriever, "cfg", None) or SearchConfig()
        self.context_tokens = cfg.context_tokens

    def _lines_of(self, path: str) -> list[str] | None:
        if self.root is None:
            return None
        try:
            data = (self.root / path).read_bytes()
        except OSError:
            return None
        return normalize_newlines(data.decode("utf-8", "replace")).split("\n")

    def _sources(self, result: RetrievalResult) -> list[Candidate]:
        if result.abstained:
            return []
        if self.context_tokens <= 0:
            return list(result.candidates)
        return pack(result.candidates, self.context_tokens, self._lines_of)

    def ask_stream(self, question: str, history: list[Turn] | None = None) -> Iterator[Event]:
        result = self.retriever.retrieve(question, history)
        start = time.perf_counter()
        sources = self._sources(result)
        timings = dict(result.timings_ms)
        timings["pack"] = (time.perf_counter() - start) * 1000.0
        yield RetrievedEvent(result, sources)
        if not sources:
            # nothing to ground an answer on (no candidates, or abstained): skip the model
            timings["generate"] = 0.0
            yield TokenEvent(NOT_FOUND)
            yield DoneEvent(NOT_FOUND, check_citations(NOT_FOUND, 0), timings, result, sources)
            return
        messages = build_messages(question, sources, history)
        parts: list[str] = []
        start = time.perf_counter()
        for token in self.chat.stream(messages):
            parts.append(token)
            yield TokenEvent(token)
        timings["generate"] = (time.perf_counter() - start) * 1000.0
        answer = "".join(parts)
        report = check_citations(answer, len(sources))
        yield DoneEvent(answer, report, timings, result, sources)

    def ask(self, question: str, history: list[Turn] | None = None) -> DoneEvent:
        done: DoneEvent | None = None
        for event in self.ask_stream(question, history):
            if isinstance(event, DoneEvent):
                done = event
        assert done is not None
        return done
