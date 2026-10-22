"""Streaming answer engine: retrieve, build the prompt, stream the answer, check citations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from clicr.answer.citations import CitationReport, check_citations
from clicr.answer.prompt import NOT_FOUND, build_messages
from clicr.providers.base import ChatModel
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.types import RetrievalResult, Turn


@dataclass(frozen=True)
class RetrievedEvent:
    result: RetrievalResult


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    answer: str
    citations: CitationReport
    timings_ms: dict[str, float]
    result: RetrievalResult


Event = RetrievedEvent | TokenEvent | DoneEvent


class Answerer:
    def __init__(self, retriever: Retriever, chat: ChatModel) -> None:
        self.retriever = retriever
        self.chat = chat

    def ask_stream(self, question: str, history: list[Turn] | None = None) -> Iterator[Event]:
        result = self.retriever.retrieve(question, history)
        yield RetrievedEvent(result)
        if not result.candidates:
            # nothing to ground an answer on: skip the model
            timings = dict(result.timings_ms)
            timings["generate"] = 0.0
            yield TokenEvent(NOT_FOUND)
            yield DoneEvent(NOT_FOUND, check_citations(NOT_FOUND, 0), timings, result)
            return
        messages = build_messages(question, result.candidates, history)
        parts: list[str] = []
        start = time.perf_counter()
        for token in self.chat.stream(messages):
            parts.append(token)
            yield TokenEvent(token)
        timings = dict(result.timings_ms)
        timings["generate"] = (time.perf_counter() - start) * 1000.0
        answer = "".join(parts)
        report = check_citations(answer, len(result.candidates))
        yield DoneEvent(answer, report, timings, result)

    def ask(self, question: str, history: list[Turn] | None = None) -> DoneEvent:
        done: DoneEvent | None = None
        for event in self.ask_stream(question, history):
            if isinstance(event, DoneEvent):
                done = event
        assert done is not None
        return done
