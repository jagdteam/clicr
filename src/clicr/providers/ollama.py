"""Ollama embedder and chat model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import numpy as np
import ollama as ollama_pkg

from clicr.providers.base import ProviderError

DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class PromptTemplate:
    query: str  # prefix put in front of every query
    document: str  # prefix put in front of every document


# Per-family prompt templates, matched by substring of the lowercased model name.
_PROMPTS: dict[str, PromptTemplate] = {
    # https://huggingface.co/google/embeddinggemma-300m (card is gated; the same text is
    # mirrored at https://huggingface.co/unsloth/embeddinggemma-300m).
    # Query form "task: {task} | query: " using the card's "code retrieval" task;
    # document form "title: {title | none} | text: ".
    "embeddinggemma": PromptTemplate(
        query="task: code retrieval | query: ",
        document="title: none | text: ",
    ),
    # https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    # Queries: "Instruct: {task}\nQuery:{query}"; documents get no instruction.
    # The card only shows a web-search task; the code wording below is our adaptation of it.
    "qwen3-embedding": PromptTemplate(
        query=(
            "Instruct: Given a code search query, retrieve relevant code snippets "
            "that answer the query\nQuery:"
        ),
        document="",
    ),
}
_NO_PROMPT = PromptTemplate("", "")


def prompts_for(model: str) -> PromptTemplate:
    lowered = model.lower()
    for family, template in _PROMPTS.items():
        if family in lowered:
            return template
    return _NO_PROMPT


def _make_client(host: str | None, read_timeout: float):
    timeout = httpx.Timeout(connect=5.0, read=read_timeout, write=30.0, pool=5.0)
    return ollama_pkg.Client(host=host or DEFAULT_HOST, timeout=timeout)


def _translate(exc: Exception, model: str, host: str | None, read_timeout: float) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, ollama_pkg.ResponseError):
        if exc.status_code == 404 or "not found" in str(exc).lower():
            return ProviderError(
                f"Ollama model '{model}' is not installed. Fix: ollama pull {model}"
            )
        return ProviderError(f"Ollama error for model '{model}': {exc}")
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(
            f"Ollama did not answer within {read_timeout:g} s "
            f"(model '{model}'). Fix: check the server load, or raise request_timeout."
        )
    if isinstance(exc, ConnectionError | httpx.ConnectError):
        return ProviderError(
            f"Cannot reach Ollama at {host or DEFAULT_HOST}. "
            "Fix: install it from https://ollama.com and start it (run: ollama serve)."
        )
    return ProviderError(f"Ollama request for model '{model}' failed ({type(exc).__name__}).")


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        host: str | None = None,
        batch_size: int = 32,
        keep_alive: str = "5m",
        request_timeout: float = 120.0,
    ) -> None:
        self.request_timeout = request_timeout
        self.model = model
        self.host = host
        self.batch_size = batch_size
        self.keep_alive = keep_alive
        self.name = f"ollama:{model}"
        self._client = None
        self._dim: int | None = None
        self._prompts = prompts_for(model)

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.host, self.request_timeout)
        return self._client

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._embed_raw(["dimension probe"]).shape[1])
        return self._dim

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        try:
            resp = self.client.embed(model=self.model, input=texts, keep_alive=self.keep_alive)
        except Exception as exc:
            raise _translate(exc, self.model, self.host, self.request_timeout) from exc
        return np.asarray(resp.embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self._prompts.document
        parts = []
        for i in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[i : i + self.batch_size]]
            parts.append(self._embed_raw(batch))
        out = np.vstack(parts)
        self._dim = int(out.shape[1])
        return out

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._embed_raw([self._prompts.query + text])[0]
        self._dim = int(vec.shape[0])
        return vec

    def unload(self) -> None:
        try:
            self.client.embed(model=self.model, input=[""], keep_alive=0)
        except Exception as exc:
            raise _translate(exc, self.model, self.host, self.request_timeout) from exc


class OllamaChat:
    def __init__(
        self,
        model: str,
        host: str | None = None,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        keep_alive: str = "5m",
        num_predict: int | None = None,
        request_timeout: float = 300.0,
    ) -> None:
        self.request_timeout = request_timeout
        self.num_predict = num_predict
        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.name = f"ollama:{model}"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _make_client(self.host, self.request_timeout)
        return self._client

    def stream(self, messages: list[dict]) -> Iterator[str]:
        options: dict = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        kwargs: dict = {}
        if "qwen3" in self.model.lower():
            kwargs["think"] = False
        try:
            chunks = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=options,
                keep_alive=self.keep_alive,
                **kwargs,
            )
            for chunk in chunks:
                piece = chunk.message.content
                if piece:
                    yield piece
        except Exception as exc:
            raise _translate(exc, self.model, self.host, self.request_timeout) from exc

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))

    def unload(self) -> None:
        try:
            self.client.chat(model=self.model, messages=[], keep_alive=0)
        except Exception as exc:
            raise _translate(exc, self.model, self.host, self.request_timeout) from exc
