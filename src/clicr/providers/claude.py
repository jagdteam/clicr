"""Anthropic Claude chat model (optional)."""

from __future__ import annotations

from collections.abc import Iterator

from clicr.providers.base import ProviderError


class ClaudeChat:
    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 2048) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"claude:{model}"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError("install with: uv sync --extra claude") from exc
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY itself
        return self._client

    def stream(self, messages: list[dict]) -> Iterator[str]:
        systems = [m["content"] for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        kwargs = {"system": "\n\n".join(systems)} if systems else {}
        client = self.client
        try:
            with client.messages.stream(
                model=self.model, max_tokens=self.max_tokens, messages=rest, **kwargs
            ) as s:
                yield from s.text_stream
        except Exception as exc:
            raise ProviderError(
                f"Claude request failed ({type(exc).__name__}). "
                "Check ANTHROPIC_API_KEY and your network."
            ) from exc

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))
