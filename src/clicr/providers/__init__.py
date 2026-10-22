"""Model providers and the factories that build them from Settings."""

from __future__ import annotations

from dataclasses import replace

from clicr.config import Settings
from clicr.ingest.chunkers import LineChunker, SyntaxChunker
from clicr.providers.base import ChatModel, Embedder, ProviderError


def make_embedder(settings: Settings) -> Embedder:
    if settings.embed_provider == "fake":
        from clicr.providers.fake import FakeEmbedder

        return FakeEmbedder()
    if settings.embed_provider == "fastembed":
        from clicr.providers.fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder(settings.fastembed_model)
    from clicr.providers.ollama import OllamaEmbedder

    return OllamaEmbedder(settings.embed_model, host=settings.ollama_host)


def embedder_from_name(name: str, settings: Settings) -> Embedder:
    """Rebuild the embedder an index was made with from its stored name."""
    if name.startswith("fake-"):
        from clicr.providers.fake import FakeEmbedder

        return FakeEmbedder(int(name.removeprefix("fake-")))
    provider, _, model = name.partition(":")
    if provider == "ollama" and model:
        return make_embedder(replace(settings, embed_provider="ollama", embed_model=model))
    if provider == "fastembed" and model:
        return make_embedder(replace(settings, embed_provider="fastembed", fastembed_model=model))
    raise ProviderError(f"Unknown embedder {name!r} in the index. Re-run: clicr index --rebuild")


def make_chat(settings: Settings) -> ChatModel:
    if settings.chat_provider == "fake":
        from clicr.providers.fake import FakeChat

        return FakeChat()
    if settings.chat_provider == "claude":
        from clicr.providers.claude import ClaudeChat

        return ClaudeChat(settings.claude_model)
    from clicr.providers.ollama import OllamaChat

    return OllamaChat(settings.chat_model, host=settings.ollama_host)


def make_chunker(settings: Settings) -> SyntaxChunker | LineChunker:
    if settings.chunker == "line":
        return LineChunker()
    return SyntaxChunker(max_lines=settings.max_lines)


__all__ = ["ProviderError", "embedder_from_name", "make_chat", "make_chunker", "make_embedder"]
