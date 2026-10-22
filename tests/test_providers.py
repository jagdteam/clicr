import sys
import types

import numpy as np
import ollama
import pytest

from clicr.config import Settings
from clicr.providers import make_chat, make_chunker, make_embedder
from clicr.providers.base import ProviderError
from clicr.providers.claude import ClaudeChat
from clicr.providers.fastembed import FastEmbedEmbedder
from clicr.providers.ollama import OllamaChat, OllamaEmbedder


def _msg(text):
    return types.SimpleNamespace(message=types.SimpleNamespace(content=text))


class FakeClient:
    def __init__(self, dim=4, chunks=("a", "b", "c"), error=None):
        self.dim, self.chunks, self.error = dim, chunks, error
        self.embed_calls, self.chat_calls = [], []

    def embed(self, **kw):
        if self.error:
            raise self.error
        self.embed_calls.append(kw)
        n = len(kw["input"])
        return types.SimpleNamespace(embeddings=[[0.5] * self.dim for _ in range(n)])

    def chat(self, **kw):
        if self.error:
            raise self.error
        self.chat_calls.append(kw)
        return iter([_msg(c) for c in self.chunks])


def with_client(obj, client):
    obj._client = client
    return obj


def test_gemma_prefixes_and_batching():
    c = FakeClient()
    e = with_client(OllamaEmbedder("embeddinggemma", batch_size=2), c)
    out = e.embed_documents(["x", "y", "z"])
    assert out.shape == (3, 4) and out.dtype == np.float32
    assert [len(k["input"]) for k in c.embed_calls] == [2, 1]
    assert c.embed_calls[0]["input"][0] == "title: none | text: x"
    q = e.embed_query("find it")
    assert q.shape == (4,) and q.dtype == np.float32
    assert c.embed_calls[-1]["input"] == ["task: code retrieval | query: find it"]
    assert e.name == "ollama:embeddinggemma"


def test_qwen_prefixes():
    c = FakeClient()
    e = with_client(OllamaEmbedder("qwen3-embedding:0.6b"), c)
    e.embed_documents(["doc"])
    assert c.embed_calls[-1]["input"] == ["doc"]
    e.embed_query("q")
    sent = c.embed_calls[-1]["input"][0]
    assert sent.startswith("Instruct: ") and sent.endswith("\nQuery:q")


def test_unknown_model_no_prefix_and_lazy_dim():
    c = FakeClient(dim=7)
    e = with_client(OllamaEmbedder("mystery"), c)
    assert c.embed_calls == []
    assert e.dim == 7
    e.embed_query("q")
    assert c.embed_calls[-1]["input"] == ["q"]


def test_chat_stream_and_think_flag():
    c = FakeClient()
    chat = with_client(OllamaChat("qwen2.5-coder:7b"), c)
    assert chat.complete([{"role": "user", "content": "hi"}]) == "abc"
    assert "think" not in c.chat_calls[0]
    assert c.chat_calls[0]["options"]["num_ctx"] == 8192
    q3 = with_client(OllamaChat("qwen3:8b"), c)
    assert list(q3.stream([])) == ["a", "b", "c"]
    assert c.chat_calls[-1]["think"] is False


def test_unload_sends_keep_alive_zero():
    c = FakeClient()
    with_client(OllamaChat("m"), c).unload()
    assert c.chat_calls[0]["keep_alive"] == 0


def test_unreachable_server_error():
    c = FakeClient(error=ConnectionError("refused"))
    with pytest.raises(ProviderError, match="ollama serve"):
        with_client(OllamaEmbedder("embeddinggemma"), c).embed_query("q")
    with pytest.raises(ProviderError, match="ollama serve"):
        with_client(OllamaChat("m"), c).complete([])


def test_missing_model_error():
    c = FakeClient(error=ollama.ResponseError("model not found", 404))
    with pytest.raises(ProviderError, match="ollama pull qwen2.5-coder:7b"):
        with_client(OllamaChat("qwen2.5-coder:7b"), c).complete([])


class FakeStreamCM:
    def __enter__(self):
        return types.SimpleNamespace(text_stream=iter(["Hel", "lo"]))

    def __exit__(self, *args):
        return False


def test_claude_splits_system_message():
    calls = {}

    def fake_stream(**kw):
        calls.update(kw)
        return FakeStreamCM()

    chat = ClaudeChat()
    chat._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=fake_stream))
    out = chat.complete(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    )
    assert out == "Hello"
    assert calls["system"] == "be brief"
    assert calls["messages"] == [{"role": "user", "content": "hi"}]
    assert calls["model"] == "claude-sonnet-5-5" and calls["max_tokens"] == 2048


def test_claude_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ProviderError, match="uv sync --extra claude"):
        ClaudeChat().complete([{"role": "user", "content": "x"}])


def test_fastembed_shapes(monkeypatch, tmp_path):
    seen = {}

    class FakeTE:
        def __init__(self, model_name, cache_dir=None, **kw):
            seen.update(model=model_name, cache=cache_dir)

        @classmethod
        def get_embedding_size(cls, name):
            return 3

        def passage_embed(self, texts):
            for _ in texts:
                yield np.ones(3, dtype=np.float64)

        def query_embed(self, query):
            yield np.zeros(3, dtype=np.float64)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTE))
    e = FastEmbedEmbedder(cache_dir=tmp_path)
    assert e.dim == 3 and e.name == "fastembed:BAAI/bge-small-en-v1.5"
    d = e.embed_documents(["a", "b"])
    assert d.shape == (2, 3) and d.dtype == np.float32
    q = e.embed_query("q")
    assert q.shape == (3,) and q.dtype == np.float32
    assert seen["cache"] == str(tmp_path)


def test_factories():
    s = Settings()
    assert isinstance(make_embedder(s), OllamaEmbedder)
    assert isinstance(make_chat(s), OllamaChat)
    assert make_embedder(Settings(embed_provider="fake")).name
    assert make_chunker(s).name.startswith("syntax")
    assert make_chunker(Settings(chunker="line")).name.startswith("line")


def test_client_built_with_timeout(monkeypatch):
    import httpx

    from clicr.providers import ollama as mod

    seen = []
    monkeypatch.setattr(
        mod.ollama_pkg, "Client", lambda host=None, **kw: seen.append((host, kw)) or object()
    )
    OllamaChat("m", host="http://h:1").client  # noqa: B018
    OllamaEmbedder("m", host="http://h:1").client  # noqa: B018
    chat_t, embed_t = seen[0][1]["timeout"], seen[1][1]["timeout"]
    assert isinstance(chat_t, httpx.Timeout)
    assert (chat_t.connect, chat_t.read, chat_t.write, chat_t.pool) == (5.0, 300.0, 30.0, 5.0)
    assert embed_t.read == 120.0
    OllamaChat("m", request_timeout=7).client  # noqa: B018
    assert seen[2][1]["timeout"].read == 7


def test_timeout_error_message():
    import httpx

    c = FakeClient(error=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError, match=r"did not answer within 120"):
        with_client(OllamaEmbedder("m"), c).embed_query("q")


def test_httpx_connect_error_message():
    import httpx

    c = FakeClient(error=httpx.ConnectError("no"))
    with pytest.raises(ProviderError, match="Cannot reach Ollama"):
        with_client(OllamaChat("m"), c).complete([])


def test_other_error_is_generic():
    c = FakeClient(error=KeyError("boom"))
    with pytest.raises(ProviderError, match="KeyError") as info:
        with_client(OllamaChat("m"), c).complete([])
    assert "Cannot reach" not in str(info.value)


def test_num_predict_option():
    c = FakeClient()
    with_client(OllamaChat("m", num_predict=20), c).complete([])
    assert c.chat_calls[-1]["options"]["num_predict"] == 20
    with_client(OllamaChat("m"), c).complete([])
    assert "num_predict" not in c.chat_calls[-1]["options"]


def test_claude_joins_all_system_messages():
    calls = {}

    def fake_stream(**kw):
        calls.update(kw)
        return FakeStreamCM()

    chat = ClaudeChat()
    chat._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=fake_stream))
    chat.complete(
        [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert calls["system"] == "a\n\nb"
    assert calls["messages"] == [{"role": "user", "content": "hi"}]


def test_factories_use_provider_specific_models(monkeypatch):
    s = Settings(
        chat_provider="claude", claude_model="cm", embed_provider="fastembed", fastembed_model="fm"
    )
    assert make_chat(s).model == "cm"
    assert make_embedder(s).model == "fm"
