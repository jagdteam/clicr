"""Live Ollama checks. Opt-in: uv run pytest -m ollama. Needs a GPU lease."""

import numpy as np
import pytest

from clicr.providers.ollama import OllamaChat, OllamaEmbedder

pytestmark = pytest.mark.ollama

EMBED_MODELS = ["embeddinggemma", "qwen3-embedding:0.6b"]
CHAT_MODEL = "qwen2.5-coder:7b"


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("model", EMBED_MODELS)
def test_embedder(model):
    e = OllamaEmbedder(model)
    assert e.dim > 0
    vecs = e.embed_documents(
        ["def add(a, b): return a + b", "def add(x, y): return x + y", "The sky is blue today."]
    )
    try:
        assert _cos(vecs[0], vecs[1]) > _cos(vecs[0], vecs[2])
    finally:
        e.unload()  # free VRAM before the next model loads (8 GB card)


def test_chat_stream():
    # Runs after the embedder tests (file order); each already unloaded itself.
    chat = OllamaChat(CHAT_MODEL, num_ctx=2048, num_predict=20)
    try:
        msgs = [{"role": "user", "content": "Say hello in five words or fewer."}]
        assert "".join(chat.stream(msgs)).strip()
    finally:
        chat.unload()
