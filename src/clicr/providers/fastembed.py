"""CPU embeddings via fastembed (ONNX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clicr.paths import clicr_home
from clicr.providers.base import ProviderError


def _text_embedding_cls():
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ProviderError("fastembed is not installed. Fix: uv add fastembed") from exc
    return TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: Path | None = None):
        self.model = model
        self.cache_dir = cache_dir if cache_dir is not None else clicr_home() / "models"
        self.name = f"fastembed:{model}"
        self._te = None
        self._dim: int | None = None

    @property
    def _model(self):
        if self._te is None:
            self._te = _text_embedding_cls()(model_name=self.model, cache_dir=str(self.cache_dir))
        return self._te

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(_text_embedding_cls().get_embedding_size(self.model))
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = [np.asarray(v, dtype=np.float32) for v in self._model.passage_embed(texts)]
        return np.vstack(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
