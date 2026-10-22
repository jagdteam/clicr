"""Incrementally bring an index in line with a repository."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from clicr.index.store import IndexStore
from clicr.ingest.chunkers import normalize_newlines
from clicr.ingest.languages import language_for
from clicr.ingest.walker import iter_files
from clicr.models import Chunk
from clicr.providers.base import Embedder

log = logging.getLogger(__name__)


class Chunker(Protocol):
    name: str

    def chunk(self, path_rel: str, text: str, language: str | None) -> list[Chunk]: ...


@dataclass
class UpdateStats:
    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0
    chunks_embedded: int = 0


def _embed_all(
    embedder: Embedder, chunks: list[Chunk], batch_size: int, header: bool
) -> np.ndarray:
    parts = [
        embedder.embed_documents([c.embed_text(header) for c in chunks[i : i + batch_size]])
        for i in range(0, len(chunks), batch_size)
    ]
    if not parts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.vstack(parts).astype(np.float32)


def update_index(
    root: Path,
    store: IndexStore,
    chunker: Chunker,
    embedder: Embedder,
    *,
    batch_size: int = 64,
    header: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> UpdateStats:
    root = Path(root)
    if not store.compatible:
        log.info("index settings changed, rebuilding from scratch")
        store.reset()
    known = store.file_hashes()
    files = list(iter_files(root))
    seen: set[str] = set()
    stats = UpdateStats()
    for done, path in enumerate(files, start=1):
        rel = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
            mtime = path.stat().st_mtime
        except OSError as exc:
            log.warning("skipping unreadable file %s: %s", rel, exc)
            continue
        seen.add(rel)
        digest = hashlib.sha256(data).hexdigest()
        if known.get(rel) == digest:
            stats.unchanged += 1
        else:
            chunks = chunker.chunk(
                rel, normalize_newlines(data.decode("utf-8", "replace")), language_for(path)
            )
            vectors = _embed_all(embedder, chunks, batch_size, header)
            store.upsert_file(rel, digest, mtime, chunks, vectors)
            stats.chunks_embedded += len(chunks)
            if rel in known:
                stats.changed += 1
            else:
                stats.added += 1
        if progress is not None:
            progress("index", done, len(files))
    for rel in sorted(set(known) - seen):
        store.delete_file(rel)
        stats.deleted += 1
    return stats
