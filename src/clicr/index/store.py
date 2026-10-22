"""SQLite index: chunks, vectors (BLOBs) and an FTS5 keyword index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from clicr.index.textsplit import search_terms
from clicr.models import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, sha256 TEXT, mtime REAL, n_chunks INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, path TEXT, start_line INTEGER,
    end_line INTEGER, language TEXT, symbol TEXT, kind TEXT, text TEXT, vector BLOB
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(terms);
"""

_COLUMNS = "rowid, path, start_line, end_line, text, language, symbol, kind"


class IndexStore:
    def __init__(self, conn: sqlite3.Connection, expected: dict[str, str]) -> None:
        self._conn = conn
        self._expected = expected
        self._dim = int(expected["dim"])
        self.compatible = True
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def open(
        cls,
        directory: Path,
        *,
        embedder_name: str,
        dim: int,
        chunker_name: str,
        header: bool = True,
    ) -> IndexStore:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(directory / "index.sqlite")
        conn.executescript(_SCHEMA)
        expected = {
            "embedder": embedder_name,
            "dim": str(dim),
            "chunker": chunker_name,
            "header": str(int(header)),
        }
        store = cls(conn, expected)
        stored = dict(conn.execute("SELECT key, value FROM meta"))
        if not stored:
            store._write_meta()
        else:
            store.compatible = stored == expected
        return store

    @staticmethod
    def read_meta(directory: Path) -> dict[str, str] | None:
        """Stored index settings, or None if there is no index. Never writes or creates."""
        db = Path(directory) / "index.sqlite"
        if not db.is_file():
            return None
        try:
            conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
            try:
                meta = dict(conn.execute("SELECT key, value FROM meta"))
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        return meta or None

    def close(self) -> None:
        self._conn.close()

    def _write_meta(self) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", self._expected.items()
            )

    def reset(self) -> None:
        with self._conn:
            for table in ("chunks_fts", "chunks", "files", "meta"):
                self._conn.execute(f"DELETE FROM {table}")
        self._write_meta()
        self.compatible = True
        self._cache = None

    # -- writes ------------------------------------------------------------
    def upsert_file(
        self, path: str, sha256: str, mtime: float, chunks: list[Chunk], vectors: np.ndarray
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vecs = np.asarray(vectors, dtype=np.float32).reshape(len(chunks), self._dim)
        with self._conn:
            self._delete_chunks(path)
            for chunk, vec in zip(chunks, vecs, strict=True):
                cur = self._conn.execute(
                    "INSERT INTO chunks(id, path, start_line, end_line, language, symbol,"
                    " kind, text, vector) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.id, chunk.path, chunk.start_line, chunk.end_line, chunk.language,
                        chunk.symbol, chunk.kind, chunk.text, vec.tobytes(),
                    ),
                )  # fmt: skip
                self._conn.execute(
                    "INSERT INTO chunks_fts(rowid, terms) VALUES (?, ?)",
                    (cur.lastrowid, search_terms(_fts_text(chunk))),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO files(path, sha256, mtime, n_chunks) VALUES (?,?,?,?)",
                (path, sha256, mtime, len(chunks)),
            )
        self._cache = None

    def delete_file(self, path: str) -> None:
        with self._conn:
            self._delete_chunks(path)
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._cache = None

    def _delete_chunks(self, path: str) -> None:
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE path = ?)",
            (path,),
        )
        self._conn.execute("DELETE FROM chunks WHERE path = ?", (path,))

    # -- reads -------------------------------------------------------------
    def file_hashes(self) -> dict[str, str]:
        return dict(self._conn.execute("SELECT path, sha256 FROM files"))

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get_chunks(self, rowids: list[int]) -> list[tuple[int, Chunk]]:
        out: list[tuple[int, Chunk]] = []
        for rowid in rowids:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM chunks WHERE rowid = ?", (int(rowid),)
            ).fetchone()
            if row is None:
                continue
            rid, path, start, end, text, language, symbol, kind = row
            out.append((rid, Chunk(path, start, end, text, language, symbol, kind)))
        return out

    # -- search ------------------------------------------------------------
    def vector_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cache is not None:
            return self._cache
        rows = self._conn.execute("SELECT rowid, vector FROM chunks ORDER BY rowid").fetchall()
        rowids = np.array([r[0] for r in rows], dtype=np.int64)
        if rows:
            matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        else:
            matrix = np.zeros((0, self._dim), dtype=np.float32)
        self._cache = (rowids, matrix)
        return self._cache

    def vector_search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        rowids, matrix = self.vector_matrix()
        if len(rowids) == 0 or k <= 0:
            return []
        q = np.asarray(qvec, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        scores = matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(rowids[i]), float(scores[i])) for i in order]

    def keyword_search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = list(dict.fromkeys(search_terms(query).split()))
        if not terms or k <= 0:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = self._conn.execute(
            "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?"
            " ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k),
        ).fetchall()
        return [(int(rid), -float(score)) for rid, score in rows]


def _fts_text(chunk: Chunk) -> str:
    """Text indexed for keywords: the symbol and path help match names."""
    return f"{chunk.symbol or ''}\n{chunk.path}\n{chunk.text}"
