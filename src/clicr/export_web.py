"""`clicr export-web`: turn a fastembed index into a static bundle for the browser demo.

The bundle (all JSON is UTF-8, compact, keys sorted):

    manifest.json   format, source, embed settings, counts, BM25 and fusion settings, file sizes
    chunks.json     [{i, path, start, end, symbol, lang, text}] in rowid order
    vectors.bin     int8, row-major n_chunks x dim; value = round(v * 127), v the stored
                    L2-normalised vector (cosine = dot product / 127^2 on dequantised values)
    bm25.json       {doc_len: [...], postings: {key: [[i, tf], ...]}}
    graph.json      {refs: {i: [names]}, defs: {name: [i, ...]}}
    questions.json  suggested questions, with recorded answers when given
    parity.json     (only with a parity suite) Python's answers for the browser self-test

`i` is the chunk's position in rowid order (0-based), the index into chunks.json and the
row of vectors.bin.

Keyword search, exactly as the browser must do it
-------------------------------------------------
The Python index stores `search_terms(_fts_text(chunk))` in an SQLite FTS5 table with the
default `unicode61` tokenizer, and `IndexStore.keyword_search` runs
`bm25(chunks_fts)` over the OR of one quoted phrase per unique search term. To reproduce it:

* `fts_tokens(text)`: lower-case, then split into runs of letters and digits (anything that
  is not a letter or digit, including `_`, is a separator); strip diacritics. In JavaScript:
  `text.normalize("NFD").replace(/\\p{M}/gu, "").toLowerCase().match(/[\\p{L}\\p{N}]+/gu) ?? []`.
* A document's length `doc_len[i]` is the number of `fts_tokens` of its keyword text
  (FTS5's column size), and `avg_len` = sum(doc_len) / n_docs.
* A posting key is `" ".join(fts_tokens(term))`. A single-token key (`get`) carries the
  token's frequency in the chunk. A key with spaces (`get user by id`, from the term
  `get_user_by_id` or `getUserById`) is a phrase: its tf is the number of places where those
  tokens occur next to each other, which is what an FTS5 phrase counts. Phrase keys exist for
  every multi-token search term found anywhere in the corpus; a query phrase nobody wrote as a
  term (it could only match by coincidence) scores nothing in the browser.
* For a query: terms = unique(search_terms(query).split()); keys = one per term (see above).
  Score of a chunk = sum over keys present in it of
  `idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))`, k1 = 1.2, b = 0.75,
  `idf = ln((N - n + 0.5) / (n + 0.5))` with N = n_docs and n = number of chunks holding the
  key, and an idf <= 0 replaced by 1e-6. This is SQLite's `fts5Bm25Function`
  (ext/fts5/fts5_aux.c, defaults k1 = 1.2, b = 0.75). SQLite returns the negated score and
  sorts ascending; here higher is better and ties break on the lower `i`.
* Hybrid ranking is `rrf_fuse` in `retrieve/search.py`: take the top `k_candidates` of the
  vector list (dot product of the normalised query vector with the stored vectors) and of the
  keyword list, fuse = sum of 1 / (rrf_k + rank) over the lists (ranks start at 1), ties break
  on the higher vector score, then the lower `i`; show the first `top_k`.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from clicr import __version__
from clicr.index.store import IndexStore, _fts_text
from clicr.index.textsplit import search_terms
from clicr.paths import index_dir_for
from clicr.providers.base import Embedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.search import SearchConfig
from clicr.ui.errors import UserError

FORMAT = 1
BM25_K1 = 1.2
BM25_B = 0.75
IDF_FLOOR = 1e-6  # FTS5 replaces an idf <= 0 with this
SEARCH = SearchConfig(mode="hybrid", k_candidates=20, rrf_k=60, top_k=6)

_ALNUM_RUN = re.compile(r"[^\W_]+")


# -- keyword side --------------------------------------------------------------------------
def fts_tokens(text: str) -> list[str]:
    """What SQLite's unicode61 tokenizer yields for `text` (letters and digits, folded)."""
    out: list[str] = []
    for run in _ALNUM_RUN.findall(text):
        if not run.isascii():
            run = "".join(
                c for c in unicodedata.normalize("NFD", run) if not unicodedata.combining(c)
            )
        out.append(run.lower())
    return out


def _key(term: str) -> str:
    return " ".join(fts_tokens(term))


def build_bm25(texts: list[str]) -> dict:
    """BM25 data from each chunk's keyword text (`search_terms(_fts_text(chunk))`)."""
    streams = [fts_tokens(text) for text in texts]
    phrases: dict[str, list[str]] = {}
    for text in texts:
        for term in text.split():
            toks = fts_tokens(term)
            if len(toks) > 1:
                phrases[" ".join(toks)] = toks
    by_first: dict[str, list[list[str]]] = defaultdict(list)
    for toks in phrases.values():
        by_first[toks[0]].append(toks)

    postings: dict[str, list[list[int]]] = defaultdict(list)
    for i, stream in enumerate(streams):
        counts: Counter[str] = Counter(stream)
        for j, tok in enumerate(stream):
            for toks in by_first.get(tok, ()):
                if stream[j : j + len(toks)] == toks:
                    counts[" ".join(toks)] += 1
        for key, tf in counts.items():
            postings[key].append([i, tf])
    return {"doc_len": [len(s) for s in streams], "postings": dict(sorted(postings.items()))}


def bm25_scores(query: str, bm25: dict) -> list[tuple[int, float]]:
    """Reference keyword search over a `bm25.json` dict; the browser must match it.

    Returns (i, score) for every chunk that holds a query term, best first. Same ranking as
    `IndexStore.keyword_search` (SQLite FTS5 `bm25()`, k1 1.2, b 0.75; see the module docstring).
    """
    doc_len = bm25["doc_len"]
    n_docs = len(doc_len)
    if n_docs == 0:
        return []
    avg_len = sum(doc_len) / n_docs
    scores: dict[int, float] = {}
    for term in dict.fromkeys(search_terms(query).split()):
        hits = bm25["postings"].get(_key(term), [])
        if not hits:
            continue
        idf = math.log((n_docs - len(hits) + 0.5) / (len(hits) + 0.5))
        if idf <= 0:
            idf = IDF_FLOOR
        for i, tf in hits:
            norm = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len[i] / avg_len)
            scores[i] = scores.get(i, 0.0) + idf * (tf * (BM25_K1 + 1)) / norm
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# -- embedder facts -------------------------------------------------------------------------
def is_fastembed(name: str) -> bool:
    return name.startswith("fastembed:")


def fastembed_prefixes(model: str) -> tuple[str, str]:
    """(query_prefix, doc_prefix) that fastembed prepends for `model`.

    fastembed's base class `TextEmbeddingBase.query_embed` / `passage_embed` call `embed`
    unchanged, so those models add no prefix. A model class that overrides either method
    (such as Jina v3's task ids) does something we cannot copy as a string, so refuse it.
    """
    from fastembed import TextEmbedding
    from fastembed.text.text_embedding_base import TextEmbeddingBase

    wanted = model.lower()
    for cls in TextEmbedding.EMBEDDINGS_REGISTRY:
        if any(d.model.lower() == wanted for d in cls._list_supported_models()):
            if (
                cls.query_embed is not TextEmbeddingBase.query_embed
                or cls.passage_embed is not TextEmbeddingBase.passage_embed
            ):
                raise UserError(
                    f"fastembed model {model!r} uses a task prompt, not a text prefix; "
                    "the browser demo cannot copy it. Use a plain model such as bge-small-en-v1.5."
                )
            return "", ""
    raise UserError(f"fastembed does not list the model {model!r}.")


# -- questions ------------------------------------------------------------------------------
def record_answer(answerer, question: str, include: str = ".") -> dict:
    """Run the real answerer once; keep the answer, the cited sources, the model and the time."""
    done = answerer.ask(question)
    sources = []
    for n in done.citations.cited:
        if 1 <= n <= len(done.sources):
            chunk = done.sources[n - 1].chunk
            sources.append(
                {
                    "path": _repo_path(include, chunk.path),
                    "start": chunk.start_line,
                    "end": chunk.end_line,
                }
            )
    return {
        "question": question,
        "answer": done.answer,
        "sources": sources,
        "model": answerer.chat.name,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _question_entry(item: str | dict) -> dict:
    if isinstance(item, str):
        item = {"question": item}
    if not isinstance(item, dict) or not isinstance(item.get("question"), str):
        raise UserError("--questions must be a JSON list of strings or {question, ...} objects.")
    return {
        "question": item["question"],
        "answer": item.get("answer"),
        "sources": [
            {"path": s["path"], "start": s["start"], "end": s["end"]}
            for s in item.get("sources") or []
        ],
        "model": item.get("model"),
        "recorded_at": item.get("recorded_at"),
    }


# -- the bundle -----------------------------------------------------------------------------
def _repo_path(include: str | None, stored: str) -> str:
    from clicr.eval.suites import repo_path

    return repo_path(include, stored)


def _dump(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _graph(index_db: Path, position: dict[int, int]) -> dict:
    conn = sqlite3.connect(f"{index_db.as_uri()}?mode=ro", uri=True)
    try:
        refs: dict[str, list[str]] = defaultdict(list)
        for rowid, name in conn.execute("SELECT rowid, name FROM refs ORDER BY _rowid_"):
            if rowid in position:
                refs[str(position[rowid])].append(name)
        defs: dict[str, list[int]] = defaultdict(list)
        for name, rowid in conn.execute("SELECT name, rowid FROM symbols ORDER BY rowid"):
            if rowid in position:
                defs[name].append(position[rowid])
    finally:
        conn.close()
    return {"refs": dict(refs), "defs": dict(defs)}


def export_web(
    root: Path,
    out: Path,
    *,
    embedder_factory: Callable[[str], Embedder],
    source: dict | None = None,
    questions: list[str | dict] | None = None,
    answerer=None,
    parity_items: list[dict] | None = None,
    max_mb: float = 3.0,
    embedder_ok: Callable[[str], bool] = is_fastembed,
    prefixes: Callable[[str], tuple[str, str]] = fastembed_prefixes,
) -> dict:
    """Write the bundle for the index of `root` into `out`; returns the manifest.

    `source` holds name/repo/commit/license/include (the pinned suite); `include` prefixes the
    paths so they are repo-root paths. `answerer` (with `questions`) records fresh answers.
    Raises UserError if the embedder is not a browser-capable one, the index is missing, or the
    bundle is larger than `max_mb`.
    """
    root = Path(root).resolve()
    directory = index_dir_for(root)
    meta = IndexStore.read_meta(directory)
    if meta is None:
        raise UserError(f"No index for {root}. Run: clicr index {root}")
    stored = meta.get("embedder", "")
    if not embedder_ok(stored):
        raise UserError(
            f"export-web needs a fastembed index (the browser runs the same model in "
            f"transformers.js); this index uses {stored!r}. "
            f"Re-run: clicr index {root} --embed-provider fastembed --rebuild"
        )
    inconsistent = UserError(
        f"The index for {root} is unreadable or inconsistent. Re-run: clicr index {root} --rebuild"
    )
    try:
        dim = int(meta["dim"])
    except (KeyError, ValueError):
        raise inconsistent from None
    header = meta.get("header", "1") == "1"
    store = IndexStore.open(
        directory,
        embedder_name=stored,
        dim=dim,
        chunker_name=meta.get("chunker", ""),
        header=header,
    )
    try:
        if not store.compatible:
            raise inconsistent
        if store.count() == 0:
            raise UserError(f"The index for {root} is empty. Run: clicr index {root}")
        model = stored.partition(":")[2] or stored
        query_prefix, doc_prefix = prefixes(model)
        src = dict(source or {})
        include = src.get("include") or "."
        src_info = {
            "name": src.get("name") or root.name,
            "repo": src.get("repo"),
            "commit": src.get("commit"),
            "license": src.get("license"),
            "include": include,
        }

        rowids, matrix = store.vector_matrix()
        position = {int(rid): i for i, rid in enumerate(rowids)}
        found = dict(store.get_chunks([int(r) for r in rowids]))
        chunks = []
        texts = []
        for i, rid in enumerate(rowids):
            c = found[int(rid)]
            chunks.append(
                {
                    "i": i,
                    "path": _repo_path(include, c.path),
                    "start": c.start_line,
                    "end": c.end_line,
                    "symbol": c.symbol,
                    "lang": c.language,
                    "text": c.text,
                }
            )
            texts.append(search_terms(_fts_text(c)))
        quant = np.clip(np.rint(matrix * 127.0), -127, 127).astype(np.int8)
        bm25 = build_bm25(texts)
        graph = _graph(directory / "index.sqlite", position)

        files: dict[str, bytes] = {
            "chunks.json": _dump(chunks),
            "vectors.bin": quant.tobytes(),
            "bm25.json": _dump(bm25),
            "graph.json": _dump(graph),
        }

        if questions is not None:
            entries = [_question_entry(q) for q in questions]
            if answerer is not None:
                entries = [record_answer(answerer, e["question"], include) for e in entries]
            files["questions.json"] = _dump(entries)
        elif answerer is not None:
            raise UserError("--record needs --questions.")

        if parity_items is not None:
            embedder = embedder_factory(stored)
            retriever = Retriever(store, embedder, None, SEARCH, rewrite=False)
            asked = [it for it in parity_items if it.get("category") != "unanswerable"]
            parity = []
            for it in asked:
                result = retriever.retrieve(it["question"])
                top = [
                    f"{_repo_path(include, c.chunk.path)}:{c.chunk.start_line}-{c.chunk.end_line}"
                    for c in result.candidates[: SEARCH.top_k]
                ]
                parity.append(
                    {
                        "question": it["question"],
                        "python_top": top,
                        "terms": search_terms(it["question"]),
                    }
                )
            qvecs = [
                [round(float(x), 6) for x in embedder.embed_query(it["question"])]
                for it in asked[:3]
            ]
            files["parity.json"] = _dump({"items": parity, "query_vectors": qvecs})
    finally:
        store.close()

    manifest = {
        "format": FORMAT,
        "source": src_info,
        "embed": {
            "fastembed_model": model,
            "dim": dim,
            "normalized": True,
            "query_prefix": query_prefix,
            "doc_prefix": doc_prefix,
            "quant": "int8/127",
        },
        "n_chunks": len(chunks),
        "bm25": {
            "k1": BM25_K1,
            "b": BM25_B,
            "n_docs": len(chunks),
            "avg_len": round(sum(bm25["doc_len"]) / len(chunks), 6),
        },
        "rrf_k": SEARCH.rrf_k,
        "k_candidates": SEARCH.k_candidates,
        "top_k": SEARCH.top_k,
        "header": header,
        "files": {name: len(data) for name, data in sorted(files.items())},
        "clicr_version": __version__,
    }
    files["manifest.json"] = _dump(manifest)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    total = sum(len(d) for d in files.values())
    if total > max_mb * 1024 * 1024:
        sizes = ", ".join(f"{n} {len(d):,}" for n, d in sorted(files.items()))
        raise UserError(
            f"The bundle is {total / 1024 / 1024:.2f} MB, over the {max_mb:g} MB limit "
            f"({sizes}). Index a smaller folder or raise --max-mb."
        )
    return manifest
