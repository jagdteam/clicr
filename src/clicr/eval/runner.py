"""Run the ablation matrix over a suite and collect the results JSON.

For each embedder and index shape (chunker + header) an index is built or reused under
<index_root>/<suite>-<sha7>-<hash>/ from the suite checkout's `include` folder. Every
config row then runs over every item. Paths in the index are relative to `include`, so
they get the `include` prefix back before they are matched against the gold.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from clicr import __version__
from clicr import models as models_mod
from clicr.answer.engine import Answerer
from clicr.eval import judge as judge_mod
from clicr.eval import metrics
from clicr.eval.configs import TOP_K, Row, history_for, search_config
from clicr.eval.suites import repo_path, source_root
from clicr.index.store import IndexStore
from clicr.index.updater import update_index
from clicr.ingest import chunkers as chunkers_mod
from clicr.ingest import languages as languages_mod
from clicr.ingest import walker as walker_mod
from clicr.ingest.chunkers import LineChunker, SyntaxChunker
from clicr.providers.base import ChatModel, Embedder
from clicr.retrieve.pipeline import Retriever
from clicr.retrieve.rerank import CrossEncoderReranker, Reranker
from clicr.retrieve.search import SearchConfig
from clicr.retrieve.types import Turn

Progress = Callable[[str, int, int], None]  # (label, done, total)
DIGITS = 4
EXACT_KEYS = frozenset({"threshold", "abstain_threshold"})  # never rounded: exact cut-offs
FINGERPRINT_PACKAGES = ("tree-sitter", "tree-sitter-language-pack")
JUDGE_NOTE = "The judge is the same local model family as the answerer, so its scores are biased."


def machine_info() -> dict:
    return {"cpu": platform.processor() or platform.machine(), "gpu": _gpu(), "os": _os()}


def _os() -> str:
    return f"{platform.system()} {platform.release()} ({platform.version()})"


def _gpu() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return ", ".join(names) or None


def make_chunker(name: str) -> LineChunker | SyntaxChunker:
    return LineChunker() if name == "line" else SyntaxChunker()


def code_fingerprint(
    files: Sequence[Path] | None = None, versions: dict[str, str] | None = None
) -> str:
    """sha1 of the code that decides what gets chunked: ingest sources + parser versions.

    Defaults: chunkers.py, walker.py, languages.py, models.py and the installed tree-sitter
    and tree-sitter-language-pack versions. Line endings are normalised so a CRLF checkout
    gives the same fingerprint.
    """
    if files is None:
        modules = (chunkers_mod, walker_mod, languages_mod, models_mod)
        files = [Path(m.__file__) for m in modules if m.__file__]
    if versions is None:
        versions = {name: importlib.metadata.version(name) for name in FINGERPRINT_PACKAGES}
    digest = hashlib.sha1()
    for path in files:
        digest.update(Path(path).name.encode() + b"\0")
        digest.update(Path(path).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    for name in sorted(versions):
        digest.update(f"{name}={versions[name]}\0".encode())
    return digest.hexdigest()


def index_dir(
    index_root: Path,
    suite: str,
    commit: str,
    embedder: str,
    chunker: str,
    header: bool,
    include: str,
    fingerprint: str,
) -> Path:
    raw = f"{embedder}|{chunker}|{int(header)}|{include}|{fingerprint}"
    key = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return Path(index_root) / f"{suite}-{commit[:7]}-{key}"


def _span_label(span: metrics.Span) -> str:
    return f"{span[0]}:{span[1]}-{span[2]}"


class _Indexes:
    """Opens (building or updating first) one index per embedder and index shape."""

    def __init__(
        self, name: str, suite: dict, src: Path, index_root: Path, fingerprint: str, progress
    ):
        self.name, self.suite, self.src = name, suite, src
        self.index_root, self.fingerprint, self.progress = index_root, fingerprint, progress

    def open(self, spec: str, embedder: Embedder, chunker_name: str, header: bool) -> IndexStore:
        chunker = make_chunker(chunker_name)
        directory = index_dir(
            self.index_root,
            self.name,
            self.suite["commit"],
            spec,
            chunker.name,
            header,
            self.suite.get("include", "."),
            self.fingerprint,
        )
        store = IndexStore.open(
            directory,
            embedder_name=embedder.name,
            dim=embedder.dim,
            chunker_name=chunker.name,
            header=header,
        )
        label = f"index {spec} {chunker.name}{' +header' if header else ''}"

        def on_progress(_stage: str, done: int, total: int) -> None:
            if self.progress is not None:
                self.progress(label, done, total)

        try:
            update_index(self.src, store, chunker, embedder, header=header, progress=on_progress)
        except BaseException:
            store.close()
            raise
        return store


def run_suite(
    name: str,
    suite: dict,
    items: Sequence[dict],
    *,
    source: Path,
    embedders: Sequence[str],
    rows: Sequence[Row],
    embedder_factory: Callable[[str], Embedder],
    chat: ChatModel | None = None,
    reranker: Reranker | None = None,
    index_root: Path,
    answers: bool = False,
    judge: ChatModel | None = None,
    top_k: int = TOP_K,
    machine: dict | None = None,
    progress: Progress | None = None,
) -> dict:
    """Run every row over every item for every embedder; return the results dict."""
    chat_rows = [row.name for row in rows if row.needs_chat]
    if chat is None and chat_rows:
        raise ValueError(f"config(s) {', '.join(chat_rows)} need a chat model")
    if chat is None and answers:
        raise ValueError("the answer track needs a chat model")
    if judge is not None and not answers:
        raise ValueError("the judge scores the answer track: run with answers")
    include = suite.get("include", ".")
    src = source_root(source, include)
    fingerprint = code_fingerprint()
    indexes = _Indexes(name, suite, src, index_root, fingerprint, progress)
    if reranker is None and any(row.rerank == "cross" for row in rows):
        reranker = CrossEncoderReranker(SearchConfig().rerank_model)

    records: list[dict] = []
    row_results: list[dict] = []
    embedder_cache: dict[str, Embedder] = {}

    def embedder_for(spec: str) -> Embedder:
        if spec not in embedder_cache:
            embedder_cache[spec] = embedder_factory(spec)
        return embedder_cache[spec]

    for spec in embedders:
        embedder = embedder_for(spec)
        for (chunker_name, header), group in itertools.groupby(rows, key=lambda r: r.shape):
            store = indexes.open(spec, embedder, chunker_name, header)
            try:
                for row in group:
                    retriever = Retriever(
                        store,
                        embedder,
                        chat if row.needs_chat else None,
                        search_config(row, top_k),
                        rewrite=row.rewrite,
                        reranker=reranker if row.rerank == "cross" else None,
                    )
                    label = f"{spec} {row.name}"
                    row_records = []
                    for done, item in enumerate(items, start=1):
                        row_records.append(_retrieve_item(retriever, row, spec, item, include))
                        if progress is not None:
                            progress(label, done, len(items))
                    records.extend(row_records)
                    row_results.append(_row_result(row, spec, row_records))
            finally:
                store.close()

    best = _best_row(row_results)
    abstain = _abstain(best, rows, records) if best else None
    answer_section = None
    if answers and best is not None:
        assert chat is not None
        threshold = abstain["threshold"] if abstain else None
        row = next(r for r in rows if r.name == best["config"])
        cfg = search_config(row, top_k, abstain=threshold)
        if threshold is None:
            cfg = replace(cfg, abstain_threshold=None)  # nothing calibrated: never abstain
        embedder = embedder_for(best["embedder"])
        store = indexes.open(best["embedder"], embedder, row.chunker, row.header)
        try:
            retriever = Retriever(
                store,
                embedder,
                chat,
                cfg,
                rewrite=row.rewrite,
                reranker=reranker if row.rerank == "cross" else None,
            )
            answerer = Answerer(retriever, chat, root=src)
            answer_section = _answer_track(answerer, items, include, judge, progress)
        finally:
            store.close()
        answer_section.update(
            chat_model=chat.name,
            judge_model=judge.name if judge is not None else None,
            judge_note=JUDGE_NOTE if judge is not None else None,
            config=best["config"],
            embedder=best["embedder"],
            abstain_threshold=threshold,
        )

    return {
        "suite": name,
        "commit": suite["commit"],
        "ref": suite.get("ref"),
        "include": include,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "clicr_version": __version__,
        "machine": machine if machine is not None else machine_info(),
        "models": {
            "embedders": list(embedders),
            "chat": chat.name if chat is not None else None,
            "reranker": (
                SearchConfig().rerank_model if any(r.rerank == "cross" for r in rows) else None
            ),
            "judge": judge.name if judge is not None else None,
        },
        "settings": {"top_k": top_k, "k_candidates": max(20, top_k)},
        "index_fingerprint": fingerprint,
        "rows": row_results,
        "best": best,
        "abstain": abstain,
        "answers": answer_section,
        "items": records,
    }


def _ranked(candidates, include: str) -> list[metrics.Span]:
    return [
        (repo_path(include, c.chunk.path), c.chunk.start_line, c.chunk.end_line) for c in candidates
    ]


def _retrieve_item(retriever: Retriever, row: Row, spec: str, item: dict, include: str) -> dict:
    result = retriever.retrieve(item["question"], history_for(row, item))
    ranked = _ranked(result.candidates, include)
    score = metrics.score_retrieval(ranked, metrics.gold_spans(item["gold"]))
    return {
        "id": item["id"],
        "category": item["category"],
        "config": row.name,
        "embedder": spec,
        "ranked": [_span_label(span) for span in ranked],
        **score,
        "signal": result.abstain_signal,
        "timings_ms": dict(result.timings_ms),
    }


def _row_result(row: Row, spec: str, records: list[dict]) -> dict:
    return {
        "config": row.name,
        "embedder": spec,
        "settings": row.as_dict(),
        "metrics": metrics.aggregate(records),
        "latency": metrics.latency([r["timings_ms"] for r in records]),
    }


def _best_row(rows: list[dict]) -> dict | None:
    """Highest overall Recall@10, then MRR; ties keep the earlier row."""
    if not rows:
        return None
    best = max(
        enumerate(rows),
        key=lambda pair: (
            pair[1]["metrics"]["overall"]["recall@10"],
            pair[1]["metrics"]["overall"]["mrr"],
            -pair[0],
        ),
    )[1]
    return {"config": best["config"], "embedder": best["embedder"]}


def _abstain(best: dict, rows: Sequence[Row], records: list[dict]) -> dict:
    """Calibrate the abstain threshold on the best row's signals."""
    mine = [
        r for r in records if r["config"] == best["config"] and r["embedder"] == best["embedder"]
    ]
    row = next(r for r in rows if r.name == best["config"])
    unanswerable = [not r["gold_ranks"] for r in mine]
    out: dict = {
        "config": best["config"],
        "embedder": best["embedder"],
        "signal": "cross-encoder score" if row.rerank == "cross" else "vector cosine",
        "n_unanswerable": sum(unanswerable),
        "n_items": len(mine),
    }
    calibrated = metrics.calibrate_abstain([r["signal"] for r in mine], unanswerable)
    if calibrated is None:
        out.update(threshold=None, precision=None, recall=None, f1=None)
        out["metrics_at_threshold"] = None
        return out
    out.update({k: calibrated[k] for k in ("threshold", "precision", "recall", "f1")})
    out["metrics_at_threshold"] = metrics.aggregate(
        metrics.apply_threshold(mine, calibrated["threshold"])
    )
    return out


def _answer_track(
    answerer: Answerer,
    items: Sequence[dict],
    include: str,
    judge: ChatModel | None,
    progress: Progress | None,
) -> dict:
    records = []
    for done, item in enumerate(items, start=1):
        history = [Turn(t["q"], t["a"]) for t in item.get("history", [])]
        result = answerer.ask(item["question"], history)
        sources = _ranked(result.sources, include)
        gold = metrics.gold_spans(item["gold"])
        rec = {
            "id": item["id"],
            "category": item["category"],
            "answer": result.answer,
            "sources": [_span_label(span) for span in sources],
            "cited": list(result.citations.cited),
            "abstained": result.result.abstained,
            "not_found": metrics.is_not_found(result.answer),
            **metrics.citation_metrics(
                result.citations.cited, result.citations.invalid, sources, gold
            ),
            "judged": False,
            "faithfulness": None,
            "correctness": None,
        }
        if judge is not None and item["gold"]:
            rec["judged"] = True
            rec.update(
                judge_mod.judge_answer(
                    judge,
                    item["question"],
                    result.sources,
                    result.answer,
                    item["reference_answer"],
                )
            )
        records.append(rec)
        if progress is not None:
            progress("answers", done, len(items))
    return {"summary": metrics.summarize_answers(records), "items": records}


def _clean(value, exact: bool = False):
    """Round floats to DIGITS places (not under EXACT_KEYS); non-finite floats become None."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value if exact else round(value, DIGITS)
    if isinstance(value, dict):
        return {str(k): _clean(v, k in EXACT_KEYS) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def dump_results(results: dict) -> str:
    """Stable JSON text: rounded floats, sorted keys, so reruns diff cleanly."""
    return json.dumps(_clean(results), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_results(path: Path, results: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_results(results), encoding="utf-8", newline="\n")
