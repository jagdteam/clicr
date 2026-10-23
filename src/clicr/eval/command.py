"""`clicr eval`: pick the suite, embedders and configs, run them, write or check the results."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from clicr.config import Settings, load_settings
from clicr.eval import configs as configs_mod
from clicr.eval import runner
from clicr.eval.check import TOLERANCE, compare_results
from clicr.eval.suites import DATASET_DIR, RESULTS_DIR, SUITES_FILE, ensure_checkout, load_items
from clicr.eval.suites import load_suites as _load_suites
from clicr.providers.base import ChatModel, Embedder
from clicr.retrieve.rerank import CrossEncoderReranker, Reranker
from clicr.ui.errors import UserError


@dataclass
class Hooks:
    """Where the models and the checkout come from; tests swap in fakes."""

    checkout: Callable[[str, dict], Path] = ensure_checkout
    embedder_factory: Callable[[str], Embedder] | None = None  # default: from the name
    chat_factory: Callable[[], ChatModel] | None = None  # default: configured chat model
    judge_factory: Callable[[], ChatModel] | None = None  # default: chat model at temperature 0
    reranker: Reranker | None = None  # default: the cross-encoder from SearchConfig
    index_root: Path | None = None  # default: ~/.clicr/eval-index
    machine: dict | None = None  # default: detected


def _split(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _settings() -> Settings:
    return load_settings(None, {})


def _default_judge() -> ChatModel:
    settings = _settings()
    if settings.chat_provider == "ollama":
        from clicr.providers.ollama import OllamaChat

        return OllamaChat(settings.chat_model, host=settings.ollama_host, temperature=0.0)
    if settings.chat_provider == "fake":
        from clicr.providers.fake import FakeChat

        return FakeChat()
    raise UserError("The judge runs on the local Ollama chat model (temperature 0).")


def _check_rerank_model(name: str) -> None:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    valid = sorted(entry["model"] for entry in TextCrossEncoder.list_supported_models())
    if name not in valid:
        raise UserError(f"Unknown rerank model {name!r}; known: {', '.join(valid)}")


def _fill(hooks: Hooks) -> Hooks:
    from clicr.paths import clicr_home
    from clicr.providers import embedder_from_name, make_chat

    return Hooks(
        checkout=hooks.checkout,
        embedder_factory=hooks.embedder_factory
        or (lambda name: embedder_from_name(name, _settings())),
        chat_factory=hooks.chat_factory or (lambda: make_chat(_settings())),
        judge_factory=hooks.judge_factory or _default_judge,
        reranker=hooks.reranker,
        index_root=hooks.index_root or clicr_home() / "eval-index",
        machine=hooks.machine,
    )


class _Bars:
    """Rich progress: one bar per label (an index build or a config row)."""

    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.tasks: dict[str, TaskID] = {}

    def __call__(self, label: str, done: int, total: int) -> None:
        if label not in self.tasks:
            self.tasks[label] = self.progress.add_task(label, total=total)
        self.progress.update(self.tasks[label], completed=done, total=total)


def _summary(console: Console, results: dict) -> None:
    table = Table(title=f"clicr eval: {results['suite']}")
    for name in ("Embedder", "Config", "Recall@5", "Recall@10", "MRR", "p95 ms"):
        table.add_column(name)
    for row in results["rows"]:
        overall = row["metrics"]["overall"]
        table.add_row(
            row["embedder"],
            row["config"],
            f"{overall['recall@5']:.3f}",
            f"{overall['recall@10']:.3f}",
            f"{overall['mrr']:.3f}",
            f"{row['latency']['total']['p95']:.1f}",
        )
    console.print(table)
    abstain = results.get("abstain")
    if abstain and abstain.get("threshold") is not None:
        console.print(
            f"Abstain: threshold {abstain['threshold']:.4f} on {abstain['signal']} "
            f"({abstain['embedder']} {abstain['config']}): precision {abstain['precision']:.3f}, "
            f"recall {abstain['recall']:.3f}, F1 {abstain['f1']:.3f}",
            markup=False,
        )
    answers = results.get("answers")
    if answers:
        console.print(
            f"Answer track: {json.dumps(answers['summary'], sort_keys=True)}", markup=False
        )


def _diff_table(console: Console, diffs: list) -> None:
    def show(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    rows = sorted({d.row for d in diffs})
    table = Table(title=f"clicr eval --check (tolerance {TOLERANCE})")
    for name in ("Row", "Metrics", "Misses", "Worst delta"):
        table.add_column(name)
    for row in rows:
        mine = [d for d in diffs if d.row == row]
        deltas = [abs(d.delta) for d in mine if d.delta is not None]
        misses = sum(1 for d in mine if not d.ok)
        table.add_row(row, str(len(mine)), str(misses), show(max(deltas) if deltas else None))
    console.print(table)
    misses = [d for d in diffs if not d.ok]
    if misses:
        detail = Table(title="Misses")
        for name in ("Row", "Metric", "Committed", "Fresh", "Delta"):
            detail.add_column(name)
        for d in misses:
            detail.add_row(d.row, d.metric, show(d.committed), show(d.fresh), show(d.delta))
        console.print(detail)


def run_command(
    suite: str,
    *,
    retrieval_only: bool,
    embedders: str | None,
    configs: str | None,
    answers: bool,
    judge: bool,
    check: bool,
    out: str | None,
    rerank_model: str | None = None,
    console: Console,
    suites_file: Path = SUITES_FILE,
    dataset_dir: Path = DATASET_DIR,
    results_dir: Path = RESULTS_DIR,
    hooks: Hooks | None = None,
) -> int:
    """Run (or --check) one suite. Returns the exit code."""
    if retrieval_only and (answers or judge):
        raise UserError("--retrieval-only cannot be combined with --answers or --judge.")
    if judge and not answers:
        raise UserError("--judge scores the answer track: add --answers.")
    if check and (answers or judge):
        raise UserError("--check reruns retrieval only: drop --answers and --judge.")
    suites = _load_suites(suites_file)
    if suite not in suites:
        raise UserError(f"Unknown suite {suite!r}; known: {', '.join(suites)}")
    spec = suites[suite]
    dataset = Path(dataset_dir) / f"{suite}.jsonl"
    if not dataset.is_file():
        raise UserError(f"Dataset not found: {dataset}")
    items = load_items(dataset)
    out_path = Path(out) if out else Path(results_dir) / f"{suite}.json"

    names = _split(embedders) or list(configs_mod.EMBEDDERS)
    try:
        rows = configs_mod.get_rows(_split(configs))
    except ValueError as exc:
        raise UserError(str(exc)) from None
    if check:
        # CPU-only and deterministic: fastembed embedders, configs without a chat model
        names = [n for n in names if configs_mod.is_cpu_embedder(n)]
        rows = [r for r in rows if not r.needs_chat]
        if not names or not rows:
            raise UserError("--check needs at least one fastembed embedder and one no-chat config.")
        if not out_path.is_file():
            console.print(f"No committed results to check against: {out_path}", markup=False)
            return 1

    hooks = _fill(hooks or Hooks())
    if rerank_model is not None:
        _check_rerank_model(rerank_model)
        if hooks.reranker is None:
            hooks.reranker = CrossEncoderReranker(rerank_model)
    assert hooks.embedder_factory and hooks.chat_factory and hooks.judge_factory
    assert hooks.index_root is not None
    needs_chat = answers or any(r.needs_chat for r in rows)
    chat = hooks.chat_factory() if needs_chat else None
    judge_model = hooks.judge_factory() if judge else None
    checkout = hooks.checkout(suite, spec)

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        results = runner.run_suite(
            suite,
            spec,
            items,
            source=checkout,
            embedders=names,
            rows=rows,
            embedder_factory=hooks.embedder_factory,
            chat=chat,
            reranker=hooks.reranker,
            index_root=hooks.index_root,
            answers=answers,
            judge=judge_model,
            machine=hooks.machine,
            progress=_Bars(progress),
        )

    if check:
        committed = json.loads(out_path.read_text(encoding="utf-8"))
        fresh = json.loads(runner.dump_results(results))  # rounded like the committed file
        diffs = compare_results(committed, fresh)
        _diff_table(console, diffs)
        missed = sum(1 for d in diffs if not d.ok)
        if missed:
            console.print(f"FAIL: {missed} metric(s) outside {TOLERANCE}", markup=False)
            return 1
        console.print(f"OK: {len(diffs)} metrics within {TOLERANCE} of {out_path}", markup=False)
        return 0

    runner.write_results(out_path, results)
    _summary(console, results)
    console.print(f"Results: {out_path}", markup=False)
    return 0
