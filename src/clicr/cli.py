import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from clicr import __version__
from clicr.config import ConfigError, Settings, load_settings
from clicr.ingest.languages import SYNTAX_LANGUAGES
from clicr.paths import clicr_home, index_dir_for
from clicr.ui.console import make_console
from clicr.ui.errors import UserError, friendly

if TYPE_CHECKING:
    from clicr.answer.engine import Answerer

app = typer.Typer(
    help="CliCR — chat with a local codebase using retrieval-augmented generation.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


def _stub(name: str) -> None:
    console.print(f"{name}: not implemented yet")
    raise typer.Exit(code=2)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """CliCR — chat with a local codebase using retrieval-augmented generation."""


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=1)


_providers: tuple[str, ...] = ()  # configured chat/embed providers, for error messages


def _note_providers(settings: Settings) -> None:
    global _providers
    _providers = (settings.chat_provider, settings.embed_provider)


@contextmanager
def _friendly_errors() -> Iterator[None]:
    """Expected failures print one red line plus a hint and exit 1; others keep their traceback."""
    try:
        yield
    except Exception as exc:
        message = friendly(exc, _providers)
        if message is None:
            raise
        first, _, hint = message.partition("\n")
        err = Console(stderr=True, highlight=False)
        err.print(first, style="red", markup=False, soft_wrap=True)
        if hint:
            err.print(hint, style="dim", markup=False, soft_wrap=True)
        raise typer.Exit(code=1) from exc


def _overrides(**values: object) -> dict[str, object]:
    return {k: v for k, v in values.items() if v is not None}


@app.command()
def index(
    path: Annotated[str, typer.Argument(help="Folder to index.")] = ".",
    chunker: Annotated[
        str | None, typer.Option("--chunker", help="Chunking strategy: syntax or line.")
    ] = None,
    embed_provider: Annotated[
        str | None, typer.Option("--embed-provider", help="ollama, fastembed or fake.")
    ] = None,
    embed_model: Annotated[str | None, typer.Option("--embed-model")] = None,
    no_header: Annotated[
        bool, typer.Option("--no-header", help="Do not prepend a path/symbol header to chunks.")
    ] = False,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Discard the existing index and start over.")
    ] = False,
    watch: Annotated[
        bool, typer.Option("--watch", help="Keep running and re-index when files change.")
    ] = False,
) -> None:
    """Index a codebase."""
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from clicr.index.store import IndexStore
    from clicr.index.updater import update_index
    from clicr.providers import make_chunker, make_embedder

    root = Path(path).resolve()
    if not root.is_dir():
        raise _fail(f"Not a folder: {root}")
    with _friendly_errors():
        settings = load_settings(
            root,
            _overrides(
                chunker=chunker,
                embed_provider=embed_provider,
                embed_model=embed_model,
                header=False if no_header else None,
            ),
        )
        _note_providers(settings)
        embedder = make_embedder(settings)
        chunker_obj = make_chunker(settings)
        directory = index_dir_for(root)
        store = IndexStore.open(
            directory,
            embedder_name=embedder.name,
            dim=embedder.dim,
            chunker_name=chunker_obj.name,
            header=settings.header,
        )
        try:
            if rebuild:
                store.reset()
            started = time.perf_counter()
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as bar:
                task = bar.add_task("Indexing", total=None)

                def on_progress(_stage: str, done: int, total: int) -> None:
                    bar.update(task, completed=done, total=total)

                stats = update_index(
                    root,
                    store,
                    chunker_obj,
                    embedder,
                    header=settings.header,
                    progress=on_progress,
                )
            elapsed = time.perf_counter() - started
            typer.echo(
                f"Indexed {root}: {stats.added} added, {stats.changed} changed, "
                f"{stats.deleted} deleted, {stats.unchanged} unchanged "
                f"({stats.chunks_embedded} chunks embedded, {store.count()} total)"
            )
            typer.echo(f"Index: {directory}")
            typer.echo(f"Elapsed: {elapsed:.1f}s")
            if watch:
                _watch(root, directory, store, chunker_obj, embedder, settings.header)
        finally:
            store.close()


def _watch(root: Path, directory: Path, store, chunker_obj, embedder, header: bool) -> None:
    import watchfiles

    from clicr.index.updater import update_index
    from clicr.watch import run_batch

    def update():
        return update_index(root, store, chunker_obj, embedder, header=header)

    typer.echo(f"Watching {root} for changes (Ctrl-C to stop)")
    try:
        for batch in watchfiles.watch(root):
            line = run_batch(
                [path for _, path in batch],
                root,
                directory,
                update,
                lambda: set(store.file_hashes()),
            )
            if line is not None:
                typer.echo(line)
    except KeyboardInterrupt:
        pass
    typer.echo("Stopped watching.")


PathOpt = Annotated[str, typer.Option("--path", help="Indexed folder.")]
TopKOpt = Annotated[int | None, typer.Option("--top-k", help="Chunks to retrieve.")]
ModeOpt = Annotated[
    Literal["hybrid", "vector", "keyword"], typer.Option("--mode", help="Retrieval mode.")
]
ChatProviderOpt = Annotated[
    str | None, typer.Option("--chat-provider", help="ollama, claude or fake.")
]
ChatModelOpt = Annotated[str | None, typer.Option("--chat-model")]
EmbedProviderOpt = Annotated[
    str | None, typer.Option("--embed-provider", help="ollama, fastembed or fake.")
]
EmbedModelOpt = Annotated[str | None, typer.Option("--embed-model")]
RerankOpt = Annotated[
    str | None, typer.Option("--rerank", help="Rerank the fused candidates: none, cross or llm.")
]
GraphOpt = Annotated[
    bool | None,
    typer.Option("--graph/--no-graph", help="Add definitions of functions the top chunks call."),
]
HydeOpt = Annotated[
    bool | None,
    typer.Option("--hyde/--no-hyde", help="Search with a model-drafted hypothetical snippet."),
]
AbstainOpt = Annotated[
    float | None,
    typer.Option("--abstain-threshold", help="Answer 'not found' below this best score."),
]
PlainOpt = Annotated[
    bool, typer.Option("--plain", help="Plain text, no colors or panels (automatic when piped).")
]


@contextmanager
def _open_answerer(
    path: str,
    mode: str,
    embed_provider: str | None,
    embed_model: str | None,
    overrides: dict[str, object],
) -> Iterator[tuple["Answerer", Settings, Path]]:
    """Open the index and build the answerer; shared by `ask` and `chat`."""
    from clicr.answer.engine import Answerer
    from clicr.index.store import IndexStore
    from clicr.providers import embedder_from_name, make_chat, make_embedder
    from clicr.retrieve.pipeline import Retriever
    from clicr.retrieve.search import SearchConfig

    root = Path(path).resolve()
    directory = index_dir_for(root)
    meta = IndexStore.read_meta(directory)
    if meta is None:
        raise UserError(f"No index for {root}. Run: clicr index {root}")
    inconsistent = UserError(
        f"The index for {root} is unreadable or inconsistent. Re-run: clicr index {root} --rebuild"
    )
    try:
        stored_dim = int(meta["dim"])
        stored = meta["embedder"]
    except (KeyError, ValueError):
        raise inconsistent from None
    settings = load_settings(
        root, _overrides(embed_provider=embed_provider, embed_model=embed_model, **overrides)
    )
    _note_providers(settings)
    if embed_provider is not None or embed_model is not None:
        embedder = make_embedder(settings)
        if embedder.name != stored:
            raise UserError(
                f"The index for {root} was built with embedder {stored!r}, not "
                f"{embedder.name!r}. Re-run: clicr index {root} --rebuild"
            )
    else:
        embedder = embedder_from_name(stored, settings)
    chat = make_chat(settings)
    # chunker and header only matter at index time: open with the stored values
    store = IndexStore.open(
        directory,
        embedder_name=stored,
        dim=stored_dim,
        chunker_name=meta.get("chunker", ""),
        header=meta.get("header", "1") == "1",
    )
    try:
        if not store.compatible:
            raise inconsistent
        if store.count() == 0:
            raise UserError(f"The index for {root} is empty. Run: clicr index {root}")
        cfg = SearchConfig(
            mode=mode,
            top_k=settings.top_k,
            k_candidates=max(20, settings.top_k),
            rerank=settings.rerank,  # validated by _CHOICES in config
            rerank_model=settings.rerank_model,
            rerank_pool=settings.rerank_pool,
            graph=settings.graph,
            graph_budget=settings.graph_budget,
            hyde=settings.hyde,
            abstain_threshold=settings.abstain_threshold,
            context_tokens=settings.context_tokens,
        )
        yield Answerer(Retriever(store, embedder, chat, cfg), chat, root=root), settings, root
    finally:
        store.close()


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    path: PathOpt = ".",
    top_k: TopKOpt = None,
    mode: ModeOpt = "hybrid",
    chat_provider: ChatProviderOpt = None,
    chat_model: ChatModelOpt = None,
    embed_provider: EmbedProviderOpt = None,
    embed_model: EmbedModelOpt = None,
    rerank: RerankOpt = None,
    graph: GraphOpt = None,
    hyde: HydeOpt = None,
    abstain_threshold: AbstainOpt = None,
    explain: Annotated[
        bool, typer.Option("--explain", help="Show how every candidate was ranked.")
    ] = False,
    plain: PlainOpt = False,
) -> None:
    """Ask a single question about the indexed codebase."""
    from clicr.ui.render import source_panels, status_line, stream_answer

    out = make_console(plain)
    overrides = dict(
        top_k=top_k,
        chat_provider=chat_provider,
        chat_model=chat_model,
        rerank=rerank,
        graph=graph,
        hyde=hyde,
        abstain_threshold=abstain_threshold,
    )
    with (
        _friendly_errors(),
        _open_answerer(path, mode, embed_provider, embed_model, overrides) as (answerer, _, _),
    ):
        streamed = stream_answer(
            out,
            answerer.ask_stream(question),
            not out.is_terminal,
            on_retrieved=(
                (lambda ev: out.print(_explain_table(ev.result, ev.sources))) if explain else None
            ),
        )
    if streamed.done is not None:
        source_panels(out, streamed.done.sources, streamed.done.citations)
        status_line(out, streamed.done.citations, streamed.done.timings_ms)


def _explain_table(result, sources) -> Table:
    """Every candidate with its rank at each stage and the source number it got (or -)."""
    source_of: dict[int, int] = {}
    for n, cand in enumerate(sources, start=1):
        for rowid in cand.trace.get("merged_from", [cand.rowid]):
            source_of.setdefault(rowid, n)
    rows = list(result.pool)
    in_pool = {c.rowid for c in rows}
    rows += [c for c in result.candidates if c.rowid not in in_pool]  # graph additions

    def show(value: object) -> str:
        return "-" if value is None else str(value)

    title = "retrieval explain"
    if result.abstained:
        title += f" (abstained: best score {result.abstain_signal:.3f})"
    table = Table(title=title)
    for name in ("path:lines", "symbol", "vector rank", "keyword rank", "fused rank"):
        table.add_column(name)
    for name in ("rerank score", "graph via", "source"):
        table.add_column(name)
    for cand in rows:
        c = cand.chunk
        rerank = cand.trace.get("rerank")
        graph = cand.trace.get("graph")
        fuse = cand.trace.get("fuse")
        table.add_row(
            escape(f"{c.path}:{c.start_line}-{c.end_line}"),
            escape(c.symbol or "-"),
            show(cand.vector_rank),
            show(cand.keyword_rank),
            show(fuse["fused_rank"] if isinstance(fuse, dict) else None),
            f"{rerank['rerank_score']:.3f}" if isinstance(rerank, dict) else "-",
            escape(str(graph["via"])) if isinstance(graph, dict) else "-",
            show(source_of.get(cand.rowid)),
        )
    return table


@app.command()
def chat(
    path: PathOpt = ".",
    top_k: TopKOpt = None,
    mode: ModeOpt = "hybrid",
    chat_provider: ChatProviderOpt = None,
    chat_model: ChatModelOpt = None,
    embed_provider: EmbedProviderOpt = None,
    embed_model: EmbedModelOpt = None,
    rerank: RerankOpt = None,
    graph: GraphOpt = None,
    hyde: HydeOpt = None,
    abstain_threshold: AbstainOpt = None,
    plain: PlainOpt = False,
) -> None:
    """Start an interactive chat session."""
    from clicr.chat import run_repl, stdin_reader
    from clicr.providers import make_chat
    from clicr.sessions import new_session_id, session_path

    out = make_console(plain)
    overrides = dict(
        top_k=top_k,
        chat_provider=chat_provider,
        chat_model=chat_model,
        rerank=rerank,
        graph=graph,
        hyde=hyde,
        abstain_threshold=abstain_threshold,
    )
    with (
        _friendly_errors(),
        _open_answerer(path, mode, embed_provider, embed_model, overrides) as (
            answerer,
            settings,
            _,
        ),
    ):
        if sys.stdin.isatty():
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory

            clicr_home().mkdir(parents=True, exist_ok=True)
            prompt = PromptSession(history=FileHistory(str(clicr_home() / "history")))
            read_line = lambda: prompt.prompt("clicr> ")  # noqa: E731
        else:  # piped input: prompt_toolkit needs a real console
            read_line = stdin_reader()
        session_id = new_session_id()

        def switch_model(name: str):
            field = "claude_model" if settings.chat_provider == "claude" else "chat_model"
            return make_chat(replace(settings, **{field: name}))

        run_repl(answerer, out, read_line, session_id, switch_model=switch_model)
    if session_path(session_id).is_file():
        typer.echo(f"Session saved: {session_path(session_id)}")


@app.command(name="eval")
def eval_(
    suite: Annotated[str, typer.Option("--suite", help="Suite name from eval/suites.toml.")],
    retrieval_only: Annotated[
        bool, typer.Option("--retrieval-only", help="Retrieval metrics only, no answer track.")
    ] = False,
    embedders: Annotated[
        str | None, typer.Option("--embedders", help="Comma-separated embedders (default: all).")
    ] = None,
    configs: Annotated[
        str | None, typer.Option("--configs", help="Comma-separated configs (default: all).")
    ] = None,
    answers: Annotated[
        bool, typer.Option("--answers", help="Also run the answer track on the best row.")
    ] = False,
    judge: Annotated[
        bool, typer.Option("--judge", help="Score the answers with the LLM judge.")
    ] = False,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Rerun the CPU-only rows; compare with the committed results (exit 1 on drift).",
        ),
    ] = False,
    out: Annotated[
        str | None,
        typer.Option("--out", help="Results file (default: eval/results/<suite>.json)."),
    ] = None,
) -> None:
    """Run retrieval and answer evaluations."""
    from clicr.eval.command import run_command

    with _friendly_errors():
        code = run_command(
            suite,
            retrieval_only=retrieval_only,
            embedders=embedders,
            configs=configs,
            answers=answers,
            judge=judge,
            check=check,
            out=out,
            console=console,
        )
    raise typer.Exit(code=code)


Check = tuple[str, str, str]  # (name, status OK|WARN|FAIL, detail)


def _http_json(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed http host
        return json.loads(resp.read().decode("utf-8"))


def _model_pulled(model: str, tags: set[str]) -> bool:
    return model in tags or (":" not in model and f"{model}:latest" in tags)


def _missing_grammars() -> list[str]:
    import tree_sitter_language_pack as tslp

    have = set(tslp.downloaded_languages())
    return sorted(SYNTAX_LANGUAGES - have)


def run_checks(settings: Settings, root: Path) -> list[Check]:
    checks: list[Check] = []
    v = sys.version_info
    checks.append(("Python", "OK" if v >= (3, 12) else "FAIL", f"{v.major}.{v.minor}.{v.micro}"))

    needs_ollama = "ollama" in (settings.embed_provider, settings.chat_provider)
    tags: set[str] | None = None
    host = settings.ollama_host.rstrip("/")
    try:
        version = _http_json(f"{host}/api/version").get("version", "?")
        checks.append(("Ollama server", "OK", f"v{version} at {host}"))
        tags = {m["name"] for m in _http_json(f"{host}/api/tags").get("models", [])}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        status = "FAIL" if needs_ollama else "WARN"
        checks.append(
            ("Ollama server", status, f"not reachable at {host} ({exc}). Fix: ollama serve")
        )

    wanted = []
    if settings.embed_provider == "ollama":
        wanted.append(("Embedding model", settings.embed_model))
    if settings.chat_provider == "ollama":
        wanted.append(("Chat model", settings.chat_model))
    for label, model in wanted:
        if tags is None:
            checks.append((label, "FAIL", f"{model}: cannot check (Ollama unreachable)"))
        elif _model_pulled(model, tags):
            checks.append((label, "OK", f"{model} is pulled"))
        else:
            checks.append((label, "FAIL", f"{model} is missing. Fix: ollama pull {model}"))

    try:
        missing = _missing_grammars()
        if missing:
            checks.append(
                (
                    "Tree-sitter grammars",
                    "WARN",
                    f"missing: {', '.join(missing)}. Fix: clicr doctor --fix",
                )
            )
        else:
            checks.append(("Tree-sitter grammars", "OK", f"all {len(SYNTAX_LANGUAGES)} present"))
    except Exception as exc:  # grammar pack unusable
        checks.append(("Tree-sitter grammars", "FAIL", str(exc)))

    try:
        import fastembed  # noqa: F401

        checks.append(("fastembed", "OK", "importable"))
    except ImportError:
        status = "FAIL" if settings.embed_provider == "fastembed" else "WARN"
        checks.append(("fastembed", status, "not installed. Fix: uv add fastembed"))

    if os.environ.get("ANTHROPIC_API_KEY"):
        checks.append(("ANTHROPIC_API_KEY", "OK", "set"))
    else:
        checks.append(("ANTHROPIC_API_KEY", "WARN", "not set (only needed for Claude)"))

    db = index_dir_for(root) / "index.sqlite"
    if db.is_file():
        try:
            conn = sqlite3.connect(db)
            try:
                n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            finally:
                conn.close()
            checks.append(("Index", "OK", f"{n} chunks ({db.parent})"))
        except sqlite3.Error as exc:
            checks.append(("Index", "FAIL", f"unreadable: {exc}"))
    else:
        checks.append(("Index", "WARN", f"none for {root.resolve()}. Fix: clicr index"))
    return checks


def _fix_grammars() -> None:
    import tree_sitter_language_pack as tslp

    missing = _missing_grammars()
    if missing:
        console.print(f"Downloading grammars: {', '.join(missing)}")
        tslp.download(missing)


@app.command()
def doctor(
    fix: Annotated[
        bool, typer.Option("--fix", help="Download missing tree-sitter grammars.")
    ] = False,
) -> None:
    """Check that the environment is set up correctly."""
    root = Path.cwd()
    extra: list[Check] = []
    try:
        settings = load_settings(root, {})
    except ConfigError as exc:
        extra.append(("Config", "FAIL", str(exc)))
        settings = Settings()
    if fix:
        try:
            _fix_grammars()
        except Exception as exc:
            extra.append(("Grammar download", "FAIL", f"{type(exc).__name__}: {exc}"))
    checks = extra + run_checks(settings, root)
    table = Table(title="clicr doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    colors = {"OK": "green", "WARN": "yellow", "FAIL": "red"}
    for name, status, detail in checks:
        table.add_row(name, f"[{colors[status]}]{status}[/{colors[status]}]", detail)
    console.print(table)
    if any(status == "FAIL" for _, status, _ in checks):
        raise typer.Exit(code=1)


@app.command(name="export-web")
def export_web() -> None:
    """Export the index for the web demo."""
    _stub("export-web")
