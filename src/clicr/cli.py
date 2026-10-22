import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from clicr import __version__
from clicr.config import ConfigError, Settings, load_settings
from clicr.ingest.languages import SYNTAX_LANGUAGES
from clicr.paths import index_dir_for
from clicr.providers.base import ProviderError

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
) -> None:
    """Index a codebase."""
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from clicr.index.store import IndexStore
    from clicr.index.updater import update_index
    from clicr.providers import make_chunker, make_embedder

    root = Path(path).resolve()
    if not root.is_dir():
        raise _fail(f"Not a folder: {root}")
    try:
        settings = load_settings(
            root,
            _overrides(
                chunker=chunker,
                embed_provider=embed_provider,
                embed_model=embed_model,
                header=False if no_header else None,
            ),
        )
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
            total_chunks = store.count()
        finally:
            store.close()
    except (ProviderError, ConfigError) as exc:
        raise _fail(f"Error: {exc}") from exc
    typer.echo(
        f"Indexed {root}: {stats.added} added, {stats.changed} changed, "
        f"{stats.deleted} deleted, {stats.unchanged} unchanged "
        f"({stats.chunks_embedded} chunks embedded, {total_chunks} total)"
    )
    typer.echo(f"Index: {directory}")
    typer.echo(f"Elapsed: {elapsed:.1f}s")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    path: Annotated[str, typer.Option("--path", help="Indexed folder.")] = ".",
    top_k: Annotated[int | None, typer.Option("--top-k", help="Chunks to retrieve.")] = None,
    mode: Annotated[
        Literal["hybrid", "vector", "keyword"], typer.Option("--mode", help="Retrieval mode.")
    ] = "hybrid",
    chat_provider: Annotated[
        str | None, typer.Option("--chat-provider", help="ollama, claude or fake.")
    ] = None,
    chat_model: Annotated[str | None, typer.Option("--chat-model")] = None,
    embed_provider: Annotated[
        str | None, typer.Option("--embed-provider", help="ollama, fastembed or fake.")
    ] = None,
    embed_model: Annotated[str | None, typer.Option("--embed-model")] = None,
    rerank: Annotated[
        str | None,
        typer.Option("--rerank", help="Rerank the fused candidates: none, cross or llm."),
    ] = None,
    graph: Annotated[
        bool | None,
        typer.Option(
            "--graph/--no-graph", help="Add definitions of functions the top chunks call."
        ),
    ] = None,
    hyde: Annotated[
        bool | None,
        typer.Option("--hyde/--no-hyde", help="Search with a model-drafted hypothetical snippet."),
    ] = None,
    abstain_threshold: Annotated[
        float | None,
        typer.Option("--abstain-threshold", help="Answer 'not found' below this best score."),
    ] = None,
    explain: Annotated[
        bool, typer.Option("--explain", help="Show how every candidate was ranked.")
    ] = False,
) -> None:
    """Ask a single question about the indexed codebase."""
    from clicr.answer.engine import Answerer, DoneEvent, RetrievedEvent, TokenEvent
    from clicr.index.store import IndexStore
    from clicr.providers import embedder_from_name, make_chat, make_embedder
    from clicr.retrieve.pipeline import Retriever
    from clicr.retrieve.search import SearchConfig

    root = Path(path).resolve()
    directory = index_dir_for(root)
    meta = IndexStore.read_meta(directory)
    if meta is None:
        raise _fail(f"No index for {root}. Run: clicr index {root}")
    try:
        stored_dim = int(meta["dim"])
        stored_embedder = meta["embedder"]
    except (KeyError, ValueError):
        raise _fail(
            f"The index for {root} is unreadable or inconsistent. "
            f"Re-run: clicr index {root} --rebuild"
        ) from None
    try:
        settings = load_settings(
            root,
            _overrides(
                top_k=top_k,
                chat_provider=chat_provider,
                chat_model=chat_model,
                embed_provider=embed_provider,
                embed_model=embed_model,
                rerank=rerank,
                graph=graph,
                hyde=hyde,
                abstain_threshold=abstain_threshold,
            ),
        )
        stored = stored_embedder
        if embed_provider is not None or embed_model is not None:
            embedder = make_embedder(settings)
            if embedder.name != stored:
                raise _fail(
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
                raise _fail(
                    f"The index for {root} is unreadable or inconsistent. "
                    f"Re-run: clicr index {root} --rebuild"
                )
            if store.count() == 0:
                raise _fail(f"The index for {root} is empty. Run: clicr index {root}")
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
            answerer = Answerer(Retriever(store, embedder, chat, cfg), chat, root=root)
            sources = []
            done = None
            for event in answerer.ask_stream(question):
                if isinstance(event, RetrievedEvent):
                    sources = event.sources
                    if explain:
                        console.print(_explain_table(event.result, sources))
                elif isinstance(event, TokenEvent):
                    typer.echo(event.text, nl=False)
                elif isinstance(event, DoneEvent):
                    done = event
            typer.echo("")
        finally:
            store.close()
    except (ProviderError, ConfigError) as exc:
        raise _fail(f"Error: {exc}") from exc
    typer.echo("\nSources:")
    for n, cand in enumerate(sources, start=1):
        c = cand.chunk
        typer.echo(f"[{n}] {c.path}:{c.start_line}-{c.end_line} {c.symbol or ''}".rstrip())
    if done is not None:
        rep = done.citations
        valid = len(rep.cited) - len(rep.invalid)
        line = f"citations: {valid} valid, {len(rep.invalid)} invalid"
        if rep.invalid:
            line += f" (invalid: {', '.join(map(str, rep.invalid))})"
        typer.echo(line)


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
def chat() -> None:
    """Start an interactive chat session."""
    _stub("chat")


@app.command(name="eval")
def eval_() -> None:
    """Run retrieval and answer evaluations."""
    _stub("eval")


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
