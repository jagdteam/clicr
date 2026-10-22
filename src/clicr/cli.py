import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from clicr import __version__
from clicr.config import ConfigError, Settings, load_settings
from clicr.ingest.languages import SYNTAX_LANGUAGES
from clicr.paths import index_dir_for

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


@app.command()
def index(path: Annotated[str, typer.Argument(help="Folder to index.")] = ".") -> None:
    """Index a codebase."""
    _stub("index")


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Your question.")]) -> None:
    """Ask a single question about the indexed codebase."""
    _stub("ask")


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
