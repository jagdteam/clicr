from typing import Annotated

import typer
from rich.console import Console

from clicr import __version__

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


@app.command()
def doctor() -> None:
    """Check that the environment is set up correctly."""
    _stub("doctor")


@app.command(name="export-web")
def export_web() -> None:
    """Export the index for the web demo."""
    _stub("export-web")
