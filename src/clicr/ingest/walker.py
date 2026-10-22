"""Walk a repository and yield the files worth indexing."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pathspec

from clicr.ingest.languages import ALLOWED_EXTENSIONS

ALWAYS_IGNORE = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".next",
        ".nuxt",
        ".idea",
        ".vscode",
        ".clicr",
    }
)
ROOT_ONLY_IGNORE = frozenset({"build", "dist", "target", "bin", "obj", "env", "coverage"})
IGNORE_FILES = frozenset({".DS_Store", "Thumbs.db", ".gitignore", ".dockerignore"})

_Specs = list[tuple[str, pathspec.GitIgnoreSpec]]


def _load_spec_file(path: Path) -> pathspec.GitIgnoreSpec | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _load_spec(directory: Path) -> pathspec.GitIgnoreSpec | None:
    return _load_spec_file(directory / ".gitignore")


def _is_ignored(rel_posix: str, specs: _Specs) -> bool:
    """rel_posix is relative to the repo root; specs are ordered shallow to deep."""
    ignored = False
    for base, spec in specs:
        if base:
            if not rel_posix.startswith(base + "/"):
                continue
            sub = rel_posix[len(base) + 1 :]
        else:
            sub = rel_posix
        verdict = spec.check_file(sub).include
        if verdict is not None:
            ignored = verdict
    return ignored


def _skip_dir(name: str, at_root: bool) -> bool:
    return name in ALWAYS_IGNORE or (at_root and name in ROOT_ONLY_IGNORE)


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(8192)
    except OSError:
        return True


def _wanted_file(path: Path, name: str, max_bytes: int) -> bool:
    if name in IGNORE_FILES or name.startswith(".env"):
        return False
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    try:
        if path.stat().st_size > max_bytes:
            return False
    except OSError:
        return False
    return not _is_binary(path)


def iter_files(root: Path, max_bytes: int = 1_000_000) -> Iterator[Path]:
    root = Path(root)
    found: list[Path] = []
    # git's per-repo exclude file goes first so .gitignore files override it, as in git
    exclude = _load_spec_file(root / ".git" / "info" / "exclude")
    stack: list[tuple[Path, _Specs]] = [(root, [("", exclude)] if exclude is not None else [])]
    while stack:
        directory, inherited = stack.pop()
        specs = list(inherited)
        spec = _load_spec(directory)
        if spec is not None:
            base = directory.relative_to(root).as_posix()
            specs.append(("" if base == "." else base, spec))
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
        for entry in entries:
            path = Path(entry.path)
            rel = path.relative_to(root).as_posix()
            if entry.is_dir(follow_symlinks=False):
                if _skip_dir(entry.name, directory == root) or _is_ignored(rel + "/", specs):
                    continue
                stack.append((path, specs))
            elif entry.is_file(follow_symlinks=False):
                if _is_ignored(rel, specs):
                    continue
                if _wanted_file(path, entry.name, max_bytes):
                    found.append(path)
    yield from sorted(found, key=lambda p: p.relative_to(root).as_posix())
