"""`clicr index --watch`: re-run the incremental updater when files change."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

from clicr.index.updater import UpdateStats
from clicr.ingest.walker import iter_files
from clicr.providers.base import ProviderError
from clicr.ui.errors import friendly


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def handle_batch(
    changed: Iterable[str | Path],
    root: Path,
    index_dir: Path,
    update: Callable[[], UpdateStats],
    known: Callable[[], set[str]],
) -> UpdateStats | None:
    """Run `update` once if the batch touches a file the walker indexes (or that was indexed).

    Paths in the index dir are skipped; ignore rules are the walker's, not re-implemented here.
    `known` gives the repo-relative paths already in the index, which covers deletions.
    """
    root, index_dir = root.resolve(), index_dir.resolve()
    paths = [Path(p).resolve() for p in changed]
    paths = [p for p in paths if not _inside(p, index_dir)]
    if not paths:
        return None
    walked = {p.resolve() for p in iter_files(root)}
    indexed = known()
    for p in paths:
        if p in walked:
            return update()
        if _inside(p, root) and p != root:
            rel = p.relative_to(root).as_posix()
            # a deleted file, or a deleted folder reported only as its own path
            if any(k == rel or k.startswith(rel + "/") for k in indexed):
                return update()
    return None


def summary_line(stats: UpdateStats) -> str:
    return (
        f"updated: {stats.added} added, {stats.changed} changed, {stats.deleted} deleted "
        f"({stats.chunks_embedded} chunks embedded)"
    )


def run_batch(
    changed: Iterable[str | Path],
    root: Path,
    index_dir: Path,
    update: Callable[[], UpdateStats],
    known: Callable[[], set[str]],
) -> str | None:
    """One watch step: the line to print for this batch (None when nothing was indexed).

    Provider and filesystem errors become a line so the watcher keeps running.
    """
    try:
        stats = handle_batch(changed, root, index_dir, update, known)
    except ProviderError as exc:
        return friendly(exc) or str(exc)
    except OSError as exc:  # a folder vanished mid-scan; the next change retries
        return f"Error: could not update the index ({exc}). Still watching."
    if stats is None:
        return None
    return f"{time.strftime('%H:%M:%S')} {summary_line(stats)}"
