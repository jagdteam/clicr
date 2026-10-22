"""Evaluation suites: pinned source checkouts and question datasets."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

from clicr.paths import clicr_home

ROOT = Path(__file__).resolve().parents[3]  # the clicr repository
SUITES_FILE = ROOT / "eval" / "suites.toml"
DATASET_DIR = ROOT / "eval" / "datasets"
RESULTS_DIR = ROOT / "eval" / "results"


def src_dir() -> Path:
    """Where pinned suite checkouts live (~/.clicr/eval-src)."""
    return clicr_home() / "eval-src"


def load_suites(path: Path = SUITES_FILE) -> dict[str, dict]:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))["suites"]


def load_items(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def ensure_checkout(name: str, suite: dict) -> Path:
    """Clone the suite repo at its pinned commit if it is not already there."""
    commit = suite["commit"]
    base = src_dir()
    target = base / f"{name}-{commit[:7]}"
    if (target / ".git").is_dir():
        return target
    base.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", suite["repo"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", commit], check=True)
    return target


def _include(include: str | None) -> str:
    return (include or ".").strip().strip("/") or "."


def source_root(checkout: Path, include: str | None) -> Path:
    """The folder that gets indexed: the checkout, or its `include` subfolder."""
    inc = _include(include)
    return checkout if inc == "." else checkout / inc


def repo_path(include: str | None, stored: str) -> str:
    """Map an index path (relative to `include`) back to a repo-root path, like the gold."""
    inc = _include(include)
    return stored if inc == "." else f"{inc}/{stored}"
