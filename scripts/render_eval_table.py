"""Render the README eval tables from eval/results/*.json.

Usage: uv run python scripts/render_eval_table.py [--check]

Rewrites the README section between <!-- eval:start --> and <!-- eval:end --> (the markers
are appended at the end of README.md when absent). Running it twice gives the same file.
--check writes nothing and exits 1 when the README is stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clicr.eval.report import default_embedder, render_section, replace_section

ROOT = Path(__file__).resolve().parent.parent
RERUN = "uv run python scripts/render_eval_table.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--check", action="store_true", help="exit 1 if the README is stale")
    args = parser.parse_args(argv)

    files = sorted(args.results_dir.glob("*.json")) if args.results_dir.is_dir() else []
    results = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    old = args.readme.read_text(encoding="utf-8")
    new = replace_section(old, render_section(results, default_embedder()))
    if args.check:
        if new != old:
            print(f"{args.readme} eval table is stale. Run: {RERUN}")
            return 1
        print(f"{args.readme} eval table is up to date")
        return 0
    if new != old:
        args.readme.write_text(new, encoding="utf-8", newline="\n")
        print(f"Updated {args.readme} from {len(files)} results file(s)")
    else:
        print(f"{args.readme} already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
