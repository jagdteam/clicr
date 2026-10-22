"""Check that every gold location in the evaluation datasets really exists.

Usage: uv run python scripts/check_gold.py [--suite NAME]

Reads eval/suites.toml and eval/datasets/<suite>.jsonl, makes sure each suite's
pinned commit is checked out under ~/.clicr/eval-src/, then validates every item.
Prints one line per failure and a summary line. Exits 1 if anything failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clicr.eval.suites import DATASET_DIR, ensure_checkout, load_suites

CATEGORIES = {"lookup", "how", "multi_hop", "unanswerable", "follow_up"}
REQUIRED = {
    "id": str,
    "question": str,
    "category": str,
    "gold": list,
    "history": list,
    "reference_answer": str,
    "notes": str,
}


def read_lines(src: Path, rel: str, cache: dict[str, list[str] | None]) -> list[str] | None:
    if rel not in cache:
        path = src / rel
        if path.is_file():
            cache[rel] = path.read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            cache[rel] = None
    return cache[rel]


def check_span(span: object, src: Path, cache: dict[str, list[str] | None]) -> list[str]:
    """Return the problems found in one gold span."""
    if not isinstance(span, dict):
        return ["gold entry is not an object"]
    problems = []
    path, start, end, symbol = (span.get(k) for k in ("path", "start_line", "end_line", "symbol"))
    if not isinstance(path, str):
        return ["gold path missing or not a string"]
    if not (isinstance(start, int) and isinstance(end, int)):
        return [f"{path}: start_line/end_line must be integers"]
    lines = read_lines(src, path, cache)
    if lines is None:
        return [f"{path}: file does not exist at the pinned commit"]
    if not 1 <= start <= end <= len(lines):
        problems.append(f"{path}: bad span {start}-{end} (file has {len(lines)} lines)")
        return problems
    if symbol is not None:
        if not isinstance(symbol, str) or not symbol:
            problems.append(f"{path}:{start}: symbol must be a non-empty string or null")
        else:
            tail = symbol.split(".")[-1]
            head = "\n".join(lines[start - 1 : start + 3])
            if tail not in head:
                problems.append(
                    f"{path}:{start}: symbol {symbol!r} not found in lines {start}-{start + 3}"
                )
    return problems


def check_item(item: object, src: Path, cache: dict[str, list[str] | None]) -> list[str]:
    """Return the problems found in one dataset item (without the id prefix)."""
    if not isinstance(item, dict):
        return ["item is not a JSON object"]
    problems = []
    for key, kind in REQUIRED.items():
        if key not in item:
            problems.append(f"missing field {key!r}")
        elif not isinstance(item[key], kind):
            problems.append(f"field {key!r} must be {kind.__name__}")
    if problems:
        return problems
    category = item["category"]
    if category not in CATEGORIES:
        problems.append(f"unknown category {category!r}")
    gold = item["gold"]
    if (category == "unanswerable") != (len(gold) == 0):
        problems.append("unanswerable must be exactly the items with empty gold")
    if category == "multi_hop":
        files = {s.get("path") for s in gold if isinstance(s, dict)}
        if len(gold) < 2 or len(files) < 2:
            problems.append("multi_hop needs at least 2 gold spans in 2 distinct files")
    if category == "follow_up":
        if not item["history"]:
            problems.append("follow_up needs a non-empty history")
    elif item["history"]:
        problems.append("history must be empty unless category is follow_up")
    for turn in item["history"]:
        if not (
            isinstance(turn, dict)
            and isinstance(turn.get("q"), str)
            and isinstance(turn.get("a"), str)
        ):
            problems.append("history entries must be {q, a} strings")
            break
    for span in gold:
        problems.extend(check_span(span, src, cache))
    return problems


def check_suite(name: str, suite: dict) -> tuple[int, int]:
    """Check one suite. Returns (item count, failure count)."""
    dataset = DATASET_DIR / f"{name}.jsonl"
    if not dataset.is_file():
        print(f"{name}: dataset file {dataset} not found")
        return 0, 1
    src = ensure_checkout(name, suite)
    cache: dict[str, list[str] | None] = {}
    seen: set[str] = set()
    questions: set[str] = set()
    count = failures = 0
    for number, raw in enumerate(dataset.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        count += 1
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"{name}:{number}: invalid JSON ({exc.msg})")
            failures += 1
            continue
        label = item.get("id", f"line {number}") if isinstance(item, dict) else f"line {number}"
        problems = check_item(item, src, cache)
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str):
                if item_id in seen:
                    problems.append("duplicate id")
                seen.add(item_id)
            question = item.get("question")
            if isinstance(question, str):
                if question.strip().lower() in questions:
                    problems.append("duplicate question")
                questions.add(question.strip().lower())
        for problem in problems:
            print(f"{name}:{label}: {problem}")
        failures += len(problems)
    return count, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", help="check only this suite")
    args = parser.parse_args()

    suites = load_suites()
    if args.suite:
        if args.suite not in suites:
            print(f"unknown suite {args.suite!r}; known: {', '.join(suites)}")
            return 1
        suites = {args.suite: suites[args.suite]}

    total = bad = 0
    for name, suite in suites.items():
        count, failures = check_suite(name, suite)
        total += count
        bad += failures
    print(f"{total} items, {bad} failures")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
