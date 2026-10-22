"""Compare fresh results with the committed results file (the `clicr eval --check` gate)."""

from __future__ import annotations

from dataclasses import dataclass

TOLERANCE = 0.02  # absolute


@dataclass(frozen=True)
class Diff:
    row: str  # "<embedder> <config>"
    metric: str  # dotted path under the row's metrics, e.g. "overall.recall@5"
    committed: float | None
    fresh: float | None
    ok: bool

    @property
    def delta(self) -> float | None:
        if self.committed is None or self.fresh is None:
            return None
        return self.fresh - self.committed


def _leaves(tree: dict, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a nested dict, keyed by dotted path (None leaves are skipped)."""
    out: dict[str, float] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_leaves(value, path + "."))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            out[path] = float(value)
    return out


def compare_results(committed: dict, fresh: dict, tol: float = TOLERANCE) -> list[Diff]:
    """One Diff per metric of every fresh row; a Diff is not ok when it misses by more than tol.

    A fresh row missing from the committed file, a metric present on one side only, or a
    different suite commit all count as misses.
    """
    diffs: list[Diff] = []
    if committed.get("commit") != fresh.get("commit"):
        diffs.append(Diff("results", "commit", None, None, False))
    old_rows = {(r["embedder"], r["config"]): r for r in committed.get("rows", [])}
    for row in fresh.get("rows", []):
        label = f"{row['embedder']} {row['config']}"
        old = old_rows.get((row["embedder"], row["config"]))
        if old is None:
            diffs.append(Diff(label, "(row missing from the committed file)", None, None, False))
            continue
        before, after = _leaves(old["metrics"]), _leaves(row["metrics"])
        for metric in sorted(before.keys() | after.keys()):
            a, b = before.get(metric), after.get(metric)
            ok = a is not None and b is not None and abs(a - b) <= tol + 1e-9
            diffs.append(Diff(label, metric, a, b, ok))
    return diffs
