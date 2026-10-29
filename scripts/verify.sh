#!/usr/bin/env sh
# Run the full verify; print only failures and the final summary line.
cd "$(dirname "$0")/.." || exit 1
out=$( { uv run ruff check . && uv run ruff format --check . && uv run pytest -q; } 2>&1 )
rc=$?
if [ $rc -ne 0 ]; then printf '%s\n' "$out" | grep -E 'FAILED|ERROR|error|Would reformat|^[A-Z]+[0-9]+ '; fi
printf '%s\n' "$out" | tail -n 1
exit $rc
