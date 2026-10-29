#!/usr/bin/env sh
# Last 50 lines of the newest log under the clicr state dir, if any.
home="${CLICR_HOME:-$HOME/.clicr}"
f=$(find "$home" -type f -name '*.log' 2>/dev/null | xargs -r ls -t 2>/dev/null | head -n 1)
if [ -n "$f" ]; then tail -n 50 "$f"; else echo "No logs found. clicr state lives in $home (set by CLICR_HOME, see src/clicr/paths.py)."; fi
