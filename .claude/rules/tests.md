---
paths: ["tests/**"]
---
- Run `uv run pytest -q`. Default addopts is `-m "not ollama and not fastembed"`, so tests marked `ollama` or `fastembed` are skipped.
- Run an opt-in group with `-m <marker>`, for example `uv run pytest -q -m fastembed` (`ollama` needs a running Ollama server).
- A new feature needs a test that fails first and passes after the change.
