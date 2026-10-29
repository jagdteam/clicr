---
paths: ["src/clicr/**/*.py"]
---
- Ruff: line length 100, target py312, rules E F I UP B SIM (pyproject.toml). Run `uv run ruff check .` and `uv run ruff format --check .`.
- `tree-sitter` stays pinned `<0.26`: 0.26.0 corrupts node memory on Windows.
- Data lives under `~/.clicr` (or `$CLICR_HOME`), resolved by `src/clicr/paths.py` (`clicr_home()`, `index_dir_for()`). Never hard-code that path; call those functions.
