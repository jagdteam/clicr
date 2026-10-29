# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-10-28

### Added
- Free local models: Ollama chat provider and fastembed CPU embeddings (`src/clicr/providers`, e0f9344), plus `clicr doctor`.
- Hybrid search: vector + BM25 (SQLite FTS5) fused with reciprocal rank fusion, with one-hop call-graph expansion on by default (`src/clicr/retrieve`, 052822f, 1d35801).
- Context packing with `[n]` citations and citation checks in answers (`src/clicr/answer`, 052822f, 40e986b).
- Optional retrieval stages: cross-encoder and LLM rerank, HyDE, abstain, `ask --explain` (40e986b).
- Evaluation harness: flask and clicr suites, ablation over every RAG stage, rerank sweep, `clicr eval` (`src/clicr/eval`, `eval/suites.toml`, 6b15b0e, 647881f).
- Terminal UI: streaming `ask`, `chat` REPL with sessions, `index --watch`, `doctor --fix` (`src/clicr/ui`, `chat.py`, `watch.py`, 31ec397).
- `clicr export-web`: static bundle for the in-browser demo, including `files.json` and `qvecs.bin` (`src/clicr/export_web.py`, 3458c3f, 8ac8e27, 8359c84).

### Changed
- Rebuilt as an installable uv package with a `clicr` command, tests and CI (9ea2aed).
- README audited against the code, with eval tables and export-web build steps (3a079bd, e4a1153).
