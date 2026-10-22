from __future__ import annotations

import re
from pathlib import Path

import pytest

from clicr.ingest.chunkers import LineChunker, SyntaxChunker
from clicr.ingest.languages import language_for
from clicr.ingest.walker import iter_files
from clicr.models import Chunk
from clicr.paths import index_dir_for


def rels(root: Path, **kw: int) -> list[str]:
    return [p.relative_to(root).as_posix() for p in iter_files(root, **kw)]


def test_walker_ignores(sample_repo: Path) -> None:
    got = rels(sample_repo)
    assert got == sorted(got)
    assert {"app.py", "big.py", "util.js", "main.go", "pkg/inner.py", "notes.md"} <= set(got)
    for bad in ("build/out.py", "debug.log", ".env", "blob.txt", ".gitignore"):
        assert bad not in got


def test_walker_nested_gitignore_and_size(sample_repo: Path) -> None:
    (sample_repo / "pkg" / ".gitignore").write_text("inner.py\n", encoding="utf-8")
    assert "pkg/inner.py" not in rels(sample_repo)
    assert "big.py" not in rels(sample_repo, max_bytes=100)


def test_chunk_id_and_embed_text() -> None:
    c = Chunk("a.py", 1, 2, "x", "python", "f", "function")
    assert len(c.id) == 16
    assert c.embed_text() == "# path: a.py\n# symbol: f\nx"
    assert Chunk("a.py", 1, 2, "x", None, None, "window").embed_text() == "# path: a.py\nx"
    assert c.embed_text(header=False) == "x"


def test_index_dir_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICR_HOME", str(tmp_path / "home"))
    d = index_dir_for(tmp_path / "proj")
    assert d.parent == tmp_path / "home" / "indexes"
    assert re.fullmatch(r"proj-[0-9a-f]{8}", d.name)


def test_line_chunker_exact_lines() -> None:
    text = "".join(f"line {i} " + "x" * 30 + "\n" for i in range(60))
    chunks = LineChunker(200, 20).chunk("f.txt", text, "text")
    assert len(chunks) > 3
    for i, c in enumerate(chunks):
        start = i * 180
        end = min(start + 200, len(text))
        assert c.text == text[start:end]
        assert c.start_line == text.count("\n", 0, start) + 1
        assert c.end_line == text.count("\n", 0, end - 1) + 1
        assert c.kind == "window" and c.symbol is None
    assert LineChunker().name == "line-500-50"


def test_line_chunker_empty() -> None:
    assert LineChunker().chunk("f.txt", "", None) == []


def by_symbol(chunks: list[Chunk]) -> dict[str | None, Chunk]:
    return {c.symbol: c for c in chunks if c.symbol}


def test_syntax_python_symbols(sample_repo: Path) -> None:
    text = (sample_repo / "app.py").read_text(encoding="utf-8")
    chunks = SyntaxChunker().chunk("app.py", text, "python")
    sym = by_symbol(chunks)
    assert sym["get_user_by_id"].kind == "function"
    dec = sym["decorated"]
    assert dec.text.startswith("@cache") and dec.kind == "function"
    assert sym["Small"].kind == "class"
    assert any(c.kind == "module" and "DEFAULT" in c.text for c in chunks)
    assert any(c.kind == "module" and "BETWEEN" in c.text for c in chunks)
    lines = text.split("\n")
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])


def test_syntax_big_class_split(sample_repo: Path) -> None:
    text = (sample_repo / "big.py").read_text(encoding="utf-8")
    chunks = SyntaxChunker(max_lines=120).chunk("big.py", text, "python")
    sym = by_symbol(chunks)
    assert sym["Big"].kind == "class"
    assert sym["Big"].end_line < sym["Big.method_0"].start_line + 1
    for i in range(10):
        assert sym[f"Big.method_{i}"].kind == "method"
    assert max(c.end_line - c.start_line + 1 for c in chunks) <= 120


def test_syntax_long_function_windows() -> None:
    text = "def f():\n" + "".join(f"    x{i} = {i}\n" for i in range(50))
    chunks = SyntaxChunker(max_lines=20).chunk("f.py", text, "python")
    assert len(chunks) == 3
    assert all(c.symbol == "f" and c.kind == "function" for c in chunks)
    assert_invariant(text, chunks)


def test_syntax_js_and_go(sample_repo: Path) -> None:
    js = (sample_repo / "util.js").read_text(encoding="utf-8")
    sym = by_symbol(SyntaxChunker().chunk("util.js", js, "javascript"))
    assert sym["greet"].text.startswith("export function")
    assert sym["Greeter"].kind == "class"
    go = (sample_repo / "main.go").read_text(encoding="utf-8")
    sym = by_symbol(SyntaxChunker().chunk("main.go", go, "go"))
    assert sym["Start"].kind == "method"
    assert sym["main"].kind == "function"
    assert sym["Server"].kind == "class"


def assert_invariant(text: str, chunks: list[Chunk]) -> None:
    lines = text.split("\n")
    covered: set[int] = set()
    for c in chunks:
        span = set(range(c.start_line, c.end_line + 1))
        assert not span & covered, f"overlap at {c}"
        covered |= span
    for n, line in enumerate(lines, start=1):
        if line.strip():
            assert n in covered, f"line {n} not covered: {line!r}"


@pytest.mark.parametrize(
    ("name", "language"),
    [("app.py", "python"), ("big.py", "python"), ("util.js", "javascript"), ("main.go", "go")],
)
def test_syntax_invariant(sample_repo: Path, name: str, language: str) -> None:
    text = (sample_repo / name).read_text(encoding="utf-8")
    assert_invariant(text, SyntaxChunker().chunk(name, text, language))
    assert_invariant(text, SyntaxChunker(max_lines=5).chunk(name, text, language))


def test_syntax_other_languages_invariant() -> None:
    samples = {
        "rust": "struct A { x: i32 }\nimpl A {\n    fn f(&self) -> i32 { 1 }\n}\nfn main() {}\n",
        "java": "class A {\n  int f() { return 1; }\n  int g() { return 2; }\n}\n",
        "c": "struct S { int x; };\nint main(void) {\n  return 0;\n}\n",
        "cpp": "class A {\n public:\n  int f() { return 1; }\n};\nint main() { return 0; }\n",
        "typescript": "export class A {\n  m(): number { return 1; }\n}\nlet x = 1;\n",
    }
    for lang, text in samples.items():
        for max_lines in (120, 2):
            chunks = SyntaxChunker(max_lines=max_lines).chunk("f", text, lang)
            assert chunks, lang
            assert_invariant(text, chunks)
    rust = by_symbol(SyntaxChunker().chunk("f.rs", samples["rust"], "rust"))
    assert {"A", "main"} <= set(rust)
    c = by_symbol(SyntaxChunker().chunk("f.c", samples["c"], "c"))
    assert "main" in c


def test_syntax_fallback_unsupported(caplog: pytest.LogCaptureFixture) -> None:
    text = "\n".join(f"row {i}" for i in range(25)) + "\n"
    with caplog.at_level("INFO", logger="clicr.ingest.chunkers"):
        chunks = SyntaxChunker(max_lines=10).chunk("a.rb", text, language_for("a.rb"))
    assert [c.kind for c in chunks] == ["window"] * 3
    assert len(caplog.records) == 1
    assert_invariant(text, chunks)
    assert SyntaxChunker().name == "syntax-120"


def test_syntax_gap_between_methods_keeps_class_symbol() -> None:
    text = (
        "class A:\n"
        "    def s(self):\n"
        "        return 1\n"
        "    class In:\n"
        "        z = 1\n"
        "    y = 2\n"
        "    def t(self):\n"
        "        return 2\n"
    )
    chunks = SyntaxChunker(max_lines=4).chunk("a.py", text, "python")
    gap = [c for c in chunks if "y = 2" in c.text]
    assert gap
    assert all(c.symbol == "A" and c.kind == "class" for c in gap)
    t = next(c for c in chunks if c.symbol == "A.t")
    assert t.start_line == 7
    assert_invariant(text, chunks)


def test_walker_root_only_ignores(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "a.py").write_text("x = 1\n")
    (tmp_path / "src" / "bin").mkdir(parents=True)
    (tmp_path / "src" / "bin" / "a.py").write_text("x = 1\n")
    (tmp_path / "src" / "node_modules").mkdir()
    (tmp_path / "src" / "node_modules" / "b.py").write_text("x = 1\n")
    assert rels(tmp_path) == ["src/bin/a.py"]


def test_walker_honours_git_info_exclude(tmp_path: Path) -> None:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("notes/\n*.secret\n", encoding="utf-8")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.py").write_text("x = 1\n")
    (tmp_path / "x.secret").write_text("x = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("x = 1\n")
    assert rels(tmp_path) == ["src/b.py"]


def test_crlf_equals_lf() -> None:
    lf = "def f():\n    return 1\n\n\nx = 2\n"
    crlf = lf.replace("\n", "\r\n")
    for chunker in (SyntaxChunker(), LineChunker(10, 2)):
        a = chunker.chunk("a.py", lf, "python")
        b = chunker.chunk("a.py", crlf, "python")
        assert a == b
        assert [c.id for c in a] == [c.id for c in b]
    cr = LineChunker(10, 2).chunk("a.py", "a\rb\rc\r", None)
    assert cr == LineChunker(10, 2).chunk("a.py", "a\nb\nc\n", None)


def test_syntax_repeated_parses_stay_stable() -> None:
    # Regression: tree-sitter 0.26.0 crashed (access violation) after a few parses of a big file.
    src = Path(__file__).resolve().parents[1] / "src" / "clicr" / "cli.py"
    text = src.read_text(encoding="utf-8")
    chunker = SyntaxChunker()
    first = chunker.chunk("cli.py", text, "python")
    for _ in range(30):
        assert chunker.chunk("cli.py", text, "python") == first
    assert all(c.start_line >= 1 for c in first)
