"""Split source files into chunks: character windows or syntax-aware spans."""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any

from clicr.ingest.languages import SYNTAX_LANGUAGES
from clicr.models import Chunk

log = logging.getLogger(__name__)


def normalize_newlines(text: str) -> str:
    """Turn CRLF and lone CR into LF so line numbers and ids do not depend on the OS."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


class LineChunker:
    """Fixed character windows with overlap; line numbers are exact."""

    def __init__(self, chunk_chars: int = 500, overlap: int = 50) -> None:
        if overlap >= chunk_chars:
            raise ValueError("overlap must be smaller than chunk_chars")
        self.chunk_chars = chunk_chars
        self.overlap = overlap
        self.name = f"line-{chunk_chars}-{overlap}"

    def chunk(self, path_rel: str, text: str, language: str | None) -> list[Chunk]:
        text = normalize_newlines(text)
        newlines = [i for i, ch in enumerate(text) if ch == "\n"]
        step = self.chunk_chars - self.overlap
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_chars, len(text))
            piece = text[start:end]
            if piece.strip():
                chunks.append(
                    Chunk(
                        path=path_rel,
                        start_line=1 + bisect_left(newlines, start),
                        end_line=1 + bisect_left(newlines, end - 1),
                        text=piece,
                        language=language,
                        symbol=None,
                        kind="window",
                    )
                )
            if end >= len(text):
                break
            start += step
        return chunks


@dataclass(frozen=True)
class LangSpec:
    defs: frozenset[str]  # top-level definitions that become chunks
    methods: frozenset[str] = frozenset()  # members found inside a long class
    wrappers: frozenset[str] = field(default_factory=frozenset)  # unwrap to the inner def


_JS = LangSpec(
    defs=frozenset({"function_declaration", "class_declaration", "method_definition"}),
    methods=frozenset({"method_definition"}),
    wrappers=frozenset({"export_statement"}),
)

LANG_SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        defs=frozenset({"function_definition", "class_definition"}),
        methods=frozenset({"function_definition", "decorated_definition"}),
        wrappers=frozenset({"decorated_definition"}),
    ),
    "javascript": _JS,
    "typescript": _JS,
    "tsx": _JS,
    "go": LangSpec(
        defs=frozenset({"function_declaration", "method_declaration", "type_declaration"})
    ),
    "rust": LangSpec(
        defs=frozenset({"function_item", "impl_item", "struct_item", "enum_item", "trait_item"}),
        methods=frozenset({"function_item"}),
    ),
    "java": LangSpec(
        defs=frozenset({"class_declaration", "interface_declaration", "method_declaration"}),
        methods=frozenset({"method_declaration"}),
    ),
    "c": LangSpec(defs=frozenset({"function_definition", "struct_specifier"})),
    "cpp": LangSpec(
        defs=frozenset({"function_definition", "struct_specifier", "class_specifier"}),
        methods=frozenset({"function_definition"}),
    ),
}

_METHOD_KINDS = frozenset({"method_definition", "method_declaration"})
_CLASS_WORDS = ("class", "struct", "interface", "impl", "trait", "enum", "type_declaration")


def _kind_of(node_type: str) -> str:
    if node_type in _METHOD_KINDS:
        return "method"
    if any(word in node_type for word in _CLASS_WORDS):
        return "class"
    return "function"


def _text_of(node: Any) -> str:
    return bytes(node.text).decode("utf-8", "replace")


def _name_of(node: Any) -> str | None:
    """Best-effort definition name across grammars."""
    if node.type == "type_declaration":  # Go
        for child in node.named_children:
            if child.type in ("type_spec", "type_alias"):
                return _name_of(child)
        return None
    for fld in ("name", "declarator", "type"):
        child = node.child_by_field_name(fld)
        if child is None:
            continue
        if fld == "declarator" and child.child_by_field_name("declarator") is not None:
            return _name_of(child)
        return _text_of(child)
    return None


def _unwrap(node: Any, spec: LangSpec) -> Any:
    """Return the inner definition of a decorated/exported node (or the node itself)."""
    if node.type not in spec.wrappers:
        return node
    inner = node.child_by_field_name("definition")
    if inner is not None:
        return _unwrap(inner, spec)
    for child in node.named_children:
        if child.type in spec.defs:
            return child
    return node


@dataclass(frozen=True)
class _Def:
    start: int  # 0-based first row
    end: int  # 0-based last row, inclusive
    symbol: str | None
    kind: str
    node: Any


class SyntaxChunker:
    """Tree-sitter based chunker; falls back to line windows."""

    def __init__(self, max_lines: int = 120) -> None:
        self.max_lines = max_lines
        self.name = f"syntax-{max_lines}"

    # -- public ------------------------------------------------------------
    def chunk(self, path_rel: str, text: str, language: str | None) -> list[Chunk]:
        text = normalize_newlines(text)
        lines = text.split("\n")
        if language not in SYNTAX_LANGUAGES or language not in LANG_SPECS:
            log.info("%s: no syntax support for %r, using line windows", path_rel, language)
            return self._windows(path_rel, lines, language)
        try:
            tree = self._parse(language, text)
        except Exception as exc:  # noqa: BLE001 - any parser failure means fall back
            log.warning("%s: parse failed (%s), using line windows", path_rel, exc)
            return self._windows(path_rel, lines, language)
        # `tree` stays referenced in this frame until chunking returns, so its nodes stay valid.
        return self._chunk_tree(path_rel, lines, language, tree.root_node, LANG_SPECS[language])

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _parse(language: str, text: str) -> Any:
        from tree_sitter_language_pack import get_parser

        return get_parser(language).parse(text.encode("utf-8"))

    # -- structure ---------------------------------------------------------
    def _chunk_tree(
        self, path: str, lines: list[str], language: str, root: Any, spec: LangSpec
    ) -> list[Chunk]:
        out: list[Chunk] = []
        emit = _Emitter(path, lines, language, self.max_lines, out)
        cursor = 0  # first row not yet covered
        for d in self._top_defs(root, spec):
            start = max(d.start, cursor)
            if d.end < start:
                continue  # entirely covered by the previous definition's line
            emit.span(cursor, start - 1, None, "module")
            if d.kind == "class" and d.end - start + 1 > self.max_lines:
                self._split_class(emit, d, start, spec)
            else:
                emit.span(start, d.end, d.symbol, d.kind)
            cursor = d.end + 1
        emit.span(cursor, len(lines) - 1, None, "module")
        return out

    def _top_defs(self, root: Any, spec: LangSpec) -> list[_Def]:
        defs: list[_Def] = []
        for child in root.children:
            inner = _unwrap(child, spec)
            if inner.type not in spec.defs:
                continue
            defs.append(
                _Def(
                    start=child.start_point.row,
                    end=child.end_point.row,
                    symbol=_name_of(inner),
                    kind=_kind_of(inner.type),
                    node=inner,
                )
            )
        return defs

    @staticmethod
    def _find_methods(node: Any, spec: LangSpec) -> list[Any]:
        found: list[Any] = []
        for child in node.children:
            if child.type in spec.methods:
                found.append(child)
            elif child.type not in spec.defs:
                found.extend(SyntaxChunker._find_methods(child, spec))
        return found

    def _split_class(self, emit: _Emitter, d: _Def, start: int, spec: LangSpec) -> None:
        methods = [m for m in self._find_methods(d.node, spec) if m.start_point.row >= start]
        if not methods:
            emit.span(start, d.end, d.symbol, d.kind)
            return
        cursor = start
        for m in methods:
            first = m.start_point.row
            emit.span(cursor, first - 1, d.symbol, "class")  # header or gap
            name = _name_of(_unwrap(m, spec))
            symbol = f"{d.symbol}.{name}" if d.symbol and name else (name or d.symbol)
            emit.span(first, m.end_point.row, symbol, "method")
            cursor = m.end_point.row + 1
        emit.span(cursor, d.end, d.symbol, "class")

    # -- fallback ----------------------------------------------------------
    def _windows(self, path: str, lines: list[str], language: str | None) -> list[Chunk]:
        out: list[Chunk] = []
        emit = _Emitter(path, lines, language, self.max_lines, out)
        emit.span(0, len(lines) - 1, None, "window")
        return out


class _Emitter:
    """Turns row ranges into chunks: trims blank edges and splits long spans."""

    def __init__(
        self, path: str, lines: list[str], language: str | None, max_lines: int, out: list[Chunk]
    ) -> None:
        self.path, self.lines, self.language = path, lines, language
        self.max_lines, self.out = max_lines, out

    def span(self, start: int, end: int, symbol: str | None, kind: str) -> None:
        end = min(end, len(self.lines) - 1)
        while start <= end and not self.lines[start].strip():
            start += 1
        while end >= start and not self.lines[end].strip():
            end -= 1
        for lo in range(start, end + 1, self.max_lines):
            hi = min(lo + self.max_lines - 1, end)
            self._add(lo, hi, symbol, kind)

    def _add(self, lo: int, hi: int, symbol: str | None, kind: str) -> None:
        # A window may start/end on blank lines after splitting; trim again.
        while lo <= hi and not self.lines[lo].strip():
            lo += 1
        while hi >= lo and not self.lines[hi].strip():
            hi -= 1
        if lo > hi:
            return
        self.out.append(
            Chunk(
                path=self.path,
                start_line=lo + 1,
                end_line=hi + 1,
                text="\n".join(self.lines[lo : hi + 1]),
                language=self.language,
                symbol=symbol,
                kind=kind,
            )
        )
