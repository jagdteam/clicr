"""File extension to language mapping."""

from __future__ import annotations

from pathlib import PurePath

EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".lua": "lua",
    ".r": "r",
    ".dart": "dart",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "conf",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".xml": "xml",
    ".m": "objective-c",
    ".mm": "objective-c",
}

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(EXTENSION_LANGUAGES)

SYNTAX_LANGUAGES: frozenset[str] = frozenset(
    {"python", "javascript", "typescript", "tsx", "go", "rust", "java", "c", "cpp"}
)


def language_for(path: str | PurePath) -> str | None:
    return EXTENSION_LANGUAGES.get(PurePath(path).suffix.lower())
