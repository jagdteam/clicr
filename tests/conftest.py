"""Shared fixtures: a generated sample repository."""

from __future__ import annotations

from pathlib import Path

import pytest

PY_MAIN = '''"""Sample module."""
import os

DEFAULT = 3


def get_user_by_id(user_id):
    """Look up a user."""
    return {"id": user_id}


@cache
def decorated(x):
    return x + 1


BETWEEN = 1
OTHER = 2


class Small:
    def only(self):
        return 1
'''


def _big_class() -> str:
    out = ["class Big:", '    """A long class."""', "    attr = 1", ""]
    for i in range(10):
        out.append(f"    def method_{i}(self):")
        out.extend(f"        x{j} = {j}" for j in range(14))
        out.append("        return 0")
        out.append("")
    return "\n".join(out) + "\n"


JS_FILE = """import fs from 'fs';

export function greet(name) {
  return 'hi ' + name;
}

class Greeter {
  hello() {
    return 1;
  }
}

const answer = 42;
"""

GO_FILE = """package main

import "fmt"

type Server struct {
	Name string
}

func (s *Server) Start() {
	fmt.Println(s.Name)
}

func main() {
	fmt.Println("hi")
}
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample_repo"
    (root / "build").mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "app.py").write_text(PY_MAIN, encoding="utf-8")
    (root / "big.py").write_text(_big_class(), encoding="utf-8")
    (root / "util.js").write_text(JS_FILE, encoding="utf-8")
    (root / "main.go").write_text(GO_FILE, encoding="utf-8")
    (root / "notes.md").write_text("# Notes\n\nSome text.\n", encoding="utf-8")
    (root / "pkg" / "inner.py").write_text("def inner():\n    return 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("build/\n*.log\n", encoding="utf-8")
    (root / "build" / "out.py").write_text("x = 1\n", encoding="utf-8")
    (root / "debug.log").write_text("log\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / "blob.txt").write_bytes(b"abc\x00def")
    return root
