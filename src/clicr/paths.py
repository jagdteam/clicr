"""Where CliCR keeps its data on disk."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def clicr_home() -> Path:
    env = os.environ.get("CLICR_HOME")
    return Path(env) if env else Path.home() / ".clicr"


def index_dir_for(root: Path) -> Path:
    digest = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
    return clicr_home() / "indexes" / f"{root.name}-{digest}"
