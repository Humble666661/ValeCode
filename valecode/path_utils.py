from __future__ import annotations

import os
import tempfile
from pathlib import Path


def platform_path(path: str | os.PathLike[str]) -> Path:
    """Return a usable local path, including POSIX temp aliases on Windows.

    Prompts, saved sessions and cross-platform tests may contain ``/tmp`` paths.
    On Windows, ``Path('/tmp')`` points at the current drive root and commonly
    fails with an access error. Map only that well-known temporary namespace;
    all other absolute and relative paths keep their native meaning.
    """

    raw = os.fspath(path)
    if os.name == "nt":
        normalized = raw.replace("\\", "/")
        for prefix in ("/tmp", "/private/tmp"):
            if normalized == prefix or normalized.startswith(prefix + "/"):
                suffix = normalized[len(prefix) :].lstrip("/")
                return Path(tempfile.gettempdir()) / suffix
    return Path(raw)
