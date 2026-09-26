from __future__ import annotations

import os
import shutil
import sys
import importlib.util

from valecode.teams.models import BackendType


class BackendDetectionError(Exception):
    pass


def detect_backend(
    teammate_mode: str = "",
    is_interactive: bool = True,
) -> BackendType:
    """External terminal creation is explicit; environment never enables it."""
    if teammate_mode in ("", "in-process"):
        return BackendType.IN_PROCESS
    if not is_interactive:
        raise BackendDetectionError("Pane teammates require an interactive local session")
    if teammate_mode == "tmux":
        if sys.platform == "win32" or shutil.which("tmux") is None:
            raise BackendDetectionError("tmux requires POSIX with tmux installed; use in-process on native Windows")
        return BackendType.TMUX
    if teammate_mode == "iterm2":
        if sys.platform != "darwin" or os.environ.get("TERM_PROGRAM") != "iTerm.app" or importlib.util.find_spec("iterm2") is None:
            raise BackendDetectionError("iTerm2 requires macOS, iTerm.app and the iterm2 Python package/API enabled")
        return BackendType.ITERM2
    raise BackendDetectionError(f"Unsupported teammate backend '{teammate_mode}'")


def detect_pane_backend(
    teammate_mode: str = "",
    is_interactive: bool = True,
) -> BackendType:
    """Compatibility alias; no automatic discovery or silent fallback."""
    return detect_backend(teammate_mode, is_interactive)
