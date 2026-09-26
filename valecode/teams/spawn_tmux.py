from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TmuxPaneInfo:
    pane_id: str
    session: str


class TmuxSpawnError(Exception):
    pass


def _run_tmux(*args: str) -> str:
    result = subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise TmuxSpawnError(result.stderr.strip() or "tmux failed")
    return result.stdout.strip()


def build_cli_command(launch_path: str | Path) -> str:
    # No prompt, keys or environment assignments reach the shell.
    return shlex.join([sys.executable, "-m", "valecode", "--teammate-launch", str(launch_path)])


def spawn_tmux_teammate(launch_path: str | Path, root: str | Path, label: str) -> TmuxPaneInfo:
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", label)[:60] or "worker"
    name = f"valecode-{slug}-{Path(launch_path).stem[:8]}"
    command = build_cli_command(launch_path)
    if os.environ.get("TMUX"):
        pane = _run_tmux("new-window", "-d", "-P", "-F", "#{pane_id}", "-n", name, "-c", str(root), command)
        session = "current"
    else:
        pane = _run_tmux("new-session", "-d", "-P", "-F", "#{pane_id}", "-s", name, "-c", str(root), command)
        session = name
    if not re.fullmatch(r"%[0-9]+", pane):
        raise TmuxSpawnError("tmux did not return an exact pane ID")
    return TmuxPaneInfo(pane, session)


def send_keys_to_pane(pane_id: str, keys: str = "") -> None:
    # Workers poll their mailbox; never send Enter into their process/shell.
    return None


def kill_pane(pane_id: str) -> None:
    if not re.fullmatch(r"%[0-9]+", pane_id):
        raise TmuxSpawnError("Refusing an ambiguous tmux target")
    _run_tmux("kill-pane", "-t", pane_id)
