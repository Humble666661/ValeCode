"""Official iTerm2 API bridge, isolated from the parent's asyncio loop."""
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ITermPaneInfo:
    session_id: str


class ITermSpawnError(Exception):
    pass


def _bridge(operation: str, target: str, root: str = "") -> str:
    result = subprocess.run([sys.executable, "-m", "valecode.teams.spawn_iterm2", operation, target, root],
        capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise ITermSpawnError(result.stderr.strip() or "iTerm2 API request failed")
    value = next((line[9:] for line in result.stdout.splitlines() if line.startswith("VALECODE:")), "")
    if not value or len(value) > 256 or any(c.isspace() for c in value):
        raise ITermSpawnError("iTerm2 did not return a session identity")
    return value


def spawn_iterm2_teammate(launch_path: str | Path, root: str | Path, label: str) -> ITermPaneInfo:
    return ITermPaneInfo(_bridge("spawn", str(launch_path), str(root)))


def kill_pane(session_id: str) -> None:
    _bridge("close", session_id)


async def _request(connection, operation: str, target: str, root: str, api):
    if operation == "spawn":
        from valecode.teams.spawn_tmux import build_cli_command
        shell = f"cd {shlex.quote(root)} && exec {build_cli_command(target)}"
        window = await api.Window.async_create(connection, command=shlex.join(["/bin/sh", "-c", shell]))
        if window is None or window.current_tab is None or window.current_tab.current_session is None:
            raise ITermSpawnError("Created window has no session")
        return window.current_tab.current_session.session_id
    if operation == "close":
        app = await api.async_get_app(connection)
        session = app.get_session_by_id(target)
        if session is not None:
            await session.async_close(force=True)
        return target
    raise ITermSpawnError("Unknown operation")


if __name__ == "__main__":
    import iterm2
    async def main(connection):
        value = await _request(connection, sys.argv[1], sys.argv[2], sys.argv[3], iterm2)
        print("VALECODE:" + value, flush=True)
    iterm2.run_until_complete(main)
