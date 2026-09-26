"""Private, one-shot launch manifests and atomic worker/parent heartbeats.

These files are execution control state, not a secrets store or message bus.
The shared SQLite Team/Session records remain the identity authority.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

from valecode.persistence import Database, TeamStore

SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
TERMINAL = {"stopped", "failed"}


def atomic_json(path: Path, data: dict) -> None:
    if path.is_symlink():
        raise ValueError("Worker state must not be a symlink")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows readers briefly hold a delete-sharing lock. Keep
                # replacement atomic; never fall back to truncation/in-place.
                if attempt == 19:
                    raise
                time.sleep(0.005)
    finally:
        temporary.unlink(missing_ok=True)


class WorkerLaunch:
    def __init__(self, path: str | Path):
        raw = Path(path).absolute()
        if raw.is_symlink() or raw.parent.name != "pane-workers" or raw.parent.parent.name != ".valecode" or not SAFE_ID.fullmatch(raw.stem) or raw.suffix != ".json":
            raise ValueError("Invalid worker launch path")
        self.root = raw.parent.parent.parent.resolve()
        directory = (self.root / ".valecode" / "pane-workers").resolve()
        if not directory.is_relative_to(self.root) or raw.resolve().parent != directory:
            raise ValueError("Worker launch directory escapes project")
        self.path = directory / raw.name
        self.state_path = self.path.with_suffix(".state")
        self.parent_path = self.path.with_suffix(".parent")
        self.claim_path = self.path.with_suffix(".claim")

    @classmethod
    def prepare(cls, root: str | Path, payload: dict) -> WorkerLaunch:
        launch_id = uuid.uuid4().hex
        launch = cls(Path(root) / ".valecode" / "pane-workers" / (launch_id + ".json"))
        launch.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "launch_id": launch_id, **payload}
        # No Provider credentials, environment or arbitrary shell command.
        launch.validate(data)
        atomic_json(launch.path, data)
        launch.heartbeat_parent()
        launch.update_state("starting", tool_count=0, token_count=0)
        return launch

    def read(self) -> dict:
        if self.path.is_symlink() or self.path.stat().st_size > 256000:
            raise ValueError("Invalid worker manifest")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.validate(data)
        return data

    def validate(self, data: dict) -> None:
        from valecode.tools.agent_tool import AgentTool
        from valecode.permissions import PermissionMode
        required = {"version", "launch_id", "session_id", "team_name", "agent_id", "member_name", "lead_id", "work_dir", "provider_name", "model", "prompt", "definition", "allowed_tools", "permission_mode", "mailbox_dir", "parent_run_id", "trace_id", "sandbox"}
        if not isinstance(data, dict) or set(data) != required or data["version"] != 1 or data["launch_id"] != self.path.stem:
            raise ValueError("Invalid worker manifest schema")
        for key in ("session_id", "team_name", "agent_id", "lead_id"):
            if not isinstance(data[key], str) or not SAFE_ID.fullmatch(data[key]):
                raise ValueError(f"Invalid worker {key}")
        for key in ("member_name", "provider_name", "model", "work_dir", "mailbox_dir", "prompt"):
            if not isinstance(data[key], str) or not data[key] or len(data[key]) > 20000:
                raise ValueError(f"Invalid worker {key}")
        if AgentTool._definition_from_resume_spec(data["definition"]) is None:
            raise ValueError("Invalid worker agent definition")
        PermissionMode(data["permission_mode"])
        if not isinstance(data["allowed_tools"], list) or len(data["allowed_tools"]) > 1000 or any(not isinstance(x, str) or len(x) > 256 for x in data["allowed_tools"]):
            raise ValueError("Invalid worker tool ceiling")
        if any(data[key] is not None and not isinstance(data[key], str) for key in ("parent_run_id", "trace_id")):
            raise ValueError("Invalid worker lineage")
        if not isinstance(data["sandbox"], dict) or set(data["sandbox"]) != {"enabled", "network_enabled", "auto_allow"} or any(type(x) is not bool for x in data["sandbox"].values()):
            raise ValueError("Invalid worker sandbox settings")
        database_path = self.root / ".valecode" / "control.db"
        if not database_path.is_file() or database_path.is_symlink():
            raise ValueError("Parent control database is missing or unsafe")
        database = Database(database_path)
        team_store = TeamStore(database)
        team = team_store.get_team(data["team_name"])
        members = team_store.list_members(data["team_name"])
        member = next((m for m in members if m.agent_id == data["agent_id"]), None)
        if team is None or team.status != "active" or team.lead_agent_id != data["lead_id"] or member is None or member.name != data["member_name"]:
            raise ValueError("Worker is not the registered member of the active team")
        work_dir = Path(data["work_dir"]).resolve()
        if not work_dir.is_dir() or work_dir != Path(member.worktree_path).resolve() or not work_dir.is_relative_to((self.root / ".valecode" / "worktrees").resolve()):
            raise ValueError("Worker worktree does not match registered managed worktree")
        # Verify actual Git ownership, not just a path prefix.
        from valecode.worktree.manager import WorktreeManager
        if WorktreeManager.read_worktree_head_sha(str(work_dir)) is None:
            raise ValueError("Worker directory is not a valid Git worktree")
        import subprocess
        def common_dir(directory):
            try:
                result = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=directory,
                    capture_output=True, text=True, timeout=5, check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError("Cannot verify worker Git ownership") from exc
            return (Path(directory) / result.stdout.strip()).resolve()
        if common_dir(work_dir) != common_dir(self.root):
            raise ValueError("Worker belongs to a different Git repository")
        from valecode.teams.models import resolve_team_dir
        import tempfile
        mailbox = Path(data["mailbox_dir"])
        expected = (resolve_team_dir(data["team_name"]) / "mailbox").resolve()
        fallback = (Path(tempfile.gettempdir()) / "valecode" / "teams" / data["team_name"] / "mailbox").resolve()
        if mailbox.is_symlink() or mailbox.resolve() not in {expected, fallback} or not mailbox.is_dir():
            raise ValueError("Worker mailbox is outside the team's managed state")
        with database.reader() as db:
            if db.execute("SELECT 1 FROM sessions WHERE id=?", (data["session_id"],)).fetchone() is None:
                raise ValueError("Worker session is missing")
            if data["parent_run_id"] and db.execute("SELECT 1 FROM runs WHERE id=? AND session_id=?", (data["parent_run_id"], data["session_id"])).fetchone() is None:
                raise ValueError("Worker parent run is outside its session")

    def claim(self) -> dict:
        data = self.read()  # Validate before creating any state or loading keys.
        if not self.parent_alive():
            raise ValueError("Worker parent is no longer alive")
        with self.claim_path.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        return data

    def heartbeat_parent(self, *, stop=False) -> None:
        atomic_json(self.parent_path, {"timestamp": time.time(), "stop": stop})

    def parent_alive(self) -> bool:
        try:
            if self.parent_path.is_symlink():
                return False
            data = json.loads(self.parent_path.read_text(encoding="utf-8"))
            stamp = data["timestamp"]
            return data.get("stop") is False and type(stamp) in (int, float) and 0 <= time.time() - stamp < 30
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def update_state(self, status: str, **progress) -> None:
        if status not in {"starting", "running", "idle", "stopped", "failed"}:
            raise ValueError("Unknown worker state")
        atomic_json(self.state_path, {"launch_id": self.path.stem, "timestamp": time.time(), "status": status, **progress})

    def state(self) -> dict:
        if self.state_path.is_symlink() or self.state_path.stat().st_size > 100000:
            raise ValueError("Invalid worker state")
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("launch_id") != self.path.stem or data.get("status") not in {"starting", "running", "idle", "stopped", "failed"}:
            raise ValueError("Invalid worker status")
        import math
        stamp = data.get("timestamp")
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Invalid worker timestamp")
        for key in ("tool_count", "token_count", "input_tokens", "output_tokens"):
            value = data.get(key, 0)
            if type(value) is not int or not 0 <= value <= 10**12:
                raise ValueError("Invalid worker progress counter")
        for key in ("last_activity", "last_message", "error"):
            if key in data and (not isinstance(data[key], str) or len(data[key]) > 2000):
                raise ValueError("Invalid worker progress text")
        return data
