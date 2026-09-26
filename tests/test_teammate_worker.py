from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from valecode.persistence import Database, SessionStore, TeamStore, RunStore, RunStatus
from valecode.teams.models import TeammateInfo
from valecode.teams.mailbox import Mailbox, create_message
from valecode.teams.worker_launch import WorkerLaunch
from valecode.agents.parser import AgentDef
from valecode.tools.agent_tool import AgentTool


@pytest.fixture
def launch_data(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    def git(*args):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    git("init")
    (root / "README.md").write_text("fixture", encoding="utf-8")
    git("add", "README.md")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
    worktree = root / ".valecode" / "worktrees" / "worker"
    git("worktree", "add", "--detach", str(worktree), "HEAD")
    database = Database(root / ".valecode" / "control.db")
    database.initialize()
    SessionStore(database).upsert("session")
    team = TeamStore(database)
    team.upsert_team("test-team", "lead-id", backend_type="tmux")
    team.upsert_member("test-team", TeammateInfo("worker", "worker-id", "probe", "fake", str(worktree), "tmux", True))
    parent_run = RunStore(database).create_run("session", input="parent", agent_id="lead-id")
    RunStore(database).transition_run(parent_run.id, RunStatus.RUNNING)
    mailbox = tmp_path / "state" / "teams" / "test-team" / "mailbox"
    mailbox.mkdir(parents=True)
    monkeypatch.setenv("VALECODE_STATE_DIR", str(tmp_path / "state"))
    data = dict(session_id="session", team_name="test-team", agent_id="worker-id", member_name="worker",
        lead_id="lead-id", work_dir=str(worktree), provider_name="fixture", model="fake", prompt="say hello",
        definition=AgentTool._resume_spec(AgentDef("probe", "test", max_turns=3), None),
        allowed_tools=["ReadFile", "WriteFile", "Grep", "ToolSearch"], permission_mode="default",
        mailbox_dir=str(mailbox), parent_run_id=parent_run.id, trace_id="trace-fixture",
        sandbox={"enabled": False, "network_enabled": False, "auto_allow": False})
    return root, database, data


def test_launch_manifest_claim_is_one_shot_and_contains_no_credentials(launch_data):
    root, database, data = launch_data
    launch = WorkerLaunch.prepare(root, data)
    assert launch.claim()["agent_id"] == "worker-id"
    with pytest.raises(FileExistsError):
        launch.claim()
    assert "api_key" not in launch.path.read_text(encoding="utf-8")
    assert launch.parent_alive()
    launch.heartbeat_parent(stop=True)
    assert not launch.parent_alive()


@pytest.mark.parametrize("change", [
    {"agent_id": "unknown"}, {"lead_id": "foreign"}, {"session_id": "other"},
    {"member_name": "other"}, {"permission_mode": "unknown"},
    {"allowed_tools": [True]}, {"api_key": "must not serialize"},
    {"version": 2}, {"parent_run_id": "foreign-run"},
])
def test_invalid_identity_or_descriptor_rejected_before_claim(launch_data, change):
    root, database, data = launch_data
    with pytest.raises(ValueError):
        WorkerLaunch.prepare(root, data | change)
    assert not list((root / ".valecode" / "pane-workers").glob("*.claim"))


def test_parent_stopped_and_unmanaged_worktree_fail_closed(launch_data, tmp_path):
    root, database, data = launch_data
    with pytest.raises(ValueError):
        WorkerLaunch.prepare(root, data | {"work_dir": str(tmp_path)})
    launch = WorkerLaunch.prepare(root, data)
    launch.heartbeat_parent(stop=True)
    with pytest.raises(ValueError, match="parent"):
        launch.claim()


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, responses):
        self.responses = responses
        self.requests = []
        super().__init__(("127.0.0.1", 0), Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        index = len(self.server.requests) - 1
        response = self.server.responses[min(index, len(self.server.responses) - 1)]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delta, finish in response:
            payload = {"id": "fixture", "object": "chat.completion.chunk", "created": 1,
                "model": "fake", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def start_worker(root, data, server, tmp_path):
    env_file = root / ".env"
    env_file.write_text("\n".join([
        "VALECODE_PROVIDER_NAME=fixture", "VALECODE_PROTOCOL=openai-compat",
        f"VALECODE_BASE_URL=http://127.0.0.1:{server.server_port}/v1",
        "VALECODE_MODEL=fake", "VALECODE_API_KEY=fixture", "VALECODE_MEMORY_ENABLED=false",
    ]), encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("VALECODE_")}
    env["USERPROFILE"] = str(home)
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VALECODE_STATE_DIR"] = str(Path(data["mailbox_dir"]).parents[2])
    launch = WorkerLaunch.prepare(root, data)
    process = subprocess.Popen([sys.executable, "-m", "valecode", "--teammate-launch", str(launch.path)],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    return launch, process


def wait_idle(launch, process, *, calls, server):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        launch.heartbeat_parent()
        state = launch.state()
        if state["status"] == "idle" and len(server.requests) >= calls:
            return state
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=2)
            pytest.fail(f"Worker exited early ({process.returncode}): {state}; {stdout}; {stderr}")
        time.sleep(0.1)
    pytest.fail(f"Worker did not become idle: {launch.state()}")


def test_real_worker_process_runs_followup_and_does_not_reconcile_parent(launch_data, tmp_path):
    root, database, data = launch_data
    server = FakeServer([
        [({"content": "initial result"}, None), ({}, "stop")],
        [({"content": "followup result"}, None), ({}, "stop")],
    ])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    launch, process = start_worker(root, data, server, tmp_path)
    try:
        wait_idle(launch, process, calls=1, server=server)
        assert RunStore(database).get_run(data["parent_run_id"]).status == RunStatus.RUNNING
        mailbox = Mailbox(data["mailbox_dir"])
        mailbox.write(data["agent_id"], create_message(data["lead_id"], data["agent_id"], "follow up", summary="retry"))
        state = wait_idle(launch, process, calls=2, server=server)
        assert "followup result" in state["last_message"]
        launch.heartbeat_parent(stop=True)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert "initial result" in stdout and "followup result" in stdout
        assert launch.state()["status"] == "stopped"
        assert TeamStore(database).list_members(data["team_name"])[0].status == "stopped"
        children = RunStore(database).list_runs(session_id=data["session_id"])
        assert sum(row.parent_run_id == data["parent_run_id"] for row in children) == 2
        assert len(mailbox.consume(data["lead_id"])) == 2
    finally:
        launch.heartbeat_parent(stop=True)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        server.shutdown()
        server.server_close()


def test_real_worker_process_does_not_auto_approve_writes(launch_data, tmp_path):
    root, database, data = launch_data
    destination = Path(data["work_dir"]) / "must-not-write.txt"
    call = {"tool_calls": [{"index": 0, "id": "write-1", "type": "function", "function": {
        "name": "WriteFile", "arguments": json.dumps({"file_path": str(destination), "content": "unsafe"})}}]}
    server = FakeServer([
        [(call, None), ({}, "tool_calls")],
        [({"content": "write was denied"}, None), ({}, "stop")],
    ])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    launch, process = start_worker(root, data, server, tmp_path)
    try:
        state = wait_idle(launch, process, calls=2, server=server)
        assert state["tool_count"] == 1 and not destination.exists()
        assert "denied" in json.dumps(server.requests[-1]["messages"]).lower()
        launch.heartbeat_parent(stop=True)
        process.communicate(timeout=10)
        assert process.returncode == 0
    finally:
        launch.heartbeat_parent(stop=True)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        server.shutdown()
        server.server_close()
