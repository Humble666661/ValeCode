from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from valecode.persistence import Database, SessionStore
from valecode.persistence.orchestration_store import OrchestrationStore, LeaseLost
from valecode.runtime.workflow import WorkflowDefinition, WorkflowRuntime
from valecode.runtime.node_runner import NodeResult
from test_harness import remote


class Runner:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak = 0
        self.fail = set()
        self.denied = False
        self.gate = None

    def snapshot(self, kind, turns):
        if kind != "fixture":
            raise ValueError("Unknown agent type")
        return {"type": kind, "max_turns": turns}

    def validate_snapshot(self, snapshot, *, max_turns=100):
        if snapshot["max_turns"] > max_turns:
            raise ValueError("invalid snapshot")

    async def run(self, snapshot, prompt, *, read_only=False):
        self.calls.append((prompt, read_only))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0.01)
            if prompt in self.fail:
                raise RuntimeError("fixture failed")
            return NodeResult(prompt, 2, 3, denied=self.denied)
        finally:
            self.active -= 1


@pytest.fixture
def runtime(tmp_path):
    database = Database(tmp_path / "control.db")
    database.initialize()
    SessionStore(database).upsert("s")
    parent = SimpleNamespace(session_id="s", work_dir=str(tmp_path))
    tool = SimpleNamespace(_parent_agent=parent, _task_manager=SimpleNamespace(task_store=SimpleNamespace(database=database)))
    return WorkflowRuntime(tool, Runner())


def node(identity, **kwargs):
    return {"id": identity, "agent_type": "fixture", "prompt": identity, **kwargs}


def definition(nodes):
    return WorkflowDefinition.model_validate({"name": "fixture", "nodes": nodes})


@pytest.mark.parametrize("nodes", [
    [node("x"), node("x")],
    [node("x", depends_on=["missing"])],
    [node("x", depends_on=["y"]), node("y", depends_on=["x"])],
    [node("x", prompt="{{steps.y.output}}"), node("y")],
    [node("x", prompt="{{__import__('os')}}")],
    [node("x", retries=1)],
    [node("x", read_only="true")],
    [{"id": "x", "kind": "branch"}],
    [node("x", depends_on=["y", "y"]), node("y")],
    [node("x", arbitrary_shell="echo hi")],
])
def test_definition_rejects_invalid_graphs(nodes):
    with pytest.raises(ValidationError):
        definition(nodes)


@pytest.mark.asyncio
async def test_branch_parallel_join_and_completed_resume(runtime):
    spec = definition([node("seed", read_only=True),
        {"id": "branch", "kind": "branch", "depends_on": ["seed"], "condition": {"step": "seed", "value": "seed"}},
        node("left", read_only=True, depends_on=["branch"], prompt="{{steps.seed.output}} left"),
        node("right", read_only=True, depends_on=["branch"]),
        node("skip", depends_on=["branch"], when={"step": "branch", "value": "false"}),
        {"id": "join", "kind": "join", "depends_on": ["left", "right", "skip"]}])
    identity = runtime.create(spec)
    result = await runtime.run(identity)
    assert result["status"] == "completed" and result["nodes"]["skip"]["status"] == "skipped"
    assert json.loads(result["nodes"]["join"]["output"])["left"] == "seed left"
    assert runtime.runner.peak == 2
    await runtime.run(identity)
    assert len(runtime.runner.calls) == 3


@pytest.mark.asyncio
async def test_writes_are_exclusive_and_read_retry_bounded(runtime):
    runtime.runner.fail.add("read")
    identity = runtime.create(definition([node("write"), node("read", read_only=True, retries=2)]))
    result = await runtime.run(identity)
    assert result["status"] == "blocked" and runtime.runner.peak == 1
    assert result["nodes"]["read"]["attempts"] == 3
    assert result["nodes"]["write"]["attempts"] == 1
    await runtime.run(identity)
    assert len(runtime.runner.calls) == 4
    with pytest.raises(ValueError, match="confirmation"):
        runtime.retry(identity, "read", False)
    runtime.runner.fail.clear()
    runtime.retry(identity, "read", True)
    assert (await runtime.run(identity))["status"] == "completed"
    assert runtime.status(identity)["nodes"]["write"]["attempts"] == 1


@pytest.mark.asyncio
async def test_cancel_waits_all_children_and_blocks_replay(runtime):
    runtime.runner.gate = asyncio.Event()
    identity = runtime.create(definition([node("a", read_only=True), node("b", read_only=True)]))
    task = asyncio.create_task(runtime.run(identity))
    while runtime.runner.active < 2:
        await asyncio.sleep(0.001)
    await runtime.cancel()
    assert task.cancelled() and runtime.runner.active == 0
    assert runtime.status(identity)["status"] == "blocked"
    await runtime.run(identity)
    assert len(runtime.runner.calls) == 2


@pytest.mark.asyncio
async def test_denied_node_does_not_complete(runtime):
    runtime.runner.denied = True
    identity = runtime.create(definition([node("a")]))
    assert (await runtime.run(identity))["status"] == "blocked"


@pytest.mark.asyncio
async def test_scope_corruption_and_unknown_agent_fail_closed(runtime, tmp_path):
    with pytest.raises(ValueError):
        runtime.create(definition([node("a", agent_type="unknown")]))
    identity = runtime.create(definition([node("a")]))
    runtime.agent_tool._parent_agent.session_id = "other"
    with pytest.raises(ValueError, match="outside"):
        runtime.status(identity)
    runtime.agent_tool._parent_agent.session_id = "s"
    with runtime.store.database.transaction() as db:
        db.execute("DELETE FROM orchestration_nodes WHERE instance_id=?", (identity,))
    with pytest.raises(ValueError, match="journal"):
        await runtime.run(identity)
    outside = tmp_path.parent / "outside-fixture.yaml"
    outside.write_text("name: no\nnodes: []", encoding="utf-8")
    with pytest.raises(ValueError, match="within"):
        runtime.load(str(outside))


def test_expired_lease_never_replays_uncertain_node_and_fences_old_owner(runtime):
    identity = runtime.create(definition([node("a"), node("b")]))
    epoch = runtime.store.claim(identity, "old")
    runtime.store.begin_node(identity, "a", "old", epoch)
    runtime.store.finish_node(identity, "a", "old", epoch, "succeeded", "keep")
    runtime.store.begin_node(identity, "b", "old", epoch)
    with pytest.raises(LeaseLost):
        runtime.store.claim(identity, "new")
    with runtime.store.database.transaction() as db:
        db.execute("UPDATE orchestrations SET lease_until=0 WHERE id=?", (identity,))
    assert runtime.store.claim(identity, "new") is None
    assert runtime.status(identity)["nodes"]["a"]["output"] == "keep"
    assert runtime.status(identity)["nodes"]["b"]["status"] == "blocked"
    with pytest.raises(LeaseLost):
        runtime.store.finish_node(identity, "b", "old", epoch, "succeeded")
    runtime.retry(identity, "b", True)
    assert runtime.store.claim(identity, "new") > epoch


def test_cross_process_claim_is_atomic(runtime):
    import sys
    identity = runtime.create(definition([node("a")]))
    code = "from valecode.persistence import Database; from valecode.persistence.orchestration_store import OrchestrationStore; import sys; s=OrchestrationStore(Database(sys.argv[1])); print(s.claim(sys.argv[2],sys.argv[3]))"
    processes = [subprocess.Popen([sys.executable, "-c", code, str(runtime.store.database.path), identity, str(index)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for index in range(2)]
    results = [process.communicate(timeout=15) for process in processes]
    assert sorted(process.returncode for process in processes) == [0, 1]
    assert sum(b"already owned" in stderr for _, stderr in results) == 1


@pytest.mark.asyncio
async def test_real_node_agent_reads_with_persisted_lineage_and_default_permissions(remote, tmp_path):
    from test_teammate_worker import FakeServer
    from valecode.config import ProviderConfig
    from valecode.client import create_client
    from valecode.permissions import PermissionMode
    (tmp_path / "artifact.txt").write_text("actual artifact", encoding="utf-8")
    call = {"tool_calls": [{"index": 0, "id": "read-1", "type": "function", "function": {
        "name": "ReadFile", "arguments": json.dumps({"file_path": str(tmp_path / "artifact.txt")})}}]}
    service = FakeServer([[(call, None), ({}, "tool_calls")], [({"content": "observed artifact"}, None), ({}, "stop")]])
    threading.Thread(target=service.serve_forever, daemon=True).start()
    provider = ProviderConfig("fixture", "openai-compat", f"http://127.0.0.1:{service.server_port}/v1", "fake", "fixture")
    remote.agent.client = create_client(provider)
    remote.agent_tool._provider_config = provider
    remote.agent.model = provider.model
    remote.agent.set_permission_mode(PermissionMode.BYPASS)
    runtime = remote.workflow_runtime
    identity = runtime.create(definition([node("a", agent_type="general-purpose", prompt="read artifact", read_only=True)]))
    try:
        result = await asyncio.wait_for(runtime.run(identity), 10)
        assert result["status"] == "completed"
        assert result["nodes"]["a"]["metadata"]["run_id"]
        assert "actual artifact" in json.dumps(service.requests[-1])
        exposed = {item["function"]["name"] for item in service.requests[0]["tools"]}
        assert exposed <= {"ReadFile", "Glob", "Grep"} and "ReadFile" in exposed
        assert remote.command_registry.find("workflow") is not None
    finally:
        await remote._shutdown()
        service.shutdown()
        service.server_close()


@pytest.mark.asyncio
async def test_command_shows_id_before_running_and_requires_retry_confirmation(runtime):
    from valecode.commands.handlers.orchestration import create_orchestration_command
    messages = []
    ui = SimpleNamespace(_streaming=False, add_system_message=messages.append)
    ctx = SimpleNamespace(args="retry id a", session=SimpleNamespace(session_id="s"), ui=ui)
    await create_orchestration_command(runtime).handler(ctx)
    assert "--confirm" in messages[-1]
