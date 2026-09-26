"""Strict DAG definitions and fenced execution; no code/shell DSL evaluation."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from valecode.persistence.orchestration_store import OrchestrationStore, LeaseLost

REFERENCE = re.compile(r"\{\{\s*steps\.([A-Za-z][A-Za-z0-9_-]*)\.output\s*\}\}")


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    operator: Literal["equals", "contains"] = "equals"
    value: str = Field(max_length=1000)

    def matches(self, outputs):
        actual = outputs[self.step]
        return actual == self.value if self.operator == "equals" else self.value in actual


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    kind: Literal["agent", "branch", "join"] = "agent"
    depends_on: list[str] = Field(default_factory=list, max_length=32)
    agent_type: str | None = Field(default=None, max_length=128)
    prompt: str = Field(default="", max_length=20000)
    when: Predicate | None = None
    condition: Predicate | None = None
    read_only: bool = False
    max_turns: int = Field(default=20, ge=1, le=100)
    retries: int = Field(default=0, ge=0, le=2)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)

    @model_validator(mode="after")
    def contract(self):
        if self.kind == "agent" and (not self.agent_type or not self.prompt.strip() or self.condition is not None):
            raise ValueError("Agent nodes need agent_type/prompt and no condition")
        if self.kind == "branch" and self.condition is None:
            raise ValueError("Branch nodes need a condition")
        if self.kind != "agent" and (self.agent_type is not None or self.prompt or self.retries):
            raise ValueError("Control nodes cannot execute Agents or retry actions")
        if self.kind == "join" and self.condition is not None:
            raise ValueError("Join cannot evaluate a branch condition")
        if self.retries and not self.read_only:
            raise ValueError("Automatic retries are only allowed for builtin-read-only nodes")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("Duplicate dependencies")
        return self


class WorkflowDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=128)
    max_parallel: int = Field(default=4, ge=1, le=8)
    nodes: list[WorkflowNode] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def dag(self):
        nodes = {node.id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError("Duplicate node IDs")
        visiting, visited, ancestors = set(), set(), {}
        def visit(identity):
            if identity in visiting:
                raise ValueError("Workflow dependency cycle")
            if identity in visited:
                return ancestors[identity]
            if identity not in nodes:
                raise ValueError("Unknown dependency")
            visiting.add(identity)
            parents = set()
            for dependency in nodes[identity].depends_on:
                parents.add(dependency)
                parents.update(visit(dependency))
            visiting.remove(identity)
            visited.add(identity)
            ancestors[identity] = parents
            return parents
        for node in self.nodes:
            parents = visit(node.id)
            references = set(REFERENCE.findall(node.prompt))
            remainder = REFERENCE.sub("", node.prompt)
            if "{{" in remainder or "}}" in remainder:
                raise ValueError("Only {{steps.ID.output}} placeholders are supported")
            references.update(predicate.step for predicate in (node.when, node.condition) if predicate)
            if not references <= parents:
                raise ValueError("References/conditions must target explicit ancestors")
        return self


class WorkflowRuntime:
    def __init__(self, agent_tool, runner):
        self.agent_tool = agent_tool
        self.runner = runner
        self.store = OrchestrationStore(agent_tool._task_manager.task_store.database)
        self._running: set[asyncio.Task] = set()
        self._closed = False

    def scope(self):
        parent = self.agent_tool._parent_agent
        if not parent.session_id:
            raise ValueError("Workflow requires an active persistent session")
        return parent.session_id, parent.work_dir

    def load(self, path):
        _, root = self.scope()
        file = Path(root) / path
        if file.is_symlink() or not file.resolve().is_relative_to(Path(root).resolve()) or file.stat().st_size > 128000:
            raise ValueError("Workflow file must be within the workspace and <=128 KB")
        return WorkflowDefinition.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")))

    def create(self, definition):
        scope = self.scope()
        snapshots = {node.id: self.runner.snapshot(node.agent_type, node.max_turns) for node in definition.nodes if node.kind == "agent"}
        return self.store.create("workflow", *scope, {"definition": definition.model_dump(), "agents": snapshots}, [node.id for node in definition.nodes])

    def status(self, identity):
        state = self.store.get(identity, *self.scope())
        if state["kind"] != "workflow":
            raise ValueError("Not a workflow instance")
        return state | {"nodes": self.store.nodes(identity)}

    def retry(self, identity, node, confirmed):
        self.status(identity)
        self.store.retry_node(identity, node, confirmed=confirmed)

    async def run(self, identity):
        if self._closed:
            raise ValueError("Workflow runtime is closed")
        initial = self.status(identity)
        scope = self.scope()
        definition = WorkflowDefinition.model_validate(initial["spec"]["definition"])
        if set(initial["nodes"]) != {node.id for node in definition.nodes}:
            raise ValueError("Workflow node journal does not match its definition")
        agents = initial["spec"]["agents"]
        if set(agents) != {node.id for node in definition.nodes if node.kind == "agent"}:
            raise ValueError("Workflow snapshots do not match its definition")
        for node in definition.nodes:
            if node.kind == "agent":
                self.runner.validate_snapshot(agents[node.id], max_turns=node.max_turns)
        owner = uuid.uuid4().hex
        epoch = self.store.claim(identity, owner)
        if epoch is None:
            return self.status(identity)
        current = asyncio.current_task()
        self._running.add(current)
        async def keepalive():
            while True:
                if self.scope() != scope:
                    raise ValueError("Active session/workspace changed during workflow")
                await asyncio.sleep(5)
                try:
                    self.store.heartbeat(identity, owner, epoch)
                except LeaseLost:
                    current.cancel()
                    return
        heartbeat = asyncio.create_task(keepalive())
        try:
            while True:
                states = self.store.nodes(identity)
                if any(row["status"] in {"blocked", "failed"} for row in states.values()):
                    self.store.finish(identity, owner, epoch, "blocked", "Node requires explicit retry")
                    break
                pending = [node for node in definition.nodes if states[node.id]["status"] == "pending"]
                if not pending:
                    self.store.finish(identity, owner, epoch, "completed")
                    break
                ready = [node for node in pending if all(states[key]["status"] in {"succeeded", "skipped"} for key in node.depends_on)]
                if not ready:
                    raise ValueError("Workflow has no runnable node")
                writes = [node for node in ready if node.kind == "agent" and not node.read_only]
                batch = writes[:1] if writes else ready[:definition.max_parallel]
                # Await all siblings before changing the instance state. Only
                # declared builtin-read-only agents may share a parallel batch.
                siblings = [asyncio.create_task(self._node(identity, node, initial["spec"], owner, epoch, states)) for node in batch]
                try:
                    await asyncio.gather(*siblings)
                finally:
                    for sibling in siblings:
                        if not sibling.done():
                            sibling.cancel()
                    await asyncio.gather(*siblings, return_exceptions=True)
        except BaseException as exc:
            try:
                self.store.finish(identity, owner, epoch, "blocked", str(exc) or "Interrupted; explicit resume required")
            except LeaseLost:
                pass
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._running.discard(current)
        return self.status(identity)

    async def _node(self, identity, node, spec, owner, epoch, states):
        outputs = {key: row["output"] for key, row in states.items()}
        self.store.begin_node(identity, node.id, owner, epoch)
        if node.when and not node.when.matches(outputs):
            self.store.finish_node(identity, node.id, owner, epoch, "skipped")
            return
        if node.kind == "branch":
            self.store.finish_node(identity, node.id, owner, epoch, "succeeded", "true" if node.condition.matches(outputs) else "false")
            return
        if node.kind == "join":
            result = json.dumps({key: outputs[key] for key in node.depends_on}, ensure_ascii=False)
            self.store.finish_node(identity, node.id, owner, epoch, "succeeded", result)
            return
        prompt = REFERENCE.sub(lambda match: outputs[match.group(1)], node.prompt)
        if len(prompt) > 128000:
            self.store.finish_node(identity, node.id, owner, epoch, "blocked", error="Rendered prompt exceeds 128000 characters")
            return
        try:
            result = await asyncio.wait_for(self.runner.run(spec["agents"][node.id], prompt, read_only=node.read_only), timeout=node.timeout_seconds)
            self.store.finish_node(identity, node.id, owner, epoch, "blocked" if result.denied else "succeeded", result.output,
                error="Node needs user permission; explicit retry required" if result.denied else "",
                metadata={"input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "run_id": result.run_id, "denied": result.denied})
        except asyncio.CancelledError:
            raise
        except LeaseLost:
            raise
        except Exception as exc:
            attempts = self.store.nodes(identity)[node.id]["attempts"]
            status = "pending" if node.read_only and attempts <= node.retries else "blocked"
            self.store.finish_node(identity, node.id, owner, epoch, status, error=str(exc))

    async def cancel(self):
        running = [task for task in self._running if task is not asyncio.current_task()]
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)

    async def close(self):
        self._closed = True
        await self.cancel()
