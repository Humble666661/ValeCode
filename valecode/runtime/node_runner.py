"""Fresh bounded Agents for orchestration nodes, without authority elevation."""
from __future__ import annotations

from dataclasses import dataclass, replace

from valecode.runtime.execution import CancellationToken


@dataclass
class NodeResult:
    output: str
    input_tokens: int = 0
    output_tokens: int = 0
    run_id: str | None = None
    denied: bool = False


class AgentNodeRunner:
    def __init__(self, agent_tool):
        self.tool = agent_tool

    def snapshot(self, agent_type, max_turns):
        definition = self.tool._agent_loader.get(agent_type)
        if definition is None:
            raise ValueError(f"Unknown agent type: {agent_type}")
        if definition.isolation:
            raise ValueError("Orchestration nodes do not implicitly create worktrees")
        model = definition.model
        if model == "inherit":
            model = self.tool._parent_agent.model or getattr(self.tool._provider_config, "model", "inherit")
        bounded = replace(definition, model=model, permission_mode="default", max_turns=min(max_turns, definition.max_turns))
        return self.tool._resume_spec(bounded, None)

    def validate_snapshot(self, snapshot, *, max_turns=100):
        definition = self.tool._definition_from_resume_spec(snapshot)
        if definition is None or definition.permission_mode != "default" or definition.max_turns > max_turns:
            raise ValueError("Invalid node execution snapshot")

    async def run(self, snapshot, prompt, *, read_only=False):
        from valecode.permissions import PermissionMode
        from valecode.tools import ToolRegistry, ToolSource
        from valecode.persistence import RunStatus
        self.validate_snapshot(snapshot)
        definition = self.tool._definition_from_resume_spec(snapshot)
        parent = self.tool._parent_agent
        child = self.tool._build_recovered_agent(definition,
            parent_run_id=parent._current_run_id, trace_id=parent.trace_id)
        child.cancellation_token = CancellationToken()
        child._owns_cancellation_token = True
        if parent.permission_checker is not None:
            child.permission_checker.rule_engine = parent.permission_checker.rule_engine.clone()
            if parent.permission_checker.mode == PermissionMode.PLAN:
                child.set_permission_mode(PermissionMode.PLAN)
        if read_only:
            registry = ToolRegistry()
            for name in ("ReadFile", "Glob", "Grep"):
                registration = child.registry.get_registration(name)
                if registration is not None and registration.source == ToolSource.BUILTIN and registration.tool.is_read_only:
                    child.registry.copy_registration_to(registry, name)
            child.registry = registry
            # Hooks/plugins/MCP can execute arbitrary side effects. A declared
            # read-only node gets only builtin readers and no HookEngine.
            child.hook_engine = None
        try:
            output = await child.run_to_completion(prompt)
            run_id = None
            denied = False
            if child.run_store is not None:
                with child.run_store.database.reader() as db:
                    row = db.execute("SELECT id,status FROM runs WHERE agent_id=? ORDER BY created_at DESC LIMIT 1", (child.agent_id,)).fetchone()
                    if row is not None:
                        run_id = row["id"]
                        if row["status"] != RunStatus.COMPLETED.value:
                            raise RuntimeError("Node Agent did not complete its run")
                        denied = db.execute("SELECT 1 FROM tool_calls WHERE run_id=? AND status='denied'", (run_id,)).fetchone() is not None
            self.tool._trace_manager.update(child.agent_id, input_tokens=child.total_input_tokens, output_tokens=child.total_output_tokens)
            self.tool._trace_manager.complete(child.agent_id, "completed")
            return NodeResult(output, child.total_input_tokens, child.total_output_tokens, run_id, denied)
        except BaseException:
            child.cancel("Orchestration node cancelled or failed")
            self.tool._trace_manager.complete(child.agent_id, "failed")
            raise
