
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from valecode.config import ConfigError, load_config
from valecode.hooks import HookConfigError, HookEngine, load_hooks
from valecode.permissions import PermissionMode


def _configure_logging(
    state_dir: Path = Path(".valecode"),
    fallback_dir: Path | None = None,
) -> Path | None:
    """配置文件日志；项目目录不可写时降级到系统临时目录。"""
    if fallback_dir is None:
        fallback_dir = Path(tempfile.gettempdir()) / "valecode"

    log_format = "%(asctime)s %(name)s %(message)s"
    for directory in (state_dir, fallback_dir):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "debug.log"
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                filename=str(log_path),
                filemode="w",
                force=True,
            )
            return log_path
        except OSError:
            continue

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.NullHandler()],
        force=True,
    )
    return None


def main() -> None:

    parser = argparse.ArgumentParser(prog="valecode", description="ValeCode AI coding assistant")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in PermissionMode],
        default=None,
        help="Permission mode (overrides config.yaml)",
    )
    parser.add_argument(
        "-p",
        metavar="PROMPT",
        default=None,
        help="Run non-interactively: execute the prompt and print the result to stdout",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "stream-json"],
        default="text",
        help="Output format for -p mode: 'text' (default) prints final text, 'stream-json' emits NDJSON events",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        default=False,
        help="Start the browser UI (defaults to 127.0.0.1:18888)",
    )
    args = parser.parse_args()

    # --help / argparse 参数错误应在触碰项目状态目录之前完成。
    _configure_logging()

    try:
        config = load_config()
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    from valecode.observability import configure_tracing_from_env

    configure_tracing_from_env(Path.cwd())

    mode_str = args.mode if args.mode else config.permission_mode
    permission_mode = PermissionMode(mode_str)

    try:
        hooks = load_hooks(config.raw_hooks)
    except HookConfigError as e:
        print(f"Hook config error: {e}", file=sys.stderr)
        sys.exit(1)

    hook_engine = HookEngine(hooks) if hooks else None

    if args.p is not None:
        output_format = getattr(args, "output_format", "text")
        asyncio.run(_run_prompt_with_cleanup(
            config, permission_mode, hook_engine, args.p, output_format,
        ))
        return

    # Remote 模式：默认只监听回环地址；非回环监听必须配置访问 Token。
    if args.remote:
        from valecode.remote import RemoteServer

        try:
            server = RemoteServer(
                providers=config.providers,
                mcp_servers=config.mcp_servers,
                hook_engine=hook_engine,
                sandbox_config=config.sandbox,
                addr=config.remote.host,
                port=config.remote.port,
                auth_token=config.remote.token,
                enable_fork=config.enable_fork,
                enable_verification_agent=config.enable_verification_agent,
                background_task_config=config.background_tasks,
            )
        except ValueError as e:
            print(f"Remote config error: {e}", file=sys.stderr)
            sys.exit(1)
        asyncio.run(server.run())
        return

    from valecode.app import ValeCodeApp
    from valecode.driver import NoAltScreenDriver

    app = ValeCodeApp(
        providers=config.providers,
        permission_mode=permission_mode,
        mcp_servers=config.mcp_servers,
        hook_engine=hook_engine,
        enable_fork=config.enable_fork,
        enable_verification_agent=config.enable_verification_agent,
        worktree_config=config.worktree,
        teammate_mode=config.teammate_mode,
        enable_coordinator_mode=config.enable_coordinator_mode,
        driver_class=NoAltScreenDriver,
        sandbox_config=config.sandbox,
        background_task_config=config.background_tasks,
    )
    app.run()


async def _run_prompt_with_cleanup(
    config, permission_mode, hook_engine, prompt: str, output_format: str,
) -> None:
    resources = _PromptResources()
    try:
        await _run_prompt(
            config, permission_mode, hook_engine, prompt, output_format,
            _resources=resources,
        )
    finally:
        await resources.close()
        if hook_engine is not None:
            await hook_engine.shutdown()


class _PromptResources:
    """Resources owned by one non-interactive prompt invocation."""

    def __init__(self) -> None:
        self.registry = None
        self.session = None
        self.mcp_manager = None
        self.task_manager = None
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.mcp_manager is not None:
            try:
                await self.mcp_manager.shutdown()
            except Exception:
                logging.warning("Failed to shut down prompt MCP manager", exc_info=True)
        if self.task_manager is not None:
            try:
                await self.task_manager.shutdown()
            except Exception:
                logging.warning("Failed to stop prompt task manager", exc_info=True)
        if self.registry is not None:
            try:
                from valecode.tools import ToolSource

                await self.registry.release_source(ToolSource.PLUGIN)
                await self.registry.release_session()
            except Exception:
                logging.warning("Failed to release prompt tool session", exc_info=True)
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                logging.warning("Failed to close prompt session", exc_info=True)


async def _run_prompt(
    config, permission_mode, hook_engine, prompt: str,
    output_format: str = "text", *, _resources: _PromptResources | None = None,
) -> None:
    from valecode.agent import (
        Agent,
        CompactNotification,
        ErrorEvent,
        LoopComplete,
        PermissionRequest,
        PermissionResponse,
        RetryEvent,
        StreamText,
        ThinkingText,
        ToolResultEvent,
        ToolUseEvent,
        TurnComplete,
        UsageEvent,
    )
    from valecode.runtime import RuntimeEvent
    from valecode.client import create_client, resolve_context_window
    from valecode.conversation import ConversationManager
    from valecode.memory.instructions import load_instructions
    from valecode.memory.session import SessionManager
    from valecode.permissions import (
        DangerousCommandDetector,
        PathSandbox,
        PermissionChecker,
        RuleEngine,
    )
    from valecode.tools import create_default_registry
    from valecode.agents.loader import AgentLoader
    from valecode.agents.durable_task_manager import DurableTaskManager
    from valecode.agents.trace import TraceManager
    from valecode.tools.agent_tool import AgentTool
    from valecode.tools.impl.tool_search import ToolSearchTool
    from valecode.teams.manager import TeamManager
    from valecode.teams.models import BackendType
    from valecode.tools.team_create import TeamCreateTool
    from valecode.tools.team_delete import TeamDeleteTool
    from valecode.tools.lead_tasks import build_lead_task_tools
    from valecode.worktree import WorktreeManager
    from valecode.config import WorktreeConfig

    is_json = output_format == "stream-json"
    owns_resources = _resources is None
    resources = _resources or _PromptResources()

    def emit_json(obj: dict) -> None:
        """输出一行 NDJSON 到 stdout"""
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    provider = config.providers[0]
    client = create_client(provider)
    # 第 2 层：尽力从 provider 自动拉取模型的 context window（缓存在 provider 上）。
    # 不会抛异常或阻塞启动；失败则退化到映射表。
    await resolve_context_window(provider)
    work_dir = os.getcwd()
    home = Path.home()

    checker = PermissionChecker(
        detector=DangerousCommandDetector(),
        sandbox=PathSandbox(work_dir),
        rule_engine=RuleEngine(
            user_rules_path=home / ".valecode" / "permissions.yaml",
            project_rules_path=Path(work_dir) / ".valecode" / "permissions.yaml",
            local_rules_path=Path(work_dir) / ".valecode" / "permissions.local.yaml",
        ),
        mode=permission_mode,
    )

    instructions = load_instructions(work_dir)
    session_manager = SessionManager(work_dir)
    session = session_manager.create()
    resources.session = session
    checker.bind_session(session.session_id)
    registry = create_default_registry(load_plugins=True)
    resources.registry = registry
    registry.bind_session(session.session_id)
    registry.register(ToolSearchTool(registry, protocol=provider.protocol))

    mcp_instructions = ""
    mcp_configs = getattr(config, "mcp_servers", [])
    if mcp_configs:
        from valecode.mcp import MCPManager

        manager = MCPManager()
        resources.mcp_manager = manager
        manager.load_configs(mcp_configs)
        connect_result = await manager.register_all_tools(registry)
        for error in connect_result.errors:
            logging.warning("MCP error: %s", error)
        sections: list[str] = []
        for server in connect_result.servers:
            body = server.instructions
            if not body:
                names = manager.tool_names_for_server(server.name)
                body = "Available tools: " + ", ".join(names) if names else ""
            sections.append(f"## {server.name}\n{body}".rstrip())
        if sections:
            mcp_instructions = (
                "# MCP Server Instructions\n\n"
                "The following MCP servers have provided instructions for how "
                "to use their tools and resources:\n\n" + "\n\n".join(sections)
            )
    if config.sandbox.enabled:
        from valecode.sandbox import attach_sandbox

        attached, reason = attach_sandbox(
            registry,
            checker,
            work_dir,
            network_enabled=config.sandbox.network_enabled,
            auto_allow=config.sandbox.auto_allow,
        )
        if not attached:
            logging.warning("OS sandbox requested but unavailable: %s", reason)

    agent = Agent(
        client=client,
        registry=registry,
        protocol=provider.protocol,
        work_dir=work_dir,
        permission_checker=checker,
        context_window=provider.get_context_window(),
        instructions_content=instructions,
        hook_engine=hook_engine,
        run_store=session_manager.run_store,
        provider_name=provider.name,
        model=provider.model,
    )
    agent.session_id = session.session_id
    from valecode.tools.todo_write import TodoWrite

    todo_tool = TodoWrite(lambda: (agent.work_dir, session.session_id))
    registry.register(todo_tool)
    agent.set_todo_state_provider(todo_tool.current_summary)

    wt_cfg = config.worktree or WorktreeConfig()
    wt_manager = WorktreeManager(
        repo_root=work_dir,
        symlink_directories=wt_cfg.symlink_directories,
    )
    trace_manager = TraceManager()
    task_manager = DurableTaskManager.from_config(
        session_manager.task_store,
        getattr(config, "background_tasks", None),
    )
    task_manager.start_maintenance()
    resources.task_manager = task_manager
    agent_loader = AgentLoader(work_dir, enable_verification=config.enable_verification_agent)
    agent_loader.load_all()
    team_manager = TeamManager(
        worktree_manager=wt_manager,
        trace_manager=trace_manager,
        task_store=session_manager.task_store,
    )

    agent_tool = AgentTool(
        agent_loader=agent_loader,
        task_manager=task_manager,
        trace_manager=trace_manager,
        parent_agent=agent,
        enable_fork=config.enable_fork,
        provider_config=provider,
        worktree_manager=wt_manager,
        team_manager=team_manager,
    )
    registry.register(agent_tool)
    registry.register(TeamCreateTool(
        team_manager=team_manager,
        parent_agent=agent,
        teammate_mode="in-process",
        is_interactive=False,
        enable_coordinator_mode=config.enable_coordinator_mode,
    ))
    registry.register(TeamDeleteTool(team_manager=team_manager, parent_agent=agent))
    for task_tool in build_lead_task_tools(team_manager, agent.agent_id):
        registry.register(task_tool)

    def drain_notifications() -> list[str]:
        notes: list[str] = []
        for t in task_manager.poll_completed():
            notes.append(
                f"<task-notification>\n<task_id>{t.id}</task_id>\n"
                f"<status>{t.status}</status>\n<result>{t.result}</result>\n"
                f"</task-notification>"
            )
        notes.extend(team_manager.drain_lead_mailbox())
        return notes

    def drain_mailbox_only() -> list[str]:
        return team_manager.drain_lead_mailbox()

    agent.notification_fn = drain_mailbox_only

    # 使用事件驱动的 agent.run()，支持 text 和 stream-json 两种输出格式
    conv = ConversationManager()
    conv.add_user_message(prompt)
    session.append(conv.history[-1])
    if mcp_instructions:
        conv.add_system_reminder(mcp_instructions)

    start = time.monotonic()
    text_buf = ""
    total_input = 0
    total_output = 0
    tool_calls: list[dict] = []

    async for event in agent.run(conv):
        if isinstance(event, StreamText):
            text_buf += event.text
            if is_json:
                emit_json({"type": "assistant", "text": event.text})

        elif isinstance(event, ThinkingText):
            if is_json:
                emit_json({"type": "thinking", "text": event.text})

        elif isinstance(event, ToolUseEvent):
            tool_calls.append({"name": event.tool_name, "is_error": False})
            if is_json:
                emit_json({
                    "type": "tool_use",
                    "tool_name": event.tool_name,
                    "tool_id": event.tool_id,
                    "args": event.arguments,
                })

        elif isinstance(event, ToolResultEvent):
            # 回填最后一个同名 tool_call 的 is_error
            if tool_calls:
                tool_calls[-1]["is_error"] = event.is_error
            if is_json:
                emit_json({
                    "type": "tool_result",
                    "tool_name": event.tool_name,
                    "tool_id": event.tool_id,
                    "output": event.output,
                    "is_error": event.is_error,
                    "elapsed": round(event.elapsed, 3),
                })

        elif isinstance(event, UsageEvent):
            total_input = event.input_tokens
            total_output = event.output_tokens
            if is_json:
                emit_json({
                    "type": "usage",
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                })

        elif isinstance(event, TurnComplete):
            if is_json:
                emit_json({"type": "turn_complete", "turn": event.turn})

        elif isinstance(event, LoopComplete):
            # 最终结果：stream-json 输出 result 行，text 模式直接打印文本
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if is_json:
                emit_json({
                    "type": "result",
                    "result": text_buf,
                    "duration_ms": elapsed_ms,
                    "num_turns": event.total_turns,
                    "tool_calls": tool_calls,
                    "usage": {
                        "input_tokens": total_input,
                        "output_tokens": total_output,
                    },
                    "stop_reason": "end_turn",
                })
            else:
                print(text_buf, end="", flush=True)
            break

        elif isinstance(event, ErrorEvent):
            if is_json:
                emit_json({"type": "error", "message": event.message})
            else:
                print(f"Error: {event.message}", file=sys.stderr, flush=True)

        elif isinstance(event, CompactNotification):
            if is_json:
                emit_json({"type": "compact", "message": event.message})

        elif isinstance(event, RetryEvent):
            if is_json:
                emit_json({"type": "retry", "reason": event.reason})

        elif isinstance(event, PermissionRequest):
            # -p 非交互模式：自动批准所有权限请求
            event.future.set_result(PermissionResponse.ALLOW)

        elif isinstance(event, RuntimeEvent) and is_json:
            emit_json({
                "type": "runtime_event",
                "event": event.to_envelope().to_dict(),
            })

    # 如果有 team 在运行，轮询等待 teammate 完成
    if not team_manager._teams:
        if owns_resources:
            await resources.close()
        return

    for i in range(90):
        await asyncio.sleep(2)
        running = {k: not t.done() for k, t in task_manager._async_tasks.items()}
        completed_ids = [t.id for t in task_manager._tasks.values() if t.status != "running"]
        print(f"[poll {i}] running={running} completed={completed_ids} teams={list(team_manager._teams.keys())} queue_size={task_manager._notify_queue.qsize()}", file=sys.stderr, flush=True)
        notes = drain_notifications()
        if not notes:
            has_running = any(v for v in running.values())
            if not has_running:
                print(f"[poll {i}] no running tasks, breaking", file=sys.stderr, flush=True)
                break
            continue
        for note in notes:
            conv.add_system_reminder(note)
        # 后续 team 轮询仍用 run_to_completion，避免重复事件循环
        last_result = await agent.run_to_completion(
            "Teammate notifications received. Process them and continue.", conv
        )
        if is_json:
            emit_json({"type": "assistant", "text": last_result})
        else:
            print(last_result, flush=True)

    if owns_resources:
        await resources.close()


if __name__ == "__main__":
    main()
