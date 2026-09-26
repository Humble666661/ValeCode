
"""
Remote Control 服务器：通过 WebSocket 桥接 Agent 事件和 Web UI。

使用 websockets 库提供 HTTP（静态 HTML）+ WebSocket 服务，
让用户在浏览器中与 ValeCode Agent 交互。
"""

from __future__ import annotations

import asyncio
import hmac
import hashlib
import secrets
import ipaddress
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import websockets
from websockets.asyncio.server import Server as WSServer, ServerConnection
from websockets.http11 import Request, Response

from valecode.agent import (
    Agent,
    CompactNotification,
    ErrorEvent,
    HookEvent,
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
from valecode.agents.durable_task_manager import DurableTaskManager
from valecode.agents.loader import AgentLoader
from valecode.agents.notification import format_task_notification
from valecode.agents.trace import TraceManager
from valecode.client import create_client, resolve_context_window
from valecode.commands import CommandContext, CommandRegistry, CommandType
from valecode.commands.handlers import register_all_commands
from valecode.commands.handlers.skill_register import register_skill_commands
from valecode.commands.handlers.tasks import create_tasks_command
from valecode.commands.handlers.trace import create_trace_command
from valecode.commands.parser import parse_command
from valecode.config import MCPServerConfig, ProviderConfig, SandboxAppConfig
from valecode.conversation import ConversationManager, Message
from valecode.hooks import HookEngine
from valecode.mcp import MCPManager
from valecode.memory import MemoryManager, load_instructions
from valecode.memory.session import Session, SessionManager
from valecode.permissions import (
    DangerousCommandDetector,
    PathSandbox,
    PermissionChecker,
    PermissionMode,
    RuleEngine,
)
from valecode.skills.loader import SkillLoader
from valecode.skills.executor import SkillExecutor
from valecode.tools import ToolRegistry, ToolSource, create_default_registry
from valecode.tools.impl.tool_search import ToolSearchTool
from valecode.tools.agent_tool import AgentTool
from valecode.tools.install_skill import InstallSkillTool
from valecode.tools.load_skill import LoadSkill
from valecode.web_content import INDEX_HTML

log = logging.getLogger(__name__)


def _is_loopback_bind(host: str) -> bool:
    """Return whether a bind target is restricted to the local machine."""
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


class RemoteServer:
    """Remote Control 核心：桥接 Agent 事件和 WebSocket 客户端。"""

    def __init__(
        self,
        providers: list[ProviderConfig],
        mcp_servers: list[MCPServerConfig] | None = None,
        hook_engine: HookEngine | None = None,
        addr: str = "127.0.0.1",
        port: int = 18888,
        sandbox_config: SandboxAppConfig | None = None,
        auth_token: str = "",
        enable_fork: bool = False,
        enable_verification_agent: bool = False,
        background_task_config: Any = None,
        worktree_config: Any = None,
        enable_coordinator_mode: bool = False,
    ) -> None:
        if not _is_loopback_bind(addr) and not auth_token:
            raise ValueError(
                "Remote access token is required when binding outside localhost"
            )
        self.providers = providers
        self._mcp_server_configs = mcp_servers or []
        self.hook_engine = hook_engine
        self.addr = addr
        self.port = port
        self.auth_token = auth_token
        self._sandbox_config = sandbox_config or SandboxAppConfig()
        self._enable_fork = enable_fork
        self._enable_verification_agent = enable_verification_agent
        self._background_task_config = background_task_config
        self._worktree_config = worktree_config
        self._enable_coordinator_mode = enable_coordinator_mode

        # WebSocket 连接池（支持多客户端广播）
        self._connections: set[ServerConnection] = set()

        # Agent 相关状态
        self.agent: Agent | None = None
        self.conversation: ConversationManager | None = None
        self.registry: ToolRegistry | None = None
        self.session_id: str = ""
        self._streaming = False
        self._cancel_event: asyncio.Event | None = None
        self._notification_task: asyncio.Task[None] | None = None

        # 权限请求的 pending 队列：id -> Future
        self._pending_perms: dict[str, asyncio.Future[PermissionResponse]] = {}
        self._pending_asks: dict[str, Any] = {}
        self._pending_plan = None
        self._pre_plan_mode = PermissionMode.DEFAULT
        self._has_exited_plan_mode = False
        self._request_tasks: set[asyncio.Task] = set()

        # 命令注册表
        self.command_registry = CommandRegistry()
        register_all_commands(self.command_registry)

        # MCP 相关
        self.mcp_manager: MCPManager | None = None
        self._mcp_instructions: str = ""

        # Skill 加载器
        self.skill_loader: SkillLoader | None = None
        self.skill_executor: SkillExecutor | None = None
        self._install_skill_tool: InstallSkillTool | None = None

        # 子 Agent / 后台任务
        self.agent_loader: AgentLoader | None = None
        self.agent_tool: AgentTool | None = None
        self.cron_runtime = None
        self.task_manager: DurableTaskManager | None = None
        self.trace_manager = TraceManager()
        self.harness = None
        self.team_manager = None
        self.worktree_manager = None

        # Memory / Session
        self.memory_manager: MemoryManager | None = None
        self.session_manager: SessionManager | None = None
        self.session: Session | None = None

    # ------------------------------------------------------------------
    # 启动入口
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """启动 HTTP + WebSocket 服务器。"""
        try:
            self._init_agent()
            await self._init_mcp()
            if self.cron_runtime is not None:
                self.cron_runtime.start()
            self._notification_task = asyncio.create_task(
                self._start_notification_polling()
            )

            display_host = "localhost" if _is_loopback_bind(self.addr) else self.addr
            try:
                if ipaddress.ip_address(display_host).version == 6:
                    display_host = f"[{display_host}]"
            except ValueError:
                pass
            auth_note = " (Token required)" if self.auth_token else ""
            print(f"\n  Remote UI: http://{display_host}:{self.port}{auth_note}\n")

            # websockets 的 serve 支持 process_request 回调来处理普通 HTTP
            async with websockets.serve(
                self._ws_handler,
                self.addr,
                self.port,
                process_request=self._process_http_request,
                max_size=4 * 1024 * 1024,  # 4MB 消息上限
            ):
                # 服务器启动后永久阻塞
                await asyncio.Future()
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Release remote runtime resources even on startup failure/cancellation."""
        from valecode.runtime.harness import close_resources
        self._settle_interactions()
        self._pending_plan = None
        if self.agent is not None:
            self.agent.cancel("Remote server shutting down")
        for task in list(self._request_tasks):
            task.cancel()
        await asyncio.gather(*self._request_tasks, return_exceptions=True)
        self._request_tasks.clear()
        await close_resources([
            ("cron", self.cron_runtime.close if self.cron_runtime is not None else None),
            ("teams", self.team_manager.close if self.team_manager is not None else None),
        ])
        if self._notification_task is not None:
            self._notification_task.cancel()
            await asyncio.gather(self._notification_task, return_exceptions=True)
            self._notification_task = None
        await close_resources([("tasks", self.task_manager.shutdown if self.task_manager is not None else None)])
        if self.mcp_manager is not None:
            try:
                await self.mcp_manager.shutdown()
            except Exception:
                log.exception("Failed to close MCP manager")
            self.mcp_manager = None
        if self.registry is not None:
            await close_resources([
                ("plugins", lambda: self.registry.release_source(ToolSource.PLUGIN)),
                ("tools", self.registry.release_session),
            ])
        if self.hook_engine is not None:
            try:
                await self.hook_engine.shutdown()
            except Exception:
                log.exception("Failed to close remote hook engine")
        if self.session is not None:
            self.session.close()

    # ------------------------------------------------------------------
    # HTTP 请求处理（为 / 路径提供前端 HTML）
    # ------------------------------------------------------------------

    def _process_http_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        """拦截 HTTP 请求，对 / 路径返回 HTML 页面。
        返回 None 表示继续走 WebSocket 升级流程。
        """
        parsed = urlsplit(request.path)
        if parsed.path == "/":
            html = INDEX_HTML.replace(
                "const remoteAuthRequired = false;",
                f"const remoteAuthRequired = {str(bool(self.auth_token)).lower()};",
                1,
            )
            return Response(
                200,
                "OK",
                websockets.Headers({"Content-Type": "text/html; charset=utf-8"}),
                html.encode("utf-8"),
            )
        if parsed.path != "/ws":
            return Response(404, "Not Found", websockets.Headers(), b"404 Not Found")
        if not self._request_is_authorized(request, parsed.query):
            return Response(
                401,
                "Unauthorized",
                websockets.Headers(
                    {
                        "Content-Type": "text/plain; charset=utf-8",
                        "WWW-Authenticate": "Bearer realm=\"ValeCode Remote\"",
                        "Cache-Control": "no-store",
                    }
                ),
                b"Unauthorized",
            )
        # /ws 路径 → 继续 WebSocket 升级
        return None

    def _request_is_authorized(self, request: Request, query: str) -> bool:
        if not self.auth_token:
            return True
        candidates: list[str] = []
        authorization = request.headers.get("Authorization", "")
        scheme, separator, credentials = authorization.partition(" ")
        if separator and scheme.lower() == "bearer":
            candidates.append(credentials.strip())
        candidates.extend(parse_qs(query, keep_blank_values=True).get("token", []))
        return any(
            hmac.compare_digest(candidate, self.auth_token) for candidate in candidates
        )

    # ------------------------------------------------------------------
    # WebSocket 连接处理
    # ------------------------------------------------------------------

    async def _ws_handler(self, websocket: ServerConnection) -> None:
        """处理单个 WebSocket 连接的全生命周期。"""
        self._connections.add(websocket)
        try:
            # 连接建立时推送会话信息
            await self._broadcast({
                "type": "connected",
                "data": {
                    "session": self.session_id,
                    "cwd": os.getcwd(),
                },
            })

            # 推送命令列表
            await self._broadcast({
                "type": "commands",
                "data": self._build_command_list(),
            })

            # 消息循环
            for identity, event in list(self._pending_asks.items()):
                await websocket.send(json.dumps({"type": "ask_user", "data": {"id": identity, "questions": event.questions}}, ensure_ascii=False))
            if self._pending_plan is not None:
                await websocket.send(json.dumps({"type": "plan_approval", "data": {
                    "id": self._pending_plan["id"], "content": self._pending_plan["content"]}}, ensure_ascii=False))
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if not isinstance(msg, dict) or not isinstance(msg.get("data", {}), dict):
                    continue
                msg_type = msg.get("type", "")
                data = msg.get("data", {})

                if msg_type == "user_message":
                    content = data.get("content", "")
                    if not isinstance(content, str):
                        continue
                    content = content.strip()
                    if content:
                        # 在后台任务中处理，不阻塞 WebSocket 读循环
                        self._spawn_request(self._handle_user_message(content))

                elif msg_type == "permission_response":
                    self._handle_permission_response(data)

                elif msg_type == "ask_user_response":
                    self._handle_ask_response(data)

                elif msg_type == "plan_response":
                    self._spawn_request(self._handle_plan_response(data))

                elif msg_type == "cancel":
                    if self._cancel_event is not None:
                        self._cancel_event.set()
                        self.agent.cancel("Cancelled by remote client")

                elif msg_type == "ping":
                    # 应用层保活
                    await self._broadcast({"type": "pong", "data": None})

        except websockets.ConnectionClosed:
            pass
        finally:
            self._connections.discard(websocket)

    # ------------------------------------------------------------------
    # Agent 初始化（复刻 TUI 的 _select_provider 流程）
    # ------------------------------------------------------------------

    def _init_agent(self) -> None:
        """初始化 Agent 及相关子系统。"""
        provider = self.providers[0]
        work_dir = os.getcwd()
        home = Path.home()

        # 权限系统
        checker = PermissionChecker(
            detector=DangerousCommandDetector(),
            sandbox=PathSandbox(work_dir),
            rule_engine=RuleEngine(
                user_rules_path=home / ".valecode" / "permissions.yaml",
                project_rules_path=Path(work_dir) / ".valecode" / "permissions.yaml",
                local_rules_path=Path(work_dir) / ".valecode" / "permissions.local.yaml",
            ),
            mode=PermissionMode.DEFAULT,
        )

        # 加载自定义指令和记忆
        instructions = load_instructions(work_dir)
        self.memory_manager = MemoryManager(work_dir)
        self.session_manager = SessionManager(work_dir)
        self.session = self.session_manager.create()
        self.session_id = self.session.session_id
        checker.bind_session(self.session_id)

        # 创建 LLM 客户端
        client = create_client(provider)

        # 工具注册表
        self.registry = create_default_registry(load_plugins=True)
        self.registry.bind_session(self.session_id)
        self.registry.register(ToolSearchTool(self.registry, protocol=provider.protocol))
        if self._sandbox_config.enabled:
            from valecode.sandbox import attach_sandbox

            attached, reason = attach_sandbox(
                self.registry,
                checker,
                work_dir,
                network_enabled=self._sandbox_config.network_enabled,
                auto_allow=self._sandbox_config.auto_allow,
            )
            if not attached:
                log.warning("OS sandbox requested but unavailable: %s", reason)

        # Skill 加载
        self.skill_loader = SkillLoader(work_dir)
        self.skill_loader.load_all()
        load_skill_tool = LoadSkill()
        self.registry.register(load_skill_tool)
        install_skill_tool = InstallSkillTool()
        self.registry.register(install_skill_tool)
        self._install_skill_tool = install_skill_tool

        # 创建 Agent
        self.agent = Agent(
            client=client,
            registry=self.registry,
            protocol=provider.protocol,
            work_dir=work_dir,
            permission_checker=checker,
            context_window=provider.get_context_window(),
            instructions_content=instructions,
            memory_manager=self.memory_manager,
            hook_engine=self.hook_engine,
            run_store=self.session_manager.run_store,
            provider_name=provider.name,
            model=provider.model,
        )
        self.agent.session_id = self.session_id
        from valecode.tools.ask_user import AskUserTool
        from valecode.tools.exit_plan_mode import ExitPlanModeTool
        self.registry.register(AskUserTool(self._request_questions))
        self.registry.register(ExitPlanModeTool(lambda: self.agent.plan_mode,
            lambda: self.agent._get_plan_path().exists()))
        from valecode.tools.todo_write import TodoWrite

        todo_tool = TodoWrite(lambda: (self.agent.work_dir, self.session_id))
        self.registry.register(todo_tool)
        self.agent.set_todo_state_provider(todo_tool.current_summary)

        from valecode.runtime.harness import HarnessOptions, assemble_harness
        self.harness = assemble_harness(self.agent, self.session_manager, provider, self.registry,
            options=HarnessOptions(enable_fork=self._enable_fork,
                enable_verification=self._enable_verification_agent,
                enable_coordinator=self._enable_coordinator_mode,
                background_tasks=self._background_task_config, worktree_config=self._worktree_config),
            trace_manager=self.trace_manager)
        self.task_manager = self.harness.task_manager
        self.agent_loader = self.harness.agent_loader
        self.agent_tool = self.harness.agent_tool
        self.team_manager = self.harness.team_manager
        self.worktree_manager = self.harness.worktree_manager
        self.cron_runtime = self.harness.cron_runtime
        from valecode.commands.handlers.worktree import create_worktree_command
        self.command_registry.register_sync(create_worktree_command(self.worktree_manager))
        from valecode.commands.handlers.cron import create_cron_command
        self.command_registry.register_sync(create_cron_command(self.cron_runtime))

        self.command_registry.register_sync(create_tasks_command(self.task_manager))
        self.command_registry.register_sync(
            create_trace_command(self.trace_manager, self.agent.agent_id)
        )

        # 连接 Skill 到 Agent
        load_skill_tool.set_loader(self.skill_loader)
        load_skill_tool.set_agent(self.agent)
        install_skill_tool.set_loader(self.skill_loader)
        self.skill_executor = SkillExecutor(
            agent=self.agent,
            client=client,
            protocol=provider.protocol,
            provider_config=provider,
        )
        register_skill_commands(
            self.command_registry,
            self.skill_loader,
            self.skill_executor,
        )

        def _on_skill_installed(_name: str) -> None:
            assert self.skill_loader is not None
            register_skill_commands(
                self.command_registry,
                self.skill_loader,
                self.skill_executor,
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(self._broadcast({
                "type": "commands",
                "data": self._build_command_list(),
            }))

        install_skill_tool.set_on_installed(_on_skill_installed)

        catalog = self.skill_loader.get_catalog()
        if catalog:
            lines = ["You can use the following Skills:", ""]
            for name, desc in catalog:
                lines.append(f"- {name}: {desc}")
            lines.append("")
            lines.append("If the user's request matches a Skill, call LoadSkill to activate it.")
            self.agent.set_skill_catalog("\n".join(lines))

        # 初始化对话管理器
        self.conversation = ConversationManager()

        log.info("Agent initialized: session=%s, model=%s", self.session_id, provider.model)

    async def _process_task_notifications(self) -> None:
        """Deliver completed background work and let the lead Agent summarize it."""
        if (
            self._streaming
            or self._pending_plan is not None
            or not self._connections
            or self.task_manager is None
            or self.agent is None
        ):
            return

        completed = self.task_manager.poll_completed()
        completed = [
            task
            for task in completed
            if task.agent.session_id == self.session_id
        ]
        for task in completed:
            if self.team_manager is not None:
                self.team_manager.on_teammate_completed(task.agent.agent_id)
        team_notes = self.team_manager.drain_lead_mailbox() if self.team_manager is not None else []
        if not completed and not team_notes:
            return

        for task in completed:
            status_icon = "✓" if task.status == "completed" else "✗"
            await self._broadcast({
                "type": "system",
                "data": {
                    "message": (
                        f"{status_icon} 后台任务完成: "
                        f"[{task.id}] {task.name} — {task.status}"
                    )
                },
            })

        notification_prompt = "\n\n".join([
            *(format_task_notification(task) for task in completed), *team_notes,
        ])
        await self._handle_user_message(
            notification_prompt,
            dispatch_commands=False,
        )

    async def _start_notification_polling(self) -> None:
        """Poll durable task completions without blocking WebSocket handling."""
        try:
            while True:
                await asyncio.sleep(1)
                await self._process_task_notifications()
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # MCP 初始化
    # ------------------------------------------------------------------

    async def _init_mcp(self) -> None:
        """连接所有配置的 MCP 服务器，注册工具。"""
        if not self._mcp_server_configs or self.registry is None:
            return

        manager = MCPManager()
        manager.load_configs(self._mcp_server_configs)
        connect_result = await manager.register_all_tools(self.registry)
        self.mcp_manager = manager

        for err in connect_result.errors:
            log.warning("MCP error: %s", err)

        # 构建 MCP 指令（首次发送消息时注入 conversation）
        if connect_result.servers:
            parts = []
            for srv_info in connect_result.servers:
                section = f"## {srv_info.name}\n"
                if srv_info.instructions:
                    section += srv_info.instructions
                else:
                    tool_names = manager.tool_names_for_server(srv_info.name)
                    if tool_names:
                        section += "Available tools: " + ", ".join(tool_names)
                parts.append(section)
            self._mcp_instructions = (
                "# MCP Server Instructions\n\n"
                "The following MCP servers have provided instructions "
                "for how to use their tools and resources:\n\n"
                + "\n\n".join(parts)
            )

    # ------------------------------------------------------------------
    # 用户消息处理
    # ------------------------------------------------------------------

    async def _handle_user_message(
        self, content: str, *, dispatch_commands: bool = True,
    ) -> None:
        """处理来自 Web UI 的用户消息或斜杠命令。"""
        if self._streaming:
            return
        if self._pending_plan is not None:
            await self._broadcast({"type": "system", "data": {"message": "请先批准、修改或拒绝当前计划。"}})
            return

        # 斜杠命令
        if dispatch_commands and content.startswith("/"):
            await self._handle_slash_command(content)
            return

        # 普通消息 → 发给 Agent
        self._streaming = True
        assert self.conversation is not None
        assert self.agent is not None

        self.conversation.add_user_message(content)

        # 首次注入 MCP 指令
        if self._mcp_instructions:
            self.conversation.add_system_reminder(self._mcp_instructions)
            self._mcp_instructions = ""

        # 创建取消事件
        self._cancel_event = asyncio.Event()
        start_time = time.monotonic()
        stream_buf = ""

        try:
            async for event in self.agent.run(self.conversation):
                # 检查取消信号
                if self._cancel_event.is_set():
                    break

                if isinstance(event, StreamText):
                    stream_buf += event.text
                    await self._broadcast({
                        "type": "stream_text",
                        "data": {"text": event.text},
                    })

                elif isinstance(event, ThinkingText):
                    await self._broadcast({
                        "type": "thinking_text",
                        "data": {"text": event.text},
                    })

                elif isinstance(event, ToolUseEvent):
                    await self._broadcast({
                        "type": "tool_use",
                        "data": {
                            "toolId": event.tool_id,
                            "toolName": event.tool_name,
                            "args": event.arguments,
                        },
                    })

                elif isinstance(event, ToolResultEvent):
                    if event.tool_name in {"EnterWorktree", "ExitWorktree"} and not event.is_error:
                        self._sync_worktree_context()
                    if event.tool_name == "ExitPlanMode" and not event.is_error and self.agent.plan_mode:
                        await self._request_plan_approval()
                    # 如果之前有累积的流式文本，先结束它
                    if stream_buf:
                        await self._broadcast({
                            "type": "stream_end",
                            "data": {"text": stream_buf},
                        })
                        stream_buf = ""
                    await self._broadcast({
                        "type": "tool_result",
                        "data": {
                            "toolId": event.tool_id,
                            "toolName": event.tool_name,
                            "output": event.output,
                            "isError": event.is_error,
                            "elapsed": event.elapsed,
                        },
                    })

                elif isinstance(event, PermissionRequest):
                    # 生成唯一 ID，等待 Web 端回复
                    perm_id = f"perm_{time.time_ns()}"
                    self._pending_perms[perm_id] = event.future
                    await self._broadcast({
                        "type": "permission_request",
                        "data": {
                            "id": perm_id,
                            "toolName": event.tool_name,
                            "description": event.description,
                        },
                    })

                elif isinstance(event, TurnComplete):
                    if stream_buf:
                        await self._broadcast({
                            "type": "stream_end",
                            "data": {"text": stream_buf},
                        })
                        stream_buf = ""
                    await self._broadcast({
                        "type": "turn_complete",
                        "data": {"turn": event.turn},
                    })

                elif isinstance(event, LoopComplete):
                    if stream_buf:
                        await self._broadcast({
                            "type": "stream_end",
                            "data": {"text": stream_buf},
                        })
                        stream_buf = ""
                    elapsed = time.monotonic() - start_time
                    await self._broadcast({
                        "type": "loop_complete",
                        "data": {
                            "totalTurns": event.total_turns,
                            "elapsed": elapsed,
                        },
                    })

                elif isinstance(event, UsageEvent):
                    await self._broadcast({
                        "type": "usage",
                        "data": {
                            "inputTokens": event.input_tokens,
                            "outputTokens": event.output_tokens,
                        },
                    })

                elif isinstance(event, ErrorEvent):
                    await self._broadcast({
                        "type": "error",
                        "data": {"message": event.message},
                    })

                elif isinstance(event, CompactNotification):
                    await self._broadcast({
                        "type": "compact",
                        "data": {"message": event.message},
                    })

                elif isinstance(event, RetryEvent):
                    await self._broadcast({
                        "type": "retry",
                        "data": {
                            "reason": event.reason,
                            "waitMs": int(event.wait * 1000),
                        },
                    })

                elif isinstance(event, HookEvent):
                    status = "ok" if event.success else "error"
                    await self._broadcast({
                        "type": "system",
                        "data": {
                            "message": f"Hook [{event.hook_id}] {status}: {event.output}"
                        },
                    })

                elif event.EVENT_TYPE in {"permission.responded", "mailbox.received"}:
                    await self._broadcast({
                        "type": "runtime_event",
                        "data": event.to_envelope().to_dict(),
                    })

        except asyncio.CancelledError:
            await self._broadcast({
                "type": "error",
                "data": {"message": "Operation cancelled"},
            })
        except Exception as exc:
            log.exception("Agent run error")
            await self._broadcast({
                "type": "error",
                "data": {"message": str(exc)},
            })
        finally:
            self._streaming = False
            self._settle_interactions()
            self._cancel_event = None

    # ------------------------------------------------------------------
    # 斜杠命令处理
    # ------------------------------------------------------------------

    async def _handle_slash_command(self, input_text: str) -> None:
        """分发斜杠命令。"""
        name, args, is_command = parse_command(input_text)
        if not is_command or not name:
            return

        cmd = self.command_registry.find(name)
        if cmd is None:
            await self._broadcast({
                "type": "error",
                "data": {"message": f"Unknown command: /{name} — type /help to see available commands"},
            })
            await self._broadcast({"type": "command_done", "data": None})
            return

        # 需要参数但没给
        if not args and cmd.arg_prompt:
            await self._broadcast({
                "type": "system",
                "data": {"message": cmd.arg_prompt},
            })
            await self._broadcast({"type": "command_done", "data": None})
            return

        if cmd.type == CommandType.LOCAL:
            # 本地命令直接执行
            ctx = self._build_command_context(args)
            try:
                await cmd.handler(ctx)
            except Exception as exc:
                await self._broadcast({
                    "type": "error",
                    "data": {"message": f"Command error: {exc}"},
                })
            await self._broadcast({"type": "command_done", "data": None})

        elif cmd.type == CommandType.LOCAL_UI:
            # UI 命令需要特殊处理
            if name == "clear":
                self.conversation = ConversationManager()
                if self.agent is not None:
                    self.agent.clear_active_skills()
                await self._broadcast({"type": "clear", "data": None})

            elif name == "compact":
                await self._handle_compact()
                return

            elif name in {"plan", "p"}:
                await cmd.handler(self._build_command_context(args))

            else:
                await self._broadcast({
                    "type": "system",
                    "data": {"message": f"/{name} is not fully supported in remote mode."},
                })

            await self._broadcast({"type": "command_done", "data": None})

        elif cmd.type == CommandType.PROMPT:
            # Prompt 类命令：handler 返回 prompt 文本，注入给 agent
            ctx = self._build_command_context(args)
            try:
                await cmd.handler(ctx)
            except Exception as exc:
                await self._broadcast({
                    "type": "error",
                    "data": {"message": f"Command error: {exc}"},
                })
                await self._broadcast({"type": "command_done", "data": None})

    def _build_command_context(self, args: str) -> CommandContext:
        """构建命令上下文。"""
        return CommandContext(
            args=args,
            agent=self.agent,
            conversation=self.conversation,
            session=self.session,
            session_manager=self.session_manager,
            memory_manager=self.memory_manager,
            ui=self,  # type: ignore[arg-type]
            config={
                "registry": self.command_registry,
                "set_session": self._set_session,
                "set_conversation": self._set_conversation,
                "clear_chat": self._clear_chat,
                "render_restored": self._render_restored_messages,
                "recover_tasks": self._recover_session_tasks,
                "skill_loader": self.skill_loader,
                "skill_executor": self.skill_executor,
            },
        )

    def _set_session(self, session: Session) -> None:
        self._settle_interactions()
        self._pending_plan = None
        self.session = session
        self.session_id = session.session_id
        if self.registry is not None:
            self.registry.bind_session(session.session_id)
        if self.agent is not None:
            self.agent.session_id = session.session_id
            if self.agent.permission_checker is not None:
                self.agent.permission_checker.bind_session(session.session_id)

    def _set_conversation(self, conversation: ConversationManager) -> None:
        self.conversation = conversation

    def _clear_chat(self) -> None:
        asyncio.create_task(self._broadcast({"type": "clear", "data": None}))

    async def _render_restored_messages(self, messages: list[Message]) -> None:
        await self._broadcast({"type": "clear", "data": None})
        for message in messages:
            if message.tool_results or not message.content:
                continue
            if message.role == "user":
                event_type = "replay_user"
            elif message.role == "assistant":
                event_type = "replay_assistant"
            else:
                continue
            await self._broadcast({
                "type": event_type,
                "data": {"content": message.content},
            })

    def _recover_session_tasks(self, session_id: str) -> list[str]:
        if self.agent_tool is None:
            return []
        return self.agent_tool.recover_persisted_tasks(session_id)

    async def _handle_compact(self) -> None:
        """处理 /compact 命令。"""
        if self.agent is None or self.conversation is None:
            await self._broadcast({
                "type": "error",
                "data": {"message": "Compact requires an active agent."},
            })
            await self._broadcast({"type": "command_done", "data": None})
            return

        await self._broadcast({
            "type": "system",
            "data": {"message": "Compacting conversation..."},
        })

        result = await self.agent.manual_compact(self.conversation)
        if isinstance(result, CompactNotification):
            await self._broadcast({
                "type": "system",
                "data": {"message": result.message},
            })
        elif isinstance(result, ErrorEvent):
            await self._broadcast({
                "type": "error",
                "data": {"message": result.message},
            })

        await self._broadcast({"type": "command_done", "data": None})

    # ------------------------------------------------------------------
    # UIController 协议实现（供命令系统回调）
    # ------------------------------------------------------------------

    def add_system_message(self, text: str) -> None:
        """同步接口 — 在事件循环中调度广播。"""
        asyncio.ensure_future(self._broadcast({
            "type": "system",
            "data": {"message": text},
        }))

    def send_user_message(self, text: str) -> None:
        """同步接口 — 注入用户消息并触发 agent。"""
        self._spawn_request(
            self._handle_user_message(text, dispatch_commands=False)
        )

    def set_plan_mode(self, enabled: bool) -> None:
        if self.agent is None:
            return
        if enabled:
            if not self.agent.plan_mode:
                self._pre_plan_mode = self.agent.permission_mode
            self.agent.set_permission_mode(PermissionMode.PLAN)
        else:
            self._pending_plan = None
            self.agent.set_permission_mode(self._pre_plan_mode)

    def get_token_count(self) -> tuple[int, int]:
        if self.agent:
            return self.agent.total_input_tokens, self.agent.total_output_tokens
        return 0, 0

    def refresh_status(self) -> None:
        pass  # Remote 模式不需要刷新 TUI 状态栏

    # ------------------------------------------------------------------
    # 权限响应处理
    # ------------------------------------------------------------------

    def _spawn_request(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._request_tasks.add(task)
        def done(completed):
            self._request_tasks.discard(completed)
            if not completed.cancelled() and completed.exception() is not None:
                log.error("Remote request failed", exc_info=completed.exception())
        task.add_done_callback(done)

    def _settle_interactions(self):
        for future in self._pending_perms.values():
            if not future.done():
                future.set_result(PermissionResponse.DENY)
        self._pending_perms.clear()
        for event in self._pending_asks.values():
            if not event.future.done():
                event.future.set_result({})
        self._pending_asks.clear()

    async def _request_questions(self, event):
        request_id = "ask_" + secrets.token_hex(12)
        self._pending_asks[request_id] = event
        event.future.add_done_callback(lambda _future: self._pending_asks.pop(request_id, None))
        await self._broadcast({"type": "ask_user", "data": {"id": request_id, "questions": event.questions}})

    def _handle_ask_response(self, data):
        identity = data.get("id")
        if not isinstance(identity, str):
            return
        event = self._pending_asks.get(identity)
        answers = data.get("answers")
        if event is None or event.future.done() or not isinstance(answers, dict):
            return
        names = {q["name"] for q in event.questions}
        if any(key not in names or not isinstance(value, str) or len(value) > 20000 for key, value in answers.items()):
            return
        self._pending_asks.pop(identity, None)
        event.future.set_result(answers)

    def _read_plan(self):
        path = self.agent._get_plan_path()
        if path.is_symlink() or not path.resolve().is_relative_to((Path(self.agent.work_dir) / ".valecode" / "plans").resolve()) or path.stat().st_size > 200000:
            raise ValueError("Plan is outside the managed directory or exceeds 200 KB")
        return path.read_text(encoding="utf-8")

    async def _request_plan_approval(self):
        content = self._read_plan()
        identity = "plan_" + secrets.token_hex(12)
        self._pending_plan = {"id": identity, "session_id": self.session_id, "content": content,
            "digest": hashlib.sha256(content.encode()).hexdigest()}
        await self._broadcast({"type": "plan_approval", "data": {"id": identity, "content": content}})

    async def _handle_plan_response(self, data):
        pending = self._pending_plan
        if self._streaming or pending is None or data.get("id") != pending["id"] or pending["session_id"] != self.session_id:
            return
        choice = data.get("choice")
        if choice not in {"approve", "feedback", "reject"}:
            return
        feedback = data.get("feedback", "")
        if not isinstance(feedback, str) or len(feedback) > 20000:
            return
        self._pending_plan = None  # Consume once before scheduling another run.
        if choice == "approve":
            try:
                content = self._read_plan()
                if hashlib.sha256(content.encode()).hexdigest() != pending["digest"]:
                    raise ValueError("计划已修改，请重新提交审批")
            except (OSError, ValueError) as exc:
                await self._broadcast({"type": "error", "data": {"message": str(exc)}})
                return
            self.agent.set_permission_mode(self._pre_plan_mode)
            self._has_exited_plan_mode = True
            from valecode.prompts import build_plan_mode_exit_reminder
            reminder = build_plan_mode_exit_reminder(str(self.agent._get_plan_path()), True)
            await self._handle_user_message(reminder + "\n\nUser approved the plan. Execute it under the existing permissions.\n\n" + content, dispatch_commands=False)
        elif choice == "feedback" and feedback.strip():
            await self._handle_user_message(feedback, dispatch_commands=False)
        else:
            await self._broadcast({"type": "system", "data": {"message": "计划未执行，仍保持 Plan 模式。"}})

    def _sync_worktree_context(self):
        from valecode.runtime.harness import sync_worktree_context
        sync_worktree_context(self.agent, self.worktree_manager)

    def _handle_permission_response(self, data: dict[str, Any]) -> None:
        """处理来自 Web UI 的权限回复。"""
        perm_id = data.get("id", "")
        response_str = data.get("response", "deny")

        future = self._pending_perms.pop(perm_id, None)
        if future is None or future.done():
            return

        # 映射字符串到枚举
        mapping = {
            "allow": PermissionResponse.ALLOW,
            "deny": PermissionResponse.DENY,
            "allowSession": PermissionResponse.ALLOW_SESSION,
            "allowAlways": PermissionResponse.ALLOW_SESSION,  # older clients
        }
        response = mapping.get(response_str, PermissionResponse.DENY)
        future.set_result(response)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _build_command_list(self) -> list[dict[str, str]]:
        """构建命令列表，推送给前端用于斜杠命令菜单。"""
        result = []
        for cmd in self.command_registry.list_commands():
            result.append({
                "name": cmd.name,
                "description": cmd.description,
            })
        return result

    async def _broadcast(self, msg: dict[str, Any]) -> None:
        """向所有已连接的 WebSocket 客户端广播消息。"""
        if not self._connections:
            return
        data = json.dumps(msg, ensure_ascii=False)
        # 复制集合避免迭代中修改
        closed = []
        for ws in list(self._connections):
            try:
                await ws.send(data)
            except websockets.ConnectionClosed:
                closed.append(ws)
            except Exception:
                closed.append(ws)
        for ws in closed:
            self._connections.discard(ws)
