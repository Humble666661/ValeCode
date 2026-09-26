from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from valecode.config import MCPServerConfig
from valecode.mcp.client import MCPClient
from valecode.mcp.catalog_tools import build_catalog_tools
from valecode.mcp.tool_wrapper import MCPToolWrapper
from valecode.tools import ToolRegistry, ToolSource
from valecode.tools.base import Tool

logger = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """单个 MCP 服务器的连接信息，包含名称和 instructions。"""
    name: str
    instructions: str = ""


@dataclass
class ConnectResult:
    """ConnectAll 的返回结果，包含已注册工具、服务器信息和错误列表。"""
    tools: list[Tool] = field(default_factory=list)
    servers: list[ServerInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class MCPManager:


    def __init__(self) -> None:
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._registry: ToolRegistry | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False


    def load_configs(self, configs: list[MCPServerConfig]) -> None:
        for cfg in configs:
            self._configs[cfg.name] = cfg


    async def connect_all(self) -> ConnectResult:
        """连接所有已加载的 MCP 服务器，返回工具列表、服务器信息和错误。

        对齐 Go 版 ConnectAll：连接后从 InitializeResult 提取 instructions，
        将其包含在 ServerInfo 中返回，供系统提示注入使用。
        """
        async with self._lifecycle_lock:
            return await self._connect_all()

    async def _connect_all(self) -> ConnectResult:
        if self._closed:
            raise RuntimeError("MCP manager is shut down")
        result = ConnectResult()
        for name, config in self._configs.items():
            client = self._clients.get(name)
            if client is None:
                client = MCPClient(config)
            try:
                await client.connect()
                tools = await client.list_tools()
                wrappers = [MCPToolWrapper(name, tool_def, client) for tool_def in tools]
                wrappers.extend(build_catalog_tools(name, client, {tool.name for tool in wrappers}))
                self._clients[name] = client
                result.servers.append(ServerInfo(name=name, instructions=client.instructions))
                result.tools.extend(wrappers)
                for tool_def in tools:
                    logger.info("Discovered MCP tool: %s/%s", name, tool_def.name)

            except Exception as e:
                await client.close()
                self._clients.pop(name, None)
                msg = f"MCP server '{name}': {e}"
                logger.warning(msg)
                result.errors.append(msg)
            except BaseException:
                await client.close()
                self._clients.pop(name, None)
                raise

        return result

    async def register_all_tools(self, registry: ToolRegistry) -> ConnectResult:
        """连接所有服务器并注册工具到 registry，返回 ConnectResult。

        与旧版签名兼容（之前返回 list[str]），现在返回 ConnectResult，
        调用方可通过 result.errors 获取错误列表，也可通过 result.servers
        获取每个服务器的 instructions。
        """
        async with self._lifecycle_lock:
            result = await self._connect_all()
            if self._registry is not None:
                for name in self._configs:
                    await self._registry.release_scope(f"mcp:{name}")
            self._registry = registry
            for tool in result.tools:
                assert isinstance(tool, MCPToolWrapper)
                registry.register(
                    tool, source=ToolSource.MCP, scope_id=f"mcp:{tool.server_name}",
                )
            return result


    async def get_client(self, name: str) -> MCPClient | None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MCP manager is shut down")
            config = self._configs.get(name)
            if config is None:
                return None
            client = self._clients.get(name)
            if client is None:
                client = MCPClient(config)
            await client.connect()
            self._clients[name] = client
            return client

    def tool_names_for_server(self, server_name: str) -> list[str]:
        """Use registry ownership, not a guessed tool-name prefix."""
        if self._registry is None:
            return []
        scope_id = f"mcp:{server_name}"
        return [
            entry.name for entry in self._registry.list_registrations()
            if entry.source == ToolSource.MCP and entry.scope_id == scope_id
        ]


    async def shutdown(self) -> None:
        async with self._lifecycle_lock:
            self._closed = True
            await self._shutdown()

    async def _shutdown(self) -> None:
        registry = self._registry
        if registry is not None:
            for name in self._configs:
                await registry.release_scope(f"mcp:{name}")
            self._registry = None
        for name, client in self._clients.items():
            try:
                await client.close()
                logger.info("MCP server '%s' closed", name)
            except Exception:
                logger.debug("Error closing MCP server '%s'", name, exc_info=True)
        self._clients.clear()
