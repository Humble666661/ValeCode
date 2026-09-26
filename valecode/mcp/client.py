from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
from anyio import BrokenResourceError, EndOfStream
from mcp import ClientSession, types
from mcp.shared.exceptions import McpError
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from valecode.config import MCPServerConfig, build_child_env, resolve_env_vars

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    method: str
    arguments: tuple[Any, ...]
    future: asyncio.Future[Any]
    deadline: float


class MCPClient:
    """One owner task enters, uses and exits the SDK's transport cancel scopes."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.name = config.name
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._alive = False
        self._init_result: types.InitializeResult | None = None
        self._connect_lock = asyncio.Lock()
        self._owner_task: asyncio.Task[None] | None = None
        self._owner_closing = False
        self._cleaning = False
        self._stop_requested = False
        self._requests: asyncio.Queue[_Request] = asyncio.Queue()

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def instructions(self) -> str:
        if self._init_result is not None and self._init_result.instructions:
            return self._init_result.instructions
        return ""

    def supports(self, capability: str) -> bool:
        capabilities = getattr(self._init_result, "capabilities", None)
        return capabilities is not None and getattr(capabilities, capability, None) is not None

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._alive:
                return
            await self._stop_owner()
            self._owner_closing = False
            self._stop_requested = False
            ready = asyncio.get_running_loop().create_future()
            self._requests = asyncio.Queue()
            self._owner_task = asyncio.create_task(
                self._own_connection(ready), name=f"mcp:{self.name}"
            )
            try:
                await asyncio.shield(ready)
            except BaseException:
                await self._stop_owner()
                if ready.done() and not ready.cancelled():
                    ready.exception()
                raise

    async def _own_connection(self, ready: asyncio.Future[None]) -> None:
        active: _Request | None = None
        try:
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with asyncio.timeout(self.config.connect_timeout):
                        await self._open_session()
                    break
                except Exception:
                    await self._cleanup_stack()
                    if self._stop_requested:
                        raise asyncio.CancelledError
                    if attempt == self.config.max_retries:
                        raise
                    logger.warning("MCP server '%s' connection attempt failed; retrying", self.name)
                    await asyncio.sleep(self.config.retry_delay)
            self._alive = True
            ready.set_result(None)
            logger.info("MCP server '%s' connected", self.name)
            while True:
                active = await self._requests.get()
                if active.future.cancelled():
                    active = None
                    continue
                if active.deadline <= asyncio.get_running_loop().time():
                    active.future.set_exception(TimeoutError("MCP request expired in queue"))
                    active = None
                    continue
                try:
                    assert self._session is not None
                    async with asyncio.timeout_at(active.deadline):
                        result = await getattr(self._session, active.method)(*active.arguments)
                    if not active.future.done():
                        active.future.set_result(result)
                except Exception as exc:
                    if not active.future.done():
                        active.future.set_exception(exc)
                    # An expired RPC has an unknown outcome. Never replay it.
                    if isinstance(exc, (TimeoutError, ConnectionError, EOFError,
                                        BrokenResourceError, EndOfStream, httpx.TransportError)) or (
                        isinstance(exc, McpError) and exc.error.code in (408, types.CONNECTION_CLOSED)
                    ):
                        return
                active = None
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP server '%s' owner stopped: %s", self.name, exc)
        finally:
            self._owner_closing = True
            self._alive = False
            self._session = None
            self._init_result = None
            error = ConnectionError(f"MCP server '{self.name}' connection closed")
            if active is not None and not active.future.done():
                active.future.set_exception(error)
            while not self._requests.empty():
                request = self._requests.get_nowait()
                if not request.future.done():
                    request.future.set_exception(error)
            await self._cleanup_stack()

    async def _open_session(self) -> None:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        if self.config.is_stdio:
            read, write = await self._connect_stdio()
        else:
            read, write = await self._connect_http()
        session = await self._stack.enter_async_context(
            ClientSession(
                read, write,
                read_timeout_seconds=timedelta(seconds=self.config.request_timeout),
            )
        )
        self._init_result = await session.initialize()
        self._session = session

    async def _connect_stdio(self) -> tuple[Any, Any]:
        assert self._stack is not None
        assert self.config.command is not None
        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=build_child_env(self.config.env),
        )
        devnull = open(os.devnull, "w")
        self._stack.callback(devnull.close)
        read, write = await self._stack.enter_async_context(stdio_client(params, errlog=devnull))
        return read, write

    async def _connect_http(self) -> tuple[Any, Any]:
        assert self._stack is not None
        assert self.config.url is not None
        headers = {k: resolve_env_vars(v) for k, v in self.config.headers.items()}
        http_client = httpx.AsyncClient(
            headers=headers, follow_redirects=True,
            timeout=httpx.Timeout(self.config.request_timeout, connect=self.config.connect_timeout),
        )
        await self._stack.enter_async_context(http_client)
        result = await self._stack.enter_async_context(
            streamable_http_client(self.config.url, http_client=http_client)
        )
        return result[0], result[1]

    async def _request(self, method: str, *arguments: Any) -> Any:
        await self.connect()
        if not self._alive:
            raise ConnectionError(f"MCP server '{self.name}' connection closed")
        future = asyncio.get_running_loop().create_future()
        deadline = asyncio.get_running_loop().time() + self.config.request_timeout
        self._requests.put_nowait(_Request(method, arguments, future, deadline))
        try:
            async with asyncio.timeout_at(deadline):
                return await future
        except TimeoutError as exc:
            raise TimeoutError(
                f"MCP server '{self.name}' {method} timed out after "
                f"{self.config.request_timeout:g}s; the operation was not retried"
            ) from exc

    async def list_tools(self) -> list[types.Tool]:
        await self.connect()
        if not self.supports("tools"):
            return []
        return await self._list_all("list_tools", "tools")

    async def _list_all(self, method: str, field: str) -> list[Any]:
        items: list[Any] = []
        cursor = None
        seen: set[str] = set()
        for _ in range(100):
            result = await self._request(method, cursor)
            items.extend(getattr(result, field))
            cursor = result.nextCursor
            if not cursor:
                return items
            if cursor in seen:
                raise RuntimeError(f"MCP server '{self.name}' repeated a pagination cursor")
            seen.add(cursor)
        raise RuntimeError(f"MCP server '{self.name}' exceeded 100 catalog pages")

    async def list_resources(self) -> list[types.Resource]:
        return await self._list_all("list_resources", "resources")

    async def list_resource_templates(self) -> list[types.ResourceTemplate]:
        return await self._list_all("list_resource_templates", "resourceTemplates")

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        return await self._request("read_resource", types.AnyUrl(uri))

    async def list_prompts(self) -> list[types.Prompt]:
        return await self._list_all("list_prompts", "prompts")

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> types.GetPromptResult:
        return await self._request("get_prompt", name, arguments)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await self._request("call_tool", name, arguments)

    async def close(self) -> None:
        async with self._connect_lock:
            self._alive = False
            await self._stop_owner()

    async def _stop_owner(self) -> None:
        task = self._owner_task
        if task is not None:
            self._stop_requested = True
            if not task.done() and not self._owner_closing and not self._cleaning:
                task.cancel()
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))
            if self._owner_task is task:
                self._owner_task = None

    async def _cleanup_stack(self) -> None:
        stack = self._stack
        self._stack = None
        if stack is not None:
            self._cleaning = True
            try:
                await stack.aclose()
            except Exception:
                logger.warning("Error closing MCP transport '%s'", self.name, exc_info=True)
            finally:
                self._cleaning = False
