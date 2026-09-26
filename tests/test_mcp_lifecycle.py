from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from mcp import types

from valecode.config import ConfigError, MCPServerConfig
from valecode.mcp.client import MCPClient
from valecode.mcp.manager import MCPManager
from valecode.mcp.tool_wrapper import MCPToolWrapper
from valecode.mcp.catalog_tools import build_catalog_tools, ResourceParams, PromptParams
from valecode.tools import ToolRegistry
from valecode.validator import validate_mcp_servers


@pytest.mark.parametrize("key,value", [
    ("connect_timeout", True), ("connect_timeout", 0),
    ("request_timeout", float("inf")), ("request_timeout", float("nan")),
    ("retry_delay", -1), ("retry_delay", "1"),
    ("max_retries", True), ("max_retries", -1), ("max_retries", 11),
])
def test_mcp_limits_reject_invalid_values(key, value):
    with pytest.raises(ConfigError, match=key):
        validate_mcp_servers([{"name": "test", "command": "unused", key: value}])


def test_mcp_limits_defaults_and_overrides():
    result = validate_mcp_servers([{
        "name": "test", "command": "unused", "max_retries": 0, "connect_timeout": 2,
    }])[0]
    assert result["max_retries"] == 0
    assert result["connect_timeout"] == 2.0
    assert result["request_timeout"] == 60.0
    assert result["retry_delay"] == 0.5


def tool_definition():
    return types.Tool(name="echo", inputSchema={"type": "object", "properties": {}})


class _FakeTransport:
    def __init__(self, *, failures=0, hanging_init=False, hanging_call=False):
        self.tasks = []
        self.opens = 0
        self.closes = 0
        self.failures = failures
        self.hanging_init = hanging_init
        self.hanging_call = hanging_call
        self.calls = 0
        self.session = SimpleNamespace(
            initialize=self.initialize, list_tools=self.list_tools, call_tool=self.call_tool,
        )

    @asynccontextmanager
    async def context(self):
        task = asyncio.current_task()
        self.tasks.append(task)
        self.opens += 1
        with anyio.CancelScope():
            try:
                yield self.session
            finally:
                assert asyncio.current_task() is task
                self.closes += 1

    async def open(self, client):
        from contextlib import AsyncExitStack
        client._stack = AsyncExitStack()
        session = await client._stack.enter_async_context(self.context())
        client._init_result = await session.initialize()
        client._session = session

    async def initialize(self):
        if self.hanging_init:
            await asyncio.Event().wait()
        if self.failures:
            self.failures -= 1
            raise ConnectionError("initialization failed")
        return SimpleNamespace(instructions="test server", capabilities=SimpleNamespace(tools=True))

    async def list_tools(self, cursor=None):
        return SimpleNamespace(tools=[tool_definition()], nextCursor=None)

    async def call_tool(self, *args):
        self.calls += 1
        if self.hanging_call:
            await asyncio.Event().wait()
        return types.CallToolResult(content=[types.TextContent(type="text", text="done")])


def make_client():
    return MCPClient(MCPServerConfig(
        name="offline", command="unused", connect_timeout=0.03,
        request_timeout=0.03, max_retries=1, retry_delay=0.001,
    ))


@pytest.mark.asyncio
async def test_retry_and_cross_task_close_keep_transport_owner():
    fake = _FakeTransport(failures=1)
    client = make_client()
    with patch.object(client, "_open_session", new=lambda: fake.open(client)):
        await asyncio.gather(client.connect(), client.connect())
        assert client.instructions == "test server"
        assert len(await client.list_tools()) == 1
        await asyncio.create_task(client.close())
    assert fake.opens == fake.closes == 2
    assert len(set(fake.tasks)) == 1
    assert not client.is_alive
    assert client._owner_task is None


@pytest.mark.asyncio
async def test_connect_timeout_cleans_every_attempt():
    fake = _FakeTransport(hanging_init=True)
    client = make_client()
    with patch.object(client, "_open_session", new=lambda: fake.open(client)):
        with pytest.raises(TimeoutError):
            await client.connect()
    assert fake.opens == fake.closes == 2
    assert client._owner_task is None
    assert not client.is_alive


@pytest.mark.asyncio
async def test_cancelled_initialization_cleans_transport():
    fake = _FakeTransport(hanging_init=True)
    client = make_client()
    with patch.object(client, "_open_session", new=lambda: fake.open(client)):
        connecting = asyncio.create_task(client.connect())
        async with asyncio.timeout(1):
            while fake.opens == 0:
                await asyncio.sleep(0)
        connecting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connecting
    assert fake.opens == fake.closes == 1
    assert client._owner_task is None


@pytest.mark.asyncio
async def test_cancel_while_failed_transport_is_closing_does_not_interrupt_exit():
    client = make_client()
    closing = asyncio.Event()
    release = asyncio.Event()
    exited = asyncio.Event()

    @asynccontextmanager
    async def failed_transport():
        with anyio.CancelScope():
            try:
                yield None
            finally:
                closing.set()
                await release.wait()
                exited.set()

    async def fail():
        from contextlib import AsyncExitStack
        client._stack = AsyncExitStack()
        await client._stack.enter_async_context(failed_transport())
        raise ConnectionError("failed initialize")

    with patch.object(client, "_open_session", new=fail):
        connecting = asyncio.create_task(client.connect())
        async with asyncio.timeout(1):
            await closing.wait()
            connecting.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await connecting
    assert exited.is_set()
    assert client._owner_task is None


@pytest.mark.asyncio
async def test_tool_timeout_never_replays_side_effect():
    fake = _FakeTransport(hanging_call=True)
    client = make_client()
    with patch.object(client, "_open_session", new=lambda: fake.open(client)):
        with pytest.raises(TimeoutError, match="not retried"):
            await client.call_tool("write", {"value": 1})
        await client.close()
    assert fake.calls == 1
    assert fake.opens == fake.closes == 1


@pytest.mark.asyncio
async def test_close_releases_pending_requests():
    fake = _FakeTransport(hanging_call=True)
    client = make_client()
    client.config.request_timeout = 10
    with patch.object(client, "_open_session", new=lambda: fake.open(client)):
        active = asyncio.create_task(client.call_tool("write", {}))
        async with asyncio.timeout(1):
            while fake.calls == 0:
                await asyncio.sleep(0)
            queued = asyncio.create_task(client.call_tool("next", {}))
            await asyncio.sleep(0)
            await client.close()
            outcomes = await asyncio.gather(active, queued, return_exceptions=True)
    assert all(isinstance(outcome, ConnectionError) for outcome in outcomes)
    assert fake.calls == 1
    assert fake.opens == fake.closes == 1


@pytest.mark.asyncio
async def test_catalog_pagination_and_repeated_cursor_refusal():
    fake = _FakeTransport()
    client = make_client()
    fake.session.list_tools = AsyncMock(side_effect=[
        SimpleNamespace(tools=[tool_definition()], nextCursor="next"),
        SimpleNamespace(tools=[types.Tool(name="second", inputSchema={})], nextCursor=None),
    ])
    with patch.object(client, "_open_session", new=lambda: fake.open(client)):
        try:
            assert [tool.name for tool in await client.list_tools()] == ["echo", "second"]
            assert fake.session.list_tools.await_args_list[1].args == ("next",)
            fake.session.list_tools.side_effect = None
            fake.session.list_tools.return_value = SimpleNamespace(tools=[], nextCursor="same")
            with pytest.raises(RuntimeError, match="repeated"):
                await client.list_tools()
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_resources_only_server_registered_without_remote_tools():
    manager = MCPManager()
    manager.load_configs([MCPServerConfig(name="catalog", command="unused")])
    client = SimpleNamespace(
        connect=AsyncMock(), close=AsyncMock(), instructions="resources only",
        list_tools=AsyncMock(return_value=[]), supports=lambda name: name == "resources",
    )
    registry = ToolRegistry()
    with patch("valecode.mcp.manager.MCPClient", return_value=client):
        result = await manager.register_all_tools(registry)
    assert result.errors == []
    assert manager.tool_names_for_server("catalog") == ["mcp_catalog_resources"]
    assert registry.get("mcp_catalog_resources").should_defer
    await manager.shutdown()
    assert registry.get("mcp_catalog_resources") is None


@pytest.mark.asyncio
async def test_discovery_failure_does_not_publish_or_leak_client():
    manager = MCPManager()
    manager.load_configs([MCPServerConfig(name="broken", command="unused")])
    client = SimpleNamespace(
        connect=AsyncMock(), close=AsyncMock(), instructions="should not publish",
        list_tools=AsyncMock(side_effect=RuntimeError("list failed")),
    )
    with patch("valecode.mcp.manager.MCPClient", return_value=client):
        result = await manager.connect_all()
    assert result.servers == result.tools == []
    assert len(result.errors) == 1
    assert manager._clients == {}
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_wrapper_release_keeps_shared_client_and_shutdown_prevents_reopen():
    manager = MCPManager()
    manager.load_configs([MCPServerConfig(name="test", command="unused")])
    client = SimpleNamespace(
        connect=AsyncMock(), close=AsyncMock(), is_alive=True, instructions="test",
        list_tools=AsyncMock(return_value=[tool_definition()]), call_tool=AsyncMock(),
    )
    registry = ToolRegistry()
    with patch("valecode.mcp.manager.MCPClient", return_value=client):
        await manager.register_all_tools(registry)
        wrapper = registry.get("mcp_test_echo")
        client.is_alive = False
        assert await manager.get_client("test") is client
        await registry.release_scope("mcp:test")
        result = await wrapper.execute(wrapper.params_model())
        assert result.is_error and "released" in result.output
        client.close.assert_not_awaited()
        await manager.shutdown()
        await manager.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            await manager.get_client("test")
    client.close.assert_awaited_once()
    client.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_stdio_transport_connect_call_and_close_from_other_task(caplog):
    script = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
    client = MCPClient(MCPServerConfig(
        name="real-stdio", command=sys.executable, args=[str(script)], max_retries=0,
    ))
    try:
        await client.connect()
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {"echo", "disconnect"}
        result = await asyncio.create_task(client.call_tool("echo", {"text": "offline-ok"}))
        assert result.content[0].text == "offline-ok"
        assert client.supports("resources") and client.supports("prompts")
        catalogs = build_catalog_tools("real-stdio", client, {"mcp_real-stdio_resources"})
        resources, prompts = catalogs
        assert resources.name == "mcp_real-stdio_resources_2"
        listed = await resources.execute(ResourceParams())
        assert "valecode-test://greeting" in listed.output and not listed.is_error
        templates = await resources.execute(ResourceParams(action="templates"))
        assert "{name}" in templates.output
        read = await resources.execute(ResourceParams(action="read", uri="valecode-test://greeting"))
        assert "hello from offline MCP" in read.output and not read.is_error
        prompt_list = await prompts.execute(PromptParams())
        assert "review" in prompt_list.output
        rendered = await prompts.execute(PromptParams(action="get", name="review", arguments={"topic": "permissions"}))
        assert "Review permissions carefully." in rendered.output and not rendered.is_error
        missing = await prompts.execute(PromptParams(action="get", name="review"))
        assert missing.is_error
        await resources.close()
        assert client.is_alive
        assert (await resources.execute(ResourceParams())).is_error
    finally:
        await asyncio.create_task(client.close())
    assert not client.is_alive
    assert client._stack is None
    assert "Error closing MCP transport" not in caplog.text


@pytest.mark.asyncio
async def test_real_stdio_disconnect_returns_error_and_reconnects_same_client(caplog):
    script = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
    client = MCPClient(MCPServerConfig(
        name="disconnect", command=sys.executable, args=[str(script)],
        request_timeout=1, max_retries=0,
    ))
    try:
        await client.connect()
        with pytest.raises(Exception):
            await client.call_tool("disconnect", {})
        await client.connect()
        result = await client.call_tool("echo", {"text": "reconnected"})
        assert result.content[0].text == "reconnected"
    finally:
        await client.close()
    assert "Error closing MCP transport" not in caplog.text
