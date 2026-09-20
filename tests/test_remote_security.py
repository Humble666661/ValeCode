from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from valecode.remote import RemoteServer, _is_loopback_bind


def _request(path: str, headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(path=path, headers=websockets.Headers(headers or {}))


def test_remote_defaults_to_loopback_without_authentication() -> None:
    server = RemoteServer(providers=[])

    assert server.addr == "127.0.0.1"
    assert server.auth_token == ""
    assert server._process_http_request(None, _request("/ws")) is None


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_bind_detection(host: str) -> None:
    assert _is_loopback_bind(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "remote.test"])
def test_non_loopback_bind_requires_token(host: str) -> None:
    with pytest.raises(ValueError, match="token is required"):
        RemoteServer(providers=[], addr=host)


def test_websocket_rejects_missing_or_incorrect_token() -> None:
    server = RemoteServer(providers=[], auth_token="expected-secret")

    missing = server._process_http_request(None, _request("/ws"))
    incorrect = server._process_http_request(None, _request("/ws?token=wrong"))

    assert missing is not None and missing.status_code == 401
    assert incorrect is not None and incorrect.status_code == 401
    assert missing.headers["WWW-Authenticate"] == 'Bearer realm="ValeCode Remote"'


def test_websocket_accepts_query_or_bearer_token() -> None:
    server = RemoteServer(providers=[], auth_token="expected-secret")

    query_result = server._process_http_request(
        None, _request("/ws?token=expected-secret")
    )
    bearer_result = server._process_http_request(
        None,
        _request("/ws", {"Authorization": "Bearer expected-secret"}),
    )

    assert query_result is None
    assert bearer_result is None


def test_remote_page_only_exposes_auth_requirement_not_token() -> None:
    server = RemoteServer(providers=[], auth_token="must-not-leak")

    response = server._process_http_request(None, _request("/"))

    assert response is not None and response.status_code == 200
    html = bytes(response.body).decode("utf-8")
    assert "const remoteAuthRequired = true;" in html
    assert "must-not-leak" not in html


@pytest.mark.asyncio
async def test_live_websocket_handshake_enforces_token() -> None:
    server = RemoteServer(providers=[], auth_token="expected-secret")
    async with websockets.serve(
        server._ws_handler,
        "127.0.0.1",
        0,
        process_request=server._process_http_request,
    ) as listening:
        port = listening.sockets[0].getsockname()[1]
        with pytest.raises(websockets.exceptions.InvalidStatus) as rejected:
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws"):
                pass
        assert rejected.value.response.status_code == 401

        async with websockets.connect(
            f"ws://127.0.0.1:{port}/ws?token=expected-secret"
        ) as websocket:
            message = json.loads(await websocket.recv())
            assert message["type"] == "connected"


@pytest.mark.asyncio
async def test_remote_shutdown_releases_resources_after_startup_error() -> None:
    server = RemoteServer(providers=[])
    server.mcp_manager = AsyncMock()
    server.mcp_manager.shutdown.side_effect = RuntimeError("MCP close failed")
    server.registry = AsyncMock()
    server.hook_engine = AsyncMock()
    server.session = MagicMock()
    with (
        patch.object(server, "_init_agent"),
        patch.object(server, "_init_mcp", new_callable=AsyncMock) as init_mcp,
    ):
        init_mcp.side_effect = RuntimeError("startup failed")
        with pytest.raises(RuntimeError, match="startup failed"):
            await server.run()

    server.registry.release_session.assert_awaited_once()
    server.hook_engine.shutdown.assert_awaited_once()
    server.session.close.assert_called_once()


@pytest.mark.asyncio
async def test_remote_registers_subagents_tasks_and_trace_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from valecode.agents.durable_task_manager import DurableTaskManager
    from valecode.config import ProviderConfig
    from valecode.tools.agent_tool import AgentTool

    monkeypatch.chdir(tmp_path)
    provider = ProviderConfig(
        name="offline",
        protocol="anthropic",
        base_url="https://example.invalid",
        model="offline",
        api_key="not-used",
    )
    with patch("valecode.remote.create_client", return_value=MagicMock()):
        server = RemoteServer(
            [provider],
            enable_fork=True,
            enable_verification_agent=True,
        )
        server._init_agent()

    assert isinstance(server.registry.get("Agent"), AgentTool)
    assert isinstance(server.task_manager, DurableTaskManager)
    assert server.agent_loader.get("Verification") is not None
    assert server.command_registry.find("tasks") is not None
    assert server.command_registry.find("trace") is not None
    assert "Leave subagent_type empty" in server.agent._agent_catalog
    await server._shutdown()


@pytest.mark.asyncio
async def test_remote_delivers_completed_background_task_to_lead() -> None:
    from valecode.agents.task_manager import BackgroundTask

    server = RemoteServer([])
    server.agent = MagicMock()
    server.session_id = "session-active"
    server.task_manager = MagicMock()
    task_agent = MagicMock()
    task_agent.session_id = "session-active"
    completed = BackgroundTask(
        id="task-1",
        name="Explore",
        agent=task_agent,
        task="inspect project",
        status="completed",
        result="found the implementation",
    )
    server.task_manager.poll_completed.return_value = [completed]
    server._connections.add(MagicMock())
    server._broadcast = AsyncMock()
    server._handle_user_message = AsyncMock()

    await server._process_task_notifications()

    server._broadcast.assert_awaited_once()
    prompt = server._handle_user_message.await_args.args[0]
    assert "<task-notification>" in prompt
    assert "found the implementation" in prompt
    server._handle_user_message.assert_awaited_once_with(
        prompt, dispatch_commands=False
    )


@pytest.mark.asyncio
async def test_remote_does_not_inject_another_sessions_task() -> None:
    from valecode.agents.task_manager import BackgroundTask

    server = RemoteServer([])
    server.agent = MagicMock()
    server.session_id = "session-active"
    server.task_manager = MagicMock()
    task_agent = MagicMock()
    task_agent.session_id = "session-old"
    server.task_manager.poll_completed.return_value = [
        BackgroundTask(
            id="task-old",
            name="Explore",
            agent=task_agent,
            task="old work",
            status="completed",
            result="old result",
        )
    ]
    server._connections.add(MagicMock())
    server._broadcast = AsyncMock()
    server._handle_user_message = AsyncMock()

    await server._process_task_notifications()

    server._broadcast.assert_not_awaited()
    server._handle_user_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_remote_keeps_completed_tasks_queued_without_clients() -> None:
    server = RemoteServer([])
    server.agent = MagicMock()
    server.task_manager = MagicMock()

    await server._process_task_notifications()

    server.task_manager.poll_completed.assert_not_called()
