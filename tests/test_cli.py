from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from valecode.__main__ import _configure_logging, _run_prompt_with_cleanup, main
from valecode.client import LLMClient
from valecode.config import ProviderConfig
from valecode.mcp import ConnectResult, ServerInfo
from valecode.permissions import PermissionMode
from valecode.tools import ToolSource
from valecode.tools.base import StreamEnd, TextDelta, Tool, ToolResult


def test_help_does_not_touch_state_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["valecode", "--help"])

    with patch("valecode.__main__._configure_logging") as configure_logging:
        with pytest.raises(SystemExit) as exc_info:
            main()

    assert exc_info.value.code == 0
    assert "ValeCode AI coding assistant" in capsys.readouterr().out
    configure_logging.assert_not_called()


def test_logging_falls_back_when_project_state_directory_is_unwritable(
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    fallback = tmp_path / "fallback"

    with patch("valecode.__main__.logging.basicConfig") as basic_config:
        log_path = _configure_logging(blocked, fallback)

    assert log_path == fallback / "debug.log"
    assert fallback.is_dir()
    assert basic_config.call_args.kwargs["filename"] == str(log_path)


@pytest.mark.asyncio
async def test_prompt_mode_shuts_down_hooks_even_on_error() -> None:
    hooks = AsyncMock()
    with patch("valecode.__main__._run_prompt", new_callable=AsyncMock) as run_prompt:
        run_prompt.side_effect = RuntimeError("prompt failed")
        with pytest.raises(RuntimeError, match="prompt failed"):
            await _run_prompt_with_cleanup(None, None, hooks, "test", "text")
    hooks.shutdown.assert_awaited_once()


class _NoParams(BaseModel):
    pass


class _MCPTool(Tool):
    name = "mcp_demo_lookup"
    description = "demo MCP tool"
    params_model = _NoParams

    async def execute(self, params: _NoParams) -> ToolResult:
        return ToolResult("ok")


class _PromptClient(LLMClient):
    def __init__(self) -> None:
        self.tool_names: list[str] = []
        self.history_text = ""

    async def stream(
        self, conversation, system: str = "", tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator:
        self.tool_names = [tool["name"] for tool in tools or []]
        self.history_text = "\n".join(message.content for message in conversation.history)
        yield TextDelta("done")
        yield StreamEnd("end_turn", input_tokens=1, output_tokens=1)


def _prompt_config() -> SimpleNamespace:
    provider = ProviderConfig(
        name="offline", protocol="anthropic", base_url="https://example.invalid",
        model="offline", api_key="not-used",
    )
    return SimpleNamespace(
        providers=[provider], sandbox=SimpleNamespace(enabled=False),
        worktree=None, enable_verification_agent=False, enable_fork=False,
        teammate_mode="", enable_coordinator_mode=False,
        mcp_servers=[SimpleNamespace(name="demo")],
    )


@pytest.mark.asyncio
async def test_prompt_mode_registers_mcp_injects_instructions_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import valecode.mcp as mcp_module

    monkeypatch.chdir(tmp_path)
    client = _PromptClient()
    manager = SimpleNamespace(shutdown=AsyncMock())
    manager.load_configs = lambda configs: setattr(manager, "configs", configs)

    async def register(registry):
        registry.register(
            _MCPTool(), source=ToolSource.MCP, scope_id="mcp:demo"
        )
        manager.registry = registry
        return ConnectResult(servers=[ServerInfo("demo", "Use demo lookup first.")])

    manager.register_all_tools = register
    manager.tool_names_for_server = lambda _name: ["mcp_demo_lookup"]

    async def no_resolve(_provider):
        return None

    with (
        patch("valecode.client.create_client", return_value=client),
        patch("valecode.client.resolve_context_window", no_resolve),
        patch.object(mcp_module, "MCPManager", return_value=manager),
    ):
        await _run_prompt_with_cleanup(
            _prompt_config(), PermissionMode.DEFAULT, None, "hello", "text"
        )

    assert "mcp_demo_lookup" in client.tool_names
    assert "Use demo lookup first." in client.history_text
    manager.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_prompt_mode_closes_mcp_when_initialization_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import valecode.mcp as mcp_module

    monkeypatch.chdir(tmp_path)
    manager = SimpleNamespace(shutdown=AsyncMock())
    manager.load_configs = lambda _configs: None

    async def fail(_registry):
        raise RuntimeError("MCP init failed")

    manager.register_all_tools = fail
    async def no_resolve(_provider):
        return None

    with (
        patch("valecode.client.create_client", return_value=_PromptClient()),
        patch("valecode.client.resolve_context_window", no_resolve),
        patch.object(mcp_module, "MCPManager", return_value=manager),
        pytest.raises(RuntimeError, match="MCP init failed"),
    ):
        await _run_prompt_with_cleanup(
            _prompt_config(), PermissionMode.DEFAULT, None, "hello", "text"
        )
    manager.shutdown.assert_awaited_once()
