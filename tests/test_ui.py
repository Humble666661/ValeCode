from __future__ import annotations

import pytest
from textual.widgets import Static

from valecode import __version__
from valecode.app import ToolCallBlock, VelaCodeApp
from valecode.commands.completion import CompletionPopup
from valecode.config import ProviderConfig


def _provider(name: str, model: str) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        protocol="anthropic",
        base_url="https://example.invalid",
        model=model,
        api_key="test-key",
    )


def test_banner_uses_package_version_and_runtime_context() -> None:
    banner = VelaCodeApp._make_banner(
        "test-model", "C:/workspace", "test-provider"
    )

    assert banner.plain == (
        f"◆ VelaCode  v{__version__}\n"
        "  test-provider / test-model  ·  C:/workspace"
    )


def test_tool_block_has_distinct_loading_success_and_error_states() -> None:
    block = ToolCallBlock("ReadFile", {"file_path": "README.md"})
    assert "◇  Read README.md" in str(block.render())

    block.set_result("content", is_error=False, elapsed=0.25)
    assert "✓  Read README.md  ·  0.2s" in str(block.render())

    failed = ToolCallBlock("Bash", {"command": "false"})
    failed.set_result("failed", is_error=True, elapsed=1.0)
    assert "×  Bash: false  ·  1.0s" in str(failed.render())
    assert failed.has_class("tool-block-error")


def test_completion_popup_window_follows_cursor() -> None:
    popup = CompletionPopup()
    pairs = [(f"/cmd{i} — 命令 {i}", f"/cmd{i}") for i in range(12)]
    popup.show_pairs(pairs)

    for _ in range(9):
        popup.move_down()

    rendered = str(popup.render())
    assert popup.get_selected() == "/cmd9"
    assert "/cmd9" in rendered
    assert "/cmd0" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_app_composes_branded_shell_and_shortcut_bar(
    size: tuple[int, int],
) -> None:
    app = VelaCodeApp(
        [_provider("first", "model-a"), _provider("second", "model-b")]
    )

    async with app.run_test(size=size):
        title = app.query_one("#title-bar", Static).render()
        assert "VelaCode" in str(title)
        assert f"v{__version__}" in str(title)
        assert "Select a provider to begin" in str(title)

        shortcut = app.query_one("#shortcut-label", Static)
        assert "Shift+Enter 换行" in str(shortcut.render())
        assert app.theme == "valecode"
