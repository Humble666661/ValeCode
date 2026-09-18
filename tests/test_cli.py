from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from valecode.__main__ import _configure_logging, _run_prompt_with_cleanup, main


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
