from __future__ import annotations

from pathlib import Path

import pytest

import valecode.config as config_module
from valecode.config import ConfigError, load_config


_CONFIG_ENV_KEYS = (
    "VALECODE_PROVIDER_NAME",
    "VALECODE_PROTOCOL",
    "VALECODE_BASE_URL",
    "VALECODE_MODEL",
    "VALECODE_API_KEY",
    "VALECODE_THINKING",
    "VALECODE_CONTEXT_WINDOW",
    "VALECODE_MAX_OUTPUT_TOKENS",
    "VALECODE_PERMISSION_MODE",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _CONFIG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _isolate_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(config_module.Path, "home", classmethod(lambda cls: home))
    return project


def test_loads_single_provider_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _isolate_paths(monkeypatch, tmp_path)
    (project / ".env").write_text(
        "\n".join(
            [
                "VALECODE_PROVIDER_NAME=local-anthropic",
                "VALECODE_PROTOCOL=anthropic",
                "VALECODE_BASE_URL=https://api.example.test",
                "VALECODE_MODEL=claude-test",
                "ANTHROPIC_API_KEY=secret-from-dotenv",
                "VALECODE_THINKING=true",
                "VALECODE_CONTEXT_WINDOW=123456",
                "VALECODE_MAX_OUTPUT_TOKENS=4096",
                "VALECODE_PERMISSION_MODE=plan",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config()

    provider = config.providers[0]
    assert provider.name == "local-anthropic"
    assert provider.protocol == "anthropic"
    assert provider.base_url == "https://api.example.test"
    assert provider.model == "claude-test"
    assert provider.resolve_api_key() == "secret-from-dotenv"
    assert provider.thinking is True
    assert provider.context_window == 123456
    assert provider.max_output_tokens == 4096
    assert config.permission_mode == "plan"


def test_environment_precedence_over_dotenv_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _isolate_paths(monkeypatch, tmp_path)
    (project / ".env").write_text(
        "VALECODE_PROTOCOL=openai\n"
        "VALECODE_BASE_URL=https://base.example.test/v1\n"
        "VALECODE_MODEL=base-model\n",
        encoding="utf-8",
    )
    (project / ".env.local").write_text(
        "VALECODE_MODEL=local-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("VALECODE_MODEL", "process-model")

    config = load_config()

    assert config.providers[0].model == "process-model"


def test_yaml_can_reference_dotenv_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _isolate_paths(monkeypatch, tmp_path)
    (project / ".env").write_text(
        "MODEL_NAME=gpt-test\nOPENAI_API_KEY=dotenv-key\n", encoding="utf-8"
    )
    config_path = project / "config.yaml"
    config_path.write_text(
        "providers:\n"
        "  - name: local-openai\n"
        "    protocol: openai\n"
        "    base_url: https://api.example.test/v1\n"
        "    model: ${MODEL_NAME}\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.providers[0].model == "gpt-test"
    assert config.providers[0].resolve_api_key() == "dotenv-key"


def test_incomplete_dotenv_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _isolate_paths(monkeypatch, tmp_path)
    (project / ".env").write_text(
        "VALECODE_PROTOCOL=anthropic\nVALECODE_MODEL=claude-test\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="VALECODE_BASE_URL"):
        load_config()


def test_later_yaml_layer_can_explicitly_disable_boolean_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _isolate_paths(monkeypatch, tmp_path)
    home = tmp_path / "home"
    (home / ".valecode").mkdir()
    (project / ".valecode").mkdir()
    provider = (
        "providers:\n"
        "  - name: test\n"
        "    protocol: anthropic\n"
        "    base_url: https://api.example.test\n"
        "    model: test-model\n"
    )
    (home / ".valecode" / "config.yaml").write_text(
        provider
        + "permission_mode: bypassPermissions\n"
        + "enable_fork: true\n"
        + "enable_verification_agent: true\n"
        + "enable_coordinator_mode: true\n"
        + "sandbox:\n"
        + "  enabled: true\n"
        + "  auto_allow: true\n"
        + "  network_enabled: true\n",
        encoding="utf-8",
    )
    (project / ".valecode" / "config.local.yaml").write_text(
        provider
        + "permission_mode: default\n"
        + "enable_fork: false\n"
        + "enable_verification_agent: false\n"
        + "enable_coordinator_mode: false\n"
        + "sandbox:\n"
        + "  enabled: false\n"
        + "  auto_allow: false\n"
        + "  network_enabled: false\n",
        encoding="utf-8",
    )

    config = load_config()

    assert config.permission_mode == "default"
    assert config.enable_fork is False
    assert config.enable_verification_agent is False
    assert config.enable_coordinator_mode is False
    assert config.sandbox.enabled is False
    assert config.sandbox.auto_allow is False
    assert config.sandbox.network_enabled is False


def test_unspecified_yaml_fields_keep_values_from_earlier_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _isolate_paths(monkeypatch, tmp_path)
    home = tmp_path / "home"
    (home / ".valecode").mkdir()
    (project / ".valecode").mkdir()
    provider = (
        "providers:\n"
        "  - name: test\n"
        "    protocol: anthropic\n"
        "    base_url: https://api.example.test\n"
        "    model: test-model\n"
    )
    (home / ".valecode" / "config.yaml").write_text(
        provider
        + "permission_mode: plan\n"
        + "enable_fork: true\n"
        + "enable_verification_agent: true\n"
        + "enable_coordinator_mode: true\n"
        + "sandbox:\n"
        + "  enabled: true\n"
        + "  auto_allow: true\n"
        + "  network_enabled: true\n",
        encoding="utf-8",
    )
    (project / ".valecode" / "config.local.yaml").write_text(
        provider, encoding="utf-8"
    )

    config = load_config()

    assert config.permission_mode == "plan"
    assert config.enable_fork is True
    assert config.enable_verification_agent is True
    assert config.enable_coordinator_mode is True
    assert config.sandbox.enabled is True
    assert config.sandbox.auto_allow is True
    assert config.sandbox.network_enabled is True
