from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import dotenv_values

from .validator import (
    ConfigError,
    DEFAULT_CONTEXT_WINDOW,
    VALID_PERMISSION_MODES,
    VALID_PROTOCOLS,
    VALID_TEAMMATE_MODES,
    lookup_model_context_window,
    validate_config_structure,
)


_ENV_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compat": "OPENAI_API_KEY",
}

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass
class ProviderConfig:
    name: str
    protocol: str
    base_url: str
    model: str
    api_key: str = ""
    thinking: bool = False
    # 0 表示"未设置" — get_context_window() 通过四层 fallback 解析真实窗口大小。
    # 正数表示配置文件里显式指定的覆盖值。
    context_window: int = 0
    max_output_tokens: int = 0
    # 运行时 cache，存放从 provider 的 /v1/models 端点自动拉取的 context window
    # （get_context_window 的第 2 层）。通过 set_fetched_context_window() 写入一次；
    # 0 表示"尚未拉取"。不会持久化。
    _fetched_context_window: int = field(default=0, repr=False)

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_var = _ENV_KEY_MAP.get(self.protocol, "")
        return os.environ.get(env_var, "")

    def set_fetched_context_window(self, window: int) -> None:
        """记录从 provider 自动拉取到的 context window（第 2 层）。

        非正数会被忽略，这样一次失败的拉取就不会污染 cache。在解析
        context window 时，每个 provider 只会调用一次。
        """
        if window > 0:
            self._fetched_context_window = window

    def get_context_window(self) -> int:
        """通过四层 fallback 解析模型的 context window，按优先级从高到低：

          1. 配置文件提供的 context_window（> 0）——显式覆盖，永远优先。
          2. 从 provider 的 /v1/models 端点自动拉取并通过 set_fetched_context_window
             缓存的值（只有 anthropic 协议的 provider 才会设置它；拉取失败或缺失时
             保持为 0 并跳过）。
          3. 内置的「模型名 -> window」映射表（按子串匹配）。
          4. 保守的默认值（claude -> 200000，其他 -> 128000）。
        """
        if self.context_window > 0:
            return self.context_window
        if self._fetched_context_window > 0:
            return self._fetched_context_window
        window = lookup_model_context_window(self.model)
        if window > 0:
            return window
        if "claude" in self.model.lower():
            return DEFAULT_CONTEXT_WINDOW
        return 128_000

    def get_max_output_tokens(self) -> int:
        if self.max_output_tokens > 0:
            return self.max_output_tokens
        if self.thinking:
            return 64000
        return 8192


def resolve_env_vars(value: str, env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    return _ENV_VAR_RE.sub(lambda m: source.get(m.group(1), m.group(0)), value)


def _load_dotenv_environment(project_dir: Path, home: Path) -> dict[str, str]:
    """读取分层 .env，但不修改进程级 ``os.environ``。

    优先级从低到高为：用户级、项目级、项目本地、进程环境变量。
    """
    layered: dict[str, str] = {}
    paths = [
        home / ".valecode" / ".env",
        project_dir / ".env",
        project_dir / ".env.local",
    ]
    for path in paths:
        try:
            is_file = path.is_file()
        except OSError:
            continue
        if not is_file:
            continue
        try:
            values = dotenv_values(path, interpolate=False)
        except OSError:
            continue
        for key, value in values.items():
            if value is None:
                continue
            # 支持后一个值引用前面文件或同一文件中已经声明的变量。
            scope = {**layered, **os.environ}
            layered[key] = resolve_env_vars(value, scope)

    return {**layered, **os.environ}


def _resolve_nested_env(value: object, env: Mapping[str, str]) -> object:
    if isinstance(value, str):
        return resolve_env_vars(value, env)
    if isinstance(value, list):
        return [_resolve_nested_env(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_nested_env(item, env) for key, item in value.items()}
    return value


def build_child_env(declared_env: dict[str, str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    path = os.environ.get("PATH", "")
    if path:
        env["PATH"] = path
    for key, value in (declared_env or {}).items():
        env[key] = resolve_env_vars(value)
    return env


@dataclass
class MCPServerConfig:
    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    connect_timeout: float = 15.0
    request_timeout: float = 60.0
    max_retries: int = 2
    retry_delay: float = 0.5


    @property
    def is_stdio(self) -> bool:
        return self.command is not None


@dataclass
class WorktreeConfig:
    symlink_directories: list[str] = field(default_factory=lambda: ["node_modules", ".venv", "vendor"])
    stale_cleanup_interval: int = 3600
    stale_cutoff_hours: int = 24


@dataclass
class SandboxAppConfig:
    """沙箱相关的配置项。"""
    enabled: bool = False         # 是否启用 OS 级沙箱
    auto_allow: bool = False      # 是否自动放行命令（沙箱兜底）
    network_enabled: bool = False  # 沙箱内是否允许网络访问
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset, repr=False, compare=False
    )


@dataclass
class RemoteAppConfig:
    """Remote Web UI 的监听与认证配置。"""

    host: str = "127.0.0.1"
    port: int = 18888
    token: str = ""
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset, repr=False, compare=False
    )


@dataclass
class BackgroundTaskConfig:
    """Durable background worker timing and concurrency limits."""

    lease_seconds: float = 30.0
    heartbeat_interval: float = 10.0
    maintenance_interval: float = 10.0
    max_concurrency: int = 8
    per_team_concurrency: int = 4
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    result_retention_days: float = 30.0
    result_gc_interval: float = 3600.0
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset, repr=False, compare=False
    )


@dataclass
class AppConfig:
    providers: list[ProviderConfig]
    permission_mode: str = "default"
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    raw_hooks: list[dict] = field(default_factory=list)
    enable_fork: bool = False
    enable_verification_agent: bool = False
    worktree: WorktreeConfig = field(default_factory=WorktreeConfig)
    teammate_mode: str = ""
    enable_coordinator_mode: bool = False
    sandbox: SandboxAppConfig = field(default_factory=SandboxAppConfig)
    remote: RemoteAppConfig = field(default_factory=RemoteAppConfig)
    background_tasks: BackgroundTaskConfig = field(
        default_factory=BackgroundTaskConfig
    )
    _specified_fields: frozenset[str] = field(
        default_factory=frozenset, repr=False, compare=False
    )


def _build_app_config(validated: dict, env: Mapping[str, str]) -> AppConfig:
    providers = []
    for p in validated["providers"]:
        api_key = p["api_key"]
        if not api_key:
            api_key = env.get(_ENV_KEY_MAP.get(p["protocol"], ""), "")
        providers.append(
            ProviderConfig(
                name=p["name"],
                protocol=p["protocol"],
                base_url=p["base_url"],
                model=p["model"],
                api_key=api_key,
                thinking=p["thinking"],
                context_window=p["context_window"],
                max_output_tokens=p["max_output_tokens"],
            )
        )

    mcp_servers = [
        MCPServerConfig(
            name=s["name"],
            command=s["command"],
            args=s["args"],
            url=s["url"],
            headers=s["headers"],
            env=s["env"],
            connect_timeout=s["connect_timeout"],
            request_timeout=s["request_timeout"],
            max_retries=s["max_retries"],
            retry_delay=s["retry_delay"],
        )
        for s in validated["mcp_servers"]
    ]

    wt = validated["worktree"]
    worktree_cfg = WorktreeConfig(
        symlink_directories=wt["symlink_directories"],
        stale_cleanup_interval=wt["stale_cleanup_interval"],
        stale_cutoff_hours=wt["stale_cutoff_hours"],
    )

    sb = validated["sandbox"]
    sandbox_cfg = SandboxAppConfig(
        enabled=sb["enabled"],
        auto_allow=sb["auto_allow"],
        network_enabled=sb["network_enabled"],
    )
    remote_data = validated["remote"]
    remote_cfg = RemoteAppConfig(
        host=remote_data["host"],
        port=remote_data["port"],
        token=remote_data["token"],
    )
    task_data = validated["background_tasks"]
    background_task_cfg = BackgroundTaskConfig(
        lease_seconds=task_data["lease_seconds"],
        heartbeat_interval=task_data["heartbeat_interval"],
        maintenance_interval=task_data["maintenance_interval"],
        max_concurrency=task_data["max_concurrency"],
        per_team_concurrency=task_data["per_team_concurrency"],
        retry_base_seconds=task_data["retry_base_seconds"],
        retry_max_seconds=task_data["retry_max_seconds"],
        result_retention_days=task_data["result_retention_days"],
        result_gc_interval=task_data["result_gc_interval"],
    )

    return AppConfig(
        providers=providers,
        permission_mode=validated["permission_mode"],
        mcp_servers=mcp_servers,
        raw_hooks=validated["hooks"],
        enable_fork=validated["enable_fork"],
        enable_verification_agent=validated["enable_verification_agent"],
        worktree=worktree_cfg,
        teammate_mode=validated["teammate_mode"],
        enable_coordinator_mode=validated["enable_coordinator_mode"],
        sandbox=sandbox_cfg,
        remote=remote_cfg,
        background_tasks=background_task_cfg,
    )


def _load_single_file(path: Path, env: Mapping[str, str] | None = None) -> AppConfig:
    effective_env = os.environ if env is None else env
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config {path}: {e}") from e

    raw = _resolve_nested_env(raw, effective_env)
    validated = validate_config_structure(raw)
    config = _build_app_config(validated, effective_env)
    assert isinstance(raw, dict)
    config._specified_fields = frozenset(raw)
    raw_sandbox = raw.get("sandbox")
    if isinstance(raw_sandbox, dict):
        config.sandbox._specified_fields = frozenset(raw_sandbox)
    raw_remote = raw.get("remote")
    if isinstance(raw_remote, dict):
        config.remote._specified_fields = frozenset(raw_remote)
    raw_background_tasks = raw.get("background_tasks")
    if isinstance(raw_background_tasks, dict):
        config.background_tasks._specified_fields = frozenset(
            name
            for name in raw_background_tasks
            if hasattr(config.background_tasks, name)
        )
    return config


def _merge_config(base: AppConfig, override: AppConfig) -> AppConfig:
    specified = override._specified_fields
    if override.providers:
        base.providers = override.providers
    if "permission_mode" in specified:
        base.permission_mode = override.permission_mode

    if override.mcp_servers:
        by_name = {s.name: i for i, s in enumerate(base.mcp_servers)}
        for s in override.mcp_servers:
            if s.name in by_name:
                base.mcp_servers[by_name[s.name]] = s
            else:
                base.mcp_servers.append(s)
                by_name[s.name] = len(base.mcp_servers) - 1

    base.raw_hooks.extend(override.raw_hooks)
    if "enable_fork" in specified:
        base.enable_fork = override.enable_fork
    if "enable_verification_agent" in specified:
        base.enable_verification_agent = override.enable_verification_agent
    if "teammate_mode" in specified:
        base.teammate_mode = override.teammate_mode
    if "enable_coordinator_mode" in specified:
        base.enable_coordinator_mode = override.enable_coordinator_mode

    sandbox_fields = override.sandbox._specified_fields
    if "enabled" in sandbox_fields:
        base.sandbox.enabled = override.sandbox.enabled
    if "auto_allow" in sandbox_fields:
        base.sandbox.auto_allow = override.sandbox.auto_allow
    if "network_enabled" in sandbox_fields:
        base.sandbox.network_enabled = override.sandbox.network_enabled

    remote_fields = override.remote._specified_fields
    if "host" in remote_fields:
        base.remote.host = override.remote.host
    if "port" in remote_fields:
        base.remote.port = override.remote.port
    if "token" in remote_fields:
        base.remote.token = override.remote.token

    task_fields = override.background_tasks._specified_fields
    for field_name in task_fields:
        setattr(
            base.background_tasks,
            field_name,
            getattr(override.background_tasks, field_name),
        )
    return base


def _parse_env_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be true/false, yes/no, on/off, or 1/0")


def _parse_env_int(value: str, key: str) -> int:
    try:
        parsed = int(value)
    except ValueError as e:
        raise ConfigError(f"{key} must be a non-negative integer") from e
    if parsed < 0:
        raise ConfigError(f"{key} must be a non-negative integer")
    return parsed


def _parse_env_port(value: str, key: str) -> int:
    port = _parse_env_int(value, key)
    if not 1 <= port <= 65535:
        raise ConfigError(f"{key} must be an integer between 1 and 65535")
    return port


def _provider_from_env(env: Mapping[str, str]) -> ProviderConfig | None:
    required = ("VALECODE_PROTOCOL", "VALECODE_BASE_URL", "VALECODE_MODEL")
    present = [key for key in required if env.get(key)]
    if not present:
        return None
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ConfigError(
            "Incomplete .env provider configuration; missing: " + ", ".join(missing)
        )

    protocol = env["VALECODE_PROTOCOL"]
    api_key = env.get("VALECODE_API_KEY", "")
    if not api_key:
        api_key = env.get(_ENV_KEY_MAP.get(protocol, ""), "")
    raw_provider = {
        "name": env.get("VALECODE_PROVIDER_NAME", "default"),
        "protocol": protocol,
        "base_url": env["VALECODE_BASE_URL"],
        "model": env["VALECODE_MODEL"],
        "api_key": api_key,
        "thinking": _parse_env_bool(env.get("VALECODE_THINKING", "false"), "VALECODE_THINKING"),
        "context_window": _parse_env_int(
            env.get("VALECODE_CONTEXT_WINDOW", "0"), "VALECODE_CONTEXT_WINDOW"
        ),
        "max_output_tokens": _parse_env_int(
            env.get("VALECODE_MAX_OUTPUT_TOKENS", "0"), "VALECODE_MAX_OUTPUT_TOKENS"
        ),
    }
    validated = validate_config_structure({"providers": [raw_provider]})
    return _build_app_config(validated, env).providers[0]


def _apply_env_overrides(config: AppConfig, env: Mapping[str, str]) -> AppConfig:
    provider_keys = {
        "VALECODE_PROVIDER_NAME",
        "VALECODE_PROTOCOL",
        "VALECODE_BASE_URL",
        "VALECODE_MODEL",
        "VALECODE_API_KEY",
        "VALECODE_THINKING",
        "VALECODE_CONTEXT_WINDOW",
        "VALECODE_MAX_OUTPUT_TOKENS",
    }
    if any(key in env for key in provider_keys):
        current = config.providers[0]
        protocol = env.get("VALECODE_PROTOCOL", current.protocol)
        api_key = env.get("VALECODE_API_KEY", current.api_key)
        if not api_key:
            api_key = env.get(_ENV_KEY_MAP.get(protocol, ""), "")
        raw_provider = {
            "name": env.get("VALECODE_PROVIDER_NAME", current.name),
            "protocol": protocol,
            "base_url": env.get("VALECODE_BASE_URL", current.base_url),
            "model": env.get("VALECODE_MODEL", current.model),
            "api_key": api_key,
            "thinking": (
                _parse_env_bool(env["VALECODE_THINKING"], "VALECODE_THINKING")
                if "VALECODE_THINKING" in env
                else current.thinking
            ),
            "context_window": (
                _parse_env_int(env["VALECODE_CONTEXT_WINDOW"], "VALECODE_CONTEXT_WINDOW")
                if "VALECODE_CONTEXT_WINDOW" in env
                else current.context_window
            ),
            "max_output_tokens": (
                _parse_env_int(
                    env["VALECODE_MAX_OUTPUT_TOKENS"], "VALECODE_MAX_OUTPUT_TOKENS"
                )
                if "VALECODE_MAX_OUTPUT_TOKENS" in env
                else current.max_output_tokens
            ),
        }
        validated = validate_config_structure({"providers": [raw_provider]})
        config.providers[0] = _build_app_config(validated, env).providers[0]

    if "VALECODE_PERMISSION_MODE" in env:
        mode = env["VALECODE_PERMISSION_MODE"]
        if mode not in VALID_PERMISSION_MODES:
            raise ConfigError(
                f"Invalid VALECODE_PERMISSION_MODE '{mode}', must be one of: "
                f"{', '.join(sorted(VALID_PERMISSION_MODES))}"
            )
        config.permission_mode = mode
    if "VALECODE_REMOTE_HOST" in env:
        host = env["VALECODE_REMOTE_HOST"].strip()
        if not host:
            raise ConfigError("VALECODE_REMOTE_HOST must be a non-empty string")
        config.remote.host = host
    if "VALECODE_REMOTE_PORT" in env:
        config.remote.port = _parse_env_port(
            env["VALECODE_REMOTE_PORT"], "VALECODE_REMOTE_PORT"
        )
    if "VALECODE_REMOTE_TOKEN" in env:
        config.remote.token = env["VALECODE_REMOTE_TOKEN"].strip()
    return config


def load_config(path: Path | None = None) -> AppConfig:
    home = Path.home()
    project_dir = path.parent.resolve() if path is not None else Path.cwd()
    env = _load_dotenv_environment(project_dir, home)

    if path is not None:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        return _apply_env_overrides(_load_single_file(path, env), env)

    cwd = project_dir
    candidates = [
        home / ".valecode" / "config.yaml",
        cwd / ".valecode" / "config.yaml",
        cwd / ".valecode" / "config.local.yaml",
    ]

    merged: AppConfig | None = None
    for p in candidates:
        if not p.exists():
            continue
        layer = _load_single_file(p, env)
        if merged is None:
            merged = layer
        else:
            merged = _merge_config(merged, layer)

    if merged is None:
        provider = _provider_from_env(env)
        if provider is not None:
            config = AppConfig(providers=[provider])
            return _apply_env_overrides(config, env)
        raise ConfigError(
            "No configuration found. Create .valecode/config.yaml, or define "
            "VALECODE_PROTOCOL, VALECODE_BASE_URL, and VALECODE_MODEL in .env"
        )
    return _apply_env_overrides(merged, env)
