# VelaCode

VelaCode 是一个基于 Python `asyncio` 的终端 AI 编程助手。它支持 Anthropic、OpenAI 与 OpenAI-Compatible 模型协议，并提供工具调用、MCP、Skills、权限控制、上下文压缩、Sub-Agent、Agent Teams、Git Worktree、持久化任务调度和 OpenTelemetry 链路追踪。

## 架构概览

```text
Terminal UI / CLI / Remote
           |
           v
       Agent Runtime ---- OpenTelemetry
        |    |    |
        |    |    +---- SQLite control plane
        |    |          (Run / Step / ToolCall / Task / Event)
        |    |
        |    +--------- Layered Tool Registry
        |               (Built-in < Plugin < MCP < Session)
        |
        +-------------- Provider adapters
                        (Anthropic / OpenAI / Compatible)
```

完整对话保存在 JSONL 中；SQLite 负责执行状态、检查点、任务 lease、attempt 和事件。对于崩溃时状态不确定的外部副作用，VelaCode 不承诺严格 exactly-once，而是通过幂等键、文件状态检查和人工确认降低重复执行风险。

## 安装与启动

需要 Python 3.11 或更高版本。推荐使用 `uv`：

```bash
uv sync
```

复制环境变量模板并填写模型配置：

```bash
cp .env.example .env
```

Windows PowerShell 可使用：

```powershell
Copy-Item .env.example .env
```

启动交互界面：

```bash
uv run valecode
```

查看所有 CLI 参数：

```bash
uv run valecode --help
```

配置按用户级 `.env`、项目 `.env`、项目 `.env.local`、系统环境变量的顺序覆盖，系统环境变量优先级最高。不要提交真实 API Key。

## 恢复与可观测性

VelaCode 启动时会扫描未完成 Run，将遗留状态收敛为可判断的恢复状态。已提交的工具结果按幂等键复用；只读调用可以安全重试；写文件调用会核对实际文件状态；无法确认的外部副作用标记为 `uncertain`，等待确认后继续。

如需本地 Trace，在 `.env` 中启用：

```dotenv
VALECODE_OTEL_ENABLED=true
VALECODE_OTEL_EXPORTER=file
VALECODE_OTEL_FILE=.valecode/traces.jsonl
```

也可使用 `console` 或配置 OTLP/HTTP endpoint。Prompt、工具参数和输出默认脱敏。

## OS 级沙箱

可在 `.valecode/config.yaml` 启用：

```yaml
sandbox:
  enabled: true
  auto_allow: false
  network_enabled: false
```

macOS 使用 Seatbelt，Linux 使用 bubblewrap，Windows 使用 WSL2 内的
bubblewrap。Windows 默认 WSL 发行版需预先安装 `bubblewrap`；VelaCode 会实际
探测 user/mount namespace，而不只检查命令是否存在。配置已启用但后端不可用时，
Bash 会 fail closed 且不会执行命令；`auto_allow` 也只在后端探测成功后生效。

## 测试

```bash
uv run pytest -q
```

测试覆盖数据库迁移与状态机、崩溃恢复、后台任务接管、模型重试、循环熔断、分层工具注册和 Trace 传播。

## 当前限制

- 外部 Shell、MCP 和网络副作用无法仅靠本地数据库严格保证 exactly-once。
- Windows 沙箱依赖可用的 WSL2 发行版与其中安装的 `bubblewrap`。
- 跨进程恢复依赖持久化控制面中已有的最后安全检查点。
