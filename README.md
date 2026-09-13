# ValeCode

ValeCode 是一个面向真实软件工程任务的终端 AI 编程助手。它以 Python `asyncio`
构建 Agent Runtime，支持多模型协议、工具调用、MCP、Skills、Sub-Agent、Agent
Teams、Git Worktree、持久化任务调度和 OpenTelemetry 链路追踪。

项目重点不只是“调用模型完成一次任务”，还包括执行状态可追踪、进程中断后可恢复、
副作用可判断以及长任务可接管的工程化运行时。

## 核心能力

- **多模型接入**：支持 Anthropic、OpenAI Responses 和 OpenAI-Compatible 协议。
- **持久化执行控制面**：使用 SQLite 管理 Session、Run、Step、ToolCall、Task、
  Event、Checkpoint、Result Artifact 以及 Team/Member 状态。
- **安全恢复**：启动时扫描未完成执行，通过幂等键、文件状态检查和人工确认决定
  重试、复用或停止。
- **耐久后台任务**：支持依赖关系、lease、heartbeat、失败重试、worker 接管以及
  全局和 Team 级并发控制。
- **可组合工具系统**：Built-in、Plugin、MCP 和 Session 四层 Registry，配合
  Skills、Hooks、权限规则和资源生命周期管理。
- **多 Agent 协作**：支持 Sub-Agent、后台 Agent、Agent Team、结构化邮箱事件以及
  Git Worktree 隔离。
- **上下文与结果管理**：支持上下文压缩、可验证 Checkpoint、工具调用链对齐、
  超长结果卸载和引用感知清理。
- **可观测性**：模型、工具、权限、Hook、压缩、恢复和任务调度均可输出
  OpenTelemetry Trace，默认对内容与密钥脱敏。
- **跨平台安全执行**：macOS 使用 Seatbelt，Linux 使用 bubblewrap，Windows 使用
  WSL2 + bubblewrap；启用但不可用时命令执行会 fail closed。

## 架构概览

```text
Terminal UI / CLI / Remote
            |
            v
      Agent Runtime ---------------- OpenTelemetry
       |      |      |
       |      |      +-------------- Runtime Events
       |      |
       |      +--------------------- Layered Tool Registry
       |                              | Permissions / Hooks
       |                              | MCP / Skills / Tools
       |
       +---------------------------- Provider Adapters
            |
            +---- JSONL Session / Compact Checkpoint
            +---- SQLite Control Plane
                  Run / Step / ToolCall / Task / Event
                  Artifact / Team / Member
```

JSONL 保存完整对话，SQLite 保存可查询的执行状态和恢复索引。状态转换与对应事件在
同一事务中提交；超长工具结果写入 Session/Run 隔离目录，并记录哈希、引用状态和
清理终态。

## 快速开始

### 环境要求

- Python 3.11+
- 推荐使用 [uv](https://docs.astral.sh/uv/)
- 至少一个可用的模型 API

### 安装

```bash
git clone https://github.com/Humble666661/ValeCode.git
cd ValeCode
uv sync
```

### 配置模型

复制环境变量模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

填写协议、模型和 API Key：

```dotenv
VALECODE_PROVIDER_NAME=my-provider
VALECODE_PROTOCOL=anthropic
VALECODE_BASE_URL=https://api.anthropic.com
VALECODE_MODEL=your-model-id
ANTHROPIC_API_KEY=your-api-key
```

支持的协议值为 `anthropic`、`openai` 和 `openai-compat`。请勿提交包含真实密钥的
`.env`、`.env.local` 或 `.valecode/config.local.yaml`。

### 启动

交互式终端界面：

```bash
uv run valecode
```

执行单次任务：

```bash
uv run valecode -p "分析当前项目并给出测试建议"
```

输出适合程序消费的 NDJSON 事件流：

```bash
uv run valecode -p "运行测试并总结失败原因" --output-format stream-json
```

远程模式：

```bash
uv run valecode --remote
```

远程模式启动 WebSocket 服务和浏览器界面，安全默认值为
`127.0.0.1:18888`。只在本机使用时无需 Token；浏览器会在启用鉴权时提示输入，
并仅把它保存在当前标签页的 `sessionStorage` 中。

需要从局域网访问时，在 `.env.local` 或系统环境变量中显式配置：

```dotenv
VALECODE_REMOTE_HOST=0.0.0.0
VALECODE_REMOTE_PORT=18888
VALECODE_REMOTE_TOKEN=replace-with-a-long-random-token
```

非回环地址未配置 Token 时，ValeCode 会拒绝启动。WebSocket 客户端可通过
`Authorization: Bearer <token>` 或 `?token=<token>` 完成认证。跨主机部署仍应放在
启用 HTTPS/WSS 的受控反向代理之后；ValeCode 本身不终止 TLS。不要把真实 Token
写入 YAML 或提交到版本库。

## 配置层级

ValeCode 同时支持 `.env` 和 YAML 配置。

环境变量从低到高依次覆盖：

1. `~/.valecode/.env`
2. 项目 `.env`
3. 项目 `.env.local`
4. 进程或系统环境变量

YAML 配置从低到高依次合并：

1. `~/.valecode/config.yaml`
2. 项目 `.valecode/config.yaml`
3. 项目 `.valecode/config.local.yaml`

单 Provider 场景可只使用 `.env`；多 Provider、MCP、Hooks、Worktree、Sandbox 和
Remote host/port 等配置适合放在 YAML 中。Remote Token 等秘密应放在 `.env.local`
或系统环境变量。YAML 字符串支持 `${ENV_NAME}` 引用环境变量。

## 常用交互命令

- `/help`：查看所有可用命令。
- `/session`：列出、恢复、新建或删除会话。
- `/tasks`：查看和管理后台任务。
- `/trace`：查看 Agent 父子追踪树。
- `/permission`：查看或切换权限模式。
- `/sandbox`：查看沙箱状态。
- `/worktree`：管理 Git Worktree 会话。
- `/mcp`、`/skill`：查看扩展能力。
- `/compact`：主动压缩当前上下文。

项目和用户还可以通过 `.valecode/commands/` 添加 Markdown 自定义命令。

## 恢复与副作用语义

ValeCode 在启动时扫描遗留 Run，并将未完成状态收敛为可恢复状态：

- 已提交结果的工具调用按稳定幂等键复用；
- 未开始或只读调用可以安全重试；
- 文件写入会对照目标文件的实际内容；
- 无法从本地状态确认的外部副作用标记为 `uncertain`，等待人工决策。

这套机制明确区分可重试操作与外部副作用，不以本地数据库对 Shell、MCP 或网络
请求作不可靠的严格 exactly-once 承诺。

## OpenTelemetry

启用本地 JSONL Trace：

```dotenv
VALECODE_OTEL_ENABLED=true
VALECODE_OTEL_EXPORTER=file
VALECODE_OTEL_FILE=.valecode/traces.jsonl
VALECODE_OTEL_SERVICE_NAME=valecode
```

Exporter 也可设置为 `console` 或 `otlp`。使用 OTLP 时，通过
`VALECODE_OTEL_ENDPOINT` 指定 HTTP endpoint。Prompt、工具参数和输出默认脱敏；
密钥字段始终脱敏。

## OS 级沙箱

在 `.valecode/config.yaml` 中启用：

```yaml
sandbox:
  enabled: true
  auto_allow: false
  network_enabled: false
```

ValeCode 会实际探测沙箱 namespace 是否可用。探测失败时不会静默退回宿主 Shell；
`auto_allow` 仅在 OS 沙箱成功附加后对 Bash 生效。Windows 环境需要默认 WSL2
发行版，并在该发行版中安装 `bubblewrap`。

## 开发与测试

```bash
uv sync --group dev
uv run pytest -q
```

当前回归基线为 **676 passed, 1 skipped**。测试覆盖数据库迁移与状态机、崩溃恢复、
任务 lease 与接管、事件一致性、模型重试、循环熔断、工具 Registry、权限与 Skills、
Hooks、Worktree 边界、沙箱以及 Trace 传播。

仓库级开发约定见 [VALECODE.md](VALECODE.md)。
