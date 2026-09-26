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
  全局和 Team 级并发控制；`TaskDispatch` 可把共享看板任务显式链接到执行实例。
- **可组合工具系统**：Built-in、Plugin、MCP 和 Session 四层 Registry，配合
  Skills、Hooks、权限规则和资源生命周期管理。
- **多 Agent 协作**：支持 Sub-Agent、后台 Agent、Agent Team、结构化邮箱事件以及
  Git Worktree 隔离；共享任务板支持依赖、优先级与百分比进度，消息仅在所属 Team 内路由；
  TUI 可读取队友的实时工具、Token 与状态进度，Team 删除会收口实际运行中的进程内任务。
  默认队友后端为 `in-process`；本地交互会话可显式配置 `teammate_mode: tmux`
  或 `iterm2` 启动独立成员进程。tmux 需要 POSIX 环境和已安装的 tmux；iTerm2
  需要 macOS、当前 iTerm.app、同一 Python 环境安装 `iterm2` 并启用
  [Python API](https://iterm2.com/python-api/tutorial/running.html)。原生 Windows 使用进程内模式。
  独立成员使用隔离 worktree、父 Session/Run 与共享邮箱；启动描述不含模型密钥。
  不继承一次性/会话授权，不提升父权限，待人工批准的操作会拒绝执行。父进程退出或
  心跳失联会停止成员，不自动重放；退出不会删除成员 worktree。pane 模式需指定
  `subagent_type`（不支持对话 fork），配置须能从父项目标准 `.env`/YAML 加载。
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
- `/trace`：查看 Agent 父子追踪树（包含当前会话重启前的持久记录）。
- `/permission`：查看或切换权限模式。
- `/sandbox`：查看沙箱状态。
- `/worktree`：管理 Git Worktree 会话。
- `/mcp`、`/skill`：查看扩展能力。
- `/compact`：主动压缩当前上下文。

项目和用户还可以通过 `.valecode/commands/` 添加 Markdown 自定义命令。

## MCP 工具、资源与 Prompt

在 `.valecode/config.yaml` 配置 stdio 或 Streamable HTTP 服务器，例如：

```yaml
mcp_servers:
  - name: local
    command: python
    args: ["path/to/mcp_server.py"]
    connect_timeout: 15
    request_timeout: 60
    max_retries: 2
    retry_delay: 0.5
  - name: remote
    url: "https://your-server.example/mcp"
    headers:
      Authorization: "Bearer ${MCP_TOKEN}"
```

时间单位为秒，上述数值也是默认值；`max_retries` 是首次连接失败后的额外尝试次数，
设为 `0` 可关闭连接重试。连接/初始化失败会释放本次 transport，单台服务器失败
不会阻止其他服务器。请求超时包含已连接服务器上的排队时间；工具调用失败或超时
会返回错误，运行时不会自动重放可能产生副作用的调用。

远端工具以 `mcp_<server>_<tool>` 注册，初始通过 `ToolSearch` 延迟发现。服务器声明
对应能力时，还会注册 `mcp_<server>_resources` 和 `mcp_<server>_prompts`：

- Resources 支持 `action: list`、`templates` 和 `read`；读取时传入 `uri`。
- Prompts 支持 `action: list` 和 `get`；获取时传入 `name` 与字符串字典 `arguments`。

只有资源/Prompt、没有远端工具的服务器也可使用。目录列表支持分页，重复游标会明确
报错；二进制资源只显示 URI/MIME 摘要。目录工具名与远端工具重名时会添加数字后缀，
可通过 `/mcp` 查看实际名称。资源与 Prompt 沿用 MCP 工具的权限、Hooks 和结果预算，
获取到的 Prompt 消息作为工具结果返回。

## 定时 Agent 任务

`CronCreate` 可为当前会话创建一次性、固定间隔或五段 Cron 计划，`CronList` 查看计划
和最近执行，`CronUpdate` 暂停/恢复，`CronDelete` 删除。TUI/Remote 也支持
`/cron list`、`/cron pause <id>`、`/cron resume <id>`、`/cron delete <id>`。

例如，向 ValeCode 明确要求“每小时用 Explore 检查项目状态，时区 Asia/Shanghai”，
由模型通过 `CronCreate` 发起并经过正常权限检查。工具参数示例：

```json
{
  "name": "项目状态检查",
  "prompt": "只读检查项目状态并报告异常，不修改文件。",
  "subagent_type": "Explore",
  "schedule_type": "cron",
  "schedule_spec": {"cron": "0 * * * *"},
  "timezone": "Asia/Shanghai"
}
```

`interval` 使用 `{"every_seconds": 3600}`（最低 60 秒）；`once` 使用 ISO 时间
`{"run_at": "2026-10-01T09:00:00+08:00"}`。时区使用 IANA 名称，默认 UTC。

计划与执行实例在 SQLite 同一事务内落盘，多个进程不会重复接纳同一触发；同一计划
不重叠执行，错过的周期合并一次，不连续补发历史。**只在 ValeCode 运行且对应 Session
打开时触发**，不是关机后仍运行的系统服务。恢复计划需恢复原 Session；单次 `-p` 退出
也会停掉计时器。每 Session 最多 50 个活动计划。

后台实例使用默认权限并保留当前项目规则，不继承父会话的一次性授权或 bypass 模式；
需交互审批的操作不会无人值守放行。暂停/删除取消尚未领取的实例，运行中实例继续，
需要终止时使用 `/tasks cancel <task-id>`。只恢复尚未开始的持久实例；已开始的定时
任务中断后不自动从头重放，避免重复外部副作用。历史执行保留用于审计。

## 记忆召回

记忆保存在用户或项目的 Markdown 文件中；大目录使用可重建的 SQLite FTS5 缓存缩小
候选清单，最后由独立模型查询选择最多 5 条。TUI 按 Session 记录实际展示过的记忆，
恢复会话后继续去重。召回总预算为 8 秒，失败不会阻止主对话。

可选的语义向量检索默认关闭。启用后，**用户查询和记忆正文会发送到单独选择的
embedding 服务**，不会默认使用聊天 Provider。支持 OpenAI-compatible `/embeddings`：

```dotenv
VALECODE_MEMORY_ENABLED=true
VALECODE_MEMORY_BASE_URL=http://127.0.0.1:11434/v1
VALECODE_MEMORY_MODEL=your-installed-embedding-model
# VALECODE_MEMORY_API_KEY=your-key
VALECODE_MEMORY_TIMEOUT_SECONDS=2
```

也可在 YAML 的 `memory_search` 段使用 `enabled/base_url/model/api_key/timeout_seconds`，
密钥支持 `${ENV_VAR}`。非本机服务默认要求 HTTPS。超过 80 条未展示记忆时，将语义
余弦排序与 FTS5 排序融合，再补少量近期文件；远端失败、超时或缓存损坏退回词法检索。
向量缓存位于 `.valecode/cache/`，按内容摘要及服务/模型身份增量更新，不替代 Markdown。
冷缓存分批建立，语义阶段最多 4 秒；已完成批次保留供后续复用。

## Skills

项目 Skill 放在 `.valecode/skills/`，用户级 Skill 放在 `~/.valecode/skills/`；支持
单文件 Markdown、目录 `SKILL.md`，以及 `skill.yaml + prompt.md`。Skill 可以由模型
通过 `LoadSkill` 按需加载，也可以直接使用同名 `/<skill> [args]` 命令。inline 模式
会把完整正文作为一次性上下文注入当前对话；fork 模式在独立上下文执行，并可通过
frontmatter 的 `model` 字段切换到当前 Provider 上的其他模型或 `haiku`、`sonnet`、
`opus` 别名。目录 Skill 激活时会向模型提供基准目录和有限的支持文件清单，正文中
引用的 `scripts/`、`references/` 等资源仍通过标准文件/Shell 工具使用。Skill 声明
的工具权限仍受项目权限、危险命令检测和沙箱边界约束。

ValeCode 仅内置一个窄范围的 `customize-valecode` Skill，用于修改 ValeCode 自身配置
和扩展；不会为普通代码任务自动套用通用 commit/review 流程。项目或用户同名 Skill
可以按现有优先级覆盖它。

## Python 工具插件

已安装的 Python 包可以通过 `valecode.tools` entry-point 组贡献工具。入口值可以是
`Tool` 实例、无参 `Tool` 子类，或返回一个/多个 `Tool` 的同步工厂。例如插件包的
`pyproject.toml`：

```toml
[project.entry-points."valecode.tools"]
my_tools = "my_valecode_plugin:create_tools"
```

插件工具以 Plugin 层注册，可覆盖同名 Built-in 工具，但仍经过 ValeCode 的权限、
Hooks、超时和输出预算路径；退出时会调用工具的 `close()`。加载错误会单独记录，
不会阻止其他插件或 ValeCode 启动。Python 插件在当前进程中运行，拥有与 ValeCode
相同的系统权限，因此只应安装和启用可信插件；需要进程隔离的外部工具应优先使用
MCP。延迟工具可以通过 `search_terms = ("alias", "中文别名")` 声明额外检索词。

同一插件包也可以通过 `valecode.agents` entry-point 提供包含 Markdown Agent 定义
的目录。入口值可以是目录路径，或返回一个/多个目录路径的同步工厂：

```toml
[project.entry-points."valecode.agents"]
my_agents = "my_valecode_plugin:agent_directories"
```

插件 Agent 沿用 `.valecode/agents/*.md` 的格式，优先级低于项目、用户和内置 Agent；
不同插件按 entry-point 名称确定性加载，单个插件失败不会阻止其他插件启动。Agent
插件同样在当前进程中加载，仅应安装可信包。

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

## 后台任务配置

后台子 Agent 使用 SQLite lease 支持重试和故障回收。可在 `.valecode/config.yaml`
中调整 worker 参数；未配置时使用以下默认值：

```yaml
background_tasks:
  lease_seconds: 30
  heartbeat_interval: 10
  maintenance_interval: 10
  max_concurrency: 8
  per_team_concurrency: 4
  retry_base_seconds: 1
  retry_max_seconds: 30
  result_retention_days: 30
  result_gc_interval: 3600
```

`heartbeat_interval` 在运行时不会超过 lease 时长的一半；重试最大间隔不能小于
基础间隔。大结果的完整文件默认保留 30 天，过期后删除完整文件但保留数据库中的
截断摘要。TUI、`-p` 和 Remote 使用同一组配置。

普通“定义型”后台子 Agent 会额外保存不含凭据的重建描述。进程异常退出或正常关闭
后，先使用 `/session resume <id>` 恢复原会话；ValeCode 会在历史消息恢复完成后自动
领取该会话中可安全重建的 queued 任务。正常关闭不会消耗任务原有的失败重试预算，
用户通过 `/tasks cancel <id>` 主动取消则是终态，不会再次领取。

对话 fork、Agent Team、Worktree 隔离任务、旧版本任务以及损坏的重建描述不会自动
续跑，因为它们缺少可验证的完整运行现场。这些记录仍可通过 `/tasks` 查看或取消；
ValeCode 不会使用主 Agent 或其他会话上下文盲目代跑。恢复后的工具集合仍以当前
Registry 为能力上限，并继续经过权限、危险命令和沙箱检查。

## 开发与测试

```bash
uv sync --group dev
uv run pytest -q
```

当前回归基线为 **967 passed, 1 skipped**。测试覆盖数据库迁移与状态机、崩溃恢复、
任务 lease 与接管、事件一致性、模型重试、循环熔断、工具 Registry、权限与 Skills、
Hooks、Worktree 边界、沙箱以及 Trace 传播。

仓库级开发约定见 [VALECODE.md](VALECODE.md)。

## 许可证

ValeCode 采用 [MIT License](LICENSE) 开源。
