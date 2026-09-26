# ValeCode 开发指南

本文档定义 ValeCode 仓库的公开开发约定，适用于贡献者和在仓库中工作的 Coding
Agent。项目使用 Python 3.11+ 与 `asyncio`，目标是构建可恢复、可观测且具备明确
安全边界的终端 AI 编程运行时。

## 开发原则

1. **持久状态优先**：跨进程需要保留的执行状态必须进入 SQLite 或 Session JSONL，
   不能只依赖内存对象。
2. **恢复必须幂等**：启动扫描、任务接管和清理逻辑应允许重复执行，不得重复提交
   已确认完成的副作用。
3. **安全边界不可降级**：权限检查、路径限制或 OS 沙箱不可用时，应拒绝危险操作，
   不得静默切换为不受控执行。
4. **事件与状态一致**：Run、Step、ToolCall 和 Task 的状态转换，应与对应 Runtime
   Event 在同一事务或同一明确提交边界中完成。
5. **默认保护内容**：日志和 Trace 默认不采集 Prompt、工具参数、输出与密钥；任何
   新增可观测字段都必须通过脱敏层。
6. **Remote 默认仅本机可见**：默认绑定 `127.0.0.1`；绑定非回环地址必须配置
   `VALECODE_REMOTE_TOKEN`，公网入口的 TLS 由受控反向代理终止。
7. **兼容现有数据**：数据库和 JSONL 格式采用向前迁移，不修改或破坏用户已有记录。

## 主要模块

```text
valecode/
├── agent.py          # 模型—工具执行循环
├── client.py         # Provider 适配与流式输出
├── persistence/      # SQLite Schema、迁移与 Store
├── runtime/          # 事件、重试、取消、幂等与恢复
├── tools/            # 工具抽象、Registry 与内置工具
├── permissions/      # 权限模式、规则与路径安全
├── sandbox/          # Seatbelt / bubblewrap / WSL2 后端
├── agents/           # Sub-Agent 与耐久后台任务
├── teams/            # Agent Team、邮箱和成员状态
├── worktree/         # Git Worktree 生命周期
├── context/          # 上下文预算与压缩
├── memory/           # Session JSONL 与恢复附件
├── observability/    # OpenTelemetry 配置与 Exporter
├── hooks/            # 生命周期 Hook
├── skills/           # Skill 加载与权限作用域
└── commands/         # 内置及自定义 Slash Command
```

## 本地开发

安装运行和开发依赖：

```bash
uv sync --group dev
```

准备本地配置：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

运行交互界面和非交互任务：

```bash
uv run valecode
uv run valecode -p "检查当前改动"
```

## 验证要求

提交前至少执行与修改范围直接相关的测试。影响 Runtime、持久化、恢复、权限、沙箱、
任务调度或 Worktree 的修改，应执行全量回归：

```bash
uv run pytest -q
```

当前完整基线为 `967 passed, 1 skipped`。新增行为必须包含正常路径和至少一个失败、
取消、重启或边界场景测试。

## Python 代码规范

- 使用 4 空格缩进和 UTF-8；公共类型与非显然返回值添加类型标注。
- 模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用
  `UPPER_SNAKE_CASE`。
- 异步 I/O 使用 `async`/`await`，不得在 Agent 主循环中加入阻塞调用。
- 优先使用小型、职责单一的 Store、Service 和数据类，避免跨层直接操作内部状态。
- 捕获异常时保留可诊断的错误类型，但不得把 API Key、Token、Prompt 或工具输出
  写入默认日志。
- 跨平台路径使用 `pathlib.Path` 和已有路径边界工具，不假定 `/tmp`、盘符大小写或
  POSIX `commonpath` 行为。

## 持久化与迁移

- 所有 Schema 变更追加到 `valecode/persistence/migrations.py`，迁移版本只能递增。
- 不修改已经发布迁移的语义；需要调整时新增迁移。
- Store 写操作使用 `Database.transaction()`，查询使用 `Database.reader()`。
- 状态机转换需要验证来源状态，并通过版本字段或原子条件避免并发覆盖。
- 恢复逻辑以 JSONL 对话为内容权威源、SQLite 为执行状态与索引来源；发现不一致时
  应记录恢复事件并采取可重复的修复策略。
- 任何文件删除都必须先验证规范化后的绝对路径位于 ValeCode 管理目录内。

## 工具、权限与副作用

- 新工具继承统一 `Tool` 抽象，声明稳定名称、参数模型和副作用类别。
- 工具必须通过 `ToolRegistry` 注册，不绕过 pre/post/error/close 生命周期。
- `read` 操作可采用自动重试；`write` 操作恢复前必须检查目标状态；Shell、MCP 和
  网络请求等外部副作用无法确认时必须进入 `uncertain` 或请求用户确认。
- Skills 只能收紧或声明自身所需权限，不能绕过用户/项目规则、危险命令检测和路径
  边界。
- 增加自动放行逻辑时，必须证明操作确实由可用的 OS 沙箱包裹。

## 并发、取消与任务

- LLM、工具和子任务使用统一取消令牌与超时策略。
- 子 Agent 继承父级 ExecutionController，避免绕过全局工具并发额度。
- 后台 Task 的 claim、lease、heartbeat、retry 和完成提交必须可由另一进程安全接管。
- 创建异步任务时明确其所有者，并在正常退出、失败和取消路径回收资源。
- MCP transport 与 `ClientSession` 的进入、请求和退出由同一个 owner task 执行；
  manager 持有共享连接，工具包装的 `close()` 只停止该包装的使用。重连保留 client
  对象身份；请求失败后不得自动重放可能产生外部副作用的调用。

## 文档与配置

- `README.md` 面向使用者，保持安装、配置、能力和示例与当前代码一致。
- `VALECODE.md` 面向贡献者，记录稳定的仓库级工程约束。
- `.env.example` 只提供占位值，不得包含真实凭据。
- 尊重 `.gitignore`；不要使用 `git add -f` 提交本地配置、调试输出或规划记录。

## Git 提交

- Commit message 使用英文，并按一个可独立验证的功能组织提交。
- 推荐格式：`feat(scope): ...`、`fix(scope): ...`、`test(scope): ...`、
  `docs(scope): ...`、`chore(scope): ...`。
- 不提交 `.env`、`.env.local`、`.valecode/`、虚拟环境、构建产物或真实运行数据。
- 不覆盖无关的用户改动；提交前确认 `git diff --check` 和 `git status`。
