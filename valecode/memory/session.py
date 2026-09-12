from __future__ import annotations

import json
import logging
import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import IO, Any

from valecode.conversation import (
    ConversationManager,
    Message,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    message_tail_id,
)
from valecode.observability import get_tracing
from valecode.persistence import (
    CheckpointStore,
    Database,
    RunStore,
    ResultArtifactStore,
    SessionStore,
    TaskStore,
    TeamStore,
)

log = logging.getLogger(__name__)

SESSIONS_DIR = ".valecode/sessions"
DEFAULT_MAX_AGE_DAYS = 30
TITLE_MAX_LENGTH = 50

SESSION_SUMMARY_PROMPT = (
    "你是一个对话摘要助手。请根据下面的对话内容，用一句话总结这个会话的主要内容。"
    "只输出摘要文本，不要加任何前缀或标点符号外的修饰。不要调用任何工具。"
)


# ---------------------------------------------------------------------------
# RecordType & SessionRecord
# ---------------------------------------------------------------------------


class RecordType(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    COMPRESSION = "compression"
    # Layer-2 compact 标记。auto_compact 压缩对话记录时写入。
    # 内容为结构化载荷（参见 make_compact_boundary / parse_compact_boundary），
    # 包含摘要文本和原样保留的 keep 尾部（以序列化 record 形式内联），
    # 使 resume 可以仅凭此标记重建压缩后的状态，无需重放标记之前的原始前缀。
    COMPACT_BOUNDARY = "compact_boundary"


@dataclass
class SessionRecord:
    type: RecordType
    content: Any
    timestamp: datetime
    tool_use_id: str | None = None
    is_error: bool = False

    def to_jsonl(self) -> str:
        data: dict[str, Any] = {
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.tool_use_id is not None:
            data["tool_use_id"] = self.tool_use_id
        if self.type == RecordType.TOOL_RESULT:
            data["is_error"] = self.is_error
        return json.dumps(data, ensure_ascii=False)


    @classmethod
    def from_jsonl(cls, line: str) -> SessionRecord | None:
        try:
            data = json.loads(line)
            return cls(
                type=RecordType(data["type"]),
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                tool_use_id=data.get("tool_use_id"),
                is_error=data.get("is_error", False),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    @classmethod
    def from_message(cls, message: Message) -> list[SessionRecord]:
        now = datetime.now(timezone.utc)
        records: list[SessionRecord] = []

        if message.tool_results:
            for tr in message.tool_results:
                records.append(
                    cls(
                        type=RecordType.TOOL_RESULT,
                        content=tr.content,
                        timestamp=now,
                        tool_use_id=tr.tool_use_id,
                        is_error=tr.is_error,
                    )
                )
        elif message.role == "assistant":
            if message.tool_uses or message.thinking_blocks:
                content_blocks: list[dict[str, Any]] = []
                for thinking in message.thinking_blocks:
                    content_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": thinking.thinking,
                            "signature": thinking.signature,
                        }
                    )
                if message.content:
                    content_blocks.append({"type": "text", "text": message.content})
                for tu in message.tool_uses:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tu.tool_use_id,
                            "name": tu.tool_name,
                            "input": tu.arguments,
                        }
                    )
                records.append(
                    cls(type=RecordType.ASSISTANT, content=content_blocks, timestamp=now)
                )
            else:
                records.append(
                    cls(type=RecordType.ASSISTANT, content=message.content, timestamp=now)
                )
        else:
            records.append(
                cls(type=RecordType.USER, content=message.content, timestamp=now)
            )

        return records


# ---------------------------------------------------------------------------
# Compact boundary 载荷（摘要 + 内联的 keep 尾部）
# ---------------------------------------------------------------------------


def _message_to_record_dicts(message: Message) -> list[dict[str, Any]]:
    """将单条 Message 序列化为与磁盘存储格式一致的 record-dict 列表。

    复用 SessionRecord.from_message，使内联的 keep 尾部与正常追加消息的持久化
    结果逐字节一致（assistant 的 tool_uses 变为 content-blocks 列表，每个
    tool_result 独立成一条 record）。这保证了 tool_use↔tool_result 配对的
    无损往返——不像纯 role+content 文本导出那样会丢失 tool call 的关联关系。
    """
    dicts: list[dict[str, Any]] = []
    for rec in SessionRecord.from_message(message):
        data: dict[str, Any] = {"type": rec.type.value, "content": rec.content}
        if rec.tool_use_id is not None:
            data["tool_use_id"] = rec.tool_use_id
        if rec.type == RecordType.TOOL_RESULT:
            data["is_error"] = rec.is_error
        dicts.append(data)
    return dicts


@dataclass(frozen=True)
class CompactBoundaryData:
    summary: str = ""
    keep: list[Message] = field(default_factory=list)
    tail_id: str = ""
    attachment: str = ""
    transcript_path: str = ""
    checkpoint_id: str = ""
    run_id: str | None = None
    step_id: str | None = None
    version: int = 1
    integrity_valid: bool = False


def make_compact_boundary(
    summary: str,
    keep: list[Message],
    *,
    tail_id: str = "",
    attachment: str = "",
    transcript_path: str = "",
    checkpoint_id: str = "",
    run_id: str | None = None,
    step_id: str | None = None,
) -> SessionRecord:
    """构建一条 COMPACT_BOUNDARY record，内联摘要和原样保留的 keep 尾部。

    `keep` 是 auto_compact 原样保留的近期尾部消息。将其存储在 boundary record
    内部（而不是依赖它在文件中的物理位置），意味着 resume 可以仅凭 boundary
    重建压缩后的状态——boundary 之前的原始前缀保留在磁盘上但不会被重放。
    """
    keep_dicts: list[dict[str, Any]] = []
    for msg in keep:
        keep_dicts.extend(_message_to_record_dicts(msg))
    computed_tail_id = message_tail_id(keep)
    payload = {
        "version": 2,
        "checkpoint_id": checkpoint_id or f"checkpoint_{uuid.uuid4().hex}",
        "summary": summary,
        "keep": keep_dicts,
        "tail_id": tail_id or computed_tail_id,
        "attachment": attachment,
        "transcript_path": transcript_path,
        "run_id": run_id,
        "step_id": step_id,
    }
    return SessionRecord(
        type=RecordType.COMPACT_BOUNDARY,
        content=payload,
        timestamp=datetime.now(timezone.utc),
    )


def parse_compact_boundary_details(record: SessionRecord) -> CompactBoundaryData:
    """Parse and verify the complete compact-checkpoint payload."""
    content = record.content
    if not isinstance(content, dict):
        return CompactBoundaryData()
    summary = content.get("summary", "")
    if not isinstance(summary, str):
        return CompactBoundaryData()
    keep_raw = content.get("keep", [])
    if not isinstance(keep_raw, list):
        return CompactBoundaryData()
    keep_records: list[SessionRecord] = []
    for item in keep_raw:
        if not isinstance(item, dict) or "type" not in item:
            continue
        try:
            keep_records.append(
                SessionRecord(
                    type=RecordType(item["type"]),
                    content=item.get("content"),
                    timestamp=record.timestamp,
                    tool_use_id=item.get("tool_use_id"),
                    is_error=item.get("is_error", False),
                )
            )
        except ValueError:
            continue
    keep = records_to_messages(keep_records)
    stored_tail_id = content.get("tail_id", "")
    if not isinstance(stored_tail_id, str):
        stored_tail_id = ""
    # Version-1 boundaries had no digest and remain valid for compatibility.
    integrity_valid = not stored_tail_id or stored_tail_id == message_tail_id(keep)

    def _text(name: str) -> str:
        value = content.get(name, "")
        return value if isinstance(value, str) else ""

    version = content.get("version", 1)
    return CompactBoundaryData(
        summary=summary,
        keep=keep,
        tail_id=stored_tail_id,
        attachment=_text("attachment"),
        transcript_path=_text("transcript_path"),
        checkpoint_id=_text("checkpoint_id"),
        run_id=content.get("run_id") if isinstance(content.get("run_id"), str) else None,
        step_id=content.get("step_id") if isinstance(content.get("step_id"), str) else None,
        version=version if isinstance(version, int) else 1,
        integrity_valid=integrity_valid,
    )


def parse_compact_boundary(record: SessionRecord) -> tuple[str, list[Message]]:
    """make_compact_boundary 的逆操作：返回 (summary, keep_messages)。

    对遗留或格式异常的 payload 降级返回 ("", [])，确保单条损坏的 boundary
    不会导致 resume 崩溃。
    """
    details = parse_compact_boundary_details(record)
    if not details.integrity_valid:
        return "", []
    return details.summary, details.keep


# ---------------------------------------------------------------------------
# Record ↔ Message 转换
# ---------------------------------------------------------------------------


def records_to_messages(records: list[SessionRecord]) -> list[Message]:
    messages: list[Message] = []
    pending_tool_results: list[ToolResultBlock] = []

    for record in records:
        if record.type == RecordType.TOOL_RESULT:
            pending_tool_results.append(
                ToolResultBlock(
                    tool_use_id=record.tool_use_id or "",
                    content=(
                        record.content
                        if isinstance(record.content, str)
                        else json.dumps(record.content)
                    ),
                    is_error=record.is_error,
                )
            )
            continue

        if pending_tool_results:
            messages.append(
                Message(role="user", content="", tool_results=pending_tool_results)
            )
            pending_tool_results = []

        if record.type == RecordType.SYSTEM_PROMPT:
            continue

        if record.type == RecordType.COMPRESSION:
            messages.append(
                Message(
                    role="user",
                    content="本次会话延续自之前的对话，因上下文空间不足进行了压缩。以下是早期对话的摘要：\n\n" + (record.content or ""),
                )
            )
            continue

        if record.type == RecordType.COMPACT_BOUNDARY:
            # 内联展开：摘要作为 user 消息，后接原样保留的 keep 尾部。
            # resume() 通常已预裁剪到最后一个 boundary，所以这里只会处理
            # 权威的那一条；但在此展开可以保证 records_to_messages 对任何
            # 直接调用者都保持自洽。
            details = parse_compact_boundary_details(record)
            if not details.integrity_valid:
                continue
            content = "本次会话延续自之前的对话，因上下文空间不足进行了压缩。以下是早期对话的摘要：\n\n" + details.summary
            if details.keep:
                content += "\n\n近期消息已原样保留。"
            if details.transcript_path:
                content += (
                    "\n\n如果你需要压缩前的具体细节（代码片段、报错信息等），"
                    f"请用 ReadFile 读取完整会话记录：{details.transcript_path}"
                )
            if details.attachment:
                content += "\n\n---\n\n" + details.attachment
            messages.append(Message(role="user", content=content))
            messages.extend(details.keep)
            continue

        if record.type == RecordType.USER:
            messages.append(Message(role="user", content=record.content or ""))
        elif record.type == RecordType.ASSISTANT:
            if isinstance(record.content, list):
                text = ""
                tool_uses: list[ToolUseBlock] = []
                thinking_blocks: list[ThinkingBlock] = []
                for block in record.content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_uses.append(
                            ToolUseBlock(
                                tool_use_id=block.get("id", ""),
                                tool_name=block.get("name", ""),
                                arguments=block.get("input", {}),
                            )
                        )
                    elif block.get("type") == "thinking":
                        thinking_blocks.append(
                            ThinkingBlock(
                                thinking=block.get("thinking", ""),
                                signature=block.get("signature", ""),
                            )
                        )
                messages.append(
                    Message(
                        role="assistant",
                        content=text,
                        tool_uses=tool_uses,
                        thinking_blocks=thinking_blocks,
                    )
                )
            else:
                messages.append(
                    Message(role="assistant", content=record.content or "")
                )

    if pending_tool_results:
        messages.append(
            Message(role="user", content="", tool_results=pending_tool_results)
        )

    return messages


# ---------------------------------------------------------------------------
# 消息链校验
# ---------------------------------------------------------------------------


def validate_message_chain(records: list[SessionRecord]) -> int:
    last_valid = 0
    pending_tool_uses: set[str] = set()

    for i, record in enumerate(records):
        if record.type == RecordType.ASSISTANT and isinstance(record.content, list):
            for block in record.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    if tool_id:
                        pending_tool_uses.add(tool_id)

        if record.type == RecordType.TOOL_RESULT and record.tool_use_id:
            pending_tool_uses.discard(record.tool_use_id)

        if not pending_tool_uses:
            last_valid = i + 1

    return last_valid


# ---------------------------------------------------------------------------
# SessionMeta
# ---------------------------------------------------------------------------


@dataclass
class SessionMeta:
    id: str
    title: str = ""
    summary: str = ""
    message_count: int = 0
    total_tokens: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def save(self, path: Path) -> None:
        data = {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> SessionMeta | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                id=data["id"],
                title=data.get("title", ""),
                summary=data.get("summary", ""),
                message_count=data.get("message_count", 0),
                total_tokens=data.get("total_tokens", 0),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_active=datetime.fromisoformat(data["last_active"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Session（活跃会话句柄）
# ---------------------------------------------------------------------------


class Session:
    def __init__(
        self,
        session_id: str,
        file: IO[str],
        meta: SessionMeta,
        sessions_dir: Path,
        state_store: SessionStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
    ) -> None:
        self.session_id = session_id
        self._file = file
        self.meta = meta
        self._sessions_dir = sessions_dir
        self._state_store = state_store
        self._checkpoint_store = checkpoint_store

    def _sync_state(self) -> None:
        if self._state_store is None:
            return
        self._state_store.upsert(
            self.session_id,
            title=self.meta.title,
            summary=self.meta.summary,
            message_count=self.meta.message_count,
            total_tokens=self.meta.total_tokens,
            created_at=self.meta.created_at.isoformat(),
        )

    def append(self, message: Message) -> None:
        records = SessionRecord.from_message(message)
        for record in records:
            self._file.write(record.to_jsonl() + "\n")
        self._file.flush()

        self.meta.message_count += 1
        self.meta.last_active = datetime.now(timezone.utc)

        if not self.meta.title and message.role == "user" and message.content:
            self.meta.title = message.content[:TITLE_MAX_LENGTH]

        self.meta.save(self._sessions_dir / f"{self.session_id}.meta")
        self._sync_state()

    def append_record(self, record: SessionRecord) -> None:
        """追加一条原始 SessionRecord（例如 compact_boundary 标记）。

        与 append() 不同，此方法不会更新 message_count/title——boundary 是
        结构性标记而非对话轮次。last_active 仍会更新，以保证 session 按最近
        使用排序。
        """
        transcript_offset = self._file.tell()
        self._file.write(record.to_jsonl() + "\n")
        self._file.flush()
        self.meta.last_active = datetime.now(timezone.utc)
        self.meta.save(self._sessions_dir / f"{self.session_id}.meta")
        self._sync_state()
        if record.type == RecordType.COMPACT_BOUNDARY:
            self._index_checkpoint(record, transcript_offset)

    def _index_checkpoint(
        self, record: SessionRecord, transcript_offset: int
    ) -> None:
        if self._checkpoint_store is None:
            return
        details = parse_compact_boundary_details(record)
        if not details.integrity_valid or not details.checkpoint_id:
            return
        try:
            self._checkpoint_store.upsert(
                details.checkpoint_id,
                self.session_id,
                kind="compact",
                tail_id=details.tail_id,
                payload=record.content,
                transcript_offset=transcript_offset,
                run_id=details.run_id,
                step_id=details.step_id,
                created_at=record.timestamp.isoformat(),
            )
        except Exception:
            # JSONL is the source of truth.  A resume pass retries this index
            # write, avoiding loss of the durable conversation boundary.
            log.exception("Failed to index checkpoint %s", details.checkpoint_id)


    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        self._sync_state()


# ---------------------------------------------------------------------------
# ResumeResult
# ---------------------------------------------------------------------------


@dataclass
class ResumeResult:
    session: Session
    messages: list[Message]
    last_active: datetime
    checkpoint_id: str | None = None
    tail_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None


# ---------------------------------------------------------------------------
# Session 摘要生成
# ---------------------------------------------------------------------------


async def generate_session_summary(
    client: Any, conversation: ConversationManager, protocol: str
) -> str:
    from valecode.tools.base import StreamEnd, TextDelta

    recent = conversation.history[-10:]
    if not recent:
        return ""

    summary_conv = ConversationManager()
    summary_conv.history = [Message(role="user", content=SESSION_SUMMARY_PROMPT)]
    for msg in recent:
        summary_conv.history.append(msg)
    summary_conv.history.append(
        Message(role="user", content="请用一句话总结上面的对话内容。不要调用工具。")
    )

    collected = ""
    try:
        async for event in client.stream(
            summary_conv, system=SESSION_SUMMARY_PROMPT
        ):
            if isinstance(event, TextDelta):
                collected += event.text
            elif isinstance(event, StreamEnd):
                pass
    except Exception:
        return ""

    return collected.strip()


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


def _generate_session_id() -> str:
    now = datetime.now()
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"session_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"


class SessionManager:
    def __init__(self, work_dir: str) -> None:
        self._sessions_dir = Path(work_dir) / SESSIONS_DIR
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self.database = Database(self._sessions_dir.parent / "control.db")
        self.database.initialize()
        self.session_store = SessionStore(self.database)
        self.checkpoint_store = CheckpointStore(self.database)
        self.result_artifact_store = ResultArtifactStore(self.database)
        self.run_store = RunStore(self.database)
        self.task_store = TaskStore(self.database)
        self.team_store = TeamStore(self.database)
        self.recovered_tasks = self.task_store.recover_expired_leases()
        # Reconcile stale running state before a new Agent can start. The
        # report is retained so UI/CLI callers can surface confirmation needs.
        from valecode.runtime.recovery import RecoveryService

        self.recovery_report = RecoveryService(
            self.run_store, work_dir
        ).scan_and_reconcile()


    def create(self) -> Session:
        session_id = _generate_session_id()
        jsonl_path = self._sessions_dir / f"{session_id}.jsonl"
        meta = SessionMeta(id=session_id)
        meta.save(self._sessions_dir / f"{session_id}.meta")
        self.session_store.upsert(
            session_id,
            created_at=meta.created_at.isoformat(),
        )

        file = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
        return Session(
            session_id=session_id,
            file=file,
            meta=meta,
            sessions_dir=self._sessions_dir,
            state_store=self.session_store,
            checkpoint_store=self.checkpoint_store,
        )


    def list(self) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for meta_path in self._sessions_dir.glob("*.meta"):
            meta = SessionMeta.load(meta_path)
            if meta is not None:
                metas.append(meta)
        metas.sort(key=lambda m: m.last_active, reverse=True)
        return metas

    def resume(self, session_id: str) -> ResumeResult | None:
        with get_tracing().span(
            "session.resume", {"session.id": session_id}
        ) as span:
            result = self._resume(session_id)
            span.set_attributes(
                {
                    "session.resume.found": result is not None,
                    "session.resume.message_count": (
                        len(result.messages) if result is not None else 0
                    ),
                }
            )
            return result

    def _resume(self, session_id: str) -> ResumeResult | None:
        jsonl_path = self._sessions_dir / f"{session_id}.jsonl"
        meta_path = self._sessions_dir / f"{session_id}.meta"

        if not jsonl_path.exists():
            return None

        meta = SessionMeta.load(meta_path)
        if meta is None:
            return None

        records: list[SessionRecord] = []
        record_offsets: list[int] = []
        with open(jsonl_path, encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                record = SessionRecord.from_jsonl(line)
                if record is not None:
                    records.append(record)
                    record_offsets.append(offset)

        # Ensure legacy JSONL-only sessions have a parent row before indexes
        # are reconciled into SQLite.
        self.session_store.upsert(
            session_id,
            title=meta.title,
            summary=meta.summary,
            message_count=meta.message_count,
            total_tokens=meta.total_tokens,
            created_at=meta.created_at.isoformat(),
        )

        # 重建压缩后的状态：仅从最后一个 compact_boundary 开始重放。
        # 该标记之前的 record 是已被摘要过的原始前缀——保留在磁盘上供审计，
        # 但不再重放。标记本身内联了摘要 + 原样 keep 尾部，标记之后追加的
        # 普通消息（续写）照常重放。没有 boundary 则全量重放（兼容旧 session）。
        last_boundary = -1
        active_checkpoint: CompactBoundaryData | None = None
        for i, rec in enumerate(records):
            if rec.type == RecordType.COMPACT_BOUNDARY:
                details = parse_compact_boundary_details(rec)
                if not details.integrity_valid:
                    continue
                last_boundary = i
                active_checkpoint = details
                if details.checkpoint_id:
                    try:
                        self.checkpoint_store.upsert(
                            details.checkpoint_id,
                            session_id,
                            kind="compact",
                            tail_id=details.tail_id,
                            payload=rec.content,
                            transcript_offset=record_offsets[i],
                            run_id=details.run_id,
                            step_id=details.step_id,
                            created_at=rec.timestamp.isoformat(),
                        )
                    except Exception:
                        log.exception(
                            "Failed to reconcile checkpoint %s",
                            details.checkpoint_id,
                        )
        if last_boundary >= 0:
            records = records[last_boundary:]

        valid_count = validate_message_chain(records)
        records = records[:valid_count]
        messages = records_to_messages(records)

        file = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
        session = Session(
            session_id=session_id,
            file=file,
            meta=meta,
            sessions_dir=self._sessions_dir,
            state_store=self.session_store,
            checkpoint_store=self.checkpoint_store,
        )
        session._sync_state()

        return ResumeResult(
            session=session,
            messages=messages,
            last_active=meta.last_active,
            checkpoint_id=(
                active_checkpoint.checkpoint_id if active_checkpoint else None
            ),
            tail_id=active_checkpoint.tail_id if active_checkpoint else None,
            run_id=active_checkpoint.run_id if active_checkpoint else None,
            step_id=active_checkpoint.step_id if active_checkpoint else None,
        )

    def delete(self, session_id: str) -> bool:
        jsonl_path = self._sessions_dir / f"{session_id}.jsonl"
        meta_path = self._sessions_dir / f"{session_id}.meta"

        deleted = False
        artifact_root = self._sessions_dir.parent / "session" / "tool-results"
        artifacts = self.result_artifact_store.list_for_session(session_id)
        if artifacts:
            self.result_artifact_store.reconcile_references(
                session_id, set(), root_dir=artifact_root
            )
        if jsonl_path.exists():
            jsonl_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        if self.session_store.delete(session_id):
            deleted = True
        return deleted

    def cleanup(self, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        removed = 0

        for meta_path in list(self._sessions_dir.glob("*.meta")):
            meta = SessionMeta.load(meta_path)
            if meta is not None and meta.last_active < cutoff:
                self.delete(meta.id)
                removed += 1

        return removed
