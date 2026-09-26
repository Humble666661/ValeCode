"""Atomic plan-to-durable-task admission, scoped to an explicitly active session."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from valecode.persistence.database import Database
from valecode.persistence.event_store import EventStore
from valecode.runtime.schedules import next_occurrence


class ScheduleStore:
    def __init__(self, database: Database):
        self.database = database
        self.events = EventStore(database)

    @staticmethod
    def _row(row):
        item = dict(row)
        for key in ("spec", "agent"):
            item[key] = json.loads(item.pop(key + "_json"))
        return item

    def create(self, *, session_id: str, name: str, prompt: str, work_dir: str, kind: str, spec: dict, timezone: str, agent_spec: dict, now: datetime | None = None) -> dict:
        from valecode.tools.agent_tool import AgentTool
        current = now or datetime.now(UTC)
        if not session_id or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000 or not name.strip() or len(name) > 128:
            raise ValueError("Schedule requires a session, name (1–128) and prompt (1–20000)")
        definition = AgentTool._definition_from_resume_spec(agent_spec)
        if definition is None or definition.permission_mode != "default":
            raise ValueError("Scheduled agents require a valid default-permission descriptor")
        due = next_occurrence(kind, spec, timezone, current)
        if due is None:
            raise ValueError("Schedule has no future occurrence")
        task_id = "cron_" + uuid.uuid4().hex
        stamp = current.astimezone(UTC).isoformat()
        with self.database.transaction(immediate=True) as db:
            count = db.execute("SELECT count(*) FROM schedules WHERE session_id=? AND status IN ('enabled','paused','running')", (session_id,)).fetchone()[0]
            if count >= 50:
                raise ValueError("At most 50 active schedules per session")
            db.execute("INSERT INTO schedules VALUES(?,?,?,?,?,?,?,?,?,'enabled',?,?,?)", (
                task_id, session_id, name.strip(), prompt.strip(), str(Path(work_dir).resolve()), kind,
                json.dumps(spec), timezone, json.dumps(agent_spec), due.timestamp(), stamp, stamp,
            ))
            self.events._append(db, "schedule.created", session_id=session_id, payload={"schedule_id": task_id})
            return self._row(db.execute("SELECT * FROM schedules WHERE id=?", (task_id,)).fetchone())

    def list(self, session_id: str) -> list[dict]:
        with self.database.reader() as db:
            return [self._row(row) for row in db.execute("SELECT * FROM schedules WHERE session_id=? AND status!='deleted' ORDER BY created_at", (session_id,))]

    def change(self, schedule_id: str, session_id: str, action: str, *, now: datetime | None = None) -> dict:
        if action not in {"pause", "resume", "delete"}:
            raise ValueError("action must be pause, resume or delete")
        current = now or datetime.now(UTC)
        stamp = current.astimezone(UTC).isoformat()
        with self.database.transaction(immediate=True) as db:
            row = db.execute("SELECT * FROM schedules WHERE id=? AND session_id=? AND status!='deleted'", (schedule_id, session_id)).fetchone()
            if row is None:
                raise KeyError("Schedule not found in current session")
            plan = self._row(row)
            due = plan["next_run"]
            status = "deleted" if action == "delete" else "paused"
            if action == "resume":
                if plan["status"] != "paused":
                    raise ValueError("Only a paused schedule can resume")
                due_time = next_occurrence(plan["kind"], plan["spec"], plan["timezone"], current)
                if due_time is None:
                    raise ValueError("One-time occurrence is already past; create a new schedule")
                due, status = due_time.timestamp(), "enabled"
            else:
                # Serialize against TaskStore.claim: cancel only unclaimed
                # instances. Running work is not silently killed by a pause.
                queued = list(db.execute("SELECT t.id FROM tasks t JOIN schedule_occurrences o ON t.id=o.task_id WHERE o.schedule_id=? AND t.status='queued'", (schedule_id,)))
                for task in queued:
                    db.execute("UPDATE tasks SET status='cancelled',completed_at=?,updated_at=?,error='Schedule paused/deleted before claim',version=version+1 WHERE id=? AND status='queued'", (stamp, stamp, task[0]))
                    self.events._append(db, "task.status_changed", session_id=session_id, task_id=task[0], payload={"from": "queued", "to": "cancelled"})
            db.execute("UPDATE schedules SET status=?,next_run=?,updated_at=? WHERE id=?", (status, due, stamp, schedule_id))
            self.events._append(db, "schedule." + action, session_id=session_id, payload={"schedule_id": schedule_id})
            return self._row(db.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone())

    def admit_due(self, session_id: str, work_dir: str, *, now: datetime | None = None, limit: int = 20) -> list[str]:
        current = now or datetime.now(UTC)
        stamp = current.astimezone(UTC).isoformat()
        admitted = []
        with self.database.transaction(immediate=True) as db:
            plans = db.execute("SELECT * FROM schedules WHERE session_id=? AND work_dir=? AND status='enabled' AND next_run<=? ORDER BY next_run LIMIT ?", (session_id, str(Path(work_dir).resolve()), current.timestamp(), limit)).fetchall()
            for row in plans:
                plan = self._row(row)
                active = db.execute("SELECT 1 FROM schedule_occurrences o JOIN tasks t ON t.id=o.task_id WHERE o.schedule_id=? AND t.status IN ('queued','leased','running') LIMIT 1", (plan["id"],)).fetchone()
                if active:
                    continue  # No overlap; coalesce missed ticks after it ends.
                task_id = "scheduled_" + uuid.uuid4().hex
                metadata = {"resumable": True, "resume_spec": plan["agent"], "schedule_id": plan["id"], "scheduled_for": plan["next_run"], "scheduled_work_dir": plan["work_dir"]}
                payload = {"task": plan["prompt"], "name": plan["name"]}
                db.execute("INSERT INTO tasks(id,session_id,status,input_json,max_attempts,metadata_json,created_at,updated_at) VALUES(?,?,'queued',?,1,?,?,?)", (task_id, session_id, json.dumps(payload), json.dumps(metadata), stamp, stamp))
                db.execute("INSERT INTO schedule_occurrences VALUES(?,?,?,?)", (plan["id"], plan["next_run"], task_id, stamp))
                following = None if plan["kind"] == "once" else next_occurrence(plan["kind"], plan["spec"], plan["timezone"], current)
                db.execute("UPDATE schedules SET status=?,next_run=?,updated_at=? WHERE id=?", ("running" if plan["kind"] == "once" else "enabled", following.timestamp() if following else None, stamp, plan["id"]))
                self.events._append(db, "task.created", session_id=session_id, task_id=task_id, payload={"schedule_id": plan["id"], "scheduled_for": plan["next_run"], "status": "queued"})
                admitted.append(task_id)
            # Once plans are completed only after their real worker finishes.
            db.execute("""UPDATE schedules SET status=CASE WHEN EXISTS(
                SELECT 1 FROM schedule_occurrences o JOIN tasks t ON t.id=o.task_id WHERE o.schedule_id=schedules.id AND t.status='succeeded'
                ) THEN 'completed' ELSE 'failed' END,updated_at=?
                WHERE session_id=? AND status='running' AND NOT EXISTS(
                SELECT 1 FROM schedule_occurrences o JOIN tasks t ON t.id=o.task_id WHERE o.schedule_id=schedules.id AND t.status IN ('queued','leased','running'))""", (stamp, session_id))
        return admitted
