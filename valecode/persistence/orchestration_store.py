"""Fenced instance/node journal shared by deterministic workflows and goals."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class LeaseLost(RuntimeError):
    pass


def encode(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(text.encode("utf-8")) > 1000000:
        raise ValueError("Orchestration state exceeds 1 MB")
    return text


class OrchestrationStore:
    def __init__(self, database):
        self.database = database

    def create(self, kind, session_id, work_dir, spec, node_ids):
        identity = kind + "_" + uuid.uuid4().hex
        snapshot = encode(spec)
        now = time.time()
        with self.database.transaction(immediate=True) as db:
            db.execute("INSERT INTO orchestrations(id,kind,session_id,work_dir,spec_json,status,created_at,updated_at) VALUES(?,?,?,?,?,'pending',?,?)",
                (identity, kind, session_id, str(Path(work_dir).resolve()), snapshot, now, now))
            for node in node_ids:
                db.execute("INSERT INTO orchestration_nodes(instance_id,node_id,status) VALUES(?,?,'pending')", (identity, node))
        return identity

    def get(self, identity, session_id, work_dir):
        with self.database.reader() as db:
            row = db.execute("SELECT * FROM orchestrations WHERE id=? AND session_id=? AND work_dir=?", (identity, session_id, str(Path(work_dir).resolve()))).fetchone()
        if row is None:
            raise ValueError("Instance is outside the active session/workspace")
        result = dict(row)
        result["spec"] = json.loads(result.pop("spec_json"))
        result["state"] = json.loads(result.pop("state_json"))
        return result

    def list(self, session_id, work_dir, kind):
        with self.database.reader() as db:
            return [dict(row) for row in db.execute("SELECT id,status,error,updated_at FROM orchestrations WHERE session_id=? AND work_dir=? AND kind=? ORDER BY updated_at DESC LIMIT 50", (session_id, str(Path(work_dir).resolve()), kind)).fetchall()]

    def nodes(self, identity):
        with self.database.reader() as db:
            rows = db.execute("SELECT * FROM orchestration_nodes WHERE instance_id=? ORDER BY rowid", (identity,)).fetchall()
        return {row["node_id"]: dict(row) | {"metadata": json.loads(row["metadata_json"])} for row in rows}

    def claim(self, identity, owner, *, ttl=30):
        now = time.time()
        with self.database.transaction(immediate=True) as db:
            row = db.execute("SELECT * FROM orchestrations WHERE id=?", (identity,)).fetchone()
            if row is None:
                raise ValueError("Unknown instance")
            if row["owner"] is not None and row["lease_until"] > now:
                raise LeaseLost("Instance is already owned")
            if row["status"] in {"completed", "cancelled"}:
                return None
            # A crash can leave effects without a recorded result. Never replay
            # running nodes on a plain resume, even when the lease has expired.
            db.execute("UPDATE orchestration_nodes SET status='blocked',error='Interrupted execution; explicit retry required' WHERE instance_id=? AND status='running'", (identity,))
            blocked = db.execute("SELECT 1 FROM orchestration_nodes WHERE instance_id=? AND status IN ('blocked','failed')", (identity,)).fetchone()
            if blocked:
                db.execute("UPDATE orchestrations SET status='blocked',owner=NULL,lease_until=NULL,error='Explicit node retry required',updated_at=? WHERE id=?", (now, identity))
                return None
            epoch = row["epoch"] + 1
            db.execute("UPDATE orchestrations SET status='running',owner=?,lease_until=?,epoch=?,error='',updated_at=? WHERE id=?", (owner, now + ttl, epoch, now, identity))
            return epoch

    @staticmethod
    def _fence(db, identity, owner, epoch):
        if db.execute("SELECT 1 FROM orchestrations WHERE id=? AND owner=? AND epoch=? AND status='running' AND lease_until>?", (identity, owner, epoch, time.time())).fetchone() is None:
            raise LeaseLost("Orchestration ownership expired or replaced")

    def heartbeat(self, identity, owner, epoch, *, ttl=30):
        with self.database.transaction(immediate=True) as db:
            self._fence(db, identity, owner, epoch)
            db.execute("UPDATE orchestrations SET lease_until=?,updated_at=? WHERE id=?", (time.time() + ttl, time.time(), identity))

    def begin_node(self, identity, node, owner, epoch):
        with self.database.transaction(immediate=True) as db:
            self._fence(db, identity, owner, epoch)
            cursor = db.execute("UPDATE orchestration_nodes SET status='running',attempts=attempts+1,error='' WHERE instance_id=? AND node_id=? AND status='pending'", (identity, node))
            if cursor.rowcount != 1:
                raise ValueError("Node is not pending")

    def finish_node(self, identity, node, owner, epoch, status, output="", error="", metadata=None):
        if status not in {"succeeded", "skipped", "pending", "failed", "blocked"}:
            raise ValueError("Invalid node terminal state")
        if not isinstance(output, str) or len(output) > 64000:
            raise ValueError("Node output exceeds 64000 characters")
        with self.database.transaction(immediate=True) as db:
            self._fence(db, identity, owner, epoch)
            cursor = db.execute("UPDATE orchestration_nodes SET status=?,output=?,error=?,metadata_json=? WHERE instance_id=? AND node_id=? AND status='running'", (status, output, error[:2000], encode(metadata or {}), identity, node))
            if cursor.rowcount != 1:
                raise ValueError("Node was not running")

    def update_state(self, identity, owner, epoch, state):
        snapshot = encode(state)
        with self.database.transaction(immediate=True) as db:
            self._fence(db, identity, owner, epoch)
            db.execute("UPDATE orchestrations SET state_json=?,updated_at=? WHERE id=?", (snapshot, time.time(), identity))

    def finish(self, identity, owner, epoch, status, error=""):
        if status not in {"blocked", "failed", "completed", "cancelled"}:
            raise ValueError("Invalid instance terminal state")
        with self.database.transaction(immediate=True) as db:
            self._fence(db, identity, owner, epoch)
            if status == "completed" and db.execute("SELECT 1 FROM orchestration_nodes WHERE instance_id=? AND status NOT IN ('succeeded','skipped')", (identity,)).fetchone():
                raise ValueError("Incomplete nodes cannot complete an instance")
            db.execute("UPDATE orchestration_nodes SET status='blocked',error='Interrupted; explicit retry required' WHERE instance_id=? AND status='running'", (identity,))
            db.execute("UPDATE orchestrations SET status=?,error=?,owner=NULL,lease_until=NULL,updated_at=? WHERE id=?", (status, error[:2000], time.time(), identity))

    def retry_node(self, identity, node, *, confirmed):
        if confirmed is not True:
            raise ValueError("Retry may replay side effects; explicit confirmation required")
        with self.database.transaction(immediate=True) as db:
            row = db.execute("SELECT * FROM orchestrations WHERE id=?", (identity,)).fetchone()
            if row is None or row["status"] in {"completed", "cancelled"} or (row["owner"] and row["lease_until"] > time.time()):
                raise ValueError("Instance is not available for retry")
            cursor = db.execute("UPDATE orchestration_nodes SET status='pending',error='',output='',metadata_json='{}' WHERE instance_id=? AND node_id=? AND status IN ('blocked','failed')", (identity, node))
            if cursor.rowcount != 1:
                raise ValueError("Only blocked/failed nodes may be retried")
            db.execute("UPDATE orchestrations SET status='pending',error='',owner=NULL,lease_until=NULL,updated_at=? WHERE id=?", (time.time(), identity))
