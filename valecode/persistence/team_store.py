from __future__ import annotations

from dataclasses import dataclass

from valecode.persistence._common import utc_now
from valecode.persistence.database import Database


@dataclass(frozen=True)
class TeamState:
    name: str
    lead_agent_id: str
    description: str
    backend_type: str
    status: str
    created_at: str
    updated_at: str
    deleted_at: str | None


@dataclass(frozen=True)
class TeamMemberState:
    team_name: str
    agent_id: str
    name: str
    agent_type: str
    model: str
    worktree_path: str
    backend_type: str
    is_active: bool | None
    status: str
    created_at: str
    updated_at: str


class TeamStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _team(row) -> TeamState:
        return TeamState(**dict(row))

    @staticmethod
    def _member(row) -> TeamMemberState:
        data = dict(row)
        if data["is_active"] is not None:
            data["is_active"] = bool(data["is_active"])
        return TeamMemberState(**data)

    def upsert_team(
        self,
        name: str,
        lead_agent_id: str,
        *,
        description: str = "",
        backend_type: str = "",
    ) -> TeamState:
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO teams(name, lead_agent_id, description, backend_type, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(name) DO UPDATE SET lead_agent_id=excluded.lead_agent_id,
                description=excluded.description, backend_type=excluded.backend_type,
                status='active', updated_at=excluded.updated_at, deleted_at=NULL""",
                (name, lead_agent_id, description, backend_type, now, now),
            )
            row = connection.execute(
                "SELECT * FROM teams WHERE name=?", (name,)
            ).fetchone()
        return self._team(row)

    def get_team(self, name: str) -> TeamState | None:
        with self.database.reader() as connection:
            row = connection.execute("SELECT * FROM teams WHERE name=?", (name,)).fetchone()
        return self._team(row) if row else None

    def list_active(self) -> list[TeamState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                "SELECT * FROM teams WHERE status='active' ORDER BY created_at"
            ).fetchall()
        return [self._team(row) for row in rows]

    def upsert_member(self, team_name: str, member) -> TeamMemberState:
        now = utc_now()
        status = (
            "running"
            if member.is_active is True
            else "idle"
            if member.is_active is False
            else "starting"
        )
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO team_members(team_name, agent_id, name, agent_type, model, worktree_path, backend_type, is_active, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team_name, agent_id) DO UPDATE SET name=excluded.name,
                agent_type=excluded.agent_type, model=excluded.model,
                worktree_path=excluded.worktree_path, backend_type=excluded.backend_type,
                is_active=excluded.is_active, status=excluded.status, updated_at=excluded.updated_at""",
                (
                    team_name,
                    member.agent_id,
                    member.name,
                    member.agent_type,
                    member.model,
                    member.worktree_path,
                    member.backend_type,
                    member.is_active,
                    status,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM team_members WHERE team_name=? AND agent_id=?",
                (team_name, member.agent_id),
            ).fetchone()
        return self._member(row)

    def list_members(self, team_name: str) -> list[TeamMemberState]:
        with self.database.reader() as connection:
            rows = connection.execute(
                "SELECT * FROM team_members WHERE team_name=? ORDER BY created_at",
                (team_name,),
            ).fetchall()
        return [self._member(row) for row in rows]

    def set_member_active(self, team_name: str, name: str, active: bool) -> None:
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE team_members SET is_active=?, status=?, updated_at=?
                WHERE team_name=? AND (name=? OR agent_id=?)""",
                (
                    active,
                    "running" if active else "idle",
                    utc_now(),
                    team_name,
                    name,
                    name,
                ),
            )

    def mark_deleted(self, name: str) -> None:
        now = utc_now()
        with self.database.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE team_members SET is_active=0, status='stopped', updated_at=?
                WHERE team_name=?""",
                (now, name),
            )
            connection.execute(
                """UPDATE teams SET status='deleted', updated_at=?, deleted_at=?
                WHERE name=?""",
                (now, now, name),
            )
