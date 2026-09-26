"""Calendar calculation only; no execution or authorization lives here."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from croniter import croniter


def next_occurrence(kind: str, spec: dict, timezone_name: str, now: datetime) -> datetime | None:
    if now.tzinfo is None:
        raise ValueError("Scheduler clock must be timezone-aware")
    zone = ZoneInfo(timezone_name)
    required = {"interval": "every_seconds", "once": "run_at", "cron": "cron"}.get(kind)
    if not isinstance(spec, dict) or required is None or set(spec) != {required}:
        raise ValueError("schedule_spec must contain only the field matching schedule_type")
    if kind == "interval":
        seconds = spec.get("every_seconds")
        if type(seconds) is not int or not 60 <= seconds <= 30 * 86400:
            raise ValueError("every_seconds must be an integer between 60 and 2592000")
        return now.astimezone(UTC) + timedelta(seconds=seconds)
    if kind == "once":
        if not isinstance(spec.get("run_at"), str):
            raise ValueError("once requires an ISO run_at")
        scheduled = datetime.fromisoformat(spec["run_at"])
        if scheduled.tzinfo is None:
            first, second = scheduled.replace(tzinfo=zone, fold=0), scheduled.replace(tzinfo=zone, fold=1)
            if first.utcoffset() != second.utcoffset() or first.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != scheduled:
                raise ValueError("Ambiguous/nonexistent local run_at; supply an explicit UTC offset")
            scheduled = first
        scheduled = scheduled.astimezone(UTC)
        return scheduled if scheduled > now else None
    if kind == "cron":
        expression = spec.get("cron")
        if not isinstance(expression, str) or len(expression) > 256 or len(expression.split()) != 5:
            raise ValueError("cron must contain exactly five fields")
        return croniter(expression, now.astimezone(zone), max_years_between_matches=5).get_next(datetime).astimezone(UTC)
    raise ValueError("schedule_type must be once, interval or cron")
