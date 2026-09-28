"""Explicit, evidence-backed goals; only goal_not_met_yet can auto-continue."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from valecode.persistence.orchestration_store import LeaseLost
from valecode.runtime.workflow import WorkflowRuntime

Blocker = Literal["none", "goal_not_met_yet", "missing_evidence", "needs_user_input", "run_failed", "external_wait"]


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    description: str = Field(min_length=1, max_length=2000)
    evidence_files: list[str] = Field(min_length=1, max_length=10)
    contains: str | None = Field(default=None, min_length=1, max_length=1000)


class GoalDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    objective: str = Field(min_length=1, max_length=4000)
    criteria: list[Criterion] = Field(min_length=1, max_length=20)
    agent_type: str = Field(default="general-purpose", min_length=1, max_length=128)
    worker_read_only: bool = False
    worker_max_turns: int = Field(default=20, ge=1, le=100)
    verifier_max_turns: int = Field(default=8, ge=2, le=30)
    max_rounds: int = Field(default=9, ge=1, le=9)
    max_no_progress: int = Field(default=2, ge=1, le=8)
    max_tokens: int = Field(default=200000, ge=1, le=2000000)
    max_seconds: int = Field(default=1800, ge=1, le=21600)
    step_timeout: int = Field(default=300, ge=1, le=3600)

    @field_validator("version", mode="before")
    @classmethod
    def version_number(cls, value):
        if type(value) is not int:
            raise ValueError("Version must be an integer, not a boolean")
        return value

    @model_validator(mode="after")
    def contract(self):
        if not self.objective.strip() or any(not item.description.strip() for item in self.criteria):
            raise ValueError("Objective and criteria must not be blank")
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("Duplicate criterion IDs")
        for item in self.criteria:
            if len(set(item.evidence_files)) != len(item.evidence_files):
                raise ValueError("Duplicate evidence paths")
            for file in item.evidence_files:
                path = Path(file)
                if not file.strip() or len(file) > 1000 or path.is_absolute() or path.drive or ".." in path.parts:
                    raise ValueError("Evidence must name workspace-relative files")
        return self


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    criterion_id: str
    satisfied: bool
    evidence_ids: list[str] = Field(min_length=1, max_length=30)


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    blocker: Blocker
    reason: str = Field(max_length=2000)
    checks: list[Check] = Field(min_length=1, max_length=20)


class GoalRuntime(WorkflowRuntime):
    kind = "goal"

    def load(self, path):
        _, root = self.scope()
        file = Path(root) / path
        if file.is_symlink() or not file.resolve().is_relative_to(Path(root).resolve()) or file.stat().st_size > 128000:
            raise ValueError("Goal definition must be within the workspace and <=128 KB")
        return GoalDefinition.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")))

    def create(self, definition):
        definition = GoalDefinition.model_validate(definition)
        self._paths(definition)
        spec = {"definition": definition.model_dump(),
            "worker": self.runner.snapshot(definition.agent_type, definition.worker_max_turns),
            "verifier": self.runner.verifier_snapshot(definition.verifier_max_turns)}
        return self.store.create("goal", *self.scope(), spec, ["worker_1", "verifier_1"])

    def status(self, identity):
        state = self.store.get(identity, *self.scope())
        if state["kind"] != "goal":
            raise ValueError("Not a goal instance")
        return state | {"nodes": self.store.nodes(identity)}

    def _paths(self, definition):
        _, root = self.scope()
        root = Path(root).resolve()
        result = {item.id: [(root / file).resolve() for file in item.evidence_files] for item in definition.criteria}
        if any(not file.is_relative_to(root) for files in result.values() for file in files):
            raise ValueError("Evidence path escapes the workspace")
        return result

    @staticmethod
    def _totals(nodes):
        for row in nodes.values():
            tokens, seconds = row["metadata"].get("spent_tokens", 0), row["metadata"].get("spent_seconds", 0)
            if type(tokens) is not int or tokens < 0 or isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Corrupt goal usage counters")
        return {"tokens": sum(row["metadata"].get("spent_tokens", 0) for row in nodes.values()),
            "seconds": sum(row["metadata"].get("spent_seconds", 0) for row in nodes.values())}

    @staticmethod
    def _budget(definition, totals):
        if totals["tokens"] >= definition.max_tokens:
            return "token_budget_exhausted"
        if totals["seconds"] >= definition.max_seconds:
            return "time_budget_exhausted"
        return ""

    def _assess(self, definition, output, evidence):
        evaluation = Evaluation.model_validate_json(output)
        if len({check.criterion_id for check in evaluation.checks}) != len(evaluation.checks):
            raise ValueError("Verifier repeated acceptance checks")
        if {check.criterion_id for check in evaluation.checks} != {item.id for item in definition.criteria}:
            raise ValueError("Verifier must assess every criterion exactly once")
        records = {record["id"]: record for record in evidence}
        paths = self._paths(definition)
        by_id = {item.id: item for item in definition.criteria}
        fingerprints = []
        for check in evaluation.checks:
            if len(set(check.evidence_ids)) != len(check.evidence_ids) or not set(check.evidence_ids) <= records.keys():
                raise ValueError("Verifier cited missing/foreign evidence IDs")
            cited = [records[key] for key in check.evidence_ids]
            if not {str(file) for file in paths[check.criterion_id]} <= {record["file"] for record in cited}:
                raise ValueError("Each criterion needs fresh reads of all declared evidence files")
            for file in paths[check.criterion_id]:
                if file.stat().st_size > 128000:
                    raise ValueError("Evidence file exceeds 128 KB")
                content = file.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if not any(record["file"] == str(file) and record["sha256"] == digest for record in cited):
                    raise ValueError("Evidence changed after verification")
                expected = by_id[check.criterion_id].contains
                if expected is not None and expected not in content.decode("utf-8"):
                    check.satisfied = False
                    if evaluation.blocker == "none":
                        evaluation.blocker = "goal_not_met_yet"
                fingerprints.append((str(file), digest))
        satisfied = all(check.satisfied for check in evaluation.checks)
        if evaluation.blocker == "none" and not satisfied:
            raise ValueError("Unsatisfied criteria cannot have blocker=none")
        signature = hashlib.sha256(json.dumps({"files": sorted(set(fingerprints)),
            "checks": sorted((check.criterion_id, check.satisfied) for check in evaluation.checks)}, sort_keys=True).encode()).hexdigest()
        return {"satisfied": satisfied and evaluation.blocker == "none", "evaluation": evaluation.model_dump(), "signature": signature}

    def _fresh(self, evidence):
        _, root = self.scope()
        try:
            for record in evidence:
                path = Path(record["file"])
                if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink() or path.stat().st_size > 128000:
                    return False
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    return False
            return bool(evidence)
        except (OSError, KeyError):
            return False

    def _worker_prompt(self, definition, feedback):
        return "完成明确目标，保持正常权限；不能实现时说明原因。验收由独立只读 Agent 完成。\n" + json.dumps({
            "objective": definition.objective, "criteria": [item.model_dump() for item in definition.criteria],
            "previous_feedback": feedback}, ensure_ascii=False)

    def _verifier_prompt(self, definition, worker_output):
        sample = {"blocker": "none", "reason": "依据实际产物", "checks": [
            {"criterion_id": item.id, "satisfied": True, "evidence_ids": ["本轮 ReadFile 原始工具调用 ID"]} for item in definition.criteria]}
        return ("逐项读取 evidence_files 的完整文本（最多128KB/2000行），独立验收。执行者自述不可信。"
            "只输出 JSON，不包代码块。所有项恰好评估一次，引用实际成功 ReadFile ID。"
            "blocker: none 仅全部达标；goal_not_met_yet 可继续改进；missing_evidence/needs_user_input/run_failed/external_wait 停止。\n"
            + json.dumps({"objective": definition.objective, "criteria": [item.model_dump() for item in definition.criteria],
                "untrusted_worker_output": worker_output, "output_schema_example": sample}, ensure_ascii=False))

    async def _phase(self, identity, node, definition, spec, owner, epoch, prompt, *, verifier=False):
        self.store.begin_node(identity, node, owner, epoch)
        started = time.monotonic()
        metadata = {}
        try:
            totals = self._totals(self.store.nodes(identity))
            timeout = min(definition.step_timeout, max(0.01, definition.max_seconds - totals["seconds"]))
            result = await asyncio.wait_for(self.runner.run(spec["verifier" if verifier else "worker"], prompt,
                read_only=verifier or definition.worker_read_only), timeout=timeout)
            metadata = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                "run_id": result.run_id, "denied": result.denied}
            if result.denied:
                raise ValueError("needs_user_input: node requires user permission")
            if verifier:
                metadata.update(self._assess(definition, result.output, result.evidence))
                metadata["evidence"] = result.evidence
            metadata["elapsed_seconds"] = time.monotonic() - started
            self.store.finish_node(identity, node, owner, epoch, "succeeded", result.output, metadata=metadata)
            return True
        except BaseException as exc:
            metadata["elapsed_seconds"] = time.monotonic() - started
            if not metadata.get("input_tokens") and not metadata.get("output_tokens"):
                metadata.update(input_tokens=getattr(exc, "input_tokens", 0), output_tokens=getattr(exc, "output_tokens", 0))
            self.store.finish_node(identity, node, owner, epoch, "blocked", error=str(exc) or "Interrupted; explicit retry required", metadata=metadata)
            if isinstance(exc, (asyncio.CancelledError, LeaseLost)) or not isinstance(exc, Exception):
                raise
            return False

    async def run(self, identity):
        if self._closed:
            raise ValueError("Goal runtime is closed")
        initial = self.status(identity)
        self._totals(initial["nodes"])
        definition = GoalDefinition.model_validate(initial["spec"]["definition"])
        self._paths(definition)
        self.runner.validate_snapshot(initial["spec"]["worker"], max_turns=definition.worker_max_turns)
        self.runner.validate_snapshot(initial["spec"]["verifier"], max_turns=definition.verifier_max_turns)
        count = len(initial["nodes"]) // 2
        expected = {f"{kind}_{number}" for number in range(1, count + 1) for kind in ("worker", "verifier")}
        if not 1 <= count <= definition.max_rounds or set(initial["nodes"]) != expected:
            raise ValueError("Goal journal has corrupt round/phase identities")
        scope = self.scope()
        owner = uuid.uuid4().hex
        epoch = self.store.claim(identity, owner)
        if epoch is None:
            return self.status(identity)
        current = asyncio.current_task()
        self._running.add(current)
        async def keepalive():
            while True:
                await asyncio.sleep(5)
                try:
                    self.store.heartbeat(identity, owner, epoch)
                except LeaseLost:
                    current.cancel()
                    return
        heartbeat = asyncio.create_task(keepalive())
        no_progress, previous_signature, feedback = 0, "", ""
        try:
            for number in range(1, definition.max_rounds + 1):
                if self.scope() != scope:
                    raise ValueError("Active session/workspace changed during goal")
                nodes = self.store.nodes(identity)
                worker, verifier = f"worker_{number}", f"verifier_{number}"
                stop = ""
                if worker not in nodes:
                    stop = self._budget(definition, self._totals(nodes))
                    if stop:
                        self._stop(identity, owner, epoch, stop, no_progress)
                        break
                    self.store.add_nodes(identity, [worker, verifier], owner, epoch)
                    nodes = self.store.nodes(identity)
                for node in (worker, verifier):
                    if nodes[node]["status"] == "succeeded":
                        continue
                    stop = self._budget(definition, self._totals(nodes))
                    if stop:
                        break
                    prompt = self._worker_prompt(definition, feedback) if node == worker else self._verifier_prompt(definition, nodes[worker]["output"])
                    if not await self._phase(identity, node, definition, initial["spec"], owner, epoch, prompt, verifier=node == verifier):
                        stop = "needs_user_input" if self.store.nodes(identity)[node]["metadata"].get("denied") else "run_failed" if node == worker else "missing_evidence"
                        break
                    nodes = self.store.nodes(identity)
                if stop:
                    self._stop(identity, owner, epoch, stop, no_progress)
                    break
                assessment = nodes[verifier]["metadata"]
                if number >= count and not self._fresh(assessment.get("evidence", [])):
                    self.store.invalidate_node(identity, verifier, owner, epoch, "Evidence changed; explicitly retry the read-only verifier")
                    self._stop(identity, owner, epoch, "missing_evidence", no_progress)
                    break
                no_progress = no_progress + 1 if previous_signature == assessment["signature"] else 0
                previous_signature = assessment["signature"]
                self.store.update_state(identity, owner, epoch, {"round": number, "totals": self._totals(nodes),
                    "no_progress": no_progress, "assessment": assessment["evaluation"], "stop_reason": ""})
                if assessment["satisfied"]:
                    self.store.finish(identity, owner, epoch, "completed")
                    break
                blocker = assessment["evaluation"]["blocker"]
                if blocker != "goal_not_met_yet":
                    self._stop(identity, owner, epoch, blocker, no_progress)
                    break
                if no_progress >= definition.max_no_progress:
                    self._stop(identity, owner, epoch, "no_progress", no_progress)
                    break
                if number == definition.max_rounds:
                    self._stop(identity, owner, epoch, "round_budget_exhausted", no_progress)
                    break
                feedback = assessment["evaluation"]
        except BaseException as exc:
            try:
                self._stop(identity, owner, epoch, "interrupted", no_progress, str(exc))
            except LeaseLost:
                pass
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._running.discard(current)
        return self.status(identity)

    def _stop(self, identity, owner, epoch, reason, no_progress, detail=""):
        state = self.status(identity)["state"]
        state.update(stop_reason=reason, no_progress=no_progress, totals=self._totals(self.store.nodes(identity)))
        self.store.update_state(identity, owner, epoch, state)
        self.store.finish(identity, owner, epoch, "blocked", detail or reason)
