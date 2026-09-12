from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from valecode.worktree.changes import (
    CleanupResult,
    Changes,
    count_worktree_changes,
    has_unpushed_commits,
    has_worktree_changes,
)
from valecode.worktree.models import Worktree, WorktreeSession
from valecode.worktree.paths import canonical_path, is_path_within, require_path_within
from valecode.worktree.session import load_worktree_session, save_worktree_session
from valecode.worktree.setup import perform_post_creation_setup
from valecode.worktree.slug import flatten_slug, validate_slug

log = logging.getLogger(__name__)

GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}


class WorktreeError(Exception):
    pass


class WorktreeManager:
    def __init__(
        self,
        repo_root: str,
        symlink_directories: list[str] | None = None,
        worktree_dir: str | None = None,
    ) -> None:
        self.repo_root = str(Path(repo_root).resolve(strict=False))
        self.symlink_directories = symlink_directories or []
        self.worktree_dir = str(
            Path(worktree_dir).resolve(strict=False)
            if worktree_dir
            else Path(self.repo_root) / ".valecode" / "worktrees"
        )
        self._valecode_dir = Path(self.repo_root) / ".valecode"
        self._lock = asyncio.Lock()
        self.active: dict[str, Worktree] = {}
        self.current_session: WorktreeSession | None = None

    def _run_git(self, args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, **GIT_ENV}
        return subprocess.run(
            ["git"] + args,
            cwd=cwd or self.repo_root,
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
            env=env,
        )

    # ------------------------------------------------------------------
    # 快速恢复：直接从文件系统读取 HEAD SHA，无需启动 git 子进程
    # ------------------------------------------------------------------

    @staticmethod
    def read_worktree_head_sha(wt_path: str) -> str | None:
        wt = Path(wt_path)
        git_file = wt / ".git"
        if not git_file.exists():
            return None

        try:
            content = git_file.read_text(encoding="utf-8").strip()
            if not content.startswith("gitdir:"):
                return None
            gitdir = Path(content.split(":", 1)[1].strip())
            if not gitdir.is_absolute():
                gitdir = (wt / gitdir).resolve()

            commondir_file = gitdir / "commondir"
            if commondir_file.exists():
                commondir_rel = commondir_file.read_text(encoding="utf-8").strip()
                commondir = (gitdir / commondir_rel).resolve()
            else:
                commondir = gitdir

            head_file = gitdir / "HEAD"
            if not head_file.exists():
                return None
            head_content = head_file.read_text(encoding="utf-8").strip()

            if head_content.startswith("ref:"):
                ref_path = head_content.split(":", 1)[1].strip()
                ref_file = gitdir / ref_path
                if not ref_file.exists():
                    ref_file = commondir / ref_path
                if ref_file.exists():
                    return ref_file.read_text(encoding="utf-8").strip()
                packed_refs = commondir / "packed-refs"
                if packed_refs.exists():
                    for line in packed_refs.read_text(encoding="utf-8").splitlines():
                        if line.strip() and not line.startswith("#"):
                            parts = line.split()
                            if len(parts) == 2 and parts[1] == ref_path:
                                return parts[0]
                return None
            return head_content
        except OSError:
            return None

    # ------------------------------------------------------------------
    # 创建 worktree
    # ------------------------------------------------------------------

    def _registered_worktrees(self) -> dict[str, dict[str, str]]:
        result = self._run_git(["worktree", "list", "--porcelain"])
        if result.returncode != 0:
            raise WorktreeError(
                f"git worktree list failed: {result.stderr.strip()}"
            )
        registered: dict[str, dict[str, str]] = {}
        current: dict[str, str] = {}
        for line in result.stdout.splitlines() + [""]:
            if not line:
                path = current.get("worktree")
                if path:
                    registered[canonical_path(path)] = current
                current = {}
                continue
            key, _, value = line.partition(" ")
            current[key] = value
        return registered

    @staticmethod
    def _branch_ref(branch_name: str) -> str:
        return f"refs/heads/{branch_name}"

    def _managed_worktree_path(self, name: str) -> Path:
        flat_slug = flatten_slug(name)
        try:
            return require_path_within(
                Path(self.worktree_dir) / flat_slug,
                self.worktree_dir,
                label="worktree path",
            )
        except ValueError as exc:
            raise WorktreeError(str(exc)) from exc

    async def create(self, name: str, base_branch: str = "HEAD") -> Worktree:
        async with self._lock:
            err = validate_slug(name)
            if err:
                raise WorktreeError(err)

            if any(existing.casefold() == name.casefold() for existing in self.active):
                raise WorktreeError(f"worktree already exists: {name}")
            if not base_branch or base_branch.startswith("-"):
                raise WorktreeError("base branch must not be empty or start with '-'")

            flat_slug = flatten_slug(name)
            wt_path = str(self._managed_worktree_path(name))
            branch_name = f"worktree-{flat_slug}"
            registered = self._registered_worktrees()
            registration = registered.get(canonical_path(wt_path))

            head_sha = self.read_worktree_head_sha(wt_path)
            if head_sha is not None:
                if registration is None:
                    raise WorktreeError(
                        f"path contains an unregistered worktree: {wt_path}"
                    )
                if registration.get("branch") != self._branch_ref(branch_name):
                    raise WorktreeError(
                        "worktree path/branch conflict: "
                        f"expected {branch_name}, found {registration.get('branch', 'detached HEAD')}"
                    )
                log.info("Fast recovery: reusing existing worktree at %s", wt_path)
                wt = Worktree(
                    name=name,
                    path=wt_path,
                    branch=branch_name,
                    based_on=base_branch,
                    head_commit=head_sha,
                )
                self.active[name] = wt
                return wt

            if Path(wt_path).exists():
                raise WorktreeError(
                    f"worktree path already exists but is not reusable: {wt_path}"
                )

            branch_check = self._run_git(
                ["show-ref", "--verify", "--quiet", self._branch_ref(branch_name)]
            )
            if branch_check.returncode == 0:
                raise WorktreeError(
                    f"worktree branch already exists: {branch_name}"
                )

            os.makedirs(self.worktree_dir, exist_ok=True)

            result = self._run_git([
                "worktree", "add",
                "-b", branch_name, "--", wt_path, base_branch,
            ])
            if result.returncode != 0:
                raise WorktreeError(
                    f"git worktree add failed: {result.stderr.strip()}"
                )

            perform_post_creation_setup(
                self.repo_root,
                wt_path,
                symlink_directories=self.symlink_directories,
            )

            head_sha = self.read_worktree_head_sha(wt_path) or ""
            wt = Worktree(
                name=name,
                path=wt_path,
                branch=branch_name,
                based_on=base_branch,
                head_commit=head_sha,
            )
            self.active[name] = wt
            return wt

    # ------------------------------------------------------------------
    # 进入 worktree
    # ------------------------------------------------------------------

    async def enter(self, name: str) -> WorktreeSession:
        async with self._lock:
            wt = self.active.get(name)
            if wt is None:
                raise WorktreeError(f"worktree not found: {name}")
            if self.current_session is not None:
                raise WorktreeError(
                    f"already in worktree: {self.current_session.worktree_name}"
                )
            if not is_path_within(wt.path, self.worktree_dir):
                raise WorktreeError("worktree path is outside the managed directory")
            registration = self._registered_worktrees().get(canonical_path(wt.path))
            if registration is None:
                raise WorktreeError("worktree is no longer registered with git")

            original_cwd = os.getcwd()
            if not is_path_within(original_cwd, self.repo_root):
                original_cwd = self.repo_root
            original_branch = self._get_current_branch()
            original_head = self._get_head_commit()

            session = WorktreeSession(
                original_cwd=original_cwd,
                worktree_path=wt.path,
                worktree_name=name,
                original_branch=original_branch,
                original_head_commit=original_head,
            )
            self.current_session = session
            save_worktree_session(self._valecode_dir, session)
            return session

    # ------------------------------------------------------------------
    # 退出 worktree
    # ------------------------------------------------------------------


    async def exit(
        self,
        name: str,
        action: str = "keep",
        discard_changes: bool = False,
    ) -> None:
        async with self._lock:
            wt = self.active.get(name)
            if wt is None:
                raise WorktreeError(f"worktree not found: {name}")
            if action not in {"keep", "remove"}:
                raise WorktreeError(f"unsupported worktree exit action: {action}")

            if action == "remove" and not discard_changes:
                changes = count_worktree_changes(wt.path, wt.head_commit)
                if changes.uncommitted > 0 or changes.new_commits > 0:
                    raise WorktreeError(
                        f"worktree has changes ({changes.uncommitted} uncommitted, "
                        f"{changes.new_commits} new commits). "
                        "Set discard_changes=True to force removal."
                    )

            # Removal must succeed before session state is forgotten.  This
            # leaves a failed operation recoverable and keeps the branch intact.
            if action == "remove":
                await self._remove_worktree(name, wt)

            self.current_session = None
            save_worktree_session(self._valecode_dir, None)

    # ------------------------------------------------------------------
    # 删除 worktree（内部方法）
    # ------------------------------------------------------------------

    async def _remove_worktree(self, name: str, wt: Worktree) -> None:
        if not is_path_within(wt.path, self.worktree_dir):
            raise WorktreeError("refusing to remove worktree outside managed directory")
        registration = self._registered_worktrees().get(canonical_path(wt.path))
        if registration is None:
            raise WorktreeError("refusing to remove an unregistered worktree path")
        if registration.get("branch") != self._branch_ref(wt.branch):
            raise WorktreeError("refusing to remove worktree with mismatched branch")

        result = self._run_git(["worktree", "remove", "--force", "--", wt.path])
        if result.returncode != 0:
            raise WorktreeError(
                f"git worktree remove failed: {result.stderr.strip()}"
            )

        await asyncio.sleep(0.1)

        flat_slug = flatten_slug(name)
        branch_name = f"worktree-{flat_slug}"
        branch_result = self._run_git(["branch", "-D", "--", branch_name])
        if branch_result.returncode != 0:
            log.warning(
                "worktree removed but branch cleanup failed for %s: %s",
                branch_name,
                branch_result.stderr.strip(),
            )

        self.active.pop(name, None)

    # ------------------------------------------------------------------
    # 自动清理
    # ------------------------------------------------------------------


    async def auto_cleanup(self, name: str, head_commit: str) -> CleanupResult:
        async with self._lock:
            wt = self.active.get(name)
            if wt is None:
                return CleanupResult(kept=False)

            if has_worktree_changes(wt.path, head_commit):
                return CleanupResult(kept=True, path=wt.path, branch=wt.branch)

            await self._remove_worktree(name, wt)
            return CleanupResult(kept=False)

    async def remove_stale(self, name: str, path: str) -> bool:
        """Remove a clean ephemeral worktree under the manager lock."""
        async with self._lock:
            if self.current_session and canonical_path(
                self.current_session.worktree_path
            ) == canonical_path(path):
                return False
            if not is_path_within(path, self.worktree_dir):
                raise WorktreeError("stale worktree is outside managed directory")
            head = self.read_worktree_head_sha(path)
            if head is None or has_worktree_changes(path, head):
                return False
            if has_unpushed_commits(path):
                return False
            wt = self.active.get(name) or Worktree(
                name=name,
                path=str(Path(path).resolve(strict=False)),
                branch=f"worktree-{flatten_slug(name)}",
                based_on="unknown",
                head_commit=head,
            )
            await self._remove_worktree(name, wt)
            return True

    # ------------------------------------------------------------------
    # 列出 / 查询
    # ------------------------------------------------------------------

    def list_worktrees(self) -> list[Worktree]:
        return list(self.active.values())


    def get_current_session(self) -> WorktreeSession | None:
        return self.current_session

    # ------------------------------------------------------------------
    # 从持久化的 session 中恢复
    # ------------------------------------------------------------------

    def restore_session(self) -> WorktreeSession | None:
        session = load_worktree_session(self._valecode_dir)
        if session is None:
            return None
        slug_error = validate_slug(session.worktree_name)
        if slug_error:
            log.warning("Ignoring invalid persisted worktree name: %s", slug_error)
            save_worktree_session(self._valecode_dir, None)
            return None
        wt_path = session.worktree_path
        if not is_path_within(wt_path, self.worktree_dir):
            log.warning("Ignoring worktree session outside managed directory: %s", wt_path)
            save_worktree_session(self._valecode_dir, None)
            return None
        try:
            registration = self._registered_worktrees().get(canonical_path(wt_path))
        except WorktreeError:
            return None
        expected_branch = self._branch_ref(
            f"worktree-{flatten_slug(session.worktree_name)}"
        )
        if registration is None or registration.get("branch") != expected_branch:
            log.warning("Ignoring stale or conflicting worktree session: %s", wt_path)
            save_worktree_session(self._valecode_dir, None)
            return None
        head_sha = self.read_worktree_head_sha(wt_path)
        if head_sha is None:
            save_worktree_session(self._valecode_dir, None)
            return None

        if not is_path_within(session.original_cwd, self.repo_root):
            session.original_cwd = self.repo_root

        wt = Worktree(
            name=session.worktree_name,
            path=wt_path,
            branch=f"worktree-{flatten_slug(session.worktree_name)}",
            based_on="unknown",
            head_commit=head_sha,
        )
        self.active[session.worktree_name] = wt
        self.current_session = session
        return session

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------


    def _get_current_branch(self) -> str:
        try:
            result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            return result.stdout.strip() if result.returncode == 0 else "HEAD"
        except (subprocess.SubprocessError, OSError):
            return "HEAD"

    def _get_head_commit(self) -> str:
        try:
            result = self._run_git(["rev-parse", "HEAD"])
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.SubprocessError, OSError):
            return ""
