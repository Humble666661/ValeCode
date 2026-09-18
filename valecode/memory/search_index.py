"""Rebuildable SQLite FTS5 index for file-backed memories.

The Markdown files remain the source of truth. This index only narrows large
selector manifests; a missing or unavailable index falls back to the full
file scan. Chinese bigrams work without an optional tokenizer dependency.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valecode.memory.recall import MemoryHeader


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+")
MAX_INDEX_CHARS = 64_000


def _terms(text: str) -> str:
    terms: list[str] = []
    for token in _TOKEN_RE.findall(text.casefold()):
        if "\u3400" <= token[0] <= "\u9fff":
            terms.extend(token)
            terms.extend(token[i : i + 2] for i in range(len(token) - 1))
        else:
            terms.append(token)
    return " ".join(terms)


def _query(text: str) -> str:
    tokens = list(dict.fromkeys(_terms(text).split()))[:32]
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class MemorySearchIndex:
    def __init__(self, work_dir: str | Path) -> None:
        root = Path(work_dir).resolve()
        cache_dir = (root / ".valecode" / "cache").resolve()
        if not cache_dir.is_relative_to(root):
            raise ValueError("Memory index directory escapes the project")
        self.path = cache_dir / "memory-search.sqlite3"

    def rank(self, query: str, headers: list[MemoryHeader], limit: int = 40) -> list[str]:
        """Sync changed files, then return matching absolute paths by BM25 rank."""
        if limit <= 0 or not headers:
            return []
        if self.path.is_symlink():
            raise ValueError("Memory index must not be a symlink")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=2.0) as db:
            db.execute("PRAGMA busy_timeout=2000")
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_files ("
                "path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
                "path UNINDEXED, title, description, body, tokenize='unicode61')"
            )
            self._sync(db, headers)
            expression = _query(query)
            if not expression:
                return []
            return [row[0] for row in db.execute(
                "SELECT path FROM memory_fts WHERE memory_fts MATCH ? "
                "ORDER BY bm25(memory_fts) LIMIT ?",
                (expression, limit),
            )]

    @staticmethod
    def _sync(db: sqlite3.Connection, headers: list[MemoryHeader]) -> None:
        known = {
            path: (mtime_ns, size)
            for path, mtime_ns, size in db.execute(
                "SELECT path, mtime_ns, size FROM memory_files"
            )
        }
        current = {header.file_path for header in headers}
        with db:
            for path in known.keys() - current:
                db.execute("DELETE FROM memory_fts WHERE path = ?", (path,))
                db.execute("DELETE FROM memory_files WHERE path = ?", (path,))
            for header in headers:
                path = Path(header.file_path)
                try:
                    stat = path.stat()
                except OSError:
                    continue
                signature = (stat.st_mtime_ns, stat.st_size)
                if known.get(header.file_path) == signature:
                    continue
                try:
                    with path.open(encoding="utf-8", errors="replace") as stream:
                        body = stream.read(MAX_INDEX_CHARS)
                except OSError:
                    continue
                db.execute("DELETE FROM memory_fts WHERE path = ?", (header.file_path,))
                db.execute(
                    "INSERT INTO memory_fts(path, title, description, body) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        header.file_path,
                        _terms(header.filename),
                        _terms(header.description),
                        _terms(body),
                    ),
                )
                db.execute(
                    "INSERT INTO memory_files(path, mtime_ns, size) VALUES (?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET "
                    "mtime_ns=excluded.mtime_ns, size=excluded.size",
                    (header.file_path, *signature),
                )
