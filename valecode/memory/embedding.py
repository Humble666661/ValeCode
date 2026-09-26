"""Opt-in semantic retrieval; Markdown is authoritative, vectors are disposable."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field, asdict
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

if TYPE_CHECKING:
    from valecode.memory.recall import MemoryHeader


@dataclass
class MemorySearchConfig:
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 2.0
    allow_insecure_http: bool = False
    _specified_fields: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)


def validate_memory_search(raw: object) -> dict:
    defaults = asdict(MemorySearchConfig())
    defaults.pop("_specified_fields")
    if raw is None:
        return defaults
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError("memory_search must be a mapping with known fields")
    result = defaults | raw
    for key in ("enabled", "allow_insecure_http"):
        if not isinstance(result[key], bool):
            raise ValueError(f"memory_search.{key} must be boolean")
    for key in ("base_url", "model", "api_key"):
        if not isinstance(result[key], str):
            raise ValueError(f"memory_search.{key} must be a string")
        result[key] = result[key].strip()
    timeout = result["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 4:
        raise ValueError("memory_search.timeout_seconds must be finite in (0, 4]")
    url = urlsplit(result["base_url"])
    if result["base_url"]:
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("memory_search.base_url must be an HTTP(S) URL without credentials/query/fragment")
        if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"} and not result["allow_insecure_http"]:
            raise ValueError("memory_search requires HTTPS outside loopback unless allow_insecure_http is explicit")
    if result["enabled"] and (not result["base_url"] or not result["model"]):
        raise ValueError("enabled memory_search requires base_url and model")
    result["base_url"] = result["base_url"].rstrip("/")
    return result


def normalize(vector: object) -> list[float]:
    if not isinstance(vector, list) or not 1 <= len(vector) <= 16384:
        raise ValueError("Invalid embedding dimension")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise ValueError("Invalid embedding values")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid embedding norm")
    return [x / norm for x in vector]


class MemoryVectorIndex:
    def __init__(self, work_dir: str | Path, config: MemorySearchConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport
        root = Path(work_dir).resolve()
        self.root = root
        cache = (root / ".valecode" / "cache").resolve()
        if not cache.is_relative_to(root):
            raise ValueError("Memory vector cache escapes project")
        self.path = cache / "memory-vectors.sqlite3"
        self.identity = hashlib.sha256((config.base_url + "\n" + config.model).encode()).hexdigest()

    def _db(self) -> sqlite3.Connection:
        if self.path.is_symlink() or not self.path.parent.resolve().is_relative_to(self.root):
            raise ValueError("Memory vector cache escapes project")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=0.2)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (provider TEXT, path TEXT, digest TEXT, vector TEXT, PRIMARY KEY(provider,path))")
            return db
        except BaseException:
            db.close()
            raise

    def _load(self, headers: list[MemoryHeader]):
        documents = {}
        for header in headers:
            try:
                with Path(header.file_path).open(encoding="utf-8", errors="replace") as source:
                    text = header.filename + "\n" + header.description + "\n" + source.read(8000)
            except OSError:
                continue
            documents[header.file_path] = (hashlib.sha256(text.encode()).hexdigest(), text)
        with closing(self._db()) as db, db:
            rows = list(db.execute("SELECT path,digest,vector FROM vectors WHERE provider=?", (self.identity,)))
            cached = {}
            for path, digest, vector in rows:
                if path not in documents:
                    db.execute("DELETE FROM vectors WHERE provider=? AND path=?", (self.identity, path))
                elif documents[path][0] == digest:
                    try:
                        cached[path] = normalize(json.loads(vector))
                    except (ValueError, TypeError):
                        pass  # A malformed row is rebuilt, not used.
        return documents, cached

    def _save(self, batch, vectors):
        with closing(self._db()) as db, db:
            db.executemany("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?)", [
                (self.identity, path, digest, json.dumps(vector, allow_nan=False))
                for (path, (digest, _)), vector in zip(batch, vectors, strict=True)
            ])

    async def _embed(self, client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
        response = await client.post(self.config.base_url + "/embeddings", json={"model": self.config.model, "input": texts})
        response.raise_for_status()
        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embedding count mismatch")
        ordered = {}
        for row in data:
            if not isinstance(row, dict) or type(row.get("index")) is not int or not 0 <= row["index"] < len(texts) or row["index"] in ordered:
                raise ValueError("Invalid embedding response index")
            ordered[row["index"]] = normalize(row.get("embedding"))
        vectors = [ordered[i] for i in range(len(texts))]
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Embedding dimensions differ")
        return vectors

    async def rank(self, query: str, headers: list[MemoryHeader], limit: int = 40) -> list[str]:
        if not self.config.enabled or not headers or not query.strip() or limit <= 0:
            return []
        # One phase-wide deadline, not a fresh timeout per batch. Completed
        # batches survive a cold-cache timeout and warm up on later turns.
        async with asyncio.timeout(self.config.timeout_seconds):
            documents, cached = await asyncio.to_thread(self._load, headers)
            auth = {"Authorization": "Bearer " + self.config.api_key} if self.config.api_key else {}
            async with httpx.AsyncClient(headers=auth, timeout=self.config.timeout_seconds, transport=self.transport, trust_env=False, follow_redirects=False) as client:
                query_vector = (await self._embed(client, [query[:8000]]))[0]
                missing = [item for item in documents.items() if len(cached.get(item[0], [])) != len(query_vector)]
                for start in range(0, len(missing), 16):
                    batch = missing[start:start + 16]
                    vectors = await self._embed(client, [text for _, (_, text) in batch])
                    if any(len(vector) != len(query_vector) for vector in vectors):
                        raise ValueError("Embedding model changed dimensions")
                    await asyncio.to_thread(self._save, batch, vectors)
                    cached.update((item[0], vector) for item, vector in zip(batch, vectors, strict=True))
            scored = [(sum(a * b for a, b in zip(query_vector, vector, strict=True)), path) for path, vector in cached.items()]
            return [path for score, path in sorted(scored, reverse=True) if score >= 0.2][:limit]


def fuse_rankings(*rankings: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, path in enumerate(dict.fromkeys(ranking)):
            scores[path] = scores.get(path, 0) + 1 / (60 + rank + 1)
    return sorted(scores, key=lambda path: (-scores[path], path))
