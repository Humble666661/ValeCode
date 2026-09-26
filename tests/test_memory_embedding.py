from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from types import SimpleNamespace

import httpx
import pytest

from valecode.config import AppConfig, _apply_env_overrides, _load_single_file, _merge_config
from valecode.memory.embedding import MemorySearchConfig, MemoryVectorIndex, normalize, fuse_rankings
from valecode.memory.recall import find_relevant_memories, scan_memory_files
from valecode.memory.search_index import MemorySearchIndex
from valecode.validator import ConfigError, validate_config_structure


def config(**kwargs):
    return MemorySearchConfig(enabled=True, base_url="http://127.0.0.1:9999/v1", model="test-vector", **kwargs)


def provider(calls):
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload["input"])
        assert request.url.path == "/v1/embeddings"
        assert payload["model"] == "test-vector"
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1, 0] if ("journey" in text or "旅游" in text) else [0, 1]}
            for i, text in reversed(list(enumerate(payload["input"])))
        ]})
    return httpx.MockTransport(respond)


@pytest.mark.asyncio
async def test_semantic_synonyms_cache_edit_delete_and_model_identity(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    target = memory / "travel.md"
    target.write_text("旅游的注意事项", encoding="utf-8")
    (memory / "database.md").write_text("数据迁移", encoding="utf-8")
    calls = []
    index = MemoryVectorIndex(tmp_path, config(), transport=provider(calls))
    headers = scan_memory_files(memory, "project")
    assert MemorySearchIndex(tmp_path).rank("journey", headers) == []
    assert await index.rank("journey", headers) == [str(target)]
    assert len(calls) == 2
    assert await index.rank("journey", headers) == [str(target)]
    assert len(calls) == 3  # Only the new query, document vectors are reused.
    target.write_text("数据库", encoding="utf-8")
    assert await index.rank("journey", scan_memory_files(memory, "project")) == []
    assert len(calls) == 5
    target.unlink()
    await index.rank("journey", scan_memory_files(memory, "project"))
    with sqlite3.connect(index.path) as db:
        assert db.execute("SELECT count(*) FROM vectors WHERE path=?", (str(target),)).fetchone()[0] == 0
        assert "journey" not in str(db.execute("SELECT * FROM vectors").fetchall())
    assert MemoryVectorIndex(tmp_path, MemorySearchConfig(model="another")).identity != index.identity


@pytest.mark.asyncio
async def test_disabled_does_not_contact_provider(tmp_path):
    async def forbidden(request):
        pytest.fail("Disabled semantic recall must not upload memory")
    memory = tmp_path / "note.md"
    memory.write_text("private", encoding="utf-8")
    index = MemoryVectorIndex(tmp_path, MemorySearchConfig(), transport=httpx.MockTransport(forbidden))
    assert await index.rank("query", scan_memory_files(tmp_path, "project")) == []
    assert not index.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", ["timeout", "malformed", "http", "cache"])
async def test_semantic_failure_keeps_lexical_and_bounded_latency(tmp_path, broken):
    memory = tmp_path / "memory"
    memory.mkdir()
    for i in range(86):
        (memory / f"note-{i}.md").write_text("boring", encoding="utf-8")
    target = memory / "target.md"
    target.write_text("旅游指南", encoding="utf-8")
    async def fail(request):
        if broken == "timeout":
            await asyncio.sleep(2)
        return httpx.Response(503 if broken == "http" else 200, json={"data": [{"index": 0, "embedding": []}]})
    index = MemoryVectorIndex(tmp_path, config(timeout_seconds=0.05), transport=httpx.MockTransport(fail))
    if broken == "cache":
        index.path.parent.mkdir(parents=True)
        index.path.write_bytes(b"invalid sqlite")
    observed = []
    async def selector(_system, message):
        observed.append(message)
        return json.dumps({"selected_memories": [str(target)]})
    start = time.perf_counter()
    result = await find_relevant_memories("旅游", None, memory, None, None, selector,
        index=MemorySearchIndex(tmp_path), vector_index=index)
    assert [entry.path for entry in result] == [str(target)]
    assert observed[0].count(".md") < 50
    assert time.perf_counter() - start < 1.5


@pytest.mark.asyncio
async def test_large_chinese_manifest_semantic_only_and_surfaced_filter(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    for i in range(85):
        (memory / f"note-{i}.md").write_text("数据库", encoding="utf-8")
    target = memory / "special.md"
    target.write_text("旅游的注意事项", encoding="utf-8")
    index = MemoryVectorIndex(tmp_path, config(), transport=provider([]))
    async def selector(_system, message):
        assert str(target) in message
        assert message.count(".md") < 50
        return json.dumps({"selected_memories": [str(target)]})
    result = await find_relevant_memories("journey", None, memory, None, None, selector,
        index=MemorySearchIndex(tmp_path), vector_index=index)
    assert [entry.path for entry in result] == [str(target)]
    async def empty(_system, message):
        assert str(target) not in message
        return '{"selected_memories": []}'
    assert await find_relevant_memories("journey", None, memory, None, {str(target)}, empty,
        vector_index=index) == []


@pytest.mark.parametrize("vector", [[], [0, 0], [True, 1], [float("nan"), 1], [float("inf"), 1], "bad"])
def test_reject_invalid_vectors(vector):
    with pytest.raises(ValueError):
        normalize(vector)


@pytest.mark.parametrize("raw", [{"enabled": "true"}, {"enabled": True}, {"timeout_seconds": True}, {"timeout_seconds": float("inf")}, {"timeout_seconds": 5}, {"oops": True}, {"base_url": "http://remote.example/v1"}, {"base_url": "https://user:secret@example.com/v1"}])
def test_strict_config(raw):
    with pytest.raises(ConfigError):
        validate_config_structure({"providers": [], "memory_search": raw})


def test_dotenv_overrides_and_partial_merge(tmp_path):
    provider_yaml = "providers:\n  - name: test\n    protocol: openai-compat\n    base_url: https://example.com/v1\n    model: chat\n"
    base = tmp_path / "base.yaml"
    base.write_text(provider_yaml + "memory_search:\n  base_url: https://example.com/v1\n  model: local\n  api_key: ${VECTOR_KEY}\n", encoding="utf-8")
    loaded = _load_single_file(base, {"VECTOR_KEY": "secret"})
    assert loaded.memory_search.api_key == "secret"
    assert "secret" not in repr(loaded.memory_search)
    override = tmp_path / "override.yaml"
    override.write_text(provider_yaml + "memory_search:\n  timeout_seconds: 1\n", encoding="utf-8")
    merged = _merge_config(loaded, _load_single_file(override, {}))
    assert merged.memory_search.model == "local"
    merged = _apply_env_overrides(merged, {"VALECODE_MEMORY_ENABLED": "true", "VALECODE_MEMORY_MODEL": "changed"})
    assert merged.memory_search.enabled and merged.memory_search.model == "changed"
    assert fuse_rankings(["a", "b"], ["b", "c"])[0] == "b"


@pytest.mark.asyncio
async def test_tui_prefetch_wires_semantic_and_uses_worker_threads(tmp_path, monkeypatch):
    from valecode import app as module
    from valecode.memory.recall import RelevantMemory
    app = module.ValeCodeApp([], memory_search_config=config())
    app.agent = SimpleNamespace(work_dir=str(tmp_path))
    app.memory_manager = SimpleNamespace(user_mem_dir=tmp_path, project_mem_dir=tmp_path)
    app._selected_provider = object()
    app.session = SimpleNamespace(session_id="s")
    memory = tmp_path / "note.md"
    memory.write_text("旅游", encoding="utf-8")
    async def find(**kwargs):
        assert isinstance(kwargs["vector_index"], MemoryVectorIndex)
        return [RelevantMemory(str(memory), 0)]
    monkeypatch.setattr(module, "find_relevant_memories", find)
    assert (await app._prefetch_relevant_memories("journey")).paths == [str(memory)]
    assert app.memory_recall_metrics["selected_count"] == 1
    assert 0 <= app.memory_recall_metrics["duration_ms"] < 1500
    assert "journey" not in repr(app.memory_recall_metrics)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [
    [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}],
    [{"index": True, "embedding": [1, 0]}, {"index": 1, "embedding": [1, 0]}],
    [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, 0, 0]}],
])
async def test_protocol_rejects_duplicates_boolean_indices_and_dimensions(tmp_path, data):
    index = MemoryVectorIndex(tmp_path, config())
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": data}))) as client:
        with pytest.raises(ValueError):
            await index._embed(client, ["first", "second"])


@pytest.mark.asyncio
async def test_cancellation_closes_pending_request_and_does_not_call_selector(tmp_path):
    memory = tmp_path / "note.md"
    memory.write_text("旅游", encoding="utf-8")
    entered, closed = asyncio.Event(), asyncio.Event()
    async def respond(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    index = MemoryVectorIndex(tmp_path, config(), transport=httpx.MockTransport(respond))
    task = asyncio.create_task(index.rank("journey", scan_memory_files(tmp_path, "project")))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
