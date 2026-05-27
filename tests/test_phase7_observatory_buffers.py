"""Phase 7 — Observatory `/v6/synthesis/buffers/*` handler tests (§P7.G).

Covers:
- 4 handlers honor the soft-fail contract (200 + snapshot status on missing)
- list_chats returns chat ids from the snapshot
- read returns the requested buffer's payload; missing chat → empty defaults
- recent_writes sorts by updated_at desc; respects limit
- snapshot caps chats; reports counters
- read rejects unknown buffer name with ok=false
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from titan_hcl.api import synthesis_buffer_handlers as h


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    # Reset the cache between tests (mtime-keyed).
    h._SNAPSHOT_CACHE.clear()
    return tmp_path


def _write_snapshot(data_dir, payload):
    path = os.path.join(str(data_dir), "buffers_snapshot.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def _make_request(query: dict = None):
    req = MagicMock()
    req.query_params = query or {}
    return req


# ── list_chats ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_chats_empty_when_no_snapshot(isolate_data_dir):
    r = await h.get_v6_synthesis_buffers_list_chats(_make_request())
    assert r["ok"] is True
    assert r["snapshot"] == "missing"
    assert r["chat_count"] == 0
    assert r["chats"] == []


@pytest.mark.asyncio
async def test_list_chats_returns_sorted_chat_ids(isolate_data_dir):
    _write_snapshot(isolate_data_dir, {
        "version": 1, "ts": 1.0, "writes_seen": 2,
        "clears_seen": 0, "chat_count": 2,
        "chats": {"bob:s1": {"goal": {}}, "alice:s1": {"goal": {}}},
    })
    r = await h.get_v6_synthesis_buffers_list_chats(_make_request())
    assert r["ok"] is True
    assert r["snapshot"] == "ok"
    assert r["chats"] == ["alice:s1", "bob:s1"]
    assert r["chat_count"] == 2


# ── read ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_returns_buffer_payload(isolate_data_dir):
    _write_snapshot(isolate_data_dir, {
        "version": 1, "ts": 1.0, "writes_seen": 1, "clears_seen": 0,
        "chat_count": 1,
        "chats": {
            "alice:s1": {
                "goal": {
                    "content": "debug rust",
                    "concept_ids": ["rust", "debugging"],
                    "embedding_hash": "abcdef",
                    "updated_at": 500.0,
                },
            },
        },
    })
    r = await h.get_v6_synthesis_buffers_read(
        _make_request({"chat_id": "alice:s1", "buffer": "goal"})
    )
    assert r["ok"] is True
    assert r["content"] == "debug rust"
    assert r["concept_ids"] == ["rust", "debugging"]
    assert r["embedding_hash"] == "abcdef"
    assert r["updated_at"] == 500.0


@pytest.mark.asyncio
async def test_read_missing_chat_returns_empty(isolate_data_dir):
    _write_snapshot(isolate_data_dir, {
        "version": 1, "ts": 1.0, "chat_count": 0, "chats": {},
    })
    r = await h.get_v6_synthesis_buffers_read(
        _make_request({"chat_id": "ghost", "buffer": "goal"})
    )
    assert r["ok"] is True
    assert r["content"] == ""
    assert r["concept_ids"] == []
    assert r["updated_at"] == 0.0


@pytest.mark.asyncio
async def test_read_unknown_buffer_returns_ok_false(isolate_data_dir):
    r = await h.get_v6_synthesis_buffers_read(
        _make_request({"chat_id": "x", "buffer": "brain"})
    )
    assert r["ok"] is False
    assert "buffer" in r["error"]


@pytest.mark.asyncio
async def test_read_missing_chat_id_returns_ok_false(isolate_data_dir):
    r = await h.get_v6_synthesis_buffers_read(_make_request())
    assert r["ok"] is False
    assert "chat_id" in r["error"]


# ── recent_writes ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recent_writes_sorts_by_updated_at_desc(isolate_data_dir):
    _write_snapshot(isolate_data_dir, {
        "version": 1, "ts": 1.0, "writes_seen": 3, "clears_seen": 1,
        "chat_count": 2,
        "chats": {
            "alice:s1": {
                "goal":      {"content": "g", "concept_ids": [],
                              "embedding_hash": "", "updated_at": 300.0},
                "retrieval": {"content": "r", "concept_ids": [],
                              "embedding_hash": "", "updated_at": 100.0},
            },
            "bob:s1": {
                "goal":      {"content": "g", "concept_ids": [],
                              "embedding_hash": "", "updated_at": 200.0},
            },
        },
    })
    r = await h.get_v6_synthesis_buffers_recent_writes(_make_request())
    assert r["ok"] is True
    tss = [w["updated_at"] for w in r["writes"]]
    assert tss == sorted(tss, reverse=True)
    assert r["writes"][0]["chat_id"] == "alice:s1"
    assert r["writes"][0]["buffer_name"] == "goal"
    assert r["writes_seen"] == 3
    assert r["clears_seen"] == 1


@pytest.mark.asyncio
async def test_recent_writes_respects_limit(isolate_data_dir):
    chats = {
        f"alice:{i}": {
            "goal": {"content": "g", "concept_ids": [],
                     "embedding_hash": "", "updated_at": float(i)},
        }
        for i in range(50)
    }
    _write_snapshot(isolate_data_dir, {
        "version": 1, "ts": 1.0, "chat_count": 50, "chats": chats,
    })
    r = await h.get_v6_synthesis_buffers_recent_writes(
        _make_request({"limit": "5"})
    )
    assert len(r["writes"]) == 5


# ── snapshot ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_returns_full_payload_capped(isolate_data_dir):
    chats = {
        f"alice:{i}": {
            "goal": {"content": f"g{i}", "concept_ids": [],
                     "embedding_hash": "", "updated_at": float(i)},
        }
        for i in range(10)
    }
    _write_snapshot(isolate_data_dir, {
        "version": 1, "ts": 1.0, "writes_seen": 10, "clears_seen": 0,
        "chat_count": 10, "chats": chats,
    })
    r = await h.get_v6_synthesis_buffers_snapshot(_make_request({"limit": "3"}))
    assert r["ok"] is True
    assert r["chat_count"] == 10
    assert len(r["chats"]) == 3
    # Most recent first.
    chat_keys = list(r["chats"].keys())
    assert chat_keys[0] == "alice:9"


@pytest.mark.asyncio
async def test_snapshot_missing_returns_safe_defaults(isolate_data_dir):
    r = await h.get_v6_synthesis_buffers_snapshot(_make_request())
    assert r["ok"] is True
    assert r["snapshot"] == "missing"
    assert r["chats"] == {}
    assert r["chat_count"] == 0
