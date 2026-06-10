"""Phase 7 — ActrBufferStore unit tests (D-SPEC-PHASE7 / INV-Syn-16/18).

Covers:
- DDL idempotent + PK-only (NO secondary ART indexes — INV-Syn-30 / D-SPEC-154)
- persist round-trips content + concept_ids + embedding_hash
- INSERT OR REPLACE semantics (re-persist updates same row)
- clear deletes row; idempotent on absent row
- read_all_for_chat returns expected shape; empty for unseen chat
- buffer_entities precedence + dedup + cap (covered in test_phase4_spreading.py)
- snapshot_export is atomic (tmp file disappears, target updated)
- snapshot payload shape (version/ts/writes_seen/clears_seen/chat_count/chats)
- embedding_hash is sha256(canonical_json({content, concept_ids}))
- ValueError on unknown buffer_name + empty chat_id
- writes_seen counter advances per persist; clears_seen per clear
"""
from __future__ import annotations

import hashlib
import json
import os

import duckdb
import pytest

from titan_hcl.synthesis.buffer_store import (
    ActrBufferStore, BUFFER_NAMES, MAX_CONTENT_BYTES,
    _canonical_payload_hash, _truncate,
)


@pytest.fixture()
def store(tmp_path):
    conn = duckdb.connect(":memory:")
    s = ActrBufferStore(
        duckdb_conn=conn,
        snapshot_path=str(tmp_path / "buffers_snapshot.json"),
    )
    yield s


# ── DDL ────────────────────────────────────────────────────────────────


def test_schema_idempotent_on_second_construct(tmp_path):
    conn = duckdb.connect(":memory:")
    s1 = ActrBufferStore(
        duckdb_conn=conn,
        snapshot_path=str(tmp_path / "s.json"),
    )
    # Second construct on the same connection re-runs DDL (idempotent).
    s2 = ActrBufferStore(
        duckdb_conn=conn,
        snapshot_path=str(tmp_path / "s.json"),
    )
    s2.persist(chat_id="x:y", buffer_name="goal", content="hello", concept_ids=[])
    rows = conn.execute("SELECT chat_id FROM actr_buffers").fetchall()
    assert rows == [("x:y",)]


def test_no_secondary_indexes_pk_only(store):
    """INV-Syn-30 / D-SPEC-154: actr_buffers is PK-ONLY — no secondary ART
    indexes (they were the runtime-corruption source behind the FATAL
    crash-loop). `duckdb_indexes()` lists only explicit secondary indexes;
    the PRIMARY KEY index is implicit + not reported → expect EMPTY."""
    rows = store._db.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE table_name='actr_buffers'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert names == set(), f"expected NO secondary indexes, found {names}"
    # The PK still enforces uniqueness (re-persist updates the same row, never
    # a duplicate) — proves the PK index is intact even though it's unlisted.
    store.persist(chat_id="c:s", buffer_name="goal", content="a", concept_ids=[])
    store.persist(chat_id="c:s", buffer_name="goal", content="b", concept_ids=[])
    cnt = store._db.execute(
        "SELECT COUNT(*) FROM actr_buffers WHERE chat_id='c:s' AND buffer_name='goal'"
    ).fetchone()[0]
    assert cnt == 1


def test_boot_self_heals_stray_secondary_indexes(tmp_path):
    """A pre-fix fleet DB carries the two stray secondary indexes; constructing
    the store DROPs them on boot (self-heal, no operator script — D-SPEC-154)."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE actr_buffers ("
        "  chat_id TEXT NOT NULL, buffer_name TEXT NOT NULL, content TEXT,"
        "  concept_ids TEXT, embedding_hash TEXT, updated_at DOUBLE NOT NULL,"
        "  PRIMARY KEY (chat_id, buffer_name))"
    )
    conn.execute("CREATE INDEX idx_actr_buffers_chat ON actr_buffers(chat_id)")
    conn.execute("CREATE INDEX idx_actr_buffers_updated ON actr_buffers(updated_at)")
    before = {r[0] for r in conn.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE table_name='actr_buffers'"
    ).fetchall()}
    assert before == {"idx_actr_buffers_chat", "idx_actr_buffers_updated"}
    # Boot the store on the corrupt-shaped DB → self-heal drops the indexes.
    ActrBufferStore(duckdb_conn=conn, snapshot_path=str(tmp_path / "s.json"))
    after = {r[0] for r in conn.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE table_name='actr_buffers'"
    ).fetchall()}
    assert after == set(), f"self-heal must drop stray secondary indexes, left {after}"


def test_degradation_probe_fires_on_latency_creep(tmp_path, monkeypatch):
    """When the rolling-avg persist latency crosses PERSIST_DEGRADE_AVG_MS the
    store fires on_degraded once (rate-limited) → the worker emits a WARN
    MODULE_ERROR on the SPEC cascade (D-SPEC-154 early-warning, INV-SDA-12)."""
    import titan_hcl.synthesis.buffer_store as bs

    conn = duckdb.connect(":memory:")
    calls: list[dict] = []
    # Monotonic clock so the cooldown logic is deterministic (start past cooldown).
    ticks = iter(range(10_000, 100_000))

    def _clock():
        return float(next(ticks))

    store = ActrBufferStore(
        duckdb_conn=conn,
        snapshot_path=str(tmp_path / "s.json"),
        clock=_clock,
        on_degraded=lambda **kw: calls.append(kw),
    )
    # Inject latencies above the threshold directly into the window — exercises
    # _maybe_emit_degraded without depending on real wall-clock persist timing.
    high = bs.PERSIST_DEGRADE_AVG_MS * 2.0
    for _ in range(bs.PERSIST_DEGRADE_MIN_SAMPLES):
        store._persist_latencies.append(high)
    store._max_persist_ms = high
    store._maybe_emit_degraded()
    assert len(calls) == 1, "degradation must fire once over-threshold + min-samples"
    assert calls[0]["avg_ms"] > bs.PERSIST_DEGRADE_AVG_MS
    assert calls[0]["samples"] >= bs.PERSIST_DEGRADE_MIN_SAMPLES
    # Rate-limited: a second immediate check inside the cooldown does NOT re-fire.
    store._maybe_emit_degraded()
    assert len(calls) == 1, "cooldown must suppress a second emit within the window"


def test_degradation_probe_silent_when_healthy(tmp_path):
    """Sub-threshold latencies never fire on_degraded (no false alarms)."""
    import titan_hcl.synthesis.buffer_store as bs

    conn = duckdb.connect(":memory:")
    calls: list[dict] = []
    store = ActrBufferStore(
        duckdb_conn=conn,
        snapshot_path=str(tmp_path / "s.json"),
        on_degraded=lambda **kw: calls.append(kw),
    )
    for _ in range(bs.PERSIST_DEGRADE_MIN_SAMPLES + 5):
        store._persist_latencies.append(0.2)  # healthy sub-ms
    store._maybe_emit_degraded()
    assert calls == []


# ── persist ────────────────────────────────────────────────────────────


def test_persist_round_trip(store):
    store.persist(
        chat_id="alice:s1", buffer_name="goal",
        content="debug rust panic", concept_ids=["rust", "panic"],
    )
    rows = store.read_all_for_chat("alice:s1")
    assert "goal" in rows
    row = rows["goal"]
    assert row["content"] == "debug rust panic"
    assert row["concept_ids"] == ["rust", "panic"]
    assert row["embedding_hash"]
    assert row["updated_at"] > 0


def test_persist_insert_or_replace_overwrites_row(store):
    store.persist(
        chat_id="alice:s1", buffer_name="goal",
        content="v1", concept_ids=["a"], ts=100.0,
    )
    store.persist(
        chat_id="alice:s1", buffer_name="goal",
        content="v2", concept_ids=["b"], ts=200.0,
    )
    rows = store.read_all_for_chat("alice:s1")
    assert rows["goal"]["content"] == "v2"
    assert rows["goal"]["concept_ids"] == ["b"]
    assert rows["goal"]["updated_at"] == 200.0


def test_persist_unknown_buffer_raises(store):
    with pytest.raises(ValueError):
        store.persist(
            chat_id="alice:s1", buffer_name="brain",
            content="x", concept_ids=[],
        )


def test_persist_empty_chat_raises(store):
    with pytest.raises(ValueError):
        store.persist(
            chat_id="", buffer_name="goal", content="x", concept_ids=[],
        )


def test_persist_truncates_oversize_content(store):
    huge = "X" * (MAX_CONTENT_BYTES + 1000)
    store.persist(
        chat_id="alice:s1", buffer_name="goal",
        content=huge, concept_ids=[],
    )
    rows = store.read_all_for_chat("alice:s1")
    assert len(rows["goal"]["content"].encode("utf-8")) <= MAX_CONTENT_BYTES


def test_writes_seen_counter_advances(store):
    assert store.stats()["writes_seen"] == 0
    store.persist(
        chat_id="alice:s1", buffer_name="goal",
        content="x", concept_ids=[],
    )
    store.persist(
        chat_id="alice:s1", buffer_name="retrieval",
        content="y", concept_ids=[],
    )
    assert store.stats()["writes_seen"] == 2


# ── clear ─────────────────────────────────────────────────────────────


def test_clear_deletes_row(store):
    store.persist(
        chat_id="alice:s1", buffer_name="goal",
        content="x", concept_ids=[],
    )
    store.clear(chat_id="alice:s1", buffer_name="goal")
    assert store.read_all_for_chat("alice:s1") == {}


def test_clear_idempotent_on_absent_row(store):
    # No prior persist — clear must not raise.
    store.clear(chat_id="alice:never", buffer_name="goal")
    assert store.read_all_for_chat("alice:never") == {}


def test_clear_advances_clears_seen(store):
    store.clear(chat_id="x:y", buffer_name="goal")
    store.clear(chat_id="x:y", buffer_name="retrieval")
    assert store.stats()["clears_seen"] == 2


def test_clear_unknown_buffer_raises(store):
    with pytest.raises(ValueError):
        store.clear(chat_id="x:y", buffer_name="nope")


# ── read_all_for_chat ─────────────────────────────────────────────────


def test_read_all_returns_empty_for_unseen_chat(store):
    assert store.read_all_for_chat("ghost") == {}


def test_read_all_handles_corrupt_concept_ids_json(store):
    store.persist(
        chat_id="x:y", buffer_name="goal",
        content="ok", concept_ids=["c1"],
    )
    # Manually inject a malformed row (simulates upstream corruption).
    store._db.execute(
        "UPDATE actr_buffers SET concept_ids = ? WHERE chat_id='x:y' AND buffer_name='goal'",
        ["not-json"],
    )
    rows = store.read_all_for_chat("x:y")
    # Falls back to empty list, doesn't crash.
    assert rows["goal"]["concept_ids"] == []


# ── snapshot_export ────────────────────────────────────────────────────


def test_snapshot_export_writes_payload(store, tmp_path):
    store.persist(
        chat_id="alice:s1", buffer_name="goal",
        content="g", concept_ids=["a"],
    )
    path = store._snapshot_path
    assert os.path.exists(path)
    with open(path) as f:
        payload = json.load(f)
    assert payload["version"] == 1
    assert payload["chat_count"] == 1
    assert payload["writes_seen"] == 1
    assert "alice:s1" in payload["chats"]
    assert payload["chats"]["alice:s1"]["goal"]["content"] == "g"


def test_snapshot_is_atomic_tmp_then_rename(store, tmp_path):
    """After a successful export there's no .tmp leftover."""
    store.persist(
        chat_id="x:y", buffer_name="goal",
        content="x", concept_ids=[],
    )
    assert not os.path.exists(store._snapshot_path + ".tmp")


def test_snapshot_export_soft_fails_on_unwritable_target(tmp_path):
    """Export soft-fails (logs WARN); persist surface does not raise."""
    conn = duckdb.connect(":memory:")
    bad_path = str(tmp_path / "nonexistent_dir_xyz/sub/snap.json")
    s = ActrBufferStore(duckdb_conn=conn, snapshot_path=bad_path)
    # The parent dir doesn't exist YET — but ActrBufferStore creates it.
    # Force an actual permission failure by making the target a directory.
    os.makedirs(bad_path, exist_ok=True)
    # Now snapshot_export will try to os.replace into a directory — fails.
    # Persist must NOT raise; just logs and continues.
    s.persist(chat_id="x:y", buffer_name="goal", content="x", concept_ids=[])


# ── embedding_hash ────────────────────────────────────────────────────


def test_embedding_hash_is_sha256_of_canonical_payload():
    h = _canonical_payload_hash("hello world", ["c1", "c2"])
    expected = hashlib.sha256(
        json.dumps(
            {"content": "hello world", "concept_ids": ["c1", "c2"]},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert h == expected


def test_embedding_hash_changes_with_concept_ids():
    h1 = _canonical_payload_hash("hello", ["a"])
    h2 = _canonical_payload_hash("hello", ["b"])
    assert h1 != h2


def test_truncate_keeps_under_cap():
    huge = "Z" * (MAX_CONTENT_BYTES + 5000)
    out = _truncate(huge)
    assert len(out.encode("utf-8")) <= MAX_CONTENT_BYTES


# ── BUFFER_NAMES ───────────────────────────────────────────────────────


def test_buffer_names_exposed_as_canonical():
    assert BUFFER_NAMES == ("goal", "retrieval", "imaginal", "perception")
