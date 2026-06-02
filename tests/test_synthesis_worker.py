"""Phase 1 synthesis_worker tests — D-SPEC-123 (SPEC v1.56.0 §25 / §9.B).

Validates: PlugRegistry contract, ActivationStore persistence + recompute,
SynthStatusWriter SHM payload round-trip + struct layout. The worker entry
function `synthesis_worker_main` is integration-tested via a short-lived
recv/send queue smoke test.
"""
from __future__ import annotations

import os
import queue
import struct
import threading
import time

import pytest

from titan_hcl import bus
from titan_hcl.core.direct_memory import TitanDuckDB
from titan_hcl.modules.synthesis_worker import (
    HEARTBEAT_INTERVAL_S,
    RECOMPUTE_INTERVAL_S,
    SYNTH_STATUS_PAYLOAD_BYTES,
    SYNTH_STATUS_SLOT_NAME,
    ActivationStore,
    PlugRegistry,
    SynthStatusWriter,
    synthesis_worker_main,
)


# ─────────────────────────────────────────────────────────────────────────
# PlugRegistry
# ─────────────────────────────────────────────────────────────────────────

def test_plug_registry_empty_at_boot():
    reg = PlugRegistry()
    counts = reg.counts()
    assert counts == {"substrate": 0, "truth_oracle": 0, "meaning_oracle": 0,
                      "proof": 0, "tool": 0}


def test_plug_registry_register_and_get():
    reg = PlugRegistry()
    sentinel = object()
    reg.register("truth_oracle", "x_oracle", sentinel)
    assert reg.get("truth_oracle", "x_oracle") is sentinel
    assert reg.list("truth_oracle") == ["x_oracle"]
    assert reg.counts()["truth_oracle"] == 1


def test_plug_registry_rejects_unknown_kind():
    reg = PlugRegistry()
    with pytest.raises(ValueError, match="unknown plug kind"):
        reg.register("not_a_kind", "x", object())


# ─────────────────────────────────────────────────────────────────────────
# ActivationStore — DuckDB persistence
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_db_path(tmp_path):
    # Construct via TitanDuckDB so the activation_state schema lands the
    # way production would; close immediately so ActivationStore opens it.
    db_path = tmp_path / "titan_memory.duckdb"
    db = TitanDuckDB(str(db_path))
    db._conn.close()
    return str(db_path)


def test_activation_store_record_access_creates_state(fresh_db_path):
    store = ActivationStore(fresh_db_path)
    try:
        store.record_access("kuzu:NODE_1", ts=100.0)
        assert store.items_tracked() == 1
    finally:
        store.close()


def test_activation_store_recompute_persists_to_duckdb(fresh_db_path):
    import duckdb
    store = ActivationStore(fresh_db_path)
    try:
        store.record_access("kuzu:NODE_1", ts=100.0)
        n = store.recompute_and_persist(now=200.0)
        assert n == 1
    finally:
        store.close()
    # Re-open the raw DB and verify the row was written.
    con = duckdb.connect(fresh_db_path, read_only=True)
    row = con.execute(
        "SELECT item_id, access_count, base_level, last_recompute "
        "FROM activation_state WHERE item_id = ?", ("kuzu:NODE_1",)
    ).fetchone()
    con.close()
    assert row is not None
    assert row[0] == "kuzu:NODE_1"
    assert row[1] == 1
    assert row[2] != 0.0
    assert row[3] == 200.0


def test_activation_store_resumes_state_on_reopen(fresh_db_path):
    # Boot 1: write some access state.
    store1 = ActivationStore(fresh_db_path)
    store1.record_access("kuzu:NODE_1", ts=100.0)
    store1.record_access("kuzu:NODE_1", ts=150.0)
    store1.record_access("kuzu:NODE_2", ts=200.0)
    store1.recompute_and_persist(now=300.0)
    store1.close()
    # Boot 2: should load both states from DuckDB.
    store2 = ActivationStore(fresh_db_path)
    try:
        assert store2.items_tracked() == 2
        # NODE_1 was accessed twice — check the log resumed.
        st = store2._cache["kuzu:NODE_1"]
        assert st.access_count == 2
        assert st.access_log == [100.0, 150.0]
        assert st.first_access == 100.0
    finally:
        store2.close()


def test_activation_store_recompute_idempotent_no_new_access(fresh_db_path):
    store = ActivationStore(fresh_db_path)
    try:
        store.record_access("kuzu:NODE_1", ts=100.0)
        n1 = store.recompute_and_persist(now=200.0)
        assert n1 == 1
        # Second recompute at same `now` should be a no-op (B_i unchanged).
        n2 = store.recompute_and_persist(now=200.0)
        assert n2 == 0
    finally:
        store.close()


# ─────────────────────────────────────────────────────────────────────────
# SynthStatusWriter — SHM watermark
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated_shm(monkeypatch, tmp_path):
    # Point SHM root at a tmp dir so the test doesn't touch real Titan SHM.
    shm_dir = tmp_path / "shm"
    monkeypatch.setenv("TITAN_SHM_ROOT", str(shm_dir))
    yield shm_dir


def test_synth_status_writer_initial_publish_zero(isolated_shm):
    writer = SynthStatusWriter(titan_id="test")
    try:
        # The StateRegistryWriter constructor publishes an all-zero buffer
        # at boot — verify the file exists at the expected path with the
        # right size envelope.
        slot_path = isolated_shm / f"{SYNTH_STATUS_SLOT_NAME}.bin"
        assert slot_path.exists()
        assert slot_path.stat().st_size > SYNTH_STATUS_PAYLOAD_BYTES
    finally:
        writer.close()


def test_synth_status_writer_publish_round_trip(isolated_shm):
    from titan_hcl.core.state_registry import RegistrySpec, StateRegistryReader
    import numpy as np
    writer = SynthStatusWriter(titan_id="test")
    try:
        writer.publish(
            last_consistent_event_ts=1234.5,
            last_recompute_ts=1234.5,
            items_tracked=7,
            recompute_count_increment=1,
        )
        # Read back via a fresh StateRegistryReader against the same spec.
        spec = RegistrySpec(
            name=SYNTH_STATUS_SLOT_NAME,
            dtype=np.dtype(np.uint8),
            shape=(SYNTH_STATUS_PAYLOAD_BYTES,),
            feature_flag="",
            schema_version=1,
            variable_size=True,
        )
        reader = StateRegistryReader(spec, isolated_shm)
        payload = reader.read_variable()
        reader.close()
        assert payload is not None
        assert len(payload) == SYNTH_STATUS_PAYLOAD_BYTES
        last_ts, recompute_ts, items, count = struct.unpack("<ddII", payload)
        assert last_ts == 1234.5
        assert recompute_ts == 1234.5
        assert items == 7
        assert count == 1
    finally:
        writer.close()


def test_synth_status_writer_publish_increments_recompute_count(isolated_shm):
    writer = SynthStatusWriter(titan_id="test")
    try:
        writer.publish(items_tracked=1, recompute_count_increment=1)
        writer.publish(items_tracked=2, recompute_count_increment=1)
        writer.publish(items_tracked=2, recompute_count_increment=1)
        # Read internal state directly (no separate reader needed for this
        # specific check — the round-trip test above proves SHM works).
        assert writer._recompute_count == 3
        assert writer._items_tracked == 2
    finally:
        writer.close()


# ─────────────────────────────────────────────────────────────────────────
# synthesis_worker_main — integration smoke test
# ─────────────────────────────────────────────────────────────────────────

def test_synthesis_worker_main_boots_and_shuts_down(fresh_db_path, isolated_shm):
    """Full worker lifecycle in-process: boot → record_access → recompute
    → shutdown. Asserts MODULE_READY emission + MEMORY_RETRIEVAL_USED
    handling + clean shutdown via MODULE_SHUTDOWN.

    Uses an absurdly short recompute_interval_s (0.5s) so the recompute
    loop fires at least once before shutdown.
    """
    recv_q: queue.Queue = queue.Queue()
    send_q: queue.Queue = queue.Queue()

    config = {
        "titan_id": "test",
        "memory_db_path": fresh_db_path,
        "recompute_interval_s": 0.5,
    }

    t = threading.Thread(
        target=synthesis_worker_main,
        args=(recv_q, send_q, "synthesis_test", config),
        daemon=True, name="synthesis_test_main")
    t.start()

    # Deterministic timing (NOT fixed sleeps — boot in a degraded test env can
    # take several seconds, and the recompute persist is submitted to the
    # SynthesisWriter thread asynchronously). We poll the SYNTHESIS_RECOMPUTE_
    # DONE payloads off send_q. MODULE_READY was RETIRED in Phase 11 §11.I.2
    # (D2 — no dual-publish; the SHM slot state is the readiness contract now).
    seen: list[dict] = []

    def _drain() -> None:
        while not send_q.empty():
            try:
                seen.append(send_q.get_nowait())
            except queue.Empty:
                break

    def _types() -> list:
        return [m.get("type") for m in seen]

    def _wait(pred, timeout: float = 25.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            _drain()
            if pred():
                return True
            time.sleep(0.1)
        return False

    # Boot complete + recompute loop alive (a recompute pass was emitted).
    assert _wait(lambda: bus.SYNTHESIS_RECOMPUTE_DONE in _types()), (
        f"worker never emitted a recompute pass (boot failed?): {_types()}")

    # Send the cited access event, then wait for a recompute pass that ACTUALLY
    # persisted a row (items_recomputed>=1) — that proves the recv loop recorded
    # the item AND a recompute materialized+submitted the persist to the writer.
    # Counting passes alone is racy: passes that fire before the recv loop
    # records the event report items_recomputed=0.
    recv_q.put({
        "type": bus.MEMORY_RETRIEVAL_USED,
        # used_by_llm=True is REQUIRED by the Phase 9 INV-Syn-23 cited gate
        # (synthesis_worker.py:2096): only LLM-CITED items are reinforced via
        # record_access; surfaced-not-cited is telemetry-only. Pre-Phase-9 the
        # field didn't exist, so this test (written then) omitted it.
        "payload": {
            "item_id": "kuzu:NODE_42", "ts": time.time(), "used_by_llm": True,
        },
    })

    def _persisted_a_row() -> bool:
        return any(
            m.get("type") == bus.SYNTHESIS_RECOMPUTE_DONE
            and (m.get("payload") or {}).get("items_recomputed", 0) >= 1
            for m in seen)
    assert _wait(_persisted_a_row), (
        f"no recompute pass persisted the access item: {_types()}")

    # Shutdown — store.close() CHECKPOINTs the writer-persisted rows (FIFO after
    # the submitted persist, so the row is durably flushed before we read).
    recv_q.put({"type": bus.MODULE_SHUTDOWN, "payload": {}})
    t.join(timeout=10.0)
    assert not t.is_alive(), "synthesis_worker did not shut down cleanly"
    _drain()

    assert bus.SYNTHESIS_RECOMPUTE_DONE in _types(), (
        f"SYNTHESIS_RECOMPUTE_DONE missing from emitted types: {_types()}")

    # Verify the access landed in DuckDB.
    import duckdb
    con = duckdb.connect(fresh_db_path, read_only=True)
    rows = con.execute(
        "SELECT item_id, access_count FROM activation_state"
    ).fetchall()
    con.close()
    found = {r[0]: r[1] for r in rows}
    assert "kuzu:NODE_42" in found, (
        f"MEMORY_RETRIEVAL_USED never landed in activation_state: {found}")
    assert found["kuzu:NODE_42"] == 1


def test_synthesis_worker_ignores_malformed_retrieval_events(fresh_db_path, isolated_shm):
    """Defensive: missing/wrong-typed item_id or ts must be silently
    ignored, not crash the recv loop."""
    recv_q: queue.Queue = queue.Queue()
    send_q: queue.Queue = queue.Queue()
    config = {
        "titan_id": "test",
        "memory_db_path": fresh_db_path,
        "recompute_interval_s": 0.5,
    }
    t = threading.Thread(
        target=synthesis_worker_main,
        args=(recv_q, send_q, "synthesis_test", config),
        daemon=True, name="synthesis_test_main")
    t.start()
    time.sleep(0.5)

    # Malformed events — should all be no-ops.
    for bad_payload in (
        {},                                  # missing both
        {"item_id": "x"},                     # missing ts
        {"ts": time.time()},                  # missing item_id
        {"item_id": 42, "ts": time.time()},   # wrong item_id type
        {"item_id": "x", "ts": "not a number"},   # wrong ts type
    ):
        recv_q.put({"type": bus.MEMORY_RETRIEVAL_USED, "payload": bad_payload})

    time.sleep(0.5)
    recv_q.put({"type": bus.MODULE_SHUTDOWN, "payload": {}})
    t.join(timeout=5.0)
    assert not t.is_alive()

    # No items should have been recorded.
    import duckdb
    con = duckdb.connect(fresh_db_path, read_only=True)
    n = con.execute("SELECT COUNT(*) FROM activation_state").fetchone()[0]
    con.close()
    assert n == 0
