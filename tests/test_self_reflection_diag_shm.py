"""Self-reflection + coding-explorer diagnostics SHM slots (2026-06-22).

Completes the Track-2 read-side migration: self_reflection_worker writes
SelfReasoningEngine.get_stats() / CodingExplorer.get_stats() to dedicated
variable-msgpack SHM slots, and the api ShmReaderBank surfaces them so
/v6/cognition/{self-reflection,coding-explorer} return live engine data
instead of a perma "not yet initialized" (the engines were always alive;
only the read path was stubbed `live = None`).

Run isolated:
  python -m pytest tests/test_self_reflection_diag_shm.py -v -p no:anchorpy
"""
import msgpack
import pytest

from titan_hcl.core.state_registry import (
    SELF_REFLECTION_STATE,
    CODING_EXPLORER_STATE,
    StateRegistryWriter,
)


@pytest.fixture
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _bank():
    # Imported lazily so the env override is already in place.
    from titan_hcl.api.shm_reader_bank import ShmReaderBank
    return ShmReaderBank(titan_id="T1")


def test_self_reflection_slot_roundtrip(shm_root):
    """Worker-side write_variable → api-side read_self_reflection_state()."""
    writer = StateRegistryWriter(SELF_REFLECTION_STATE, shm_root)
    payload = {
        "titan_id": "T1",
        "stats": {"total_introspections": 42, "observed_chains": 7,
                  "has_profile": True, "prediction_accuracy_ema": 0.83},
        "last_dream_state": None,
        "ts": 123.0,
    }
    writer.write_variable(msgpack.packb(payload, use_bin_type=True))

    got = _bank().read_self_reflection_state()
    assert got is not None, "slot must read back after a write"
    assert got["stats"]["total_introspections"] == 42
    assert got["stats"]["observed_chains"] == 7
    assert got["ts"] == 123.0


def test_coding_explorer_slot_roundtrip(shm_root):
    writer = StateRegistryWriter(CODING_EXPLORER_STATE, shm_root)
    payload = {
        "titan_id": "T1",
        "stats": {"total_experiments": 5, "sandbox_runs": 3},
        "sandbox_disabled": False,
        "sandbox_last_status": "ok",
        "ts": 456.0,
    }
    writer.write_variable(msgpack.packb(payload, use_bin_type=True))

    got = _bank().read_coding_explorer_state()
    assert got is not None
    assert got["stats"]["total_experiments"] == 5
    assert got["sandbox_disabled"] is False
    assert got["sandbox_last_status"] == "ok"


def test_cold_read_returns_none(shm_root):
    """No write yet → reader returns None (handler falls back to its
    'not yet initialized' message; never raises)."""
    assert _bank().read_self_reflection_state() is None
    assert _bank().read_coding_explorer_state() is None


def test_counters_survive_restart(tmp_path):
    """Cumulative counters must NOT reset to 0 on engine restart (2026-06-22):
    total_introspections reloads from COUNT(self_insights); observed_chains /
    observed_meta_chains reload from the self_reasoning_counters table."""
    import sqlite3
    from titan_hcl.logic.self_reasoning import SelfReasoningEngine

    db = str(tmp_path / "inner_memory.db")
    eng = SelfReasoningEngine(config={}, db_path=db)
    eng._observed_chains = 5
    eng._observed_meta_chains = 2
    # two persisted insights → cumulative introspection truth
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO self_insights(sub_mode,epoch,timestamp,data) "
                 "VALUES('audit',1,1.0,'{}')")
    conn.execute("INSERT INTO self_insights(sub_mode,epoch,timestamp,data) "
                 "VALUES('audit',2,2.0,'{}')")
    conn.commit()
    conn.close()
    eng.persist_counters()

    # simulate a Titan restart — fresh engine, same DB
    eng2 = SelfReasoningEngine(config={}, db_path=db)
    stats = eng2.get_stats()
    assert stats["total_introspections"] == 2  # from COUNT(self_insights)
    assert stats["observed_chains"] == 5       # from counters table
    assert stats["observed_meta_chains"] == 2
