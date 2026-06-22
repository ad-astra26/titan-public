"""Phase A — RFP_worker_telemetry §7.A. The telemetry core lib.

Verifies: timed() records (duration/feature/trigger/ctx/rss), the stall flag,
the bounded ring, persistence surviving a re-open (restart sim), retention prune,
never-raises (INV-TEL-5), and the RssAnon parse.

Run isolated: python -m pytest tests/test_worker_telemetry.py -v -p no:anchorpy
"""
import json
import sqlite3
import time

import pytest

from titan_hcl.logic.worker_telemetry import (
    Telemetry, _read_rss_anon_mb, _read_proc_status_kb)


def _tel(tmp_path, **over):
    cfg = {"db_dir": str(tmp_path), "flush_s": 0.05, "mem_sample_s": 0.05}
    cfg.update(over)
    return Telemetry("test", cfg)


def _rows(tel, table="op_events"):
    conn = sqlite3.connect(tel._db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    finally:
        conn.close()


# ── timed() records the full row ─────────────────────────────────────────────

def test_timed_records_op(tmp_path):
    tel = _tel(tmp_path)
    with tel.timed("consolidation", feature="consolidation", trigger_id="t1",
                   clusters=42):
        time.sleep(0.01)
    tel.flush_now()
    rows = _rows(tel)
    assert len(rows) == 1
    r = rows[0]
    assert r["op"] == "consolidation" and r["feature"] == "consolidation"
    assert r["trigger_id"] == "t1"
    assert r["duration_ms"] >= 8
    assert json.loads(r["ctx_json"])["clusters"] == 42
    assert r["rss_anon_mb"] > 0          # real RssAnon read
    tel.close()


def test_stall_flag(tmp_path):
    tel = _tel(tmp_path, warn_ms=5)      # tiny threshold so a 10ms op is a "stall"
    with tel.timed("slow", feature="x"):
        time.sleep(0.01)
    tel.flush_now()
    assert _rows(tel)[0]["stall"] == 1
    tel.close()


def test_record_stage(tmp_path):
    tel = _tel(tmp_path)
    tel.record_stage("prehook:after_vcb_recall", 1100.0, rss_mb=384.0,
                     feature="recall", trigger_id="t9")
    tel.flush_now()
    r = _rows(tel)[0]
    assert r["op"] == "prehook:after_vcb_recall" and r["feature"] == "recall"
    assert r["duration_ms"] == 1100.0 and r["rss_anon_mb"] == 384.0
    tel.close()


# ── ring is bounded, never blocks ────────────────────────────────────────────

def test_ring_bounded_drops_oldest(tmp_path):
    # tiny ring; pump more than capacity before any flush → oldest dropped, no block
    tel = _tel(tmp_path, ring_cap=5, flush_s=999)   # flusher won't fire during test
    for i in range(50):
        tel.record_stage(f"op{i}", 1.0, rss_mb=1.0, feature="f")
    assert len(tel._ring) == 5            # bounded — never grew unbounded
    tel.flush_now()
    rows = _rows(tel)
    assert len(rows) == 5
    assert {r["op"] for r in rows} == {f"op{i}" for i in range(45, 50)}  # newest 5
    tel.close()


# ── persistence survives a restart (re-open the same DB) ─────────────────────

def test_survives_restart(tmp_path):
    tel = _tel(tmp_path)
    tel.record_stage("before_restart", 5.0, rss_mb=1.0, feature="f")
    tel.flush_now()
    tel.close()
    # "restart": a fresh Telemetry on the SAME db_dir → prior rows still there
    tel2 = _tel(tmp_path)
    tel2.record_stage("after_restart", 5.0, rss_mb=1.0, feature="f")
    tel2.flush_now()
    ops = {r["op"] for r in _rows(tel2)}
    assert "before_restart" in ops and "after_restart" in ops   # spans restart
    tel2.close()


# ── retention prune caps rows ────────────────────────────────────────────────

def test_prune_removes_old(tmp_path):
    tel = _tel(tmp_path, retention_days=1.0)
    tel.record_stage("recent", 1.0, rss_mb=1.0, feature="f")
    tel.flush_now()
    # inject an ancient row directly + prune
    conn = tel._connect()
    try:
        conn.execute("INSERT INTO op_events(ts,op,feature,trigger_id,duration_ms,"
                     "rss_anon_mb,ctx_json,stall) VALUES(?,?,?,?,?,?,?,?)",
                     (time.time() - 10 * 86400, "ancient", "f", None, 1, 1, None, 0))
        conn.commit()
        tel._prune(conn)
        conn.commit()
    finally:
        conn.close()
    ops = {r["op"] for r in _rows(tel)}
    assert "recent" in ops and "ancient" not in ops    # >retention pruned
    tel.close()


# ── never raises (INV-TEL-5) ─────────────────────────────────────────────────

def test_never_raises_on_bad_ctx(tmp_path):
    tel = _tel(tmp_path)
    class Unjsonable:
        pass
    # non-JSON ctx must NOT raise (default=str handles it)
    with tel.timed("op", feature="f", weird=Unjsonable()):
        pass
    tel.flush_now()
    assert len(_rows(tel)) == 1
    tel.close()


def test_disabled_is_noop(tmp_path):
    tel = Telemetry("test", {"db_dir": str(tmp_path), "enabled": False})
    with tel.timed("op", feature="f"):       # no-op, no DB
        pass
    tel.record_stage("op2", 1.0)
    assert tel._enabled is False
    tel.close()                              # no error


# ── RssAnon parse ────────────────────────────────────────────────────────────

def test_rss_anon_parse():
    # the live process has a RssAnon line → > 0 and < VmRSS
    anon = _read_rss_anon_mb()
    assert anon > 0
    assert _read_proc_status_kb("VmRSS") >= _read_proc_status_kb("RssAnon")


def test_proc_status_missing_field_is_zero():
    assert _read_proc_status_kb("NoSuchField_xyz") == 0.0


# ══ Phase C ══════════════════════════════════════════════════════════════════

# ── C2: heartbeat-gap (only abnormal gaps recorded) ──────────────────────────

def test_record_heartbeat_gap(tmp_path):
    tel = _tel(tmp_path, hb_gap_warn_ms=20000)
    tel.record_heartbeat_gap(25000.0)     # > threshold → a missed-beat row
    tel.record_heartbeat_gap(10000.0)     # normal ~10s beat → NOT recorded
    tel.flush_now()
    rows = _rows(tel)
    assert len(rows) == 1
    assert rows[0]["op"] == "HEARTBEAT_GAP" and rows[0]["feature"] == "heartbeat"
    assert rows[0]["duration_ms"] == 25000.0 and rows[0]["stall"] == 1
    tel.close()


def test_get_active_telemetry_single_instance(tmp_path):
    import titan_hcl.logic.worker_telemetry as wt
    saved = dict(wt._INSTANCES)
    wt._INSTANCES.clear()
    try:
        assert wt.get_active_telemetry() is None        # none yet → no writer
        a = wt.get_telemetry("synthesis", {"db_dir": str(tmp_path)})
        assert wt.get_active_telemetry() is a            # the one worker instance
        wt.get_telemetry("agno", {"db_dir": str(tmp_path)})
        assert wt.get_active_telemetry() is None         # ambiguous → None
    finally:
        for t in wt._INSTANCES.values():
            t.close()
        wt._INSTANCES.clear()
        wt._INSTANCES.update(saved)


# ── C3: record_boot derives prev-uptime + downtime from the DB ───────────────

def test_record_boot_derives_prev_marks(tmp_path):
    tel = _tel(tmp_path)
    now = time.time()
    conn = tel._connect()
    try:
        conn.execute("INSERT INTO restart_events(ts,reason,prev_uptime_s) "
                     "VALUES(?,?,?)", (now - 3600, "boot", 0.0))   # prior boot 1h ago
        conn.execute("INSERT INTO memory_samples(ts,rss_anon_mb,vmrss_mb,"
                     "sizes_json) VALUES(?,?,?,?)",
                     (now - 60, 400.0, 500.0, None))               # last alive 60s ago
        conn.commit()
    finally:
        conn.close()
    tel.record_boot("synthesis_boot")
    tel.flush_now()
    newest = max(_rows(tel, "restart_events"), key=lambda r: r["ts"])
    assert "synthesis_boot" in newest["reason"]
    assert "prev_uptime" in newest["reason"] and "downtime" in newest["reason"]
    assert newest["prev_uptime_s"] > 3000          # 3600-60 ≈ 3540s prior uptime
    tel.close()


# ── C4: component-size provider fills sizes_json (best-effort) ────────────────

def test_size_provider_fills_and_tolerates_fault(tmp_path):
    tel = _tel(tmp_path)
    assert tel._sample_sizes_json() is None        # no provider → None
    tel.set_size_provider(lambda: {"faiss": 1234, "wiki_queue": 7})
    assert json.loads(tel._sample_sizes_json())["faiss"] == 1234
    def _boom():
        raise RuntimeError("provider blew up")
    tel.set_size_provider(_boom)                    # faulty provider → None, no raise
    assert tel._sample_sizes_json() is None
    tel.close()


# ── C5: freeze-dump capture (prior-run + live ingest) ────────────────────────

def test_freeze_dump_live_ingest(tmp_path):
    tel = _tel(tmp_path)
    fp = tel.freeze_dump_file()
    assert fp is not None
    fp.write("Thread 0x7f (most recent call first):\n  File a.py line 9\n")
    fp.flush()
    conn = tel._connect()
    try:
        tel._ingest_freeze_dumps(conn)
        conn.commit()
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM freeze_dumps")]
    finally:
        conn.close()
    assert len(rows) == 1 and "Thread 0x7f" in rows[0]["stack_text"]
    tel.close()


def test_freeze_dump_prior_run_captured_then_truncated(tmp_path):
    import os
    path = os.path.join(str(tmp_path), "freeze_test.txt")   # worker name = "test"
    with open(path, "w") as f:
        f.write("PRIOR-RUN FREEZE STACK\n  File frozen_op.py line 42\n")
    tel = _tel(tmp_path)
    fp = tel.freeze_dump_file()                  # captures prior content + truncates
    assert fp is not None
    assert tel._boot_freeze_text and "PRIOR-RUN" in tel._boot_freeze_text
    assert os.path.getsize(path) == 0            # truncated → never re-ingested next boot
    tel.close()


# ── Analysis: cross-worker --trace joins agno + synthesis by the same id ──────

def test_analysis_cross_worker_trace(tmp_path, capsys):
    import importlib.util
    import os
    for w, op in (("agno", "prehook:entry"), ("synthesis", "knowledge_moment")):
        t = Telemetry(w, {"db_dir": str(tmp_path)})
        t.record_stage(op, 5.0, rss_mb=1.0, feature="recall", trigger_id="tX")
        t.flush_now()
        t.close()
    spec = importlib.util.spec_from_file_location(
        "awt_mod", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                "analyze_worker_telemetry.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.trace("tX", lambda w: os.path.join(str(tmp_path), f"telemetry_{w}.db"))
    out = capsys.readouterr().out
    assert "prehook:entry" in out and "knowledge_moment" in out   # both workers
    assert "[agno" in out and "[synthesis" in out                  # labeled per worker
