"""RFP_backup_redesign_spine Phase D — BackupOrchestrator brain.

Unit-tests the readiness machine (idle drip / deadline force / the one gate),
the disk-persisted drip (persist → resume → baseline-staleness discard), and the
single-flight-guarded Sunday restore-test — with fakes for the build/ship
mechanics (those are BackupWorker, covered by test_backup_worker_pipeline).

Mapped to RFP §8: GD1 (readiness — idle-drip advances; deadline-force off-path),
GD2 (bounded — ≤1 build_slice/tick), INV-BRS-6/7 (one concurrency model;
disk-persisted side-state), and the audit H-bug fix (restore-test single-flight
+ events on the REAL send_queue).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading

import pytest

from titan_hcl.logic.backup_worker_pipeline import (
    DRIP_PROGRESS_FILENAME, StagedBuild)
from titan_hcl.logic.backup_upload_pipeline import TierFileSpec, TierShipResult
from titan_hcl.modules.backup_orchestrator import BackupOrchestrator


# ── fakes ──────────────────────────────────────────────────────────────────
class _Q:
    def __init__(self):
        self.msgs = []

    def put_nowait(self, m):
        self.msgs.append(m)


def _make_staged(scratch_dir, n_pending=3, baseline_event_id="base1"):
    return StagedBuild(
        event_id="evt-test1", event_type="incremental", baseline_trigger=None,
        baseline_event_id=baseline_event_id, prev_event_id="prev1",
        soul_present=False, scratch_dir=scratch_dir, titan_id="T1",
        pending={"personality": [
            TierFileSpec(source_path=f"/x/{i}", arc_name=f"a{i}")
            for i in range(n_pending)]},
        artifacts={"personality": []})


class _FakeWorker:
    """Drives a real StagedBuild deterministically (one spec per build_slice)."""
    def __init__(self):
        self.slice_calls = 0
        self.finalize_calls = 0

    def build_slice(self, staged, resolver, *, byte_budget=None):
        self.slice_calls += 1
        specs = staged.pending.get("personality") or []
        if specs:
            spec = specs.pop(0)
            staged.artifacts["personality"].append(
                (spec.arc_name,
                 {"patch_path": None, "patch_owned": False,
                  "patch_size_bytes": 1}))
        return not staged.fully_encoded

    def finalize_pack(self, staged):
        self.finalize_calls += 1
        staged.tier_results = {
            "personality": TierShipResult(
                tier="personality", tarball_path="/x/p.tar.zst",
                tarball_size_bytes=10)}


class _FakeBackup:
    def __init__(self, worker, staged, *, landed=False, halted=False):
        self._titan_id = "T1"
        self._staged_event = None
        self._last_restore_test_date = None
        self._worker = worker
        self._staged = staged
        self._landed = landed
        self._halted = halted
        self.plan_calls = 0
        self.restore_test_calls = 0

    def _is_backups_halted(self):
        return self._halted

    def _todays_backup_already_landed(self):
        return self._landed

    def _baseline_working_dir(self):
        return "/tmp/baseline_T1_test"

    def _make_diff_base_resolver(self, base, known):
        return lambda c, a: None

    def _plan_staged_build_v2(self, weekday, scratch_dir=None, byte_budget=None):
        self.plan_calls += 1
        if scratch_dir:
            self._staged.scratch_dir = scratch_dir
            os.makedirs(scratch_dir, exist_ok=True)
        return (self._worker, self._staged, (lambda c, a: None), set())

    def stage_built_event(self, staged, day):
        self._staged_event = {"staged": staged, "date": day}

    async def _run_weekly_restore_test(self, *, bus_emit=None):
        self.restore_test_calls += 1
        if bus_emit:
            bus_emit("BACKUP_RESTORE_TEST_PASS", {"ok": True})
        return True


def _orch(backup, tmp_path, monkeypatch, **cfg):
    """Build a BackupOrchestrator wired to a fake backup, drip_dir in tmp, and
    CWD in tmp (so _record_stage_dry_run_result writes to tmp/data)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    q = _Q()
    state = {"backup": backup, "_backup_lock": threading.Lock(),
             "titan_id": "T1", "send_queue": q, "name": "backup"}
    oc = {"staged_by_utc": "23:59", "boot_settle_s": 0, "byte_budget_mb": 1}
    oc.update(cfg)
    o = BackupOrchestrator(state, {"backup": {"orchestrator": oc}}, q, "backup")
    o.drip_dir = str(tmp_path / "drip")
    return o


# ── readiness machine (GD1/GD2) ─────────────────────────────────────────────
def test_idle_drip_advances_to_staged(tmp_path, monkeypatch):
    """Idle ticks drip one batch each (≤1 build_slice/tick — GD2 bounded) until
    fully encoded, then finalize + park for the meditation ship (GD1)."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"), n_pending=3)
    backup = _FakeBackup(worker, staged)
    o = _orch(backup, tmp_path, monkeypatch)
    o._is_idle = lambda: True  # force idle

    for _ in range(3):                  # 3 pending → 3 drip ticks
        o._drip_tick()
        assert backup._staged_event is None  # not parked until fully encoded
    assert worker.slice_calls == 3      # exactly one batch per tick (bounded)
    o._drip_tick()                      # fully encoded → finalize + park
    assert worker.finalize_calls == 1
    assert backup._staged_event is not None
    assert backup._staged_event["staged"] is staged
    assert o._drip is None              # cleared after parking


def test_not_idle_no_drip_before_deadline(tmp_path, monkeypatch):
    """Not idle + before staged_by deadline → plan happens but NO build_slice."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"), n_pending=3)
    backup = _FakeBackup(worker, staged)
    o = _orch(backup, tmp_path, monkeypatch, staged_by_utc="23:59")
    o._is_idle = lambda: False

    o._drip_tick()
    assert backup.plan_calls == 1       # drip planned (so it's ready to advance)
    assert worker.slice_calls == 0      # but no work while busy + pre-deadline


def test_deadline_force_builds_when_not_idle(tmp_path, monkeypatch):
    """Past staged_by deadline → drop the idle-gate, force the build off-path."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"), n_pending=2)
    backup = _FakeBackup(worker, staged)
    o = _orch(backup, tmp_path, monkeypatch, staged_by_utc="00:00")  # always past
    o._is_idle = lambda: False          # busy

    o._drip_tick()
    assert worker.slice_calls == 1      # deadline-force advanced despite busy


def test_gate_already_landed_no_work(tmp_path, monkeypatch):
    """The ONE gate: today's backup already landed → no plan, no drip."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"))
    backup = _FakeBackup(worker, staged, landed=True)
    o = _orch(backup, tmp_path, monkeypatch)
    o._is_idle = lambda: True

    o._drip_tick()
    assert backup.plan_calls == 0
    assert worker.slice_calls == 0
    assert o._drip is None


def test_halted_no_drip(tmp_path, monkeypatch):
    """A failed restore-test halt (INV-BR-4) → orchestrator idle."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"))
    backup = _FakeBackup(worker, staged, halted=True)
    o = _orch(backup, tmp_path, monkeypatch)
    o._is_idle = lambda: True

    o._drip_tick()
    assert backup.plan_calls == 0
    assert o._drip is None


def test_already_staged_today_no_rebuild(tmp_path, monkeypatch):
    """A fresh stage parked for today → tick is a no-op (meditation ships it
    WITHOUT the orchestrator rebuilding)."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"))
    backup = _FakeBackup(worker, staged)
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    backup._staged_event = {"staged": staged, "date": today}
    o = _orch(backup, tmp_path, monkeypatch)
    o._is_idle = lambda: True

    o._drip_tick()
    assert backup.plan_calls == 0
    assert worker.slice_calls == 0


# ── disk-persisted drip (INV-BRS-7) ──────────────────────────────────────────
def test_persist_then_resume_roundtrip(tmp_path, monkeypatch):
    """A partially-dripped StagedBuild persists to disk and a fresh orchestrator
    resumes it (baseline current + owned artifacts intact)."""
    import titan_hcl.modules.backup_orchestrator as bo

    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"), n_pending=3,
                          baseline_event_id="base1")
    backup = _FakeBackup(worker, staged)
    o1 = _orch(backup, tmp_path, monkeypatch)
    o1._is_idle = lambda: True
    o1._drip_tick()                     # plan + 1 slice → 2 pending left
    o1._drip_tick()                     # +1 slice → 1 pending left
    assert os.path.exists(os.path.join(o1.drip_dir, DRIP_PROGRESS_FILENAME))
    assert len(staged.pending["personality"]) == 1

    # Fresh orchestrator (= a restart) with a baseline-matching manifest.
    class _FakeManifest:
        current_baseline_event_id = "base1"

    monkeypatch.setattr(
        "titan_hcl.logic.backup_unified_manifest.UnifiedManifest.load",
        classmethod(lambda cls, **kw: _FakeManifest()))
    o2 = _orch(backup, tmp_path, monkeypatch)
    o2.drip_dir = o1.drip_dir
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    assert o2._try_resume_drip(today) is True
    assert o2._drip is not None
    # resumed with the SAME remaining pending count (no re-encode of done work)
    assert len(o2._drip["staged"].pending["personality"]) == 1


def test_resume_discards_on_baseline_mismatch(tmp_path, monkeypatch):
    """If the manifest baseline moved since the drip was persisted, the resume
    DISCARDS + re-plans (the persisted diffs are stale)."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"), n_pending=3,
                          baseline_event_id="base1")
    backup = _FakeBackup(worker, staged)
    o1 = _orch(backup, tmp_path, monkeypatch)
    o1._is_idle = lambda: True
    o1._drip_tick()
    assert os.path.exists(os.path.join(o1.drip_dir, DRIP_PROGRESS_FILENAME))

    class _MovedManifest:
        current_baseline_event_id = "base2"   # moved!

    monkeypatch.setattr(
        "titan_hcl.logic.backup_unified_manifest.UnifiedManifest.load",
        classmethod(lambda cls, **kw: _MovedManifest()))
    o2 = _orch(backup, tmp_path, monkeypatch)
    o2.drip_dir = o1.drip_dir
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    assert o2._try_resume_drip(today) is False
    assert o2._drip is None
    # progress file discarded
    assert not os.path.exists(
        os.path.join(o1.drip_dir, DRIP_PROGRESS_FILENAME))


def test_resume_discards_on_stale_day(tmp_path, monkeypatch):
    """A drip persisted for a prior UTC day is dropped (the day rolled over)."""
    worker = _FakeWorker()
    staged = _make_staged(str(tmp_path / "drip"))
    backup = _FakeBackup(worker, staged)
    o = _orch(backup, tmp_path, monkeypatch)
    os.makedirs(o.drip_dir, exist_ok=True)
    with open(os.path.join(o.drip_dir, DRIP_PROGRESS_FILENAME), "w") as f:
        json.dump({"staged": staged.to_dict(), "known_arcs": [],
                   "today": "2000-01-01", "weekday": 0}, f)
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    assert o._try_resume_drip(today) is False
    assert o._drip is None


# ── Sunday restore-test (D.5 single-flight guard; audit H-bug fix) ──────────
def test_restore_test_runs_sunday_and_emits_on_real_queue(tmp_path, monkeypatch):
    """On Sunday, lock-free → the restore-test runs AND its PASS event lands on
    the REAL send_queue (the old _bus_emit probed non-existent state['bus'] →
    silent no-op; audit H-bug)."""
    import titan_hcl.modules.backup_orchestrator as bo

    worker = _FakeWorker()
    backup = _FakeBackup(worker, _make_staged(str(tmp_path / "drip")))
    o = _orch(backup, tmp_path, monkeypatch)

    class _Sun:   # 2026-06-07 is a Sunday (weekday 6)
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 6, 7, 12, 0, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(bo, "datetime", _Sun)

    o.run_restore_test_if_due()
    assert backup.restore_test_calls == 1
    types = [m["type"] for m in o.send_queue.msgs]
    assert "BACKUP_RESTORE_TEST_PASS" in types  # event reached the real queue


def test_restore_test_single_flight_guarded(tmp_path, monkeypatch):
    """A ship/cascade in flight (lock held) → the restore-test is SKIPPED
    (single-flight — the audit H-bug was that it ran UNGUARDED)."""
    import titan_hcl.modules.backup_orchestrator as bo

    worker = _FakeWorker()
    backup = _FakeBackup(worker, _make_staged(str(tmp_path / "drip")))
    o = _orch(backup, tmp_path, monkeypatch)

    class _Sun:
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(2026, 6, 7, 12, 0, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(bo, "datetime", _Sun)

    o.state["_backup_lock"].acquire()   # simulate a ship in flight
    try:
        o.run_restore_test_if_due()
    finally:
        o.state["_backup_lock"].release()
    assert backup.restore_test_calls == 0   # guarded — did not run
