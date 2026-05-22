"""SPEC §24.12 + rFP §3.5 — weekly restore-test tests."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Optional

import pytest

from titan_hcl.logic import diff_encoders
from titan_hcl.logic.backup_event_tarball import (
    FileDiffSpec,
    pack_event_tarball,
)
from titan_hcl.logic.backup_restore_test import (
    EVENT_BACKUP_RESTORE_TEST_FAIL,
    EVENT_BACKUP_RESTORE_TEST_PASS,
    RestoreTestResult,
    build_fail_bus_payload,
    build_pass_bus_payload,
    is_due_for_test,
    run_weekly_restore_test,
    telegram_fail_message,
)
from titan_hcl.logic.backup_unified_manifest import (
    UnifiedManifest,
    make_event,
)
from titan_hcl.logic.backup_zk_commit import (
    build_zk_memo,
    compute_event_merkle_root,
)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeArweave:
    def __init__(self):
        self._store: dict[str, bytes] = {}
        self.fail_for: set[str] = set()

    def upload(self, data: bytes) -> str:
        tx = "ar_" + uuid.uuid4().hex[:16]
        self._store[tx] = data
        return tx

    async def download(self, tx_id: str) -> bytes:
        if tx_id in self.fail_for:
            raise RuntimeError(f"sim 503 {tx_id}")
        return self._store[tx_id]


class FakeMemos:
    def __init__(self):
        self._memos: dict[str, str] = {}

    def record(self, sig: str, memo: str): self._memos[sig] = memo
    async def fetch(self, sig: str) -> str: return self._memos[sig]


def _full(b: bytes) -> dict:
    return {"diff_mode": "full", "patch_bytes": b, "merkle_root": _sha256(b),
            "size_bytes": len(b), "encoder": "full_ship"}


def _build_event(*, event_id, ev_type, prev_id, prev_root,
                 p_files, t_files, arweave, memos, tmp_path,
                 trigger=None):
    def _pack(comp, files):
        out = tmp_path / f"{event_id}_{comp}.tar.gz"
        specs = [FileDiffSpec(arc, dd) for arc, dd in files]
        info = pack_event_tarball(
            event_id=event_id, event_type=ev_type, component=comp,
            file_specs=specs, output_path=str(out),
        )
        tx = arweave.upload(out.read_bytes())
        return {"tx_id": tx, "merkle_root": info["tarball_sha256"],
                "size_bytes": info["size_bytes"], "diff_mode": ev_type}
    p = _pack("personality", p_files)
    t = _pack("timechain", t_files)
    root = compute_event_merkle_root(
        personality_merkle_root=p["merkle_root"],
        timechain_merkle_root=t["merkle_root"],
    )
    memo = build_zk_memo(event_id=event_id, event_merkle_root=root,
                         prev_event_merkle_root=prev_root)
    sig = "sig_" + uuid.uuid4().hex[:16]
    memos.record(sig, memo)
    prev_short = prev_root[:16] if prev_root else "genesis"
    ev = make_event(
        event_id=event_id, event_type=ev_type, prev_event_id=prev_id,
        baseline_trigger=trigger if ev_type == "baseline" else None,
        personality=p, timechain=t, soul=None,
        zk_commit_tx=sig, zk_memo_prev_short=prev_short,
    )
    ev["_root"] = root
    return ev


def _arc_to_target(target_dir):
    def _m(comp, arc): return os.path.join(target_dir, comp, arc)
    return _m


# ── happy path ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weekly_test_pass_emits_pass_callback(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        p_files=[("config.txt", _full(b"v=1"))],
        t_files=[("chain.bin", _full(b"data"))],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    pass_calls: list[RestoreTestResult] = []
    fail_calls: list[RestoreTestResult] = []
    result = await run_weekly_restore_test(
        titan_id="T1", manifest=m,
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(tmp_path / "scratch")),
        on_pass=pass_calls.append, on_fail=fail_calls.append,
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert result.status == "pass"
    assert result.chain_depth == 1
    assert result.target_event_id == "e1"
    assert result.bytes_fetched > 0
    assert len(pass_calls) == 1
    assert pass_calls[0] is result
    assert fail_calls == []


@pytest.mark.asyncio
async def test_weekly_test_cleans_up_scratch_on_pass(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        p_files=[("f.txt", _full(b"x"))],
        t_files=[("c.bin", _full(b"y"))],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    # No scratch_dir → run_weekly creates one and owns it; auto-cleanup on pass
    result = await run_weekly_restore_test(
        titan_id="T1", manifest=m,
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        arc_to_target=lambda c, a: os.path.join("/tmp/notused", c, a),
        cleanup_scratch_on_pass=True,
    )
    assert result.status == "pass"
    # Scratch dir was created and then cleaned
    assert not os.path.exists(result.scratch_dir)
    assert any("cleaned" in n for n in result.notes)


# ── fail paths ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_weekly_test_fail_emits_fail_callback(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        p_files=[("f.txt", _full(b"x"))],
        t_files=[("c.bin", _full(b"y"))],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    # Tamper Arweave-stored personality tarball
    arweave._store[e1["personality"]["tx_id"]] = b"TAMPERED"
    pass_calls = []
    fail_calls = []
    result = await run_weekly_restore_test(
        titan_id="T1", manifest=m,
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(tmp_path / "scratch")),
        on_pass=pass_calls.append, on_fail=fail_calls.append,
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert result.status == "fail"
    assert result.restore_result is not None
    assert result.restore_result.halt_reason == "tarball_hash_mismatch"
    assert len(fail_calls) == 1
    assert pass_calls == []


@pytest.mark.asyncio
async def test_weekly_test_callback_exception_is_recorded(tmp_path):
    """Callback raising must not crash the test — exception is captured
    in result.notes so observability still gets the bus payload."""
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        p_files=[("f.txt", _full(b"x"))],
        t_files=[("c.bin", _full(b"y"))],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    def _boom(r): raise RuntimeError("telegram dead")
    result = await run_weekly_restore_test(
        titan_id="T1", manifest=m,
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(tmp_path / "scratch")),
        on_pass=_boom,
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert result.status == "pass"  # test itself passed
    assert any("telegram dead" in n for n in result.notes)


@pytest.mark.asyncio
async def test_weekly_test_empty_manifest_skipped(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))

    async def _noop(x): return b""

    pass_calls = []; fail_calls = []
    result = await run_weekly_restore_test(
        titan_id="T1", manifest=m,
        arweave_fetch=_noop, memo_fetch=_noop,
        arc_to_target=lambda c, a: "/tmp/x",
        on_pass=pass_calls.append, on_fail=fail_calls.append,
    )
    assert result.status == "skipped"
    assert result.skipped_reason == "manifest_empty_no_events_to_restore"
    # Skipped doesn't fire either callback
    assert pass_calls == []
    assert fail_calls == []


@pytest.mark.asyncio
async def test_weekly_test_scratch_preserved_on_fail_by_default(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        p_files=[("f.txt", _full(b"x"))],
        t_files=[("c.bin", _full(b"y"))],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    arweave._store[e1["personality"]["tx_id"]] = b"TAMPERED"
    # cleanup_scratch_on_fail=False (default) — scratch preserved for Maker
    result = await run_weekly_restore_test(
        titan_id="T1", manifest=m,
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        arc_to_target=lambda c, a: os.path.join("/tmp/notused", c, a),
    )
    assert result.status == "fail"
    assert os.path.exists(result.scratch_dir)  # preserved
    import shutil
    shutil.rmtree(result.scratch_dir, ignore_errors=True)


# ── is_due_for_test ──────────────────────────────────────────────────────


def test_is_due_for_test_first_run():
    assert is_due_for_test(last_test_ts=None) is True


def test_is_due_for_test_recent_pass():
    now = 1_000_000.0
    one_hour_ago = now - 3600
    assert is_due_for_test(last_test_ts=one_hour_ago, now=now) is False


def test_is_due_for_test_eight_days_ago():
    now = 1_000_000.0
    eight_days = now - (8 * 86400)
    assert is_due_for_test(last_test_ts=eight_days, now=now) is True


def test_is_due_for_test_exactly_at_cadence():
    now = 1_000_000.0
    seven_days = now - (7 * 86400)
    assert is_due_for_test(last_test_ts=seven_days, now=now) is True


def test_is_due_for_test_custom_cadence():
    now = 1_000_000.0
    three_days = now - (3 * 86400)
    # 7d cadence → not due yet
    assert is_due_for_test(last_test_ts=three_days, cadence_days=7, now=now) is False
    # 1d cadence → due
    assert is_due_for_test(last_test_ts=three_days, cadence_days=1, now=now) is True


# ── bus payload builders ─────────────────────────────────────────────────


def test_build_pass_bus_payload_shape():
    r = RestoreTestResult(
        status="pass", ts_unix=12345.0, target_event_id="e9",
        chain_depth=5, bytes_fetched=1024, duration_s=2.5,
        scratch_dir="/tmp/x",
    )
    p = build_pass_bus_payload(r, "T1")
    assert p["event"] == EVENT_BACKUP_RESTORE_TEST_PASS
    assert p["titan_id"] == "T1"
    assert p["target_event_id"] == "e9"
    assert p["chain_depth"] == 5
    assert p["bytes_fetched"] == 1024
    assert p["duration_s"] == 2.5
    assert p["scratch_dir"] == "/tmp/x"


def test_build_fail_bus_payload_shape():
    from titan_hcl.logic.backup_restore import RestoreResult
    rr = RestoreResult(
        status="halted", target_event_id="e9", applied_events=["e1"],
        bytes_fetched=512, errors=["bad merkle"],
        halt_reason="tarball_hash_mismatch", halt_event_id="e2",
    )
    r = RestoreTestResult(
        status="fail", ts_unix=999.0, target_event_id="e9",
        chain_depth=1, bytes_fetched=512, duration_s=0.7,
        restore_result=rr, scratch_dir="/tmp/x",
        notes=["scratch preserved"],
    )
    p = build_fail_bus_payload(r, "T1")
    assert p["event"] == EVENT_BACKUP_RESTORE_TEST_FAIL
    assert p["halt_reason"] == "tarball_hash_mismatch"
    assert p["halt_event_id"] == "e2"
    assert p["errors"] == ["bad merkle"]
    assert "scratch preserved" in p["notes"]


def test_telegram_fail_message_format():
    payload = {
        "titan_id": "T1",
        "halt_reason": "zk_disconnect",
        "halt_event_id": "e42",
        "errors": ["sim solana down sig_abc"],
        "chain_depth": 12,
        "duration_s": 3.4,
        "scratch_dir": "/tmp/scratch_X",
    }
    msg = telegram_fail_message(payload)
    assert "BACKUP RESTORE TEST FAILED (T1)" in msg
    assert "zk_disconnect" in msg
    assert "e42" in msg
    assert "sim solana down" in msg
    assert "HALTED" in msg
