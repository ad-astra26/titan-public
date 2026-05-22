"""SPEC §24.11 + rFP §3.6 — backup verify / audit-coverage CLI tests."""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Optional

import pytest

from titan_hcl.logic.backup_event_tarball import (
    FileDiffSpec,
    pack_event_tarball,
)
from titan_hcl.logic.backup_unified_manifest import (
    UnifiedManifest,
    make_event,
)
from titan_hcl.logic.backup_verify_cli import (
    TIER_ORDER,
    AuditCoverageResult,
    RestoreSimResult,
    VerifyTierResult,
    allbackups_exit_code,
    audit_coverage,
    audit_coverage_exit_code,
    restore_sim_exit_code,
    tier_result_exit_code,
    verify_allbackups,
    verify_restore_sim,
    verify_tier,
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
        self.fail_for: set[str] = set()

    def record(self, sig, memo): self._memos[sig] = memo

    async def fetch(self, sig):
        if sig in self.fail_for:
            raise RuntimeError(f"sim solana {sig}")
        return self._memos[sig]


def _full(b: bytes) -> dict:
    return {"diff_mode": "full", "patch_bytes": b, "merkle_root": _sha256(b),
            "size_bytes": len(b), "encoder": "full_ship"}


def _build_event(*, event_id, ev_type, prev_id, prev_root,
                 personality, timechain, soul, arweave, memos, tmp_path,
                 trigger=None):
    def _pack(comp, files):
        if not files:
            return None
        out = tmp_path / f"{event_id}_{comp}.tar.gz"
        specs = [FileDiffSpec(arc, dd) for arc, dd in files]
        info = pack_event_tarball(
            event_id=event_id, event_type=ev_type, component=comp,
            file_specs=specs, output_path=str(out),
        )
        tx = arweave.upload(out.read_bytes())
        return {"tx_id": tx, "merkle_root": info["tarball_sha256"],
                "size_bytes": info["size_bytes"], "diff_mode": ev_type}
    p = _pack("personality", personality)
    t = _pack("timechain", timechain)
    s = _pack("soul", soul) if soul else None
    root = compute_event_merkle_root(
        personality_merkle_root=p["merkle_root"],
        timechain_merkle_root=t["merkle_root"],
        soul_merkle_root=s["merkle_root"] if s else None,
    )
    memo = build_zk_memo(event_id=event_id, event_merkle_root=root,
                         prev_event_merkle_root=prev_root)
    sig = "sig_" + uuid.uuid4().hex[:16]
    memos.record(sig, memo)
    prev_short = prev_root[:16] if prev_root else "genesis"
    ev = make_event(
        event_id=event_id, event_type=ev_type, prev_event_id=prev_id,
        baseline_trigger=trigger if ev_type == "baseline" else None,
        personality=p, timechain=t, soul=s,
        zk_commit_tx=sig, zk_memo_prev_short=prev_short,
    )
    ev["_root"] = root
    return ev


# ── verify_tier ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_tier_personality_all_ok(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    e2 = _build_event(
        event_id="e2", ev_type="incremental", prev_id="e1",
        prev_root=e1["_root"],
        personality=[("f.txt", _full(b"x2"))],
        timechain=[("c.bin", _full(b"y2"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
    )
    m.append_event(e2)
    result = await verify_tier(
        manifest=m, tier="personality",
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
    )
    assert result.total_events == 2
    assert result.ok_events == 2
    assert result.fail_events == 0
    assert tier_result_exit_code(result) == 0
    assert all("[OK]" in line for line in result.lines)


@pytest.mark.asyncio
async def test_verify_tier_detects_tampered_tarball(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    # Tamper personality tarball
    arweave._store[e1["personality"]["tx_id"]] = b"TAMPERED"
    result = await verify_tier(
        manifest=m, tier="personality",
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
    )
    assert result.fail_events == 1
    assert result.ok_events == 0
    assert tier_result_exit_code(result) == 1
    assert any("sha256 mismatch" in line for line in result.lines)


@pytest.mark.asyncio
async def test_verify_tier_detects_zk_memo_drift(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    # Rewrite the memo to claim a different event_merkle_root
    sig = e1["zk_commit_tx"]
    memos._memos[sig] = build_zk_memo(
        event_id="e1", event_merkle_root="f" * 64,
        prev_event_merkle_root=None,
    )
    result = await verify_tier(
        manifest=m, tier="personality",
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
    )
    assert result.fail_events == 1
    assert any("ZK Merkle mismatch" in line for line in result.lines)


@pytest.mark.asyncio
async def test_verify_tier_soul_skips_non_weekly(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=None,  # no soul on this event
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    result = await verify_tier(
        manifest=m, tier="soul",
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
    )
    assert result.total_events == 1
    assert result.skipped_events == 1
    assert result.ok_events == 0
    assert result.fail_events == 0
    # Skipped events don't fail
    assert tier_result_exit_code(result) == 0


@pytest.mark.asyncio
async def test_verify_tier_no_zk_chain(tmp_path):
    """verify_zk_chain=False skips Solana round-trip — works with broken memos."""
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    # Memo unfetchable but verify still passes
    memos.fail_for.add(e1["zk_commit_tx"])
    result = await verify_tier(
        manifest=m, tier="personality",
        arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        verify_zk_chain=False,
    )
    assert result.ok_events == 1
    assert result.fail_events == 0


@pytest.mark.asyncio
async def test_verify_tier_invalid_tier_raises():
    m = UnifiedManifest("T1", base_dir="/tmp/x")
    async def _noop(x): return b""
    with pytest.raises(ValueError, match="tier must be"):
        await verify_tier(
            manifest=m, tier="garbage",
            arweave_fetch=_noop, memo_fetch=_noop,
        )


# ── verify_allbackups ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_allbackups_runs_all_three_tiers(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=[("soul.json", _full(b"{}"))],  # weekly event
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    results = await verify_allbackups(
        manifest=m, arweave_fetch=arweave.download, memo_fetch=memos.fetch,
    )
    assert set(results.keys()) == set(TIER_ORDER)
    for tier in TIER_ORDER:
        assert results[tier].ok_events == 1
        assert results[tier].fail_events == 0
    assert allbackups_exit_code(results) == 0


@pytest.mark.asyncio
async def test_verify_allbackups_exit_nonzero_on_any_fail(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    # Break only the timechain tier
    arweave._store[e1["timechain"]["tx_id"]] = b"BAD"
    results = await verify_allbackups(
        manifest=m, arweave_fetch=arweave.download, memo_fetch=memos.fetch,
    )
    assert results["personality"].fail_events == 0
    assert results["timechain"].fail_events == 1
    assert allbackups_exit_code(results) == 1


# ── verify_restore_sim ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_restore_sim_full_walk_passes(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"hello world"))],
        timechain=[("c.bin", _full(b"abcdef"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    scratch = tmp_path / "sim_scratch"

    def _arc(comp, arc):
        return str(scratch / comp / arc)

    result = await verify_restore_sim(
        manifest=m, arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        arc_to_target=_arc, scratch_dir=str(scratch),
    )
    assert result.status == "ok"
    assert result.files_verified == 2  # f.txt + c.bin
    assert restore_sim_exit_code(result) == 0
    assert (scratch / "personality" / "f.txt").read_bytes() == b"hello world"
    assert (scratch / "timechain" / "c.bin").read_bytes() == b"abcdef"


@pytest.mark.asyncio
async def test_verify_restore_sim_empty_manifest_fails(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))

    async def _noop(x): return b""

    result = await verify_restore_sim(
        manifest=m, arweave_fetch=_noop, memo_fetch=_noop,
        arc_to_target=lambda c, a: "/tmp/x",
    )
    assert result.status == "failed"
    assert any("empty" in line.lower() for line in result.lines)


@pytest.mark.asyncio
async def test_verify_restore_sim_tampered_tarball_halts(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    e1 = _build_event(
        event_id="e1", ev_type="baseline", prev_id=None, prev_root=None,
        personality=[("f.txt", _full(b"x"))],
        timechain=[("c.bin", _full(b"y"))],
        soul=None, arweave=arweave, memos=memos, tmp_path=tmp_path,
        trigger="first_event",
    )
    m.append_event(e1)
    arweave._store[e1["personality"]["tx_id"]] = b"TAMPERED"
    scratch = tmp_path / "sim_scratch"
    result = await verify_restore_sim(
        manifest=m, arweave_fetch=arweave.download, memo_fetch=memos.fetch,
        arc_to_target=lambda c, a: str(scratch / c / a),
        scratch_dir=str(scratch),
    )
    assert result.status == "failed"
    assert restore_sim_exit_code(result) == 1
    assert result.restore_result.halt_reason == "tarball_hash_mismatch"


# ── audit_coverage ───────────────────────────────────────────────────────


def test_audit_coverage_all_declared(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "inner_memory.db").write_bytes(b"x")
    (data / "titan_identity.json").write_bytes(b"{}")
    declared = [
        str(data / "inner_memory.db"),
        str(data / "titan_identity.json"),
    ]
    result = audit_coverage(
        declared_paths=declared, data_dir=str(data),
    )
    assert result.status == "ok"
    assert result.found_count == 2
    assert result.undeclared_files == []
    assert audit_coverage_exit_code(result) == 0


def test_audit_coverage_flags_undeclared(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "declared.db").write_bytes(b"x")
    (data / "rogue.json").write_bytes(b"{}")
    declared = [str(data / "declared.db")]
    result = audit_coverage(
        declared_paths=declared, data_dir=str(data),
    )
    assert result.status == "drift"
    assert any("rogue.json" in p for p in result.undeclared_files)
    assert audit_coverage_exit_code(result) == 1


def test_audit_coverage_directory_declaration(tmp_path):
    """A directory declaration like 'data/sage_memory/' covers everything
    inside, including subdirectories."""
    data = tmp_path / "data"
    sage = data / "sage_memory"
    sage.mkdir(parents=True)
    (sage / "buffer.bin").write_bytes(b"x")
    (sage / "meta.json").write_bytes(b"{}")
    (sage / "sub" / "deep.bin").parent.mkdir()
    (sage / "sub" / "deep.bin").write_bytes(b"z")
    declared = [str(sage) + "/"]
    result = audit_coverage(
        declared_paths=declared, data_dir=str(data),
    )
    assert result.status == "ok"
    assert result.found_count == 3
    assert result.undeclared_files == []


def test_audit_coverage_ignores_bak_files(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "declared.db").write_bytes(b"x")
    (data / "declared.db.bak").write_bytes(b"x")
    (data / "declared.db.bak.prev").write_bytes(b"x")
    (data / "scratch.tmp").write_bytes(b"x")
    declared = [str(data / "declared.db")]
    result = audit_coverage(
        declared_paths=declared, data_dir=str(data),
    )
    # .bak / .tmp filtered out → no drift
    assert result.status == "ok"
    assert result.undeclared_files == []


def test_audit_coverage_missing_data_dir(tmp_path):
    result = audit_coverage(
        declared_paths=[], data_dir=str(tmp_path / "does_not_exist"),
    )
    assert result.status == "drift"
    assert any("not found" in line for line in result.lines)
    assert audit_coverage_exit_code(result) == 1


def test_audit_coverage_recursive_walk(tmp_path):
    data = tmp_path / "data"
    sub = data / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "secret.bin").write_bytes(b"x")
    declared = []
    result = audit_coverage(
        declared_paths=declared, data_dir=str(data),
    )
    assert result.status == "drift"
    assert any("nested" in p and "secret.bin" in p for p in result.undeclared_files)
