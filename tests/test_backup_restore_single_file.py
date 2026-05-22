"""SPEC §11.H.4 / rFP §3.2 + §3.4 — single-file restore tests.

Covers:
  - boot-time Tier-3 Arweave fallback for a single corrupted file
  - on-demand fetch_file_at_event for arbitrary past event reconstruction
  - skipped-file walk (file unchanged across some incrementals)
  - expected_post_merkle_root guardrail
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Optional

import pytest

from titan_hcl.logic import diff_encoders
from titan_hcl.logic.backup_event_tarball import (
    FileDiffSpec,
    pack_event_tarball,
)
from titan_hcl.logic.backup_restore import (
    HALT_APPLY_FAILED,
    HALT_MANIFEST_EMPTY,
    HALT_POST_RESTORE_HASH_MISMATCH,
    HALT_TARBALL_HASH_MISMATCH,
    HALT_ZK_DISCONNECT,
    SingleFileRestoreResult,
    fetch_file_at_event,
    restore_single_file,
    select_single_file_chain,
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
        self.fetch_log: list[str] = []
        self.fail_for: set[str] = set()

    def upload_sync(self, data: bytes) -> str:
        tx = "ar_" + uuid.uuid4().hex[:16]
        self._store[tx] = data
        return tx

    async def download(self, tx_id: str) -> bytes:
        self.fetch_log.append(tx_id)
        if tx_id in self.fail_for:
            raise RuntimeError(f"sim failure {tx_id}")
        return self._store[tx_id]


class FakeMemos:
    def __init__(self):
        self._memos: dict[str, str] = {}
        self.fail_for: set[str] = set()

    def record(self, sig: str, memo: str):
        self._memos[sig] = memo

    async def fetch(self, sig: str) -> str:
        if sig in self.fail_for:
            raise RuntimeError(f"sim solana down {sig}")
        return self._memos[sig]


def _full(content: bytes, encoder: str = "full_ship") -> dict:
    return {
        "diff_mode": "full",
        "patch_bytes": content,
        "merkle_root": _sha256(content),
        "size_bytes": len(content),
        "encoder": encoder,
    }


def _skipped(prev_content: bytes) -> dict:
    return {
        "diff_mode": "skipped",
        "patch_bytes": b"",
        "merkle_root": _sha256(prev_content),
        "size_bytes": len(prev_content),
        "encoder": "full_ship",
    }


def _build_event(
    *, event_id: str, event_type: str,
    prev_event_id: Optional[str],
    prev_event_merkle_root: Optional[str],
    personality_files: list[tuple[str, dict]],
    timechain_files: list[tuple[str, dict]],
    arweave: FakeArweave, memos: FakeMemos, tmp_path,
    baseline_trigger: Optional[str] = None,
) -> dict:
    def _pack(component: str, files: list[tuple[str, dict]]):
        if not files:
            return None
        out = tmp_path / f"{event_id}_{component}.tar.gz"
        specs = [FileDiffSpec(arc, dd) for arc, dd in files]
        info = pack_event_tarball(
            event_id=event_id, event_type=event_type,
            component=component, file_specs=specs,
            output_path=str(out),
        )
        tx = arweave.upload_sync(out.read_bytes())
        return {
            "tx_id": tx, "merkle_root": info["tarball_sha256"],
            "size_bytes": info["size_bytes"], "diff_mode": event_type,
        }

    p_sub = _pack("personality", personality_files)
    t_sub = _pack("timechain", timechain_files)
    root = compute_event_merkle_root(
        personality_merkle_root=p_sub["merkle_root"],
        timechain_merkle_root=t_sub["merkle_root"],
    )
    memo = build_zk_memo(
        event_id=event_id, event_merkle_root=root,
        prev_event_merkle_root=prev_event_merkle_root,
    )
    sig = "sig_" + uuid.uuid4().hex[:16]
    memos.record(sig, memo)
    prev_short = prev_event_merkle_root[:16] if prev_event_merkle_root \
        else "genesis"
    ev = make_event(
        event_id=event_id, event_type=event_type,
        prev_event_id=prev_event_id,
        baseline_trigger=baseline_trigger if event_type == "baseline" else None,
        personality=p_sub, timechain=t_sub, soul=None,
        zk_commit_tx=sig, zk_memo_prev_short=prev_short,
    )
    ev["_root"] = root
    return ev


def _chain_3(arweave, memos, tmp_path):
    """Build a 3-event chain (baseline + 2 incrementals) and return
    (manifest, expected state at latest event)."""
    m = UnifiedManifest("T1", base_dir=str(tmp_path))

    # Event 1 — baseline
    p1_config = b"version=1.0\n"
    p1_state = b'{"epoch": 0}'
    t1_chain = b"BLOCK0" * 8
    e1 = _build_event(
        event_id="e1", event_type="baseline",
        prev_event_id=None, prev_event_merkle_root=None,
        personality_files=[
            ("config.txt", _full(p1_config)),
            ("state.json", _full(p1_state)),
        ],
        timechain_files=[("chain.bin", _full(t1_chain))],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        baseline_trigger="first_event",
    )
    m.append_event(e1)

    # Event 2 — incremental — state changed, config skipped, chain grew
    p2_state = b'{"epoch": 1, "x": true}'
    t2_chain = t1_chain + b"BLOCK1" * 8
    p1s_path = tmp_path / "p1s.bin"; p1s_path.write_bytes(p1_state)
    p2s_path = tmp_path / "p2s.bin"; p2s_path.write_bytes(p2_state)
    state_diff = diff_encoders.xdelta3.encode_diff(str(p2s_path), str(p1s_path))
    state_diff["encoder"] = "xdelta3"
    t1c_path = tmp_path / "t1c.bin"; t1c_path.write_bytes(t1_chain)
    t2c_path = tmp_path / "t2c.bin"; t2c_path.write_bytes(t2_chain)
    chain_diff = diff_encoders.timechain_tail.encode_diff(
        str(t2c_path), str(t1c_path), block_range=(1, 1),
    )
    chain_diff["encoder"] = "timechain_tail"
    e2 = _build_event(
        event_id="e2", event_type="incremental",
        prev_event_id="e1", prev_event_merkle_root=e1["_root"],
        personality_files=[
            ("config.txt", _skipped(p1_config)),
            ("state.json", state_diff),
        ],
        timechain_files=[("chain.bin", chain_diff)],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
    )
    m.append_event(e2)

    # Event 3 — incremental — state changed again
    p3_state = b'{"epoch": 2, "x": true, "y": 99}'
    t3_chain = t2_chain + b"BLOCK2" * 8
    p3s_path = tmp_path / "p3s.bin"; p3s_path.write_bytes(p3_state)
    t3c_path = tmp_path / "t3c.bin"; t3c_path.write_bytes(t3_chain)
    state_diff_3 = diff_encoders.xdelta3.encode_diff(
        str(p3s_path), str(p2s_path),
    )
    state_diff_3["encoder"] = "xdelta3"
    chain_diff_3 = diff_encoders.timechain_tail.encode_diff(
        str(t3c_path), str(t2c_path), block_range=(2, 2),
    )
    chain_diff_3["encoder"] = "timechain_tail"
    e3 = _build_event(
        event_id="e3", event_type="incremental",
        prev_event_id="e2", prev_event_merkle_root=e2["_root"],
        personality_files=[
            ("config.txt", _skipped(p1_config)),
            ("state.json", state_diff_3),
        ],
        timechain_files=[("chain.bin", chain_diff_3)],
        arweave=arweave, memos=memos, tmp_path=tmp_path,
    )
    m.append_event(e3)

    expected = {
        "config.txt": p1_config,        # never changed since baseline
        "state.json": p3_state,         # latest
        "chain.bin": t3_chain,          # latest
        "state.json@e2": p2_state,
        "state.json@e1": p1_state,
        "chain.bin@e2": t2_chain,
    }
    return m, expected


# ── basic restore_single_file ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restore_single_file_state_json_latest(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m, expected = _chain_3(arweave, memos, tmp_path)
    out_path = tmp_path / "restored_state.json"
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(out_path),
        arweave_fetch=arweave.download,
    )
    assert result.status == "success", (result.halt_reason, result.errors)
    assert out_path.read_bytes() == expected["state.json"]
    assert result.final_merkle_root == _sha256(expected["state.json"])
    assert result.target_event_id == "e3"
    # all 3 events walked (even though only e1+e2+e3 touched state.json)
    assert len(result.applied_events) == 3


@pytest.mark.asyncio
async def test_restore_single_file_chain_bin_latest(tmp_path):
    """Timechain tail encoder restore path (different from xdelta3)."""
    arweave = FakeArweave(); memos = FakeMemos()
    m, expected = _chain_3(arweave, memos, tmp_path)
    out_path = tmp_path / "restored_chain.bin"
    result = await restore_single_file(
        manifest=m, component="timechain", arc_name="chain.bin",
        output_path=str(out_path),
        arweave_fetch=arweave.download,
    )
    assert result.status == "success", (result.halt_reason, result.errors)
    assert out_path.read_bytes() == expected["chain.bin"]


@pytest.mark.asyncio
async def test_restore_single_file_to_past_event(tmp_path):
    """target_event_id reconstructs to a prior state — §3.4 use case."""
    arweave = FakeArweave(); memos = FakeMemos()
    m, expected = _chain_3(arweave, memos, tmp_path)
    out_path = tmp_path / "restored_state_at_e2.json"
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(out_path),
        arweave_fetch=arweave.download,
        target_event_id="e2",
    )
    assert result.status == "success", (result.halt_reason, result.errors)
    assert out_path.read_bytes() == expected["state.json@e2"]
    assert result.target_event_id == "e2"


@pytest.mark.asyncio
async def test_restore_single_file_skipped_throughout(tmp_path):
    """config.txt was full in baseline + skipped in e2 + skipped in e3 —
    should still reconstruct to baseline content."""
    arweave = FakeArweave(); memos = FakeMemos()
    m, expected = _chain_3(arweave, memos, tmp_path)
    out_path = tmp_path / "restored_config.txt"
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="config.txt",
        output_path=str(out_path),
        arweave_fetch=arweave.download,
    )
    assert result.status == "success", (result.halt_reason, result.errors)
    assert out_path.read_bytes() == expected["config.txt"]


@pytest.mark.asyncio
async def test_restore_single_file_with_expected_hash(tmp_path):
    """expected_post_merkle_root passes when match, halts when mismatch."""
    arweave = FakeArweave(); memos = FakeMemos()
    m, expected = _chain_3(arweave, memos, tmp_path)
    out_path = tmp_path / "with_hash.json"
    # Correct hash → success
    correct = _sha256(expected["state.json"])
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(out_path),
        arweave_fetch=arweave.download,
        expected_post_merkle_root=correct,
    )
    assert result.status == "success"

    # Wrong hash → halt
    out_path2 = tmp_path / "with_bad_hash.json"
    bad = "f" * 64
    result2 = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(out_path2),
        arweave_fetch=arweave.download,
        expected_post_merkle_root=bad,
    )
    assert result2.status == "halted"
    assert result2.halt_reason == HALT_POST_RESTORE_HASH_MISMATCH
    assert not out_path2.exists()  # scratch cleaned up


@pytest.mark.asyncio
async def test_restore_single_file_unknown_arc_halts(tmp_path):
    """File not present anywhere in chain → HALT_APPLY_FAILED."""
    arweave = FakeArweave(); memos = FakeMemos()
    m, _ = _chain_3(arweave, memos, tmp_path)
    out_path = tmp_path / "nope.json"
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="ghost.json",
        output_path=str(out_path),
        arweave_fetch=arweave.download,
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_APPLY_FAILED
    assert any("no physical presence" in e for e in result.errors)


@pytest.mark.asyncio
async def test_restore_single_file_zk_verification_optional(tmp_path):
    """verify_zk_chain=True wires the memo round-trip; failing memos halt."""
    arweave = FakeArweave(); memos = FakeMemos()
    m, _ = _chain_3(arweave, memos, tmp_path)
    # Without ZK verify (default) → still works even if Solana down
    memos.fail_for.add(m.events[0]["zk_commit_tx"])
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(tmp_path / "no_zk.json"),
        arweave_fetch=arweave.download,
        memo_fetch=memos.fetch, verify_zk_chain=False,
    )
    assert result.status == "success"

    # With ZK verify → halts on ZK disconnect
    result2 = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(tmp_path / "with_zk.json"),
        arweave_fetch=arweave.download,
        memo_fetch=memos.fetch, verify_zk_chain=True,
    )
    assert result2.status == "halted"
    assert result2.halt_reason == HALT_ZK_DISCONNECT


@pytest.mark.asyncio
async def test_restore_single_file_tarball_tamper_halts(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m, _ = _chain_3(arweave, memos, tmp_path)
    # Tamper the personality tarball for e2
    p_tx = m.events[1]["personality"]["tx_id"]
    arweave._store[p_tx] = b"TAMPERED_BYTES"
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(tmp_path / "tampered.json"),
        arweave_fetch=arweave.download,
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_TARBALL_HASH_MISMATCH
    assert result.halt_event_id == "e2"


@pytest.mark.asyncio
async def test_restore_single_file_empty_manifest_halts(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))

    async def _noop(x): return b""

    result = await restore_single_file(
        manifest=m, component="personality", arc_name="x",
        output_path=str(tmp_path / "x.json"),
        arweave_fetch=_noop,
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_MANIFEST_EMPTY


# ── fetch_file_at_event (§3.4) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_file_at_event_returns_bytes(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m, expected = _chain_3(arweave, memos, tmp_path)
    data = await fetch_file_at_event(
        manifest=m, component="personality", arc_name="state.json",
        target_event_id="e2", arweave_fetch=arweave.download,
    )
    assert data == expected["state.json@e2"]


@pytest.mark.asyncio
async def test_fetch_file_at_event_raises_on_failure(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m, _ = _chain_3(arweave, memos, tmp_path)
    with pytest.raises(ValueError, match="halted"):
        await fetch_file_at_event(
            manifest=m, component="personality", arc_name="ghost.json",
            target_event_id="e3", arweave_fetch=arweave.download,
        )


# ── SPEC §11.H.4 Tier-3 boot integrity fallback scenario ─────────────────


@pytest.mark.asyncio
async def test_spec_11h4_tier3_boot_fallback_scenario(tmp_path):
    """End-to-end §3.2: simulate a boot where data/state.json + .bak +
    .bak.prev are all corrupted. Caller invokes restore_single_file with
    expected_post_merkle_root pinned to what SPEC §11.H.5 expected (e.g.
    from the manifest itself), and gets a clean reconstruction."""
    arweave = FakeArweave(); memos = FakeMemos()
    m, expected = _chain_3(arweave, memos, tmp_path)

    # Simulate the corrupted state on disk
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "state.json").write_bytes(b"corrupt")
    (data_dir / "state.json.bak").write_bytes(b"also_corrupt")
    (data_dir / "state.json.bak.prev").write_bytes(b"also_also_corrupt")

    # Tier-3: fetch from Arweave using the unified manifest. Expected hash
    # is what the manifest's latest event says state.json should hash to.
    expected_root = _sha256(expected["state.json"])
    result = await restore_single_file(
        manifest=m, component="personality", arc_name="state.json",
        output_path=str(data_dir / "state.json"),  # overwrite the corrupt file
        arweave_fetch=arweave.download,
        expected_post_merkle_root=expected_root,
    )
    assert result.status == "success"
    assert (data_dir / "state.json").read_bytes() == expected["state.json"]
    # Restored file matches what the manifest committed to
    assert result.final_merkle_root == expected_root


# ── select_single_file_chain unit ────────────────────────────────────────


def test_select_single_file_chain_delegates_to_full_chain(tmp_path):
    arweave = FakeArweave(); memos = FakeMemos()
    m, _ = _chain_3(arweave, memos, tmp_path)
    chain = select_single_file_chain(m, "personality", "state.json")
    # Single-file chain is the full chain by design (skips are no-ops)
    assert [e["event_id"] for e in chain] == ["e1", "e2", "e3"]
