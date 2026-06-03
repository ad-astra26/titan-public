"""SPEC §24.8 + rFP §3.1 — full crash-recovery restore protocol tests.

Hermetic 5-deep chain end-to-end + halt-on-mismatch coverage. No network,
no Solana, no Arweave gateway — all I/O stubbed with dict-backed fetchers.
"""

from __future__ import annotations

import asyncio
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
    HALT_BROKEN_CHAIN,
    HALT_EVENT_MERKLE_MISMATCH,
    HALT_MANIFEST_EMPTY,
    HALT_TARBALL_FETCH_FAILED,
    HALT_TARBALL_HASH_MISMATCH,
    HALT_ZK_DISCONNECT,
    RestoreResult,
    apply_event_components,
    atomic_swap_target_into_data,
    fetch_event_components,
    restore_full,
    select_restore_chain,
    verify_component_merkle,
    verify_event_merkle,
    verify_event_zk_commit,
)
from titan_hcl.logic.backup_unified_manifest import (
    UnifiedManifest,
    make_event,
)
from titan_hcl.logic.backup_zk_commit import (
    build_zk_memo,
    compute_event_merkle_root,
)


# ── Synthetic chain harness ──────────────────────────────────────────────


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeArweave:
    """Dict-backed Arweave store. Async download_file (callable for
    arweave_fetch kw of restore_full)."""

    def __init__(self):
        self._store: dict[str, bytes] = {}
        self.fetch_log: list[str] = []
        self.fail_for: set[str] = set()  # tx_ids to simulate fetch failure

    def upload_sync(self, data: bytes) -> str:
        tx_id = "ar_" + uuid.uuid4().hex[:16]
        self._store[tx_id] = data
        return tx_id

    async def download(self, tx_id: str) -> bytes:
        self.fetch_log.append(tx_id)
        if tx_id in self.fail_for:
            raise RuntimeError(f"simulated Arweave 503 for {tx_id}")
        if tx_id not in self._store:
            raise KeyError(f"unknown Arweave tx_id {tx_id}")
        return self._store[tx_id]


class FakeMemoStore:
    """Dict-backed Solana memo store. Maps sig → memo_text."""

    def __init__(self):
        self._memos: dict[str, str] = {}
        self.fetch_log: list[str] = []
        self.fail_for: set[str] = set()

    def record(self, sig: str, memo: str) -> None:
        self._memos[sig] = memo

    async def fetch(self, sig: str) -> str:
        self.fetch_log.append(sig)
        if sig in self.fail_for:
            raise RuntimeError(f"simulated Solana RPC failure for {sig}")
        if sig not in self._memos:
            raise KeyError(f"unknown sig {sig}")
        return self._memos[sig]


def _arc_to_target(target_dir: str):
    """Build the arc_to_target callable for restore_full. Files land at
    target_dir/<component>/<arc_name> so tests can assert clean separation."""
    def _map(component: str, arc_name: str) -> str:
        return os.path.join(target_dir, component, arc_name)
    return _map


def _build_event(
    *,
    titan_id: str,
    event_id: str,
    event_type: str,
    prev_event_id: Optional[str],
    prev_event_merkle_root: Optional[str],
    personality_files: list[tuple[str, dict]],   # (arc, diff_dict)
    timechain_files: list[tuple[str, dict]],
    soul_files: Optional[list[tuple[str, dict]]],
    arweave: FakeArweave,
    memos: FakeMemoStore,
    tmp_path,
    baseline_trigger: Optional[str] = None,
) -> dict:
    """Pack tarballs, upload, build manifest event + ZK memo. Returns the
    event dict (ready to .append_event() on manifest)."""

    def _pack_component(component: str, files: list[tuple[str, dict]]) -> dict:
        if not files:
            return None
        out = tmp_path / f"{event_id}_{component}.tar.gz"
        specs = [FileDiffSpec(arc, dd) for arc, dd in files]
        info = pack_event_tarball(
            event_id=event_id, event_type=event_type,
            component=component, file_specs=specs,
            output_path=str(out),
        )
        tarball_bytes = out.read_bytes()
        tx_id = arweave.upload_sync(tarball_bytes)
        # Use the file's per-file merkle as component aggregator (simple
        # sum-of-hashes is overkill — the tarball's own sha256 IS what the
        # event Merkle should commit to per rFP §24.7. We store
        # tarball-sha256 in merkle_root because that's what
        # verify_component_merkle checks).
        return {
            "tx_id": tx_id,
            "merkle_root": info["tarball_sha256"],
            "size_bytes": info["size_bytes"],
            "diff_mode": event_type,  # "baseline" or "incremental" naming reused
        }

    personality_sub = _pack_component("personality", personality_files)
    timechain_sub = _pack_component("timechain", timechain_files)
    soul_sub = _pack_component("soul", soul_files) if soul_files else None

    # Recompose event_merkle_root per §24.7 (Phase 5 logic)
    event_merkle_root = compute_event_merkle_root(
        personality_merkle_root=personality_sub["merkle_root"],
        timechain_merkle_root=timechain_sub["merkle_root"],
        soul_merkle_root=soul_sub["merkle_root"] if soul_sub else None,
    )

    # Build v=2 memo and record it as if Solana confirmed
    memo = build_zk_memo(
        event_id=event_id,
        event_merkle_root=event_merkle_root,
        prev_event_merkle_root=prev_event_merkle_root,
    )
    zk_sig = "sig_" + uuid.uuid4().hex[:16]
    memos.record(zk_sig, memo)
    prev_short = prev_event_merkle_root[:16] if prev_event_merkle_root \
        else "genesis"

    event = make_event(
        event_id=event_id,
        event_type=event_type,
        prev_event_id=prev_event_id,
        baseline_trigger=baseline_trigger if event_type == "baseline" else None,
        personality=personality_sub,
        timechain=timechain_sub,
        soul=soul_sub,
        zk_commit_tx=zk_sig,
        zk_memo_prev_short=prev_short,
    )
    # Stash the recomposed root in-event for the test harness to chain
    event["_test_event_merkle_root"] = event_merkle_root
    return event


def _full(content: bytes, encoder: str = "full_ship") -> dict:
    return {
        "diff_mode": "full",
        "patch_bytes": content,
        "merkle_root": _sha256(content),
        "size_bytes": len(content),
        "encoder": encoder,
    }


def _xdelta_inc(patch: bytes, baseline_hash: str, post: bytes) -> dict:
    return {
        "diff_mode": "incremental",
        "patch_bytes": patch,
        "merkle_root": _sha256(post),
        "size_bytes": len(post),
        "baseline_merkle_root": baseline_hash,
        "encoder": "xdelta3",
    }


def _tail_inc(tail: bytes, prev_offset: int, post: bytes,
              block_range=None) -> dict:
    return {
        "diff_mode": "tail",
        "patch_bytes": tail,
        "merkle_root": _sha256(post),
        "size_bytes": len(post),
        "prev_offset_bytes": prev_offset,
        "block_range": list(block_range) if block_range else None,
        "encoder": "timechain_tail",
    }


# ── select_restore_chain unit tests ───────────────────────────────────────


def test_select_chain_empty_manifest_raises(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    with pytest.raises(ValueError, match=HALT_MANIFEST_EMPTY):
        select_restore_chain(m)


def test_select_chain_single_baseline_returns_baseline(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    ev = make_event(
        event_id="e1", event_type="baseline",
        prev_event_id=None, baseline_trigger="first_event",
        personality={"tx_id": "x", "merkle_root": "a" * 64,
                     "size_bytes": 1, "diff_mode": "baseline"},
        timechain={"tx_id": "y", "merkle_root": "b" * 64,
                   "size_bytes": 1, "diff_mode": "baseline"},
    )
    m.append_event(ev)
    chain = select_restore_chain(m)
    assert len(chain) == 1
    assert chain[0]["event_id"] == "e1"


def test_select_chain_walks_back_to_baseline(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    base = make_event(
        event_id="b", event_type="baseline",
        prev_event_id=None, baseline_trigger="first_event",
        personality={"tx_id": "x1", "merkle_root": "a" * 64,
                     "size_bytes": 1, "diff_mode": "baseline"},
        timechain={"tx_id": "y1", "merkle_root": "b" * 64,
                   "size_bytes": 1, "diff_mode": "baseline"},
    )
    m.append_event(base)
    for i in range(3):
        prev_id = "b" if i == 0 else f"i{i-1}"
        inc = make_event(
            event_id=f"i{i}", event_type="incremental",
            prev_event_id=prev_id, baseline_trigger=None,
            personality={"tx_id": f"x{i+2}", "merkle_root": "a" * 64,
                         "size_bytes": 1, "diff_mode": "incremental"},
            timechain={"tx_id": f"y{i+2}", "merkle_root": "b" * 64,
                       "size_bytes": 1, "diff_mode": "incremental"},
        )
        m.append_event(inc)
    chain = select_restore_chain(m)
    assert [e["event_id"] for e in chain] == ["b", "i0", "i1", "i2"]


def test_select_chain_to_specific_event(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    base = make_event(
        event_id="b", event_type="baseline",
        prev_event_id=None, baseline_trigger="first_event",
        personality={"tx_id": "x", "merkle_root": "a" * 64,
                     "size_bytes": 1, "diff_mode": "baseline"},
        timechain={"tx_id": "y", "merkle_root": "b" * 64,
                   "size_bytes": 1, "diff_mode": "baseline"},
    )
    m.append_event(base)
    prev_id = "b"
    for i in range(4):
        inc = make_event(
            event_id=f"i{i}", event_type="incremental",
            prev_event_id=prev_id, baseline_trigger=None,
            personality={"tx_id": f"x{i}", "merkle_root": "a" * 64,
                         "size_bytes": 1, "diff_mode": "incremental"},
            timechain={"tx_id": f"y{i}", "merkle_root": "b" * 64,
                       "size_bytes": 1, "diff_mode": "incremental"},
        )
        m.append_event(inc)
        prev_id = f"i{i}"
    chain = select_restore_chain(m, target_event_id="i1")
    assert [e["event_id"] for e in chain] == ["b", "i0", "i1"]


def test_select_chain_broken_chain_raises(tmp_path):
    m = UnifiedManifest("T1", base_dir=str(tmp_path))
    base = make_event(
        event_id="b", event_type="baseline",
        prev_event_id=None, baseline_trigger="first_event",
        personality={"tx_id": "x", "merkle_root": "a" * 64,
                     "size_bytes": 1, "diff_mode": "baseline"},
        timechain={"tx_id": "y", "merkle_root": "b" * 64,
                   "size_bytes": 1, "diff_mode": "baseline"},
    )
    m.append_event(base)
    inc = make_event(
        event_id="i", event_type="incremental",
        prev_event_id="b", baseline_trigger=None,
        personality={"tx_id": "x2", "merkle_root": "a" * 64,
                     "size_bytes": 1, "diff_mode": "incremental"},
        timechain={"tx_id": "y2", "merkle_root": "b" * 64,
                   "size_bytes": 1, "diff_mode": "incremental"},
    )
    m.append_event(inc)
    # Corrupt the chain: change inc's prev_event_id to a non-existent one
    m._data["events"][1]["prev_event_id"] = "ghost"
    with pytest.raises(ValueError, match="manifest_chain_broken"):
        select_restore_chain(m)


# ── end-to-end synthetic 5-deep restore ──────────────────────────────────


@pytest.mark.asyncio
async def test_synthetic_5deep_full_restore_byte_identical(tmp_path):
    """Hermetic gate test: synth 5-event chain (baseline + 4 incrementals),
    walk restore_full end-to-end, assert byte-identical reconstruction
    matches what a direct snapshot at event #5 would have produced."""

    arweave = FakeArweave()
    memos = FakeMemoStore()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    # ── Event 1: BASELINE ────────────────────────────────────────────
    # personality has 2 files (a tiny config + a small JSON state)
    # timechain has 1 file (a chain.bin with 100 bytes)
    # soul: none (only weekly events have soul)
    p1_config = b"version=1.0\n"
    p1_state = b'{"epoch": 0, "mood": "neutral"}'
    t1_chain = b"BLOCK0" * 16  # 96 bytes
    e1 = _build_event(
        titan_id="T1", event_id="e1", event_type="baseline",
        prev_event_id=None, prev_event_merkle_root=None,
        personality_files=[
            ("config.txt", _full(p1_config)),
            ("state.json", _full(p1_state)),
        ],
        timechain_files=[("chain.bin", _full(t1_chain))],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        baseline_trigger="first_event",
    )
    manifest.append_event(e1)

    # ── Event 2: INCREMENTAL — config unchanged, state changed,
    #    chain appended ───────────────────────────────────────────────
    p2_state = b'{"epoch": 1, "mood": "curious"}'
    t2_chain = t1_chain + b"BLOCK1" * 16
    # Use REAL diff_encoders for the state.json incremental
    # (xdelta3) and chain.bin (tail). config.txt skipped.
    p1_state_path = tmp_path / "p1_state.bin"
    p1_state_path.write_bytes(p1_state)
    p2_state_path = tmp_path / "p2_state.bin"
    p2_state_path.write_bytes(p2_state)
    state_diff = diff_encoders.xdelta3.encode_diff(
        current_path=str(p2_state_path),
        baseline_path=str(p1_state_path),
    )
    state_diff["encoder"] = "xdelta3"
    # Tail diff for chain.bin
    t1_chain_path = tmp_path / "t1_chain.bin"
    t1_chain_path.write_bytes(t1_chain)
    t2_chain_path = tmp_path / "t2_chain.bin"
    t2_chain_path.write_bytes(t2_chain)
    chain_diff = diff_encoders.timechain_tail.encode_diff(
        current_path=str(t2_chain_path),
        baseline_path=str(t1_chain_path),
        block_range=(1, 1),
    )
    chain_diff["encoder"] = "timechain_tail"
    # config skipped — diff_encoders doesn't have a "skipped" encoding
    # built in, but the Phase 4 content_hash cache constructs a skipped
    # pointer dict. We mimic the shape the restore engine expects.
    config_skip = {
        "diff_mode": "skipped",
        "patch_bytes": b"",
        "merkle_root": _sha256(p1_config),
        "size_bytes": len(p1_config),
        "encoder": "full_ship",  # encoder is irrelevant when skipped
    }
    e2 = _build_event(
        titan_id="T1", event_id="e2", event_type="incremental",
        prev_event_id="e1", prev_event_merkle_root=e1["_test_event_merkle_root"],
        personality_files=[
            ("config.txt", config_skip),
            ("state.json", state_diff),
        ],
        timechain_files=[("chain.bin", chain_diff)],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
    )
    manifest.append_event(e2)

    # ── Event 3: INCREMENTAL — state changes again, chain grows again
    p3_state = b'{"epoch": 2, "mood": "calm"}'
    t3_chain = t2_chain + b"BLOCK2" * 16
    p3_state_path = tmp_path / "p3_state.bin"
    p3_state_path.write_bytes(p3_state)
    t3_chain_path = tmp_path / "t3_chain.bin"
    t3_chain_path.write_bytes(t3_chain)
    state_diff_3 = diff_encoders.xdelta3.encode_diff(
        current_path=str(p3_state_path),
        baseline_path=str(p2_state_path),
    )
    state_diff_3["encoder"] = "xdelta3"
    chain_diff_3 = diff_encoders.timechain_tail.encode_diff(
        current_path=str(t3_chain_path),
        baseline_path=str(t2_chain_path),
        block_range=(2, 2),
    )
    chain_diff_3["encoder"] = "timechain_tail"
    e3 = _build_event(
        titan_id="T1", event_id="e3", event_type="incremental",
        prev_event_id="e2", prev_event_merkle_root=e2["_test_event_merkle_root"],
        personality_files=[
            ("config.txt", config_skip),
            ("state.json", state_diff_3),
        ],
        timechain_files=[("chain.bin", chain_diff_3)],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
    )
    manifest.append_event(e3)

    # ── Event 4: INCREMENTAL — state changes, chain grows ──────────
    p4_state = b'{"epoch": 3, "mood": "focused", "extra": "yes"}'
    t4_chain = t3_chain + b"BLOCK3" * 16
    p4_state_path = tmp_path / "p4_state.bin"
    p4_state_path.write_bytes(p4_state)
    t4_chain_path = tmp_path / "t4_chain.bin"
    t4_chain_path.write_bytes(t4_chain)
    state_diff_4 = diff_encoders.xdelta3.encode_diff(
        current_path=str(p4_state_path),
        baseline_path=str(p3_state_path),
    )
    state_diff_4["encoder"] = "xdelta3"
    chain_diff_4 = diff_encoders.timechain_tail.encode_diff(
        current_path=str(t4_chain_path),
        baseline_path=str(t3_chain_path),
        block_range=(3, 3),
    )
    chain_diff_4["encoder"] = "timechain_tail"
    e4 = _build_event(
        titan_id="T1", event_id="e4", event_type="incremental",
        prev_event_id="e3", prev_event_merkle_root=e3["_test_event_merkle_root"],
        personality_files=[
            ("config.txt", config_skip),
            ("state.json", state_diff_4),
        ],
        timechain_files=[("chain.bin", chain_diff_4)],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
    )
    manifest.append_event(e4)

    # ── Event 5: INCREMENTAL — final state ──────────────────────────
    p5_state = b'{"epoch": 4, "mood": "joy", "extra": "yes", "ts": 9999}'
    t5_chain = t4_chain + b"BLOCK4" * 16
    p5_state_path = tmp_path / "p5_state.bin"
    p5_state_path.write_bytes(p5_state)
    t5_chain_path = tmp_path / "t5_chain.bin"
    t5_chain_path.write_bytes(t5_chain)
    state_diff_5 = diff_encoders.xdelta3.encode_diff(
        current_path=str(p5_state_path),
        baseline_path=str(p4_state_path),
    )
    state_diff_5["encoder"] = "xdelta3"
    chain_diff_5 = diff_encoders.timechain_tail.encode_diff(
        current_path=str(t5_chain_path),
        baseline_path=str(t4_chain_path),
        block_range=(4, 4),
    )
    chain_diff_5["encoder"] = "timechain_tail"
    e5 = _build_event(
        titan_id="T1", event_id="e5", event_type="incremental",
        prev_event_id="e4", prev_event_merkle_root=e4["_test_event_merkle_root"],
        personality_files=[
            ("config.txt", config_skip),
            ("state.json", state_diff_5),
        ],
        timechain_files=[("chain.bin", chain_diff_5)],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
    )
    manifest.append_event(e5)

    # ── Run the restore ──────────────────────────────────────────────
    target_dir = tmp_path / "restored"
    progress: list[dict] = []
    result = await restore_full(
        manifest=manifest,
        target_dir=str(target_dir),
        arweave_fetch=arweave.download,
        memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(target_dir)),
        progress_callback=progress.append,
    )

    # ── Assert success ───────────────────────────────────────────────
    assert result.status == "success", (
        f"halt_reason={result.halt_reason}, errors={result.errors}"
    )
    assert result.halt_reason is None
    assert len(result.applied_events) == 5
    assert result.applied_events == ["e1", "e2", "e3", "e4", "e5"]
    assert result.bytes_fetched > 0
    # 2 personality + 1 timechain per event × 5 events = 15 file applies
    # but config.txt is "skipped" in events 2-5 (no apply), so:
    # event1: config + state + chain = 3
    # event2-5: state + chain = 2 × 4 = 8
    # total: 11
    assert result.restored_files == 11

    # ── Byte-identical reconstruction ────────────────────────────────
    assert (target_dir / "personality" / "config.txt").read_bytes() == p1_config
    assert (target_dir / "personality" / "state.json").read_bytes() == p5_state
    assert (target_dir / "timechain" / "chain.bin").read_bytes() == t5_chain

    # ── progress callback fired for each phase ──────────────────────
    phases = [p["phase"] for p in progress]
    assert phases[0] == "chain_selected"
    assert phases[-1] == "complete"
    assert "fetching_event" in phases
    assert "event_applied" in phases


@pytest.mark.asyncio
async def test_restore_halts_on_tarball_hash_mismatch(tmp_path):
    """If a tarball's bytes don't match the manifest's recorded merkle_root,
    restore halts at HALT_TARBALL_HASH_MISMATCH on that event."""
    arweave = FakeArweave()
    memos = FakeMemoStore()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    e1 = _build_event(
        titan_id="T1", event_id="e1", event_type="baseline",
        prev_event_id=None, prev_event_merkle_root=None,
        personality_files=[("f.txt", _full(b"hello"))],
        timechain_files=[("chain.bin", _full(b"data"))],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        baseline_trigger="first_event",
    )
    manifest.append_event(e1)

    # Tamper the Arweave-stored bytes for the personality tarball
    p_tx = e1["personality"]["tx_id"]
    arweave._store[p_tx] = b"TAMPERED"

    target_dir = tmp_path / "restored"
    result = await restore_full(
        manifest=manifest, target_dir=str(target_dir),
        arweave_fetch=arweave.download,
        memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(target_dir)),
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_TARBALL_HASH_MISMATCH
    assert result.halt_event_id == "e1"
    assert any(HALT_TARBALL_HASH_MISMATCH in err for err in result.errors)


@pytest.mark.asyncio
async def test_restore_halts_on_zk_memo_event_root_mismatch(tmp_path):
    """If the on-chain memo's root fragment doesn't match the recomposed
    event_merkle_root, restore halts at HALT_EVENT_MERKLE_MISMATCH."""
    arweave = FakeArweave()
    memos = FakeMemoStore()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    e1 = _build_event(
        titan_id="T1", event_id="e1", event_type="baseline",
        prev_event_id=None, prev_event_merkle_root=None,
        personality_files=[("f.txt", _full(b"hi"))],
        timechain_files=[("chain.bin", _full(b"chain"))],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        baseline_trigger="first_event",
    )
    manifest.append_event(e1)

    # Tamper the recorded memo: rewrite to claim a different root
    sig = e1["zk_commit_tx"]
    bad_memo = build_zk_memo(
        event_id="e1",
        event_merkle_root="d" * 64,  # not the real recomposed root
        prev_event_merkle_root=None,
    )
    memos._memos[sig] = bad_memo

    target_dir = tmp_path / "restored"
    result = await restore_full(
        manifest=manifest, target_dir=str(target_dir),
        arweave_fetch=arweave.download,
        memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(target_dir)),
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_EVENT_MERKLE_MISMATCH
    assert result.halt_event_id == "e1"


@pytest.mark.asyncio
async def test_restore_halts_on_zk_disconnect(tmp_path):
    """If the memo fetcher raises (Solana RPC unavailable), restore halts
    cleanly at HALT_ZK_DISCONNECT — distinct from MERKLE_MISMATCH."""
    arweave = FakeArweave()
    memos = FakeMemoStore()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    e1 = _build_event(
        titan_id="T1", event_id="e1", event_type="baseline",
        prev_event_id=None, prev_event_merkle_root=None,
        personality_files=[("f.txt", _full(b"hi"))],
        timechain_files=[("chain.bin", _full(b"chain"))],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        baseline_trigger="first_event",
    )
    manifest.append_event(e1)

    # Make the Solana RPC fail
    memos.fail_for.add(e1["zk_commit_tx"])

    target_dir = tmp_path / "restored"
    result = await restore_full(
        manifest=manifest, target_dir=str(target_dir),
        arweave_fetch=arweave.download,
        memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(target_dir)),
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_ZK_DISCONNECT
    assert result.halt_event_id == "e1"


@pytest.mark.asyncio
async def test_restore_halts_on_arweave_fetch_failed(tmp_path):
    arweave = FakeArweave()
    memos = FakeMemoStore()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    e1 = _build_event(
        titan_id="T1", event_id="e1", event_type="baseline",
        prev_event_id=None, prev_event_merkle_root=None,
        personality_files=[("f.txt", _full(b"hi"))],
        timechain_files=[("chain.bin", _full(b"chain"))],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        baseline_trigger="first_event",
    )
    manifest.append_event(e1)

    arweave.fail_for.add(e1["personality"]["tx_id"])

    target_dir = tmp_path / "restored"
    result = await restore_full(
        manifest=manifest, target_dir=str(target_dir),
        arweave_fetch=arweave.download,
        memo_fetch=memos.fetch,
        arc_to_target=_arc_to_target(str(target_dir)),
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_TARBALL_FETCH_FAILED
    assert result.halt_event_id == "e1"


@pytest.mark.asyncio
async def test_restore_can_skip_zk_verification(tmp_path):
    """verify_zk_chain=False bypasses memo round-trip — used by Phase 7
    single-file restore where outer chain was already verified."""
    arweave = FakeArweave()
    memos = FakeMemoStore()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    e1 = _build_event(
        titan_id="T1", event_id="e1", event_type="baseline",
        prev_event_id=None, prev_event_merkle_root=None,
        personality_files=[("f.txt", _full(b"hello"))],
        timechain_files=[("chain.bin", _full(b"data"))],
        soul_files=None,
        arweave=arweave, memos=memos, tmp_path=tmp_path,
        baseline_trigger="first_event",
    )
    manifest.append_event(e1)

    # Even if Solana is down, restore should succeed when verify_zk_chain=False
    memos.fail_for.add(e1["zk_commit_tx"])

    target_dir = tmp_path / "restored"

    async def _noop_memo(sig: str) -> str:
        return ""

    result = await restore_full(
        manifest=manifest, target_dir=str(target_dir),
        arweave_fetch=arweave.download,
        memo_fetch=_noop_memo,
        arc_to_target=_arc_to_target(str(target_dir)),
        verify_zk_chain=False,
    )
    assert result.status == "success"
    assert (target_dir / "personality" / "f.txt").read_bytes() == b"hello"


@pytest.mark.asyncio
async def test_restore_empty_manifest_halts(tmp_path):
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    target_dir = tmp_path / "restored"

    async def _noop(x):
        return b""

    async def _noop_memo(x):
        return ""

    result = await restore_full(
        manifest=manifest, target_dir=str(target_dir),
        arweave_fetch=_noop, memo_fetch=_noop_memo,
        arc_to_target=_arc_to_target(str(target_dir)),
    )
    assert result.status == "halted"
    assert result.halt_reason == HALT_MANIFEST_EMPTY


# ── component-level unit tests ────────────────────────────────────────


def test_verify_component_merkle_passes_on_match():
    data = b"hello"
    event = {
        "event_id": "e",
        "personality": {"merkle_root": _sha256(data)},
    }
    verify_component_merkle(event, "personality", data)  # no raise


def test_verify_component_merkle_raises_on_mismatch():
    event = {
        "event_id": "e",
        "personality": {"merkle_root": "a" * 64},
    }
    with pytest.raises(ValueError, match="tarball_hash_mismatch"):
        verify_component_merkle(event, "personality", b"other")


def test_verify_event_merkle_recomposes_correctly():
    p = b"p_tarball"
    t = b"t_tarball"
    p_root = _sha256(p)
    t_root = _sha256(t)
    event = {
        "event_id": "e",
        "personality": {"merkle_root": p_root},
        "timechain": {"merkle_root": t_root},
        "soul": None,
    }
    result = verify_event_merkle(event, {"personality": p, "timechain": t})
    expected = compute_event_merkle_root(
        personality_merkle_root=p_root,
        timechain_merkle_root=t_root,
        soul_merkle_root=None,
    )
    assert result == expected


# ── atomic_swap_target_into_data tests ──────────────────────────────────


def test_atomic_swap_swaps_cleanly(tmp_path):
    target = tmp_path / "restored"
    target.mkdir()
    (target / "marker.txt").write_text("new state")
    data = tmp_path / "data"
    data.mkdir()
    (data / "old.txt").write_text("old state")
    result = atomic_swap_target_into_data(
        target_dir=str(target), data_dir=str(data),
        keep_old_as="data.pre",
    )
    assert result["swapped"] is True
    assert result["preserved_at"] == str(tmp_path / "data.pre")
    assert (data / "marker.txt").read_text() == "new state"
    assert (tmp_path / "data.pre" / "old.txt").read_text() == "old state"


def test_atomic_swap_no_existing_data_dir(tmp_path):
    target = tmp_path / "restored"
    target.mkdir()
    (target / "marker.txt").write_text("new state")
    data = tmp_path / "data"  # does not exist
    result = atomic_swap_target_into_data(
        target_dir=str(target), data_dir=str(data),
        keep_old_as="data.pre",
    )
    assert result["swapped"] is True
    assert result["preserved_at"] is None
    assert (data / "marker.txt").read_text() == "new state"


def test_atomic_swap_missing_target_errors(tmp_path):
    target = tmp_path / "nope"
    data = tmp_path / "data"
    result = atomic_swap_target_into_data(
        target_dir=str(target), data_dir=str(data),
    )
    assert result["swapped"] is False
    assert any("does not exist" in e for e in result["errors"])


def test_apply_event_components_best_effort_skips_unreplayable(tmp_path):
    """An unreplayable per-file diff (a 'tail' with no baseline on disk — the same
    failure shape as a divergent-baseline xdelta in a damaged chain) HARD-RAISES in
    strict mode, but is SKIPPED + recorded in best_effort mode (file keeps last-good
    bytes), recovering the MAXIMUM restorable state."""
    out = tmp_path / "tc.tar.gz"
    tail_dd = {
        "diff_mode": "tail", "patch_bytes": b"APPEND", "prev_offset_bytes": 10,
        "size_bytes": 16, "merkle_root": "aa" * 32, "encoder": "timechain_tail",
    }
    pack_event_tarball(
        event_id="ev_be", event_type="incremental", component="timechain",
        file_specs=[FileDiffSpec("idx.db", tail_dd)], output_path=str(out))
    target = tmp_path / "scratch"
    a2t = _arc_to_target(str(target))
    # strict (default) → halts
    with pytest.raises(ValueError, match="apply_failed"):
        apply_event_components({"timechain": str(out)}, str(target), a2t,
                               verify_patch_hash=False)
    # best-effort → skips the unreplayable file, no raise, records it
    res = apply_event_components({"timechain": str(out)}, str(target), a2t,
                                 verify_patch_hash=False, best_effort=True)
    assert res["skipped"] == ["timechain/idx.db"]
    assert res["restored_files"] == 0
