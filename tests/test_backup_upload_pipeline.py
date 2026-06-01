"""SPEC §24 — Production upload pipeline tests (Phase 5.5 wiring).

Includes the load-bearing end-to-end test: ship 3 events through the
production upload pipeline → restore them via Phase 6 restore_full →
assert byte-identical reconstruction. This is the producer/consumer
contract test that gates Phase 11 flip safety.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Optional

import pytest

from titan_hcl.logic.backup_restore import restore_full
from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
from titan_hcl.logic.backup_upload_pipeline import (
    EVENT_BACKUP_EVENT_COMPLETE,
    EVENT_BACKUP_EVENT_FAILED,
    TierFileSpec,
    run_unified_event,
    ship_tier,
)
from titan_hcl.logic.backup_zk_commit import (
    build_zk_memo,
    compute_event_merkle_root,
)


# ── Fake dependencies ───────────────────────────────────────────────────


class FakeArweaveBackend:
    """In-memory Arweave: upload returns a tx_id, download returns bytes."""
    def __init__(self):
        self._store: dict[str, bytes] = {}
        self._tags: dict[str, dict] = {}
        self.upload_log: list[str] = []
        self.fail_uploads = False

    async def upload(self, data: bytes, tags: dict) -> str:
        if self.fail_uploads:
            raise RuntimeError("simulated upload failure")
        tx = "ar_" + uuid.uuid4().hex[:16]
        self._store[tx] = bytes(data)
        self._tags[tx] = dict(tags)
        self.upload_log.append(tx)
        return tx

    async def download(self, tx_id: str) -> bytes:
        return self._store[tx_id]


_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _fake_base58_sig() -> str:
    """A realistic base58 Solana-style signature (no 0/O/I/l/_), so the v=3
    memo's base58 `prev=` parses exactly as a real on-chain sig would."""
    import hashlib
    h = hashlib.sha256(uuid.uuid4().bytes).digest()
    return "".join(_B58_ALPHABET[b % 58] for b in h)[:24]


class FakeSolanaBackend:
    """In-memory Solana memo store. commit returns sig, memo retrievable."""
    def __init__(self):
        self._memos: dict[str, str] = {}
        self.commit_log: list[str] = []
        self.fail_commits = False

    async def commit(self, event_id: str, ts: int, event_type: str,
                     event_root: str, components: list,
                     prev_sig: Optional[str]) -> Optional[dict]:
        """v=3 chain committer (5J-2 contract): one memo per component,
        returns {"head_sig", "component_sigs"}. Mode follows the component's iv:
        an encrypted component (iv set) → Mode B with that iv; a plaintext
        component → Mode A with a fixed dummy url_key. Exercises the real v=3
        encoder + per-component emission + the head/prev contract."""
        if self.fail_commits:
            return None
        from titan_hcl.logic.backup_memo_v3 import build_v3_memo
        component_sigs: dict = {}
        head_sig: Optional[str] = None
        for comp in components:
            sig = _fake_base58_sig()
            iv = comp.get("iv")
            mode = "B" if iv else "A"
            memo = build_v3_memo(
                event_id=event_id, ts=ts, event_type=event_type,
                tier=comp["tier"], archive_hash=comp["arc"],
                merkle_root=event_root, arweave_tx=comp["tx_id"],
                mode=mode, prev_sig=prev_sig, iv_b64=iv,
                url_key=(b"\x00" * 32 if mode == "A" else None))
            self._memos[sig] = memo
            self.commit_log.append(sig)
            component_sigs[comp["tier"]] = sig
            if head_sig is None:
                head_sig = sig
        return {"head_sig": head_sig, "component_sigs": component_sigs}

    async def fetch(self, sig: str) -> str:
        return self._memos[sig]


def _build_source_files(tmp_path, spec: dict[str, bytes]) -> dict[str, str]:
    """Write each (arc_name → bytes) entry to tmp_path/<arc_name> and
    return {arc_name → absolute_path}."""
    paths: dict[str, str] = {}
    for arc, data in spec.items():
        full = tmp_path / arc
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        paths[arc] = str(full)
    return paths


def _personality_specs_for(paths: dict[str, str]) -> list[TierFileSpec]:
    return [
        TierFileSpec(source_path=p, arc_name=arc) for arc, p in paths.items()
    ]


def _timechain_specs_for(paths: dict[str, str]) -> list[TierFileSpec]:
    return [
        TierFileSpec(source_path=p, arc_name=arc,
                     format_hint="timechain_bin")
        for arc, p in paths.items()
    ]


# ── basic ship_tier tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ship_tier_baseline_packs_and_uploads(tmp_path):
    arweave = FakeArweaveBackend()
    p_paths = _build_source_files(tmp_path / "src", {
        "config.txt": b"v=1\n",
        "state.json": b'{"k": 1}',
    })
    specs = _personality_specs_for(p_paths)
    result = await ship_tier(
        tier="personality", event_id="e1", event_type="baseline",
        specs=specs, baseline_resolver=None,
        arweave_uploader=arweave.upload,
        scratch_dir=str(tmp_path / "scratch"),
        titan_id="T1",
    )
    assert result.error is None
    assert result.tx_id is not None
    assert result.tx_id.startswith("ar_")
    assert result.files_packed == 2
    assert result.files_skipped == 0
    assert result.tarball_sha256 is not None
    assert len(arweave._store) == 1
    # Tags carry the routing metadata
    tags = arweave._tags[result.tx_id]
    assert tags["App-Name"] == "TitanBackupUnified"
    assert tags["Titan-Id"] == "T1"
    assert tags["Tier"] == "personality"
    assert tags["Event-Id"] == "e1"


@pytest.mark.asyncio
async def test_ship_tier_timechain_extracts_block_ranges(tmp_path):
    arweave = FakeArweaveBackend()
    # Synthetic block-range diff: full payload encoder doesn't set
    # block_range automatically, so use the timechain_tail encoder by
    # passing format_hint="timechain_bin" + baseline_path=None which
    # produces diff_mode="full" with block_range=None. To exercise the
    # block_range extraction, we need a tail-mode incremental.
    src = tmp_path / "src"
    src.mkdir()
    chain_path = src / "chain_main.bin"
    chain_path.write_bytes(b"BLOCK0" * 10)
    # baseline file (smaller)
    baseline_src = tmp_path / "baseline_src"
    baseline_src.mkdir()
    baseline_main = baseline_src / "chain_main.bin"
    baseline_main.write_bytes(b"BLOCK0" * 5)

    specs = [TierFileSpec(
        source_path=str(chain_path),
        arc_name="timechain/chain_main.bin",
        format_hint="timechain_bin",
    )]
    # Incremental against baseline — produces tail diff
    result = await ship_tier(
        tier="timechain", event_id="e2", event_type="incremental",
        specs=specs,
        baseline_resolver=lambda arc: str(baseline_main),
        arweave_uploader=arweave.upload,
        scratch_dir=str(tmp_path / "scratch"),
        titan_id="T1",
    )
    assert result.error is None
    assert result.files_packed == 1
    # block_ranges populated for fork "main" (no explicit range was set
    # by the caller, so it stays None even after extraction)
    # Without an explicit block_range passed to encode_diff, the field
    # is None, so it should NOT appear in block_ranges. Verify the
    # extractor handles this gracefully.
    assert isinstance(result.block_ranges, dict)


@pytest.mark.asyncio
async def test_ship_tier_handles_upload_failure(tmp_path):
    arweave = FakeArweaveBackend()
    arweave.fail_uploads = True
    p_paths = _build_source_files(tmp_path / "src", {"f.txt": b"x"})
    specs = _personality_specs_for(p_paths)
    result = await ship_tier(
        tier="personality", event_id="e1", event_type="baseline",
        specs=specs, baseline_resolver=None,
        arweave_uploader=arweave.upload,
        scratch_dir=str(tmp_path / "scratch"),
        titan_id="T1",
    )
    assert result.tx_id is None
    assert "upload failed" in result.error


@pytest.mark.asyncio
async def test_ship_tier_no_files_on_disk(tmp_path):
    arweave = FakeArweaveBackend()
    specs = [TierFileSpec(
        source_path=str(tmp_path / "does_not_exist.bin"),
        arc_name="missing.bin",
    )]
    result = await ship_tier(
        tier="personality", event_id="e1", event_type="baseline",
        specs=specs, baseline_resolver=None,
        arweave_uploader=arweave.upload,
        scratch_dir=str(tmp_path / "scratch"),
        titan_id="T1",
    )
    assert result.tx_id is None
    assert "no in-scope files" in result.error


# ── run_unified_event end-to-end ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_unified_event_first_event_is_baseline(tmp_path):
    arweave = FakeArweaveBackend()
    solana = FakeSolanaBackend()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    p_paths = _build_source_files(tmp_path / "src", {
        "config.txt": b"v=1\n",
        "state.json": b'{"epoch": 0}',
    })
    t_paths = _build_source_files(tmp_path / "src_t", {
        "timechain/chain_main.bin": b"BLOCK0" * 10,
    })

    bus_events: list[tuple[str, dict]] = []

    result = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_personality_specs_for(p_paths),
        timechain_specs=_timechain_specs_for(t_paths),
        soul_specs=None,
        baseline_resolver=None,
        arweave_uploader=arweave.upload,
        zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch"),
        cleanup_scratch=False,
        bus_emit=lambda name, payload: bus_events.append((name, payload)),
    )
    assert result.status == "shipped", (result.errors, result.event_type)
    assert result.event_type == "baseline"
    assert result.baseline_trigger == "first_event"
    assert result.zk_commit_tx is not None
    assert result.event_merkle_root is not None
    # Manifest now has the event
    assert len(manifest.events) == 1
    assert manifest.events[0]["event_id"] == result.event_id
    assert manifest.events[0]["type"] == "baseline"
    # Bus emit fired BACKUP_EVENT_COMPLETE
    assert len(bus_events) == 1
    assert bus_events[0][0] == EVENT_BACKUP_EVENT_COMPLETE
    assert bus_events[0][1]["event_type"] == "baseline"


@pytest.mark.asyncio
async def test_run_unified_event_subsequent_is_incremental(tmp_path):
    arweave = FakeArweaveBackend()
    solana = FakeSolanaBackend()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    p_paths = _build_source_files(tmp_path / "src", {
        "config.txt": b"v=1\n",
    })
    t_paths = _build_source_files(tmp_path / "src_t", {
        "timechain/chain_main.bin": b"BLOCK0" * 10,
    })

    # First event = baseline
    r1 = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_personality_specs_for(p_paths),
        timechain_specs=_timechain_specs_for(t_paths),
        baseline_resolver=None,
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert r1.status == "shipped"
    assert r1.event_type == "baseline"

    # Second event — baseline_resolver returns the unchanged source paths
    # so the content_hash check fires "skipped" for unchanged files
    def _resolver(component, arc_name):
        if component == "personality":
            return p_paths.get(arc_name)
        if component == "timechain":
            return t_paths.get(arc_name)
        return None

    r2 = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_personality_specs_for(p_paths),
        timechain_specs=_timechain_specs_for(t_paths),
        baseline_resolver=_resolver,
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert r2.status == "shipped"
    assert r2.event_type == "incremental"
    assert r2.baseline_trigger is None
    # All files unchanged → all skipped in this event
    assert len(manifest.events) == 2
    assert manifest.events[1]["prev_event_id"] == r1.event_id


@pytest.mark.asyncio
async def test_run_unified_event_weekly_includes_soul(tmp_path):
    arweave = FakeArweaveBackend()
    solana = FakeSolanaBackend()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    p_paths = _build_source_files(tmp_path / "src", {"f.txt": b"x"})
    t_paths = _build_source_files(tmp_path / "src_t", {
        "timechain/chain_main.bin": b"y",
    })
    s_paths = _build_source_files(tmp_path / "src_s", {
        "soul/inner_memory.db": b"soul_data",
    })

    result = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_personality_specs_for(p_paths),
        timechain_specs=_timechain_specs_for(t_paths),
        soul_specs=[TierFileSpec(source_path=p, arc_name=arc)
                    for arc, p in s_paths.items()],
        baseline_resolver=None,
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert result.status == "shipped"
    assert "soul" in result.tiers
    assert result.tiers["soul"].tx_id is not None
    assert manifest.events[0]["soul"] is not None


@pytest.mark.asyncio
async def test_run_unified_event_arweave_failure_aborts(tmp_path):
    arweave = FakeArweaveBackend()
    arweave.fail_uploads = True
    solana = FakeSolanaBackend()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    p_paths = _build_source_files(tmp_path / "src", {"f.txt": b"x"})
    t_paths = _build_source_files(tmp_path / "src_t", {"c.bin": b"y"})
    bus = []
    result = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_personality_specs_for(p_paths),
        timechain_specs=_personality_specs_for(t_paths),
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch"),
        bus_emit=lambda n, p: bus.append((n, p)),
    )
    assert result.status == "failed"
    # Manifest UNCHANGED on failure
    assert len(manifest.events) == 0
    assert bus and bus[-1][0] == EVENT_BACKUP_EVENT_FAILED


@pytest.mark.asyncio
async def test_run_unified_event_zk_failure_aborts(tmp_path):
    arweave = FakeArweaveBackend()
    solana = FakeSolanaBackend()
    solana.fail_commits = True
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    p_paths = _build_source_files(tmp_path / "src", {"f.txt": b"x"})
    t_paths = _build_source_files(tmp_path / "src_t", {"c.bin": b"y"})
    result = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_personality_specs_for(p_paths),
        timechain_specs=_personality_specs_for(t_paths),
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch"),
    )
    assert result.status == "failed"
    assert any("chain_write_failed" in e for e in result.errors)
    assert len(manifest.events) == 0  # not appended on ZK failure


# ── LOAD-BEARING CONTRACT TEST ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_producer_consumer_contract_ship3_then_restore(tmp_path):
    """The load-bearing test: ship 3 events through the production
    upload pipeline, then run Phase 6 restore_full to walk those events,
    verify byte-identical reconstruction.

    This is the contract gate for Phase 11 flip — proves the producer
    side (upload pipeline) and consumer side (restore engine) speak the
    same on-disk tarball/manifest/ZK-memo language.
    """
    arweave = FakeArweaveBackend()
    solana = FakeSolanaBackend()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    # ── Event 1: baseline (first event) ────────────────────────────
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    p1_files = {
        "config.txt": b"version=1.0\n",
        "state.json": b'{"epoch": 0, "mood": "neutral"}',
    }
    t1_files = {
        "timechain/chain_main.bin": b"BLOCK0" * 32,
    }
    for arc, data in {**p1_files, **t1_files}.items():
        (src_dir / arc).parent.mkdir(parents=True, exist_ok=True)
        (src_dir / arc).write_bytes(data)
    p_specs = [TierFileSpec(source_path=str(src_dir / arc), arc_name=arc)
               for arc in p1_files]
    t_specs = [TierFileSpec(source_path=str(src_dir / arc), arc_name=arc,
                            format_hint="timechain_bin")
               for arc in t1_files]

    r1 = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=p_specs, timechain_specs=t_specs,
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch_1"),
    )
    assert r1.status == "shipped"

    # ── Event 2: incremental, state changes + chain grows ──────────
    (src_dir / "state.json").write_bytes(b'{"epoch": 1, "mood": "curious"}')
    (src_dir / "timechain" / "chain_main.bin").write_bytes(
        b"BLOCK0" * 32 + b"BLOCK1" * 32
    )

    # Materialize baseline files for the resolver (in real life this is
    # the prior-baseline-extracted dir; we simulate by using a snapshot
    # taken just before event 2's edits)
    baseline_dir = tmp_path / "baseline_e1"
    baseline_dir.mkdir()
    for arc, data in {**p1_files, **t1_files}.items():
        (baseline_dir / arc).parent.mkdir(parents=True, exist_ok=True)
        (baseline_dir / arc).write_bytes(data)

    def _resolver(component, arc_name):
        return str(baseline_dir / arc_name)

    r2 = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=p_specs, timechain_specs=t_specs,
        baseline_resolver=_resolver,
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch_2"),
    )
    assert r2.status == "shipped"
    assert r2.event_type == "incremental"

    # ── Event 3: incremental, state changes again, chain grows again
    (src_dir / "state.json").write_bytes(
        b'{"epoch": 2, "mood": "joy", "ts": 999}')
    (src_dir / "timechain" / "chain_main.bin").write_bytes(
        b"BLOCK0" * 32 + b"BLOCK1" * 32 + b"BLOCK2" * 32
    )

    # New baseline dir is the post-event-2 reconstructed state
    baseline_dir_2 = tmp_path / "baseline_e2"
    baseline_dir_2.mkdir()
    (baseline_dir_2 / "config.txt").write_bytes(b"version=1.0\n")
    (baseline_dir_2 / "state.json").write_bytes(
        b'{"epoch": 1, "mood": "curious"}'
    )
    (baseline_dir_2 / "timechain").mkdir()
    (baseline_dir_2 / "timechain" / "chain_main.bin").write_bytes(
        b"BLOCK0" * 32 + b"BLOCK1" * 32
    )

    def _resolver_2(component, arc_name):
        return str(baseline_dir_2 / arc_name)

    r3 = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=p_specs, timechain_specs=t_specs,
        baseline_resolver=_resolver_2,
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "scratch_3"),
    )
    assert r3.status == "shipped"
    assert r3.event_type == "incremental"

    # ── Now restore the chain end-to-end ────────────────────────────
    assert len(manifest.events) == 3

    target_dir = tmp_path / "restored"

    def _arc_to_target(component, arc_name):
        return str(target_dir / component / arc_name)

    restore_result = await restore_full(
        manifest=manifest, target_dir=str(target_dir),
        arweave_fetch=arweave.download, memo_fetch=solana.fetch,
        arc_to_target=_arc_to_target,
    )

    assert restore_result.status == "success", (
        restore_result.halt_reason, restore_result.errors,
    )
    assert len(restore_result.applied_events) == 3

    # ── Byte-identical reconstruction at event 3 state ─────────────
    assert (target_dir / "personality" / "config.txt").read_bytes() == \
        b"version=1.0\n"
    assert (target_dir / "personality" / "state.json").read_bytes() == \
        b'{"epoch": 2, "mood": "joy", "ts": 999}'
    assert (target_dir / "timechain" / "timechain" / "chain_main.bin").read_bytes() == \
        b"BLOCK0" * 32 + b"BLOCK1" * 32 + b"BLOCK2" * 32
