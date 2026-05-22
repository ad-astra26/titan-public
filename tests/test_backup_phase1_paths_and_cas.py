"""Tests for SPEC §24 / rFP_backup_diff_baseline_unified_v1 Phase 1:

(a) PERSONALITY_PATHS / WEEKLY_EXTRA_PATHS / TIMECHAIN_PATHS canonical inventory
    matches SPEC §24.4.B/C/D (11 dropped + 8 added on personality, after
    titan_chronicles.md re-added 2026-05-22; 5 dropped +
    sage_memory dir added on weekly; new TIMECHAIN_PATHS list with 7 chain .bin
    files + index.db + maker_proposals.db).

(b) Daily-personality dedup race CLOSED via async CAS lock per `on_meditation_
    complete` — regression guard against AUDIT_irys_arweave_costs_20260514 §4
    BUG-2 pattern (15 personality uploads on 2026-05-12 from concurrent
    MEDITATION_COMPLETE events all passing the bare `if today != _last_*_date`
    check before any of them could set the date).
"""

import asyncio

import pytest

from titan_hcl.logic.backup import RebirthBackup


# ── §24.4.B — PERSONALITY_PATHS canonical inventory ──────────────────────


def _personality_path_set():
    return {p[0] for p in RebirthBackup.PERSONALITY_PATHS}


@pytest.mark.parametrize("dropped", [
    "titan_constitution.md",
    # NOTE: titan_chronicles.md was RE-ADDED 2026-05-22 once the chronicle
    # writer was restored (BUG-CHRONICLE-WRITER-DEAD-POST-A87 fixed) — see
    # test_personality_readds_chronicle_after_writer_restored below.
    "data/titan_directives.sig",
    "data/genesis_record.json",
    "data/genesis_nft_metadata.json",
    "data/birth_dna_snapshot.json",
    "data/titan_identity.json",
    "data/runtime_keypair.json",
    "data/zk_queue/pending.json",
    "data/sage_memory/buffer_metadata.json",
    "data/sage_memory/meta.json",
    "data/timechain/contract_stats.json",
])
def test_personality_drops_static_birth_and_ephemeral_files(dropped):
    """SPEC §24.4.B / rFP §4.1 — 11 entries dropped from PERSONALITY_PATHS.

    (Was 12; titan_chronicles.md re-added 2026-05-22 — writer restored.)
    """
    assert dropped not in _personality_path_set(), (
        f"{dropped!r} should be dropped from PERSONALITY_PATHS per SPEC §24.4.B "
        f"(static birth identity → §24.4.A on Arweave via GenesisNFT, OR "
        f"ephemeral / reconstructable)"
    )


def test_personality_readds_chronicle_after_writer_restored():
    """SPEC §24.4.B — titan_chronicles.md RE-ADDED 2026-05-22.

    BUG-CHRONICLE-WRITER-DEAD-POST-A87 is fixed (titan_HCL._append_to_chronicle
    writes meditation reflections on MEDITATION_COMPLETE again), so the
    narrative diary must be captured in the daily personality backup again.
    """
    assert "titan_chronicles.md" in _personality_path_set(), (
        "titan_chronicles.md should be back in PERSONALITY_PATHS now that the "
        "chronicle writer is restored (BUG-CHRONICLE-WRITER-DEAD fixed)"
    )


@pytest.mark.parametrize("added", [
    "data/word_recipes.json",                       # SPEC §11.H critical
    "data/outer_interface_state.json",              # SPEC §11.H critical
    "data/prediction/novelty_state.json",           # SPEC §11.H critical
    "data/word_resonance_dynamic.json",
    "data/dim_history/assessment_history.json",
    "data/meta_teacher/adoption_metrics.json",
    "data/mini_reasoning/",
    "data/titan_vm_v2/",                            # Maker decision 2026-05-15 Q5
])
def test_personality_adds_critical_substrate_files(added):
    """SPEC §24.4.B / rFP §4.1 — 8 entries added to PERSONALITY_PATHS."""
    assert added in _personality_path_set(), (
        f"{added!r} should be added to PERSONALITY_PATHS per SPEC §24.4.B"
    )


# ── §24.4.C — WEEKLY_EXTRA_PATHS canonical inventory ─────────────────────


def _weekly_path_set():
    return {p[0] for p in RebirthBackup.WEEKLY_EXTRA_PATHS}


@pytest.mark.parametrize("dropped", [
    "data/timechain/",          # → §24.4.D dedicated TIMECHAIN_PATHS
    "data/meditation_memos/",   # on-chain Solana memos
    "data/daily_nfts/",         # on-chain NFTs
    "data/testaments/",         # already on-chain AND on Arweave
    "data/backup_records/",     # reconstructable from unified manifest
])
def test_weekly_drops_redundant_or_dedicated_paths(dropped):
    """SPEC §24.4.C / rFP §4.1 — 5 entries dropped from WEEKLY_EXTRA_PATHS."""
    assert dropped not in _weekly_path_set(), (
        f"{dropped!r} should be dropped from WEEKLY_EXTRA_PATHS per SPEC §24.4.C"
    )


def test_weekly_adds_sage_memory_full_dir():
    """SPEC §24.4.C — `data/sage_memory/` FULL DIR replaces the 2 pointer-only
    entries previously in PERSONALITY_PATHS; closes the half-broken state
    where pointer JSONs were in tarball but .memmap data was not."""
    assert "data/sage_memory/" in _weekly_path_set()


# ── §24.4.D — NEW TIMECHAIN_PATHS list ────────────────────────────────────


def test_timechain_paths_lists_7_chain_bin_files():
    """SPEC §24.4.D — separate physical tarball, 7 chain .bin files."""
    t_set = {p[0] for p in RebirthBackup.TIMECHAIN_PATHS}
    for fork in ("conversation", "declarative", "episodic", "main",
                 "meta", "procedural", "system"):
        assert f"data/timechain/chain_{fork}.bin" in t_set


def test_timechain_paths_includes_index_db_and_auxiliary():
    """SPEC §24.4.D — index.db (sqlite over chains) + auxiliary maker_proposals."""
    t_set = {p[0] for p in RebirthBackup.TIMECHAIN_PATHS}
    assert "data/timechain/index.db" in t_set
    assert "data/timechain/auxiliary/maker_proposals.db" in t_set


def test_timechain_paths_count_matches_spec():
    """SPEC §24.4.D — 7 chain .bin + index.db + maker_proposals.db = 9."""
    assert len(RebirthBackup.TIMECHAIN_PATHS) == 9


# ── Dedup race — async CAS lock regression test ─────────────────────────


@pytest.mark.asyncio
async def test_personality_cas_lock_dedup_blocks_concurrent_meditations(monkeypatch):
    """SPEC §24 / rFP_backup_diff_baseline_unified_v1 Phase 1 — closes
    AUDIT_irys_arweave_costs_20260514 §4 BUG-2 (concurrent MEDITATION_COMPLETE
    events all passing the bare date check before any could set the date,
    observed 15 personality uploads on 2026-05-12).

    Fires N concurrent on_meditation_complete events for the same UTC day.
    Asserts upload_personality_to_arweave is invoked EXACTLY ONCE despite
    the concurrent fan-out — the CAS lock + atomic date-write closes the
    window between the read and the write.
    """
    rb = RebirthBackup(network_client=None, titan_id="T1",
                       arweave_store=None, full_config={
                           "backup": {"local_diff_enabled": False},
                       })

    # Disable all the side-effects we don't care about for this test
    upload_call_count = {"n": 0}

    async def _fake_upload(*a, **kw):
        upload_call_count["n"] += 1
        # Simulate non-trivial upload duration so concurrent calls have
        # time to race at the CAS gate
        await asyncio.sleep(0.05)
        return {"arweave_tx": "fake_tx", "size_mb": 1.0, "archive_hash": "fake_hash"}

    async def _noop_async(*a, **kw):
        return None

    def _noop_sync(*a, **kw):
        return None

    monkeypatch.setattr(rb, "upload_personality_to_arweave", _fake_upload)
    monkeypatch.setattr(rb, "_compute_sovereignty", lambda: _noop_async())
    monkeypatch.setattr(rb, "anchor_backup_hash", _noop_async)
    monkeypatch.setattr(rb, "_update_vault_shadow_hash", _noop_async)
    monkeypatch.setattr(rb, "_update_titan_frontmatter", _noop_sync)
    monkeypatch.setattr(rb, "_alert_backup_success", _noop_sync)
    monkeypatch.setattr(rb, "_alert_backup_failure", _noop_sync)
    monkeypatch.setattr(rb, "_save_backup_state", _noop_sync)
    # Block timechain path so test focuses on personality CAS
    rb._last_timechain_date = "2099-12-31"
    # Block soul path — not Sunday
    rb._last_soul_date = "2099-12-31"

    # Fire 10 concurrent meditation events
    payload = {"success": True, "epoch": 1, "promoted": 0}
    results = await asyncio.gather(*[
        rb.on_meditation_complete(payload) for _ in range(10)
    ])

    assert upload_call_count["n"] == 1, (
        f"CAS lock should permit exactly 1 personality upload across 10 "
        f"concurrent meditations; observed {upload_call_count['n']}. "
        f"Regression: AUDIT §4 BUG-2 dedup race re-opened."
    )


@pytest.mark.asyncio
async def test_timechain_cas_lock_dedup_blocks_concurrent_meditations(monkeypatch):
    """Companion to the personality CAS test — timechain gate has its own lock."""
    rb = RebirthBackup(network_client=None, titan_id="T1",
                       arweave_store=None, full_config={
                           "backup": {"local_diff_enabled": False, "local_rolling_days": 30},
                           "mainnet_budget": {"backup_arweave_enabled": False},
                           "network": {"solana_network": "devnet"},
                       })

    timechain_call_count = {"n": 0}

    async def _fake_snapshot_to_arweave(*a, **kw):
        timechain_call_count["n"] += 1
        await asyncio.sleep(0.05)
        return None

    # Block personality + soul paths
    rb._last_personality_date = "2099-12-31"
    rb._last_soul_date = "2099-12-31"

    async def _noop_async(*a, **kw):
        return None

    def _noop_sync(*a, **kw):
        return None

    monkeypatch.setattr(rb, "_compute_sovereignty", lambda: _noop_async())
    monkeypatch.setattr(rb, "_save_backup_state", _noop_sync)
    rb.memory = None  # short-circuit memory.get_persistent_count

    # Patch TimeChainBackup to capture concurrent calls without doing real work
    import titan_hcl.logic.timechain_backup as tcb_module

    class _FakeTCB:
        def __init__(self, *a, **kw):
            pass

        async def snapshot_to_arweave(self, *a, **kw):
            return await _fake_snapshot_to_arweave()

    monkeypatch.setattr(tcb_module, "TimeChainBackup", _FakeTCB)

    payload = {"success": True, "epoch": 1, "promoted": 0}
    await asyncio.gather(*[
        rb.on_meditation_complete(payload) for _ in range(10)
    ])

    assert timechain_call_count["n"] == 1, (
        f"CAS lock should permit exactly 1 timechain upload across 10 "
        f"concurrent meditations; observed {timechain_call_count['n']}"
    )


def test_cas_locks_are_lazy_constructed():
    """Locks must be created lazily on first async use (not in sync __init__)
    so RebirthBackup can be instantiated at plugin boot before the event
    loop exists."""
    rb = RebirthBackup(network_client=None, titan_id="T1",
                       arweave_store=None, full_config={})
    assert rb._personality_cas_lock is None
    assert rb._soul_cas_lock is None
    assert rb._timechain_cas_lock is None
    # First call should construct
    lock = rb._get_personality_cas_lock()
    assert lock is not None
    assert isinstance(lock, asyncio.Lock)
    # Second call should return same instance
    assert rb._get_personality_cas_lock() is lock
