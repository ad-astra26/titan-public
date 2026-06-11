"""RFP_backup_arweave_sustainability Phase B — chained incrementals on a
self-healing diff-base + Mode-B decryption.

Hermetic (no live Arweave/Solana — the pipeline is fully dependency-injected):
  A. chained round-trip — ship baseline + N chained incrementals, advancing the
     rolling mirror via _advance_mirror_from_tarballs, then restore_full →
     BYTE-IDENTICAL (the resurrection-safety contract).
  B. make_event accepts the new self_heal baseline_trigger.
  C. the new-vs-known resolver: RAISE for a KNOWN-missing arc (fail-closed,
     INV-BR-9); None for a NEW arc (legit per-file full-ship).
  D. precheck decisions — flag-off missing-known → labeled baseline; all-present
     → incremental; chained steady (mirror verifies) → incremental.
  E. the decrypting fetch round-trips Mode-B ciphertext via the local keypair +
     manifest iv (no Solana RPC).
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import pytest

from titan_hcl.logic import diff_encoders
from titan_hcl.logic.backup import RebirthBackup
from titan_hcl.logic.backup_restore import restore_full
from titan_hcl.logic.backup_unified_manifest import UnifiedManifest, make_event
from titan_hcl.logic.backup_upload_pipeline import (
    MissingDiffBaseError,
    TierFileSpec,
    run_unified_event,
)


# ── in-memory backends (same shape as test_backup_upload_pipeline.py) ──────

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _fake_sig() -> str:
    h = hashlib.sha256(uuid.uuid4().bytes).digest()
    return "".join(_B58[b % 58] for b in h)[:24]


class _FakeArweave:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    async def upload(self, data: bytes, tags: dict) -> str:
        tx = "ar_" + uuid.uuid4().hex[:16]
        self._store[tx] = bytes(data)
        return tx

    async def fetch(self, tx_id: str) -> bytes:
        return self._store[tx_id]


class _FakeSolana:
    def __init__(self):
        self._memos: dict[str, str] = {}

    async def commit(self, event_id, ts, event_type, event_root, components,
                     prev_sig):
        from titan_hcl.logic.backup_memo_v3 import build_v3_memo
        head = None
        for comp in components:
            sig = _fake_sig()
            iv = comp.get("iv")
            mode = "B" if iv else "A"
            self._memos[sig] = build_v3_memo(
                event_id=event_id, ts=ts, event_type=event_type,
                tier=comp["tier"], archive_hash=comp["arc"],
                merkle_root=event_root, arweave_tx=comp["tx_id"], mode=mode,
                prev_sig=prev_sig, iv_b64=iv,
                url_key=(b"\x00" * 32 if mode == "A" else None))
            if head is None:
                head = sig
        return {"head_sig": head, "component_sigs": {}}

    async def fetch(self, sig: str) -> str:
        return self._memos[sig]


class _PhaseBBackup(RebirthBackup):
    """Minimal carrier: inherits the real Phase-B methods, overrides the leaf
    dependencies so they can run without a full RebirthBackup init."""

    def __init__(self, mirror_dir, chained, store=None, full_config=None):
        self._mirror = mirror_dir
        self._titan_id = "T1"
        self._full_config = full_config or {"backup": {"chained_incrementals": chained}}
        self._store = store
        self.alerts: list[str] = []
        # Phase D (INV-BRS-7): the halt/force-baseline/mirror methods are now
        # in-memory + flush via _save_backup_state. This carrier bypasses
        # RebirthBackup.__init__, so seed the side-state attrs + stub the flush
        # (persistence isn't under test here — the in-memory logic is).
        self._halted = False
        self._halt_reason = ""
        self._halt_failed_event_id = None
        self._force_baseline_pending = False
        self._last_dry_run = None
        self._last_restore_test_date = None
        self._mirror_state = None

    def _save_backup_state(self):
        pass  # no-op: don't touch the real data/backup_state.json in unit tests

    def _baseline_working_dir(self):
        return self._mirror

    # _send_telegram_alert was DELETED in production (RFP Phase E — alerts route
    # through titan_hcl.utils.maker_alert.send_maker_alert). Tests capture alerts
    # by monkeypatching backup.send_maker_alert → self.alerts (see _capture_alerts).

    def _ensure_arweave_store_for_unified(self):
        return self._store


class _FakeManifest:
    def __init__(self, latest_id=None, rebase=False, baseline_id="base1"):
        self._latest = latest_id
        self._rebase = rebase
        self.current_baseline_event_id = baseline_id
        self.events: list = []

    def should_rebase(self, now=None, *, cadence="monthly", depth_cap=30):
        return (self._rebase, None)

    def get_latest_event(self):
        return {"event_id": self._latest} if self._latest else None


def _pspecs(src, arcs):
    return [TierFileSpec(source_path=str(src / a), arc_name=a) for a in arcs]


def _tspecs(src, arcs):
    return [TierFileSpec(source_path=str(src / a), arc_name=a,
                         format_hint="timechain_bin") for a in arcs]


def _write(src, files):
    for arc, data in files.items():
        (src / arc).parent.mkdir(parents=True, exist_ok=True)
        (src / arc).write_bytes(data)


# ── A. chained round-trip via the rolling mirror ───────────────────────────


@pytest.mark.asyncio
async def test_chained_roundtrip_via_rolling_mirror(tmp_path):
    arweave, solana = _FakeArweave(), _FakeSolana()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    mirror = str(tmp_path / "mirror")
    backup = _PhaseBBackup(mirror, chained=True)

    src = tmp_path / "src"
    src.mkdir()
    p_arcs = ["config.txt", "state.json"]
    t_arcs = ["timechain/chain_main.bin"]

    def _resolver(component, arc_name):
        cand = os.path.join(mirror, arc_name)
        return cand if os.path.exists(cand) else None

    async def _ship(n, force_baseline_resolver):
        return await run_unified_event(
            titan_id="T1", manifest=manifest,
            personality_specs=_pspecs(src, p_arcs),
            timechain_specs=_tspecs(src, t_arcs),
            baseline_resolver=force_baseline_resolver,
            arweave_uploader=arweave.upload, zk_committer=solana.commit,
            scratch_dir=str(tmp_path / f"scratch_{n}"), cleanup_scratch=False)

    def _advance(result):
        paths = [t.tarball_path for t in result.tiers.values()
                 if getattr(t, "tarball_path", None)]
        backup._advance_mirror_from_tarballs(paths, result.event_id)

    # Event 1 — baseline (resolver=None → full-ship), then advance the mirror.
    _write(src, {"config.txt": b"v=1\n", "state.json": b'{"epoch":0}',
                 "timechain/chain_main.bin": b"BLOCK0" * 32})
    r1 = await _ship(1, None)
    assert r1.status == "shipped" and r1.event_type == "baseline", r1.errors
    _advance(r1)
    # the mirror now equals event-1's reconstructed state + a sidecar @ e1.
    assert (Path(mirror) / "state.json").read_bytes() == b'{"epoch":0}'
    st = backup._load_mirror_state()
    assert st["event_id"] == r1.event_id and "config.txt" in st["arcs"]

    # Event 2 — chained incremental (diff vs the mirror == e1), then advance.
    _write(src, {"state.json": b'{"epoch":1}',
                 "timechain/chain_main.bin": b"BLOCK0" * 32 + b"BLOCK1" * 32})
    r2 = await _ship(2, _resolver)
    assert r2.event_type == "incremental", r2.errors
    _advance(r2)
    assert (Path(mirror) / "state.json").read_bytes() == b'{"epoch":1}'

    # Event 3 — chained incremental again, advance.
    _write(src, {"state.json": b'{"epoch":2}',
                 "timechain/chain_main.bin":
                     b"BLOCK0" * 32 + b"BLOCK1" * 32 + b"BLOCK2" * 32})
    r3 = await _ship(3, _resolver)
    assert r3.event_type == "incremental", r3.errors
    _advance(r3)

    assert len(manifest.events) == 3
    assert manifest.events[2]["prev_event_id"] == r2.event_id

    # Restore the whole chain → BYTE-IDENTICAL to event-3 state.
    target = tmp_path / "restored"

    def _a2t(component, arc_name):
        return str(target / component / arc_name)

    rr = await restore_full(
        manifest=manifest, target_dir=str(target),
        arweave_fetch=arweave.fetch, memo_fetch=solana.fetch,
        arc_to_target=_a2t)
    assert rr.status == "success", (rr.halt_reason, rr.errors)
    assert len(rr.applied_events) == 3
    assert (target / "personality" / "state.json").read_bytes() == b'{"epoch":2}'
    assert (target / "personality" / "config.txt").read_bytes() == b"v=1\n"
    assert (target / "timechain" / "timechain" / "chain_main.bin").read_bytes() \
        == b"BLOCK0" * 32 + b"BLOCK1" * 32 + b"BLOCK2" * 32


# ── B. make_event accepts self_heal ────────────────────────────────────────


def test_make_event_accepts_self_heal_trigger():
    sub = {"tx_id": "ar_x", "merkle_root": "abc", "size_bytes": 1,
           "diff_mode": "baseline"}
    ev = make_event(
        event_id="e1", event_type="baseline", prev_event_id="e0",
        baseline_trigger="self_heal", personality=sub, timechain=sub)
    assert ev["baseline_trigger"] == "self_heal"
    with pytest.raises(ValueError):
        make_event(event_id="e2", event_type="baseline", prev_event_id="e1",
                   baseline_trigger="bogus", personality=sub, timechain=sub)


# ── C. new-vs-known resolver (the fail-closed backstop) ────────────────────


def test_resolver_raises_for_known_missing_returns_none_for_new(tmp_path):
    mirror = str(tmp_path / "mirror")
    os.makedirs(mirror)
    backup = _PhaseBBackup(mirror, chained=True)
    # "present.txt" exists in the mirror; "config.txt" is KNOWN but missing;
    # "brand_new.txt" is NEW (not known).
    Path(mirror, "present.txt").write_bytes(b"hi")
    resolver = backup._make_diff_base_resolver(mirror, {"config.txt", "present.txt"})

    assert resolver("personality", "present.txt") == os.path.join(mirror, "present.txt")
    assert resolver("personality", "brand_new.txt") is None  # NEW → per-file full-ship
    with pytest.raises(MissingDiffBaseError):
        resolver("personality", "config.txt")  # KNOWN but gone → fail closed


# ── D. precheck decisions ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_precheck_flag_off_missing_known_forces_self_heal_baseline(tmp_path, monkeypatch):
    mirror = str(tmp_path / "mirror")
    os.makedirs(mirror)
    backup = _PhaseBBackup(mirror, chained=False)
    # Phase E: alerts route through send_maker_alert — capture into backup.alerts.
    monkeypatch.setattr("titan_hcl.logic.backup.send_maker_alert",
                        lambda text, *a, **k: backup.alerts.append(text))
    # sidecar says config.txt is KNOWN, but it is absent from the mirror.
    backup._write_mirror_state("base1", {"config.txt": "deadbeef"})
    force_et, force_trig, known = await backup._precheck_diff_base(
        _FakeManifest(latest_id="base1", rebase=False))
    assert (force_et, force_trig) == ("baseline", "self_heal")
    assert backup.alerts, "a self-heal baseline must alarm (never silent)"


@pytest.mark.asyncio
async def test_precheck_flag_off_all_present_stays_incremental(tmp_path):
    mirror = str(tmp_path / "mirror")
    os.makedirs(mirror)
    backup = _PhaseBBackup(mirror, chained=False)
    Path(mirror, "config.txt").write_bytes(b"v=1\n")
    backup._write_mirror_state(
        "base1", {"config.txt": diff_encoders.file_merkle_root(
            os.path.join(mirror, "config.txt"))})
    force_et, force_trig, _ = await backup._precheck_diff_base(
        _FakeManifest(latest_id="base1"))
    assert force_et is None and force_trig is None


@pytest.mark.asyncio
async def test_precheck_chained_steady_verifies_and_stays_incremental(tmp_path):
    mirror = str(tmp_path / "mirror")
    os.makedirs(mirror)
    backup = _PhaseBBackup(mirror, chained=True)
    Path(mirror, "config.txt").write_bytes(b"v=1\n")
    sha = diff_encoders.file_merkle_root(os.path.join(mirror, "config.txt"))
    backup._write_mirror_state("evt7", {"config.txt": sha})
    force_et, force_trig, _ = await backup._precheck_diff_base(
        _FakeManifest(latest_id="evt7"))
    assert force_et is None and force_trig is None  # mirror verifies @ latest


@pytest.mark.asyncio
async def test_precheck_chained_drift_no_store_falls_back_to_baseline(tmp_path):
    mirror = str(tmp_path / "mirror")
    os.makedirs(mirror)
    backup = _PhaseBBackup(mirror, chained=True, store=None)  # no store → no L2
    Path(mirror, "config.txt").write_bytes(b"DRIFTED")
    backup._write_mirror_state("evtOLD", {"config.txt": "expected_other_sha"})
    # sidecar.event_id (evtOLD) != latest (evtNEW) AND sha drift → recover fails
    # (no store) → labeled baseline.
    force_et, force_trig, _ = await backup._precheck_diff_base(
        _FakeManifest(latest_id="evtNEW"))
    assert (force_et, force_trig) == ("baseline", "self_heal")


# ── D2. boot-ordering: a TRANSIENT reconstruct failure DEFERS, never self_heals ─
# RFP_backup_redesign_spine (2026-06-11): the T1 false self_heal baseline — the
# chained mirror reconstruct ran at early boot before the chain provider was
# fetch-ready → fetch failed → expensive 2.4GB baseline. Fix: classify the
# reconstruct failure. GENUINE corruption / definitively-missing chain → self_heal;
# a transient fetch failure while the chain is reachable/unproven-gone → DEFER
# (retry next tick → cheap catch-up incremental). Maker rule: baseline only on a
# real integrity failure, never because the chain couldn't be fetched right now.
from titan_hcl.logic.backup import BackupReconstructDeferred  # noqa: E402
from titan_hcl.logic.backup_restore import (  # noqa: E402
    HALT_TARBALL_FETCH_FAILED, HALT_TARBALL_HASH_MISMATCH)


class _ReconCtl(_PhaseBBackup):
    """Drives the chained reconstruct + head-probe outcomes to exercise the
    transient-vs-genuine self_heal classification (the boot-ordering fix)."""

    def __init__(self, mirror_dir, *, recon, head="present"):
        super().__init__(mirror_dir, chained=True, store=_FakeArweave())
        self._recon = recon              # (recovered: bool, halt_reason)
        self._head_status = head         # 'present' | 'missing' | 'unverified'
        self._recover_ran = False

    async def _recover_mirror_to_latest(self, manifest, latest_id):
        self._recover_ran = True
        return self._recon

    async def _probe_latest_chain_head(self, manifest, latest_id):
        return self._head_status

    def _mirror_verifies(self, *a, **k):
        # steady-state (pre-recover) → False to force the reconstruct path;
        # post-recover → True iff the (mocked) reconstruct succeeded.
        return self._recover_ran and self._recon[0]


def _drift_backup(tmp_path, *, recon, head="present"):
    mirror = str(tmp_path / "mirror")
    os.makedirs(mirror)
    backup = _ReconCtl(mirror, recon=recon, head=head)
    Path(mirror, "config.txt").write_bytes(b"DRIFTED")
    backup._write_mirror_state("evtOLD", {"config.txt": "expected_other_sha"})
    return backup


@pytest.mark.asyncio
async def test_precheck_transient_fetch_chain_reachable_DEFERS(tmp_path, monkeypatch):
    """Reconstruct fails to FETCH but the chain is reachable (head=present) →
    DEFER (raise), NEVER self_heal. This is the T1 early-boot case."""
    backup = _drift_backup(tmp_path, recon=(False, HALT_TARBALL_FETCH_FAILED),
                           head="present")
    monkeypatch.setattr("titan_hcl.logic.backup.send_maker_alert",
                        lambda text, *a, **k: backup.alerts.append(text))
    with pytest.raises(BackupReconstructDeferred):
        await backup._precheck_diff_base(_FakeManifest(latest_id="evtNEW"))
    assert not backup.alerts, "a DEFER must NOT alarm a self_heal baseline"


@pytest.mark.asyncio
async def test_precheck_transient_fetch_head_unverified_DEFERS(tmp_path):
    """Reconstruct raised + head=unverified (uncertain, not proven gone) →
    DEFER. INV: never self_heal on uncertainty."""
    backup = _drift_backup(tmp_path, recon=(False, "exception"), head="unverified")
    with pytest.raises(BackupReconstructDeferred):
        await backup._precheck_diff_base(_FakeManifest(latest_id="evtNEW"))


@pytest.mark.asyncio
async def test_precheck_fetch_failed_but_chain_DEFINITIVELY_gone_self_heals(tmp_path, monkeypatch):
    """Reconstruct fetch failed AND head=missing (definitive 404 → genuinely
    gone) → self_heal baseline (legit) + alarm."""
    backup = _drift_backup(tmp_path, recon=(False, HALT_TARBALL_FETCH_FAILED),
                           head="missing")
    monkeypatch.setattr("titan_hcl.logic.backup.send_maker_alert",
                        lambda text, *a, **k: backup.alerts.append(text))
    et, trig, _ = await backup._precheck_diff_base(_FakeManifest(latest_id="evtNEW"))
    assert (et, trig) == ("baseline", "self_heal")
    assert backup.alerts, "a genuine self_heal must alarm"


@pytest.mark.asyncio
async def test_precheck_genuine_corruption_self_heals_without_probe(tmp_path, monkeypatch):
    """Reconstruct failed on DATA corruption (hash mismatch) → self_heal
    immediately (the chain is reachable but its bytes are bad); no head probe."""
    backup = _drift_backup(tmp_path, recon=(False, HALT_TARBALL_HASH_MISMATCH),
                           head="present")  # head would say present, but corruption wins

    async def _boom(*a, **k):  # the probe must NOT be consulted for corruption
        raise AssertionError("head probe must not run for genuine corruption")
    backup._probe_latest_chain_head = _boom
    monkeypatch.setattr("titan_hcl.logic.backup.send_maker_alert",
                        lambda text, *a, **k: backup.alerts.append(text))
    et, trig, _ = await backup._precheck_diff_base(_FakeManifest(latest_id="evtNEW"))
    assert (et, trig) == ("baseline", "self_heal")
    assert backup.alerts


@pytest.mark.asyncio
async def test_precheck_reconstruct_success_stays_incremental(tmp_path):
    """Reconstruct succeeds → mirror at latest → incremental (no baseline)."""
    backup = _drift_backup(tmp_path, recon=(True, None))
    et, trig, _ = await backup._precheck_diff_base(_FakeManifest(latest_id="evtNEW"))
    assert et is None and trig is None


def test_plan_staged_build_v2_returns_none_on_defer(tmp_path, monkeypatch):
    """The orchestrator-facing plan returns None on a DEFER (skip this tick) —
    so NO drip is built (no baseline), and it retries next tick."""
    backup = _drift_backup(tmp_path, recon=(False, HALT_TARBALL_FETCH_FAILED),
                           head="present")
    # gate on: backup_arweave_enabled so _plan_staged_build_v2 proceeds to precheck
    backup._full_config = {"backup": {"chained_incrementals": True},
                           "mainnet_budget": {"backup_arweave_enabled": True}}
    monkeypatch.setattr(backup, "_baseline_working_dir", lambda: backup._mirror)
    monkeypatch.setattr(backup, "_tier_specs_from_paths", lambda *a, **k: [])
    monkeypatch.setattr(backup, "_rebase_params", lambda: ("none", 90))

    class _M(_FakeManifest):
        def __init__(self):
            super().__init__(latest_id="evtNEW")
    monkeypatch.setattr(
        "titan_hcl.logic.backup_unified_manifest.UnifiedManifest.load",
        classmethod(lambda cls, **k: _M()))
    assert backup._plan_staged_build_v2(weekday=0) is None


# ── E. Mode-B decrypting fetch ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decrypting_fetch_roundtrips_mode_b(tmp_path):
    from titan_hcl.logic.backup_crypto import (
        derive_master_key, encrypt_component_tarball)

    # A test keypair (Solana id.json shape: list of 64 ints).
    kp_path = tmp_path / "kp.json"
    kp_bytes = bytes((i * 7 + 13) % 256 for i in range(64))
    kp_path.write_text(json.dumps(list(kp_bytes)))
    from titan_hcl.logic.backup_crypto import load_keypair_bytes
    _, pubkey = load_keypair_bytes(str(kp_path))
    master = derive_master_key(kp_bytes, pubkey)

    plaintext = b"PERSONALITY-TARBALL-PLAINTEXT" * 500
    arc = hashlib.sha256(plaintext).hexdigest()
    ciphertext, iv_b64 = encrypt_component_tarball(plaintext, master, "personality")
    assert ciphertext != plaintext

    store = _FakeArweave()
    tx = await store.upload(ciphertext, {})

    manifest = _FakeManifest()
    manifest.events = [{
        "personality": {"tx_id": tx, "merkle_root": arc, "iv": iv_b64},
        "timechain": {"tx_id": "ar_other", "merkle_root": "x", "iv": None},
    }]
    backup = _PhaseBBackup(
        str(tmp_path / "mirror"), chained=True, store=store,
        full_config={"backup": {"chained_incrementals": True,
                                 "wallet_keypair_path": str(kp_path)}})

    fetch = backup._build_decrypting_fetch(manifest)
    out = await fetch(tx)
    assert out == plaintext  # Mode-B decrypt via local keypair + manifest iv


# ── R4 — weekly restore-test + self-heal halt ──────────────────────────────


def test_halt_mechanism_and_force_baseline_marker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # halt/marker paths are relative to data/backups/
    backup = _PhaseBBackup(str(tmp_path / "mirror"), chained=True)

    assert backup._is_backups_halted() is False
    backup._set_backups_halt(reason="HALT_BROKEN_CHAIN", failed_event_id="evtX")
    assert backup._is_backups_halted() is True
    assert backup._force_baseline_pending is True   # Phase D: in-memory token

    # A clear lifts the halt but LEAVES the force-baseline token → resume rebases.
    backup._clear_backups_halt()
    assert backup._is_backups_halted() is False
    assert backup._force_baseline_pending is True

    # Marker is one-shot.
    assert backup._take_force_baseline() is True
    assert backup._take_force_baseline() is False


@pytest.mark.asyncio
async def test_precheck_forces_baseline_after_failed_restore_test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backup = _PhaseBBackup(str(tmp_path / "mirror"), chained=True)
    backup._set_backups_halt(reason="restore_failed", failed_event_id="evtY")  # arms the marker
    # Even a healthy-looking manifest must rebase: the chain is suspect post-FAIL.
    force_et, force_trig, _ = await backup._precheck_diff_base(
        _FakeManifest(latest_id="evtZ", rebase=False))
    assert (force_et, force_trig) == ("baseline", "self_heal")
    assert backup._take_force_baseline() is False  # consumed by the precheck (one-shot)


@pytest.mark.asyncio
async def test_restore_test_pass_on_green_chain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arweave, solana = _FakeArweave(), _FakeSolana()
    manifest = UnifiedManifest("T1", base_dir="data")  # → tmp_path/data (chdir'd)
    src = tmp_path / "src"
    src.mkdir()
    _write(src, {"config.txt": b"v=1\n", "timechain/chain_main.bin": b"BLOCK0" * 16})
    r1 = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_pspecs(src, ["config.txt"]),
        timechain_specs=_tspecs(src, ["timechain/chain_main.bin"]),
        baseline_resolver=None, arweave_uploader=arweave.upload,
        zk_committer=solana.commit, scratch_dir=str(tmp_path / "s1"))
    assert r1.status == "shipped"

    backup = _PhaseBBackup(str(tmp_path / "mirror"), chained=True, store=arweave)
    emitted: list = []
    ok = await backup._run_weekly_restore_test(
        memo_fetch=solana.fetch, bus_emit=lambda n, p: emitted.append((n, p)))
    assert ok is True
    assert backup._is_backups_halted() is False
    assert any(n == "BACKUP_RESTORE_TEST_PASS" for n, _ in emitted)


@pytest.mark.asyncio
async def test_restore_test_fail_halts_backups(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arweave, solana = _FakeArweave(), _FakeSolana()
    manifest = UnifiedManifest("T1", base_dir="data")
    src = tmp_path / "src"
    src.mkdir()
    _write(src, {"config.txt": b"v=1\n", "timechain/chain_main.bin": b"BLOCK0" * 16})
    r1 = await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_pspecs(src, ["config.txt"]),
        timechain_specs=_tspecs(src, ["timechain/chain_main.bin"]),
        baseline_resolver=None, arweave_uploader=arweave.upload,
        zk_committer=solana.commit, scratch_dir=str(tmp_path / "s1"))
    assert r1.status == "shipped"

    arweave._store.clear()  # tarballs gone → reconstruct fetch fails → restore FAILS
    backup = _PhaseBBackup(str(tmp_path / "mirror"), chained=True, store=arweave)
    # Phase E: the restore-test-failed alert routes through send_maker_alert.
    monkeypatch.setattr("titan_hcl.logic.backup.send_maker_alert",
                        lambda text, *a, **k: backup.alerts.append(text))
    emitted: list = []
    ok = await backup._run_weekly_restore_test(
        memo_fetch=solana.fetch, bus_emit=lambda n, p: emitted.append((n, p)))
    assert ok is False
    assert backup._is_backups_halted() is True                 # INV-BR-4 halt
    assert backup._force_baseline_pending is True              # INV-BKP-5 recovery armed
    assert backup.alerts, "a FAILED restore-test must maker_notify"
    assert any(n == "BACKUP_RESTORE_TEST_FAIL" for n, _ in emitted)
