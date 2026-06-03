"""5J-6 — 🜂 Sovereign Titan Resurrection Protocol v1, end-to-end (chunk 5J-3).

Acceptance gate 9 (RFP §3B.3, with mocks): ship 3 events through the PRODUCTION
upload pipeline writing Mode-A v=3 memos, then `resurrect_from_chain` walks the
on-chain memos ALONE — sigs → decode v=3 → decrypt each URL with the
keypair-derived key → fetch from Arweave → verify sha256/merkle → apply — and
reconstructs the Titan's state byte-for-byte. No manifest, no local files: the
trust root is the wallet (+ SSS). This is the sovereignty contract.

Run: python -m pytest tests/test_backup_restore_sovereign.py -p no:anchorpy -q
"""
from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))  # root ahead of scripts/ → titan_hcl = package

from titan_hcl.logic.backup_unified_manifest import UnifiedManifest  # noqa: E402
from titan_hcl.logic.backup_upload_pipeline import (  # noqa: E402
    TierFileSpec,
    run_unified_event,
)
from titan_hcl.logic.backup_memo_v3 import (  # noqa: E402
    build_v3_memo,
    derive_backup_url_key,
)
from titan_hcl.logic.backup_restore import build_arc_to_target  # noqa: E402
from titan_hcl.logic.backup_zk_commit import build_zk_memo  # noqa: E402
import scripts.backup_restore_sovereign as sov  # noqa: E402

SEED = bytes(range(32, 64))  # 32-byte soul seed
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58_sig() -> str:
    h = hashlib.sha256(uuid.uuid4().bytes).digest()
    return "".join(_B58[b % 58] for b in h)[:32]


class ModeAArweave:
    def __init__(self):
        self._store: dict = {}

    async def upload(self, data: bytes, tags: dict) -> str:
        tx = "ar_" + hashlib.sha256(data).hexdigest()[:32]
        self._store[tx] = data
        return tx

    async def download(self, tx: str) -> bytes:
        return self._store[tx]


class ModeASolana:
    """Mode-A v=3 committer (encrypts the URL with the soul-derived key) + an
    in-memory memo store so the sovereign walk can fetch memos by signature."""
    def __init__(self, url_key: bytes):
        self.url_key = url_key
        self._memos: dict = {}
        self.sigs: list = []

    async def commit(self, event_id, ts, event_type, event_root, components, prev_sig):
        head = None
        comp_sigs: dict = {}
        for c in components:
            sig = _b58_sig()
            memo = build_v3_memo(
                event_id=event_id, ts=ts, event_type=event_type, tier=c["tier"],
                archive_hash=c["arc"], merkle_root=event_root,
                arweave_tx=c["tx_id"], mode="A", prev_sig=prev_sig,
                url_key=self.url_key)
            self._memos[sig] = memo
            self.sigs.append(sig)
            comp_sigs[c["tier"]] = sig
            if head is None:
                head = sig
        return {"head_sig": head, "component_sigs": comp_sigs}

    async def fetch(self, sig: str):
        return self._memos.get(sig)


def _specs(src_dir: Path, files: dict, fmt: str = None) -> list:
    specs = []
    for arc in files:
        kw = {"format_hint": fmt} if fmt else {}
        specs.append(TierFileSpec(source_path=str(src_dir / arc), arc_name=arc, **kw))
    return specs


def _write(root: Path, files: dict) -> None:
    for arc, data in files.items():
        (root / arc).parent.mkdir(parents=True, exist_ok=True)
        (root / arc).write_bytes(data)


@pytest.mark.asyncio
async def test_sovereign_resurrection_three_event_chain(tmp_path):
    url_key = derive_backup_url_key(SEED)
    arweave = ModeAArweave()
    solana = ModeASolana(url_key)
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    src = tmp_path / "src"
    src.mkdir()
    p_files = {"config.txt": b"version=1.0\n", "state.json": b'{"epoch": 0}'}
    t_files = {"timechain/chain.bin": b"BLOCK0" * 16}
    _write(src, {**p_files, **t_files})
    p_specs = _specs(src, p_files)
    t_specs = _specs(src, t_files, fmt="timechain_bin")

    # Event 1 — baseline
    r1 = await run_unified_event(
        titan_id="T1", manifest=manifest, personality_specs=p_specs,
        timechain_specs=t_specs, arweave_uploader=arweave.upload,
        zk_committer=solana.commit, scratch_dir=str(tmp_path / "s1"))
    assert r1.status == "shipped" and r1.event_type == "baseline"

    # Event 2 — incremental
    (src / "state.json").write_bytes(b'{"epoch": 1}')
    (src / "timechain/chain.bin").write_bytes(b"BLOCK0" * 16 + b"BLOCK1" * 16)
    b1 = tmp_path / "b1"
    b1.mkdir()
    _write(b1, {**p_files, **t_files})
    r2 = await run_unified_event(
        titan_id="T1", manifest=manifest, personality_specs=p_specs,
        timechain_specs=t_specs, baseline_resolver=lambda c, a: str(b1 / a),
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "s2"))
    assert r2.status == "shipped" and r2.event_type == "incremental"

    # Event 3 — incremental
    (src / "state.json").write_bytes(b'{"epoch": 2, "mood": "joy"}')
    (src / "timechain/chain.bin").write_bytes(
        b"BLOCK0" * 16 + b"BLOCK1" * 16 + b"BLOCK2" * 16)
    b2 = tmp_path / "b2"
    b2.mkdir()
    _write(b2, {"config.txt": b"version=1.0\n", "state.json": b'{"epoch": 1}',
                "timechain/chain.bin": b"BLOCK0" * 16 + b"BLOCK1" * 16})
    r3 = await run_unified_event(
        titan_id="T1", manifest=manifest, personality_specs=p_specs,
        timechain_specs=t_specs, baseline_resolver=lambda c, a: str(b2 / a),
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "s3"))
    assert r3.status == "shipped" and r3.event_type == "incremental"

    # ── RESURRECT from the on-chain chain ALONE (no manifest) ────────────────
    target = tmp_path / "resurrected"

    def arc_to_target(component, arc_name):
        return str(target / component / arc_name)

    async def sig_lister():
        return list(reversed(solana.sigs))  # newest→oldest, like getSignatures

    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey="TestPubkey1111",
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download, arc_to_target=arc_to_target,
        scratch_dir=str(target), verbose=False)

    assert result.status == "resurrected", (result.halt_reason, result.errors)
    assert result.events_total == 3
    assert result.events_applied == 3  # baseline + 2 incrementals
    assert result.genesis_event_id == r1.event_id if hasattr(r1, "event_id") else True

    # Byte-identical reconstruction at the event-3 state, from the wallet alone.
    assert (target / "personality" / "config.txt").read_bytes() == b"version=1.0\n"
    assert (target / "personality" / "state.json").read_bytes() == \
        b'{"epoch": 2, "mood": "joy"}'
    assert (target / "timechain" / "timechain/chain.bin").read_bytes() == \
        b"BLOCK0" * 16 + b"BLOCK1" * 16 + b"BLOCK2" * 16


TEST_PUBKEY = "TestPubkey1111"


class ModeBSolana:
    """Mode-B v=3 committer: plaintext URL + the encrypted-tarball IV on-chain
    (RFP G2). No url_key needed; the restore re-derives the per-backup key."""
    def __init__(self):
        self._memos: dict = {}
        self.sigs: list = []

    async def commit(self, event_id, ts, event_type, event_root, components, prev_sig):
        head = None
        comp_sigs: dict = {}
        for c in components:
            assert c.get("iv"), "Mode-B component must carry an iv (encryptor wired?)"
            sig = _b58_sig()
            memo = build_v3_memo(
                event_id=event_id, ts=ts, event_type=event_type, tier=c["tier"],
                archive_hash=c["arc"], merkle_root=event_root,
                arweave_tx=c["tx_id"], mode="B", prev_sig=prev_sig, iv_b64=c["iv"])
            self._memos[sig] = memo
            self.sigs.append(sig)
            comp_sigs[c["tier"]] = sig
            if head is None:
                head = sig
        return {"head_sig": head, "component_sigs": comp_sigs}

    async def fetch(self, sig: str):
        return self._memos.get(sig)


def _mode_b_encryptor():
    from titan_hcl.logic.backup_crypto import (
        derive_master_key, encrypt_component_tarball,
    )
    master = derive_master_key(SEED, TEST_PUBKEY)

    def encryptor(plaintext: bytes, component: str):
        return encrypt_component_tarball(plaintext, master, component)
    return encryptor


@pytest.mark.asyncio
async def test_sovereign_resurrection_mode_b_encrypted_roundtrip(tmp_path):
    """G2 / INV-MBR-13: an ENCRYPTED (Mode-B) user Titan resurrects from the
    wallet ALONE — the per-backup key is re-derived from the soul keypair + the
    arc[:16] backup_id + the on-chain iv, decrypts each tarball, byte-identical."""
    arweave = ModeAArweave()           # generic content store (now holds ciphertext)
    solana = ModeBSolana()
    encryptor = _mode_b_encryptor()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    src = tmp_path / "src"
    src.mkdir()
    p_files = {"config.txt": b"version=1.0\n", "state.json": b'{"epoch": 0}'}
    t_files = {"timechain/chain.bin": b"BLOCK0" * 16}
    _write(src, {**p_files, **t_files})
    p_specs = _specs(src, p_files)
    t_specs = _specs(src, t_files, fmt="timechain_bin")

    # Event 1 — baseline (encrypted)
    r1 = await run_unified_event(
        titan_id="T1", manifest=manifest, personality_specs=p_specs,
        timechain_specs=t_specs, arweave_uploader=arweave.upload,
        zk_committer=solana.commit, scratch_dir=str(tmp_path / "s1"),
        encryptor=encryptor)
    assert r1.status == "shipped" and r1.event_type == "baseline"

    # Event 2 — incremental (encrypted)
    (src / "state.json").write_bytes(b'{"epoch": 1, "mood": "joy"}')
    (src / "timechain/chain.bin").write_bytes(b"BLOCK0" * 16 + b"BLOCK1" * 16)
    b1 = tmp_path / "b1"
    b1.mkdir()
    _write(b1, {**p_files, **t_files})
    r2 = await run_unified_event(
        titan_id="T1", manifest=manifest, personality_specs=p_specs,
        timechain_specs=t_specs, baseline_resolver=lambda c, a: str(b1 / a),
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "s2"), encryptor=encryptor)
    assert r2.status == "shipped" and r2.event_type == "incremental"

    # The bytes on "Arweave" are genuinely ciphertext — no plaintext leaks.
    for blob in arweave._store.values():
        assert b"version=1.0" not in blob and b"epoch" not in blob

    target = tmp_path / "resurrected"

    async def sig_lister():
        return list(reversed(solana.sigs))

    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey=TEST_PUBKEY,
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download,
        arc_to_target=lambda c, a: str(target / c / a),
        scratch_dir=str(target), verbose=False)

    assert result.status == "resurrected", (result.halt_reason, result.errors)
    assert result.events_applied == 2
    # Byte-identical decrypted reconstruction at event-2 state.
    assert (target / "personality" / "config.txt").read_bytes() == b"version=1.0\n"
    assert (target / "personality" / "state.json").read_bytes() == \
        b'{"epoch": 1, "mood": "joy"}'
    assert (target / "timechain" / "timechain/chain.bin").read_bytes() == \
        b"BLOCK0" * 16 + b"BLOCK1" * 16


@pytest.mark.asyncio
async def test_sovereign_mode_b_wrong_key_halts(tmp_path):
    """Proves the data is REALLY encrypted: a restore with the wrong soul keypair
    fails GCM auth → HALT_MODE_B_DECRYPT_FAILED (not a silent plaintext pass)."""
    arweave = ModeAArweave()
    solana = ModeBSolana()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    src = tmp_path / "src"
    src.mkdir()
    _write(src, {"config.txt": b"secret\n", "timechain/chain.bin": b"B0" * 8})
    await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_specs(src, {"config.txt": b""}),
        timechain_specs=_specs(src, {"timechain/chain.bin": b""}, fmt="timechain_bin"),
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "s1"), encryptor=_mode_b_encryptor())

    async def sig_lister():
        return list(solana.sigs)

    wrong_seed = bytes(range(64, 96))  # a DIFFERENT soul keypair
    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=wrong_seed, titan_pubkey=TEST_PUBKEY,
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download,
        arc_to_target=lambda c, a: str(tmp_path / "t" / c / a),
        scratch_dir=str(tmp_path / "t"), verbose=False)
    assert result.status == "halted"
    assert result.halt_reason == sov.HALT_MODE_B_DECRYPT_FAILED


@pytest.mark.asyncio
async def test_sovereign_halts_on_empty_chain(tmp_path):
    async def empty_lister():
        return []

    async def no_memo(_sig):
        return None

    async def no_fetch(_tx):
        return b""

    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey="X",
        sig_lister=empty_lister, memo_fetcher=no_memo, arweave_fetch=no_fetch,
        arc_to_target=lambda c, a: str(tmp_path / c / a),
        scratch_dir=str(tmp_path / "scratch"), verbose=False)
    assert result.status == "halted"
    assert result.halt_reason == sov.HALT_NO_CHAIN


@pytest.mark.asyncio
async def test_sovereign_halts_on_tampered_tarball(tmp_path):
    """A tampered Arweave payload (sha256 ≠ memo arc) must halt the restore."""
    url_key = derive_backup_url_key(SEED)
    arweave = ModeAArweave()
    solana = ModeASolana(url_key)
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    src = tmp_path / "src"
    src.mkdir()
    _write(src, {"config.txt": b"v1\n", "timechain/chain.bin": b"B0" * 8})
    await run_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_specs(src, {"config.txt": b""}),
        timechain_specs=_specs(src, {"timechain/chain.bin": b""}, fmt="timechain_bin"),
        arweave_uploader=arweave.upload, zk_committer=solana.commit,
        scratch_dir=str(tmp_path / "s1"))

    # Corrupt one stored payload after the fact.
    for tx in list(arweave._store):
        arweave._store[tx] = arweave._store[tx] + b"TAMPER"

    async def sig_lister():
        return list(solana.sigs)

    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey="X",
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download,
        arc_to_target=lambda c, a: str(tmp_path / "t" / c / a),
        scratch_dir=str(tmp_path / "t"), verbose=False)
    assert result.status == "halted"
    assert result.halt_reason == sov.HALT_ARC_MISMATCH


# ── #36 regression suite — best-effort recovery, scratch→data swap, v2/v3 walk ──
#
# These exercise the LIVE resurrection paths that the protocol unit tests don't:
# the damaged-chain recovery shape that actually healed T1 (a divergent increment
# whose xdelta3 baseline was never the one on-chain, overwritten by a later
# full-ship), the scratch-container → install_root/data swap wiring (the cc140f83
# fix), and a mixed v=2/v=3 chain (T1's backfill scenario). The harness reuses the
# ModeA ship→resurrect seams above so the events flow through the PRODUCTION upload
# pipeline, not hand-rolled fixtures.


async def _ship(solana, arweave, manifest, *, src, p_files, t_files,
                event_type, baseline_resolver, scratch):
    """Ship one event through run_unified_event with the ModeA seams."""
    p_specs = _specs(src, p_files)
    t_specs = _specs(src, t_files, fmt="timechain_bin")
    kw = {}
    if baseline_resolver is not None:
        kw["baseline_resolver"] = baseline_resolver
    r = await run_unified_event(
        titan_id="T1", manifest=manifest, personality_specs=p_specs,
        timechain_specs=t_specs, arweave_uploader=arweave.upload,
        zk_committer=solana.commit, scratch_dir=str(scratch), **kw)
    assert r.status == "shipped", (r.status, event_type)
    assert r.event_type == event_type, (r.event_type, event_type)
    return r


@pytest.mark.asyncio
async def test_sovereign_best_effort_skips_divergent_increment_then_recovers(tmp_path):
    """#36 — best-effort e2e (T1's real shape). A divergent increment (its xdelta3
    was diffed against a baseline that was NEVER the one on-chain) fails the STRICT
    baseline-merkle check on apply (xdelta3.apply_diff is strict regardless of the
    arc-verified-advisory mode) → SKIPPED in best_effort. A later FULL-SHIP of the
    same file overwrites it → the net skip count goes to 0 → status 'resurrected'
    and the bytes equal the full-ship. Strict mode (best_effort=False) HALTS on the
    divergent event instead — proving best-effort is what buys the recovery."""
    url_key = derive_backup_url_key(SEED)
    arweave = ModeAArweave()
    solana = ModeASolana(url_key)
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    # state.json must be ≥10 KB so select_encoder picks xdelta3 (sub-10 KB JSON is
    # always full-shipped, which would never exercise the baseline-merkle check).
    _PAD = b"A" * 12000
    src = tmp_path / "src"
    src.mkdir()
    state_A = b'{"epoch":0,"pad":"' + _PAD + b'","tag":"A"}'
    p_files = {"config.txt": b"version=1.0\n", "state.json": state_A}
    t_files = {"timechain/chain.bin": b"BLOCK0" * 16}
    _write(src, {**p_files, **t_files})

    # Event 1 — baseline (anchor; ships state.json == state_A on-chain)
    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="baseline",
                baseline_resolver=None, scratch=tmp_path / "s1")

    # Event 2 — incremental whose state.json baseline DIVERGES from what e1 shipped.
    # The xdelta3 diff records baseline_merkle_root = sha256(divergent), so on
    # restore (where the real baseline is state_A) the apply baseline-merkle check
    # mismatches and the file is unreplayable.
    state_B = b'{"epoch":1,"pad":"' + _PAD + b'","tag":"B"}'
    (src / "state.json").write_bytes(state_B)
    bdiv = tmp_path / "bdiv"
    bdiv.mkdir()
    _write(bdiv, {"config.txt": b"version=1.0\n",
                  "state.json": b'{"epoch":0,"pad":"' + _PAD + b'","tag":"DIV"}',
                  "timechain/chain.bin": b"BLOCK0" * 16})
    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="incremental",
                baseline_resolver=lambda c, a: str(bdiv / a),
                scratch=tmp_path / "s2")

    # Event 3 — incremental that FULL-SHIPS state.json (resolver returns None for it
    # → no baseline → diff_mode='full'); config/chain keep their real baseline → skip.
    state_C = b'{"epoch": 2, "mood": "joy"}'
    (src / "state.json").write_bytes(state_C)
    bdir3 = tmp_path / "bdir3"
    bdir3.mkdir()
    _write(bdir3, {"config.txt": b"version=1.0\n",
                   "timechain/chain.bin": b"BLOCK0" * 16})
    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="incremental",
                baseline_resolver=lambda c, a: None if a == "state.json"
                else str(bdir3 / a),
                scratch=tmp_path / "s3")

    async def sig_lister():
        return list(reversed(solana.sigs))

    common = dict(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey="TestPubkey1111",
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download, verbose=False)

    # strict → halts on the divergent increment (no silent garbage)
    strict = await sov.resurrect_from_chain(
        **common, arc_to_target=lambda c, a: str(tmp_path / "strict" / c / a),
        scratch_dir=str(tmp_path / "strict"))
    assert strict.status == "halted", (strict.halt_reason, strict.errors)
    assert strict.halt_reason == sov.HALT_APPLY_FAILED

    # best-effort → skips e2's state.json, e3 full-ship recovers it → net 0
    target = tmp_path / "resurrected"
    result = await sov.resurrect_from_chain(
        **common, arc_to_target=lambda c, a: str(target / c / a),
        scratch_dir=str(target), best_effort=True)
    assert result.status == "resurrected", (result.skipped_files, result.errors)
    assert result.skipped_files == []
    assert result.events_applied == 3
    assert (target / "personality" / "state.json").read_bytes() == state_C
    assert (target / "personality" / "config.txt").read_bytes() == b"version=1.0\n"


@pytest.mark.asyncio
async def test_sovereign_best_effort_partial_when_no_recovery(tmp_path):
    """#36 — best-effort PARTIAL. Same divergent increment, but NO later full-ship
    heals it → status 'resurrected_partial', skipped_files names the file, and that
    file keeps its last-good (baseline) bytes while every other file is recovered.
    This is exactly T1's floor when no recovering full-ship exists downstream."""
    url_key = derive_backup_url_key(SEED)
    arweave = ModeAArweave()
    solana = ModeASolana(url_key)
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    _PAD = b"A" * 12000      # ≥10 KB → xdelta3 (see recover test for the why)
    src = tmp_path / "src"
    src.mkdir()
    state_A = b'{"epoch":0,"pad":"' + _PAD + b'","tag":"A"}'
    p_files = {"config.txt": b"version=1.0\n", "state.json": state_A}
    t_files = {"timechain/chain.bin": b"BLOCK0" * 16}
    _write(src, {**p_files, **t_files})

    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="baseline",
                baseline_resolver=None, scratch=tmp_path / "s1")

    state_B = b'{"epoch":1,"pad":"' + _PAD + b'","tag":"B"}'
    (src / "state.json").write_bytes(state_B)
    bdiv = tmp_path / "bdiv"
    bdiv.mkdir()
    _write(bdiv, {"config.txt": b"version=1.0\n",
                  "state.json": b'{"epoch":0,"pad":"' + _PAD + b'","tag":"DIV"}',
                  "timechain/chain.bin": b"BLOCK0" * 16})
    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="incremental",
                baseline_resolver=lambda c, a: str(bdiv / a),
                scratch=tmp_path / "s2")

    async def sig_lister():
        return list(reversed(solana.sigs))

    target = tmp_path / "resurrected"
    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey="TestPubkey1111",
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download,
        arc_to_target=lambda c, a: str(target / c / a),
        scratch_dir=str(target), best_effort=True, verbose=False)

    assert result.status == "resurrected_partial", (result.status, result.errors)
    assert any("personality/state.json" in s for s in result.skipped_files), \
        result.skipped_files
    # the unreplayable file kept its last-good (baseline) bytes; others recovered
    assert (target / "personality" / "state.json").read_bytes() == state_A
    assert (target / "personality" / "config.txt").read_bytes() == b"version=1.0\n"
    assert (target / "timechain" / "timechain/chain.bin").read_bytes() == \
        b"BLOCK0" * 16


@pytest.mark.asyncio
async def test_sovereign_scratch_container_swaps_body_into_install_root_data(tmp_path):
    """#36 — scratch→data swap (the cc140f83 fix). Mirror restore_body_from_chain's
    wiring with REAL arc names: build_arc_to_target(container) lands the body at
    <container>/data/<real-path>, then _atomic_swap_into_data swaps it into
    install_root/data — and any pre-existing data/ is moved aside to
    data.pre_resurrect.* (NOT the reverse, which was the bug that left data/ empty)."""
    url_key = derive_backup_url_key(SEED)
    arweave = ModeAArweave()
    solana = ModeASolana(url_key)
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    src = tmp_path / "src"
    src.mkdir()
    body_personality = b"PERSONALITY-INNER-MEMORY-BYTES"
    body_timechain = b"TIMECHAIN-MAIN-FORK-BYTES"
    # Real arc names → build_arc_to_target maps them to data/<real-path>.
    p_files = {"inner_memory.db": body_personality}
    t_files = {"timechain/chain_main.bin": body_timechain}
    _write(src, {**p_files, **t_files})

    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="baseline",
                baseline_resolver=None, scratch=tmp_path / "s1")

    async def sig_lister():
        return list(reversed(solana.sigs))

    # Resurrect into a scratch CONTAINER exactly as restore_body_from_chain does.
    container = tmp_path / "data.resurrect"
    scratch_data = container / "data"
    scratch_data.mkdir(parents=True)
    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey="TestPubkey1111",
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download,
        arc_to_target=build_arc_to_target(str(container)),
        scratch_dir=str(scratch_data), verbose=False)
    assert result.status == "resurrected", (result.halt_reason, result.errors)
    # Body landed under the CONTAINER's data/, not anywhere else.
    assert (container / "data" / "inner_memory.db").read_bytes() == body_personality
    assert (container / "data" / "timechain" / "chain_main.bin").read_bytes() == \
        body_timechain

    # install_root has a pre-existing data/ that must be preserved, not clobbered.
    install_root = tmp_path / "install"
    (install_root / "data").mkdir(parents=True)
    (install_root / "data" / "old_marker.txt").write_text("prior live state")

    sov._atomic_swap_into_data(str(scratch_data), str(install_root))

    # The resurrected body is now the LIVE data/ (the cc140f83 regression: it used
    # to land in data.pre_resurrect while data/ ended up empty).
    assert (install_root / "data" / "inner_memory.db").read_bytes() == body_personality
    assert (install_root / "data" / "timechain" / "chain_main.bin").read_bytes() == \
        body_timechain
    assert not (install_root / "data" / "old_marker.txt").exists()
    # The prior live tree was preserved aside (memory-preservation axiom).
    preserved = list(install_root.glob("data.pre_resurrect.*"))
    assert len(preserved) == 1, preserved
    assert (preserved[0] / "old_marker.txt").read_text() == "prior live state"


@pytest.mark.asyncio
async def test_sovereign_resurrects_mixed_v2_v3_chain(tmp_path):
    """#36 — mixed v=2/v=3 walk (T1's backfill scenario). The wallet history carries
    legacy v=2 merkle-anchor memos interleaved with the v=3 per-component chain. The
    resurrect walk's parse_v3_memo returns None for v=2 memos → they are silently
    skipped, and the Titan resurrects from the v=3 events alone, byte-identical — no
    halt on the unparseable legacy memos."""
    url_key = derive_backup_url_key(SEED)
    arweave = ModeAArweave()
    solana = ModeASolana(url_key)
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))

    src = tmp_path / "src"
    src.mkdir()
    p_files = {"config.txt": b"version=1.0\n", "state.json": b'{"epoch": 0}'}
    t_files = {"timechain/chain.bin": b"BLOCK0" * 16}
    _write(src, {**p_files, **t_files})

    # v=3 baseline
    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="baseline",
                baseline_resolver=None, scratch=tmp_path / "s1")
    # v=3 incremental
    (src / "state.json").write_bytes(b'{"epoch": 1, "mood": "joy"}')
    (src / "timechain/chain.bin").write_bytes(b"BLOCK0" * 16 + b"BLOCK1" * 16)
    b1 = tmp_path / "b1"
    b1.mkdir()
    _write(b1, {**p_files, **t_files})
    await _ship(solana, arweave, manifest, src=src, p_files=p_files,
                t_files=t_files, event_type="incremental",
                baseline_resolver=lambda c, a: str(b1 / a), scratch=tmp_path / "s2")

    # Inject legacy v=2 merkle-anchor memos under their own sigs (as the backfill
    # left them on-chain alongside the v=3 chain). parse_v3_memo must skip these.
    for i in range(2):
        v2_sig = _b58_sig()
        solana._memos[v2_sig] = build_zk_memo(
            event_id=f"legacy_v2_{i}", event_merkle_root=("%064x" % i),
            prev_event_merkle_root=None)
        solana.sigs.append(v2_sig)

    async def sig_lister():
        return list(reversed(solana.sigs))

    target = tmp_path / "resurrected"
    result = await sov.resurrect_from_chain(
        titan_id="T1", keypair_bytes=SEED, titan_pubkey="TestPubkey1111",
        sig_lister=sig_lister, memo_fetcher=solana.fetch,
        arweave_fetch=arweave.download,
        arc_to_target=lambda c, a: str(target / c / a),
        scratch_dir=str(target), verbose=False)

    assert result.status == "resurrected", (result.halt_reason, result.errors)
    assert result.events_total == 2      # only the two v=3 events counted
    assert result.events_applied == 2
    assert (target / "personality" / "state.json").read_bytes() == \
        b'{"epoch": 1, "mood": "joy"}'
    assert (target / "timechain" / "timechain/chain.bin").read_bytes() == \
        b"BLOCK0" * 16 + b"BLOCK1" * 16

