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
