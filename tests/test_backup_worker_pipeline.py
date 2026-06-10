"""BackupWorker — RFP_backup_redesign_spine Phase B (§7.B, gate GB1).

Proves build→ship→anchor→manifest is ONE gate-free path through the carved
worker (plan_build → build_slice [resumable] → finalize_pack → ship_event), on
the Phase-A snapshot + a streamed ChainProvider, with the on-chain finalize
reused verbatim from ship_staged_event. B-2: Mode-A uploads stream straight from
disk (no f.read whole-tarball RAM load).
"""
import asyncio
import os
import sqlite3
import uuid
from typing import Optional

from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
from titan_hcl.logic.backup_upload_pipeline import TierFileSpec
from titan_hcl.logic.backup_worker_pipeline import BackupWorker


class StubChainProvider:
    """Records whether put() received a PATH (Mode-A streamed) or BYTES (an
    f.read whole-tarball load) — the B-2 proof."""
    def __init__(self):
        self.put_calls: list[tuple[str, object]] = []

    async def put(self, src, *, content_type="application/octet-stream", tags=None):
        if isinstance(src, (bytes, bytearray)):
            self.put_calls.append(("bytes", len(src)))
        else:
            self.put_calls.append(("path", src))
        return "ar_" + uuid.uuid4().hex[:16]


class StubZk:
    """v=3 chain committer (5J-2 contract) — one sig per component, head=first."""
    def __init__(self):
        self.commits: list[str] = []

    async def commit(self, event_id, ts, event_type, event_root, components,
                     prev_sig: Optional[str]):
        self.commits.append(event_id)
        sigs = ["zksig_" + uuid.uuid4().hex[:16] for _ in components]
        return {"head_sig": sigs[0],
                "component_sigs": {c["tier"]: s for c, s in zip(components, sigs)}}


def _make_sqlite(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
    conn.executemany("INSERT INTO t(v) VALUES(?)", [(f"v{i}",) for i in range(rows)])
    conn.commit()
    conn.close()


def test_backup_worker_one_gate_free_path(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    db = src / "mem.db"
    _make_sqlite(db, 50)                       # SQLite → Phase-A consistent snapshot
    js = src / "state.json"
    js.write_text('{"k": 1}')
    tc = src / "chain_0.bin"
    tc.write_bytes(os.urandom(2048))
    p_specs = [TierFileSpec(source_path=str(db), arc_name="mem.db"),
               TierFileSpec(source_path=str(js), arc_name="state.json")]
    t_specs = [TierFileSpec(source_path=str(tc), arc_name="timechain/chain_0.bin",
                            format_hint="timechain_bin")]
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    chain = StubChainProvider()
    zk = StubZk()
    worker = BackupWorker(titan_id="T1", chain_provider=chain, byte_budget=8 * 1024 * 1024)

    staged = worker.plan_build(
        manifest=manifest, personality_specs=p_specs, timechain_specs=t_specs,
        soul_specs=None, scratch_dir=str(tmp_path / "scratch"),
        force_event_type="baseline", force_trigger="first_event")

    # drip: resumable build_slice until fully encoded
    guard = 0
    while worker.build_slice(staged, None) and guard < 100:
        guard += 1
    assert staged.fully_encoded
    assert len(staged.artifacts["personality"]) == 2
    assert len(staged.artifacts["timechain"]) == 1

    worker.finalize_pack(staged)
    pr = staged.tier_results["personality"]
    assert pr.tarball_path and os.path.exists(pr.tarball_path)
    assert staged.tier_results["timechain"].tarball_sha256

    res = asyncio.run(worker.ship_event(staged, manifest=manifest, zk_committer=zk.commit))
    assert res.status == "shipped", res.errors
    assert res.zk_commit_tx

    # the anchor landed in the manifest — gate-free, ONE path
    ev = manifest.get_latest_event()
    assert ev["event_id"] == staged.event_id
    assert ev["type"] == "baseline"
    assert ev["zk_commit_tx"] == res.zk_commit_tx
    assert zk.commits == [staged.event_id]

    # B-2: Mode-A streamed the tarball PATHS (no f.read whole-tarball RAM load)
    assert chain.put_calls, "no uploads happened"
    assert all(kind == "path" for kind, _ in chain.put_calls), chain.put_calls
    assert len(chain.put_calls) == 2   # personality + timechain


def test_build_slice_is_resumable_drip(tmp_path):
    """A tiny byte_budget forces multiple drip slices — proving build_slice is
    resumable (the primitive Phase-D's Orchestrator drives across idle ticks)."""
    src = tmp_path / "src"
    src.mkdir()
    specs = []
    for i in range(6):
        p = src / f"f{i}.json"
        p.write_text(('{"n":%d}' % i) + (" " * 5000))   # ~5 KB each
        specs.append(TierFileSpec(source_path=str(p), arc_name=f"f{i}.json"))
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    worker = BackupWorker(titan_id="T1", chain_provider=None, byte_budget=8000)
    staged = worker.plan_build(
        manifest=manifest, personality_specs=specs, timechain_specs=[],
        soul_specs=None, scratch_dir=str(tmp_path / "s"),
        force_event_type="baseline", force_trigger="first_event")

    slices = 0
    while True:
        more = worker.build_slice(staged, None)
        slices += 1
        if not more:
            break
        assert slices < 50
    assert slices >= 2, "a tiny budget should require multiple drip slices"
    assert len(staged.artifacts["personality"]) == 6   # all encoded across slices
    assert staged.fully_encoded
