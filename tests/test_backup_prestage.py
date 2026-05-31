"""Phase 2 (2026-05-31) — pre-stage path: build_unified_event + ship_staged_event.

build_unified_event does the heavy diff/pack with NO upload + NO manifest mutation
(run by the stager off the recv loop, ahead of meditation). ship_staged_event then
uploads the pre-built tarballs + runs the identical merkle→ZK→manifest finalize as
run_unified_event, with a stale-baseline guard.
"""
from __future__ import annotations

import pytest

from titan_hcl.logic.backup_unified_manifest import UnifiedManifest
from titan_hcl.logic.backup_upload_pipeline import (
    EVENT_BACKUP_EVENT_COMPLETE,
    StagedEvent,
    TierFileSpec,
    build_unified_event,
    ship_staged_event,
)


class _FakeArweave:
    def __init__(self):
        self.uploads = []

    async def upload(self, data: bytes, tags: dict) -> str:
        self.uploads.append((tags.get("Tier"), len(data)))
        return f"ar_tx_{len(self.uploads)}_{tags.get('Tier')}"


class _FakeZk:
    def __init__(self):
        self.commits = []

    async def commit(self, event_id, ts, event_type, event_root, components,
                     prev_sig):
        self.commits.append(event_id)
        return {"head_sig": f"sig_{event_id[:10]}", "component_sigs": {}}


def _w(p, b: bytes) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    return str(p)


def _pspecs(paths):
    return [TierFileSpec(source_path=p, arc_name=a) for a, p in paths.items()]


def _tspecs(paths):
    return [TierFileSpec(source_path=p, arc_name=a, format_hint="timechain_bin")
            for a, p in paths.items()]


def _srcs(tmp_path):
    p = {"config.txt": _w(tmp_path / "src" / "config.txt", b"v=1\n")}
    t = {"timechain/chain_main.bin":
         _w(tmp_path / "src_t" / "chain_main.bin", b"BLOCK0" * 10)}
    return p, t


@pytest.mark.asyncio
async def test_build_is_pure_no_upload_no_manifest_mutation(tmp_path):
    """build_unified_event must NOT upload or touch the manifest — pure build."""
    ar, zk = _FakeArweave(), _FakeZk()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    p, t = _srcs(tmp_path)

    staged = build_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_pspecs(p), timechain_specs=_tspecs(t),
        scratch_dir=str(tmp_path / "stage"))

    assert isinstance(staged, StagedEvent)
    assert staged.event_type == "baseline"          # first-ever event
    assert staged.baseline_trigger == "first_event"
    assert ar.uploads == []                          # NO uploads during build
    assert len(manifest.events) == 0                 # manifest untouched
    # Tarballs were actually built on disk, tx_id still None.
    for tier in ("personality", "timechain"):
        r = staged.tier_results[tier]
        assert r.tarball_path is not None
        assert r.tx_id is None
        assert r.tarball_sha256 is not None


@pytest.mark.asyncio
async def test_build_then_ship_completes_and_anchors(tmp_path):
    ar, zk = _FakeArweave(), _FakeZk()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    p, t = _srcs(tmp_path)
    bus = []

    staged = build_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_pspecs(p), timechain_specs=_tspecs(t),
        scratch_dir=str(tmp_path / "stage"))
    out = await ship_staged_event(
        staged, manifest=manifest, arweave_uploader=ar.upload,
        zk_committer=zk.commit,
        bus_emit=lambda n, pl: bus.append((n, pl)), cleanup_scratch=False)

    assert out.status == "shipped", out.errors
    assert out.event_type == "baseline"
    assert out.event_id == staged.event_id
    assert out.zk_commit_tx is not None
    assert out.event_merkle_root is not None
    # Uploads happened at SHIP time (not build).
    assert {tier for tier, _ in ar.uploads} == {"personality", "timechain"}
    # Manifest now carries exactly this event.
    assert len(manifest.events) == 1
    assert manifest.events[0]["event_id"] == staged.event_id
    assert manifest.events[0]["type"] == "baseline"
    assert manifest.current_baseline_event_id == staged.event_id
    # Bus completion fired.
    assert any(n == EVENT_BACKUP_EVENT_COMPLETE for n, _ in bus)


@pytest.mark.asyncio
async def test_ship_rejects_stale_baseline(tmp_path):
    """If the manifest baseline changed since build, ship must refuse (forces
    a rebuild) rather than commit a diff against the wrong baseline."""
    ar, zk = _FakeArweave(), _FakeZk()
    manifest = UnifiedManifest("T1", base_dir=str(tmp_path))
    p, t = _srcs(tmp_path)

    # Two stages built from the empty manifest — both pin baseline_event_id=None.
    staged1 = build_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_pspecs(p), timechain_specs=_tspecs(t),
        scratch_dir=str(tmp_path / "stage1"))
    staged2 = build_unified_event(
        titan_id="T1", manifest=manifest,
        personality_specs=_pspecs(p), timechain_specs=_tspecs(t),
        scratch_dir=str(tmp_path / "stage2"))
    assert staged1.baseline_event_id is None
    assert staged2.baseline_event_id is None

    # Ship the first → becomes the baseline; current_baseline_event_id advances.
    out1 = await ship_staged_event(
        staged1, manifest=manifest, arweave_uploader=ar.upload,
        zk_committer=zk.commit, cleanup_scratch=False)
    assert out1.status == "shipped"
    assert manifest.current_baseline_event_id == staged1.event_id

    # staged2's pinned baseline (None) != current (staged1) → STALE, refused.
    out2 = await ship_staged_event(
        staged2, manifest=manifest, arweave_uploader=ar.upload,
        zk_committer=zk.commit, cleanup_scratch=False)
    assert out2.status == "stale_baseline", out2.status
    assert len(manifest.events) == 1          # no second event committed
