"""BackupWorker — the carved build→ship→anchor pipeline.

RFP_backup_redesign_spine Phase B (§7.B). Consolidates the THREE overlapping
god-class entrypoints (`backup.py:_run_unified_event_v2` / `_build_staged_event_v2`
/ `_ship_staged_event_v2`) into ONE gate-free, resumable pipeline of three
methods, built on the Phase-A snapshot (via `_build_*_payload`) + ChainProvider:

    plan_build()    → decide event_type/id/prev/baseline; seed the pending specs
    build_slice()   → encode the next ≤byte_budget batch → patch artifacts (RESUMABLE)
    finalize_pack() → ONE streamed .tar.zst per component over the artifacts
    ship_event()    → STREAMED upload (Mode-A straight from disk — no f.read, B-2)
                      + the proven §24.7 merkle / v=3 ZK-commit / manifest-append

This worker is **gate-free** — the BackupOrchestrator (Phase D) owns the one
cadence/single-flight gate + drives the drip across idle ticks, persisting the
partial `StagedBuild` (INV-BRS-7). The crypto + manifest are KEPT verbatim
(INV-BRS-8): `ship_event` reuses `ship_staged_event`'s finalize unchanged, so the
on-chain event is byte-identical (INV-BR-6).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from titan_hcl.logic.backup_unified_manifest import UnifiedManifest, new_event_id
from titan_hcl.logic.backup_event_tarball import FileDiffSpec, pack_event_tarball
from titan_hcl.logic.backup_upload_pipeline import (
    StagedEvent,
    TierFileSpec,
    TierShipResult,
    _build_full_payload,
    _build_incremental_payload,
    _extract_block_ranges,
    ship_staged_event,
)

logger = logging.getLogger("titan.backup.worker")

# Default drip slice budget (Q-BRS-D / D-BRS-D — per-file-batch, bounded bytes).
_DEFAULT_BYTE_BUDGET = 64 * 1024 * 1024  # 64 MB / build_slice tick
_TIER_ORDER = ("personality", "timechain", "soul")


@dataclass
class StagedBuild:
    """A resumable, gate-free unified-event build (RFP §7.B). `build_slice`
    encodes `pending` specs into `artifacts` per tick (the Orchestrator, Phase D,
    persists this + drives the drip — INV-BRS-7); `finalize_pack` packs them into
    `tier_results`; `ship_event` ships. `baseline_event_id` pins which baseline
    the incrementals diffed against (the ship staleness guard)."""
    event_id: str
    event_type: str
    baseline_trigger: Optional[str]
    baseline_event_id: Optional[str]
    prev_event_id: Optional[str]
    soul_present: bool
    scratch_dir: str
    titan_id: str
    pending: dict                  # component -> list[TierFileSpec] not yet encoded
    artifacts: dict                # component -> list[(arc_name, diff_dict)]
    tier_results: dict = field(default_factory=dict)  # component -> TierShipResult (post-pack)
    built_at: float = 0.0

    @property
    def fully_encoded(self) -> bool:
        return all(not specs for specs in self.pending.values())


class BackupWorker:
    """Stateless build/ship pipeline (no cadence gate — Phase-D owns it)."""

    def __init__(self, *, titan_id: str, chain_provider,
                 byte_budget: int = _DEFAULT_BYTE_BUDGET) -> None:
        self.titan_id = titan_id
        self.chain = chain_provider
        self.byte_budget = byte_budget

    def plan_build(self, *, manifest: UnifiedManifest,
                   personality_specs: Iterable[TierFileSpec],
                   timechain_specs: Iterable[TierFileSpec],
                   soul_specs: Optional[Iterable[TierFileSpec]] = None,
                   scratch_dir: str,
                   force_event_type: Optional[str] = None,
                   force_trigger: Optional[str] = None) -> StagedBuild:
        """Decide event_type (baseline vs incremental) + identity from the
        manifest, and seed the per-component pending spec lists. Mirrors
        `build_unified_event`'s header — no encode/pack yet (that is build_slice)."""
        if force_event_type in ("baseline", "incremental"):
            event_type = force_event_type
            trigger = (force_trigger or "self_heal") if event_type == "baseline" else None
        else:
            should_rebase, trigger = manifest.should_rebase()
            event_type = "baseline" if should_rebase else "incremental"
        prev_event = manifest.get_latest_event()
        pending = {
            "personality": list(personality_specs),
            "timechain": list(timechain_specs),
        }
        if soul_specs is not None:
            pending["soul"] = list(soul_specs)
        os.makedirs(scratch_dir, exist_ok=True)
        return StagedBuild(
            event_id=new_event_id(), event_type=event_type,
            baseline_trigger=(trigger if event_type == "baseline" else None),
            baseline_event_id=manifest.current_baseline_event_id,
            prev_event_id=(prev_event["event_id"] if prev_event else None),
            soul_present=(soul_specs is not None), scratch_dir=scratch_dir,
            titan_id=self.titan_id, pending=pending,
            artifacts={k: [] for k in pending}, built_at=time.time(),
        )

    def build_slice(self, staged: StagedBuild,
                    baseline_resolver: Optional[Callable[[str, str], Optional[str]]] = None,
                    *, byte_budget: Optional[int] = None) -> bool:
        """Encode the next ≤`byte_budget` batch of pending specs into immutable
        patch artifacts (the §24.5 diff over the Phase-A consistent snapshot, via
        `_build_*_payload`). RESUMABLE — advances `staged.pending` → `artifacts`.
        Returns True if specs remain (Orchestrator calls again next idle tick).

        `baseline_resolver(component, arc_name) → prior-baseline file path` (the
        2-arg god-class resolver); None / baseline event → full-ship. Heavy work,
        bounded (INV-BRS-3)."""
        budget = byte_budget if byte_budget is not None else self.byte_budget
        spent = 0
        for component in _TIER_ORDER:
            specs = staged.pending.get(component)
            if not specs:
                continue
            while specs and spent < budget:
                spec = specs.pop(0)
                if staged.event_type == "baseline" or baseline_resolver is None:
                    dd = _build_full_payload(spec)
                else:
                    base_path = baseline_resolver(component, spec.arc_name)
                    dd = _build_incremental_payload(spec, base_path)
                if dd is None:
                    continue   # source absent on disk — skip (matches build_tier)
                staged.artifacts[component].append((spec.arc_name, dd))
                spent += int(dd.get("patch_size_bytes", 0) or 0)
            if spent >= budget:
                break
        return not staged.fully_encoded

    def finalize_pack(self, staged: StagedBuild) -> None:
        """Once every batch is encoded: pack ONE streamed `.tar.zst` per component
        over the staged artifacts → `staged.tier_results`. A finalized tar can't
        be appended-to, so pack is a single final pass (cheap, bounded). The
        §24.5 encode / §24.7 Merkle / manifest schema are unchanged."""
        os.makedirs(staged.scratch_dir, exist_ok=True)
        for component in _TIER_ORDER:
            if component not in staged.artifacts:
                continue
            artifacts = staged.artifacts[component]
            result = TierShipResult(tier=component)
            if not artifacts:
                result.error = f"{component}: no in-scope files found on disk"
                staged.tier_results[component] = result
                continue
            out_path = os.path.join(
                staged.scratch_dir, f"event_{staged.event_id}_{component}.tar.zst")
            pack_specs = [FileDiffSpec(arc, dd) for arc, dd in artifacts]
            pack_info = pack_event_tarball(
                event_id=staged.event_id, event_type=staged.event_type,
                component=component, file_specs=pack_specs, output_path=out_path)
            result.tarball_path = out_path
            result.tarball_size_bytes = pack_info["size_bytes"]
            result.tarball_sha256 = pack_info["tarball_sha256"]
            result.files_packed = len(artifacts)
            result.files_skipped = sum(
                1 for _, dd in artifacts if dd.get("diff_mode") == "skipped")
            if component == "timechain":
                result.block_ranges = _extract_block_ranges(artifacts)
            staged.tier_results[component] = result

    async def ship_event(self, staged: StagedBuild, *, manifest: UnifiedManifest,
                         zk_committer, bus_emit=None, encryptor=None):
        """Ship the packed event — STREAMED (Mode-A uploads each tarball straight
        from disk via `ChainProvider.put(path)`, no `f.read` whole-tarball RAM
        load — B-2). Reuses `ship_staged_event`'s finalize VERBATIM (staleness
        guard → §24.7 merkle → v=3 ZK-commit → manifest append) so the on-chain
        event is byte-identical (INV-BR-6/8). Gate-free."""
        staged_event = StagedEvent(
            event_id=staged.event_id, event_type=staged.event_type,
            baseline_trigger=staged.baseline_trigger,
            baseline_event_id=staged.baseline_event_id,
            prev_event_id=staged.prev_event_id, soul_present=staged.soul_present,
            tier_results=staged.tier_results, scratch_dir=staged.scratch_dir,
            built_at=staged.built_at, titan_id=staged.titan_id,
        )

        async def _bytes_uploader(data: bytes, tags: dict) -> str:
            return await self.chain.put(data, tags=tags)

        async def _path_uploader(path: str, tags: dict) -> str:
            return await self.chain.put(path, tags=tags)

        return await ship_staged_event(
            staged_event, manifest=manifest, arweave_uploader=_bytes_uploader,
            zk_committer=zk_committer, bus_emit=bus_emit, cleanup_scratch=False,
            encryptor=encryptor, path_uploader=_path_uploader)
