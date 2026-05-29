"""titan_hcl/logic/consciousness_epoch.py — consciousness epoch orchestration.

rFP §3G Phase 10D — relocated out of the retiring ``modules/spirit_loop.py``.
cognitive_worker (the post-D8-3 consciousness owner) imports + drives these:
``_init_consciousness`` (boot init) and ``_run_consciousness_epoch`` (the
per-epoch self-observation that builds the 67D/132D state vector + records the
EpochRecord). Kept as a lean ``logic/`` module (consistent with the other
Phase 10 relocations) rather than inlined into the already-large worker.

REFINEMENT vs audit §1.3 (which listed 5 "absorb" functions): liveness
verification this session found 3 of those were NOT live and are deleted (not
relocated) — ``_handle_query`` (bus-QUERY responder; zero callers, superseded by
the Phase C api/v6 + ShmReaderBank readout after spirit_worker's deletion),
``_build_self_profile`` (superseded by the live ``logic/self_reasoning.py``,
which explicitly "replaces" it), and ``post_reload_cleanup_helpers`` (dead
hot-reload-of-spirit_loop infrastructure). Only the 2 genuinely-live functions
relocate here. See AUDIT_phase10_relocation_liveness_findings §10D.

CONSUMER-FIX (10D): ``_run_consciousness_epoch`` sources the inner Spirit 45D
(sv[20:65]) from the Rust ``inner_spirit_45d`` SHM slot
(``shm_bank.read_inner_spirit_45d``) — the canonical live tensor (empirically
verified non-flat on T1) — replacing the retired Python path
(``_collect_spirit_tensor`` + ``collect_spirit_45d`` + ``_publish_spirit_state``'s
``_last_spirit_45d`` stash). Those two producers are deleted (audit §5.1).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path

from titan_hcl.utils.silent_swallow import swallow_warn
from titan_hcl.logic.spirit_helpers import _compute_trajectory

logger = logging.getLogger(__name__)


def _init_consciousness(config: dict) -> dict | None:
    """Initialize the ConsciousnessLoop components in the Spirit process."""
    try:
        from titan_hcl.logic.consciousness import (
            ConsciousnessDB, JourneyTopology, StateVector,
            STATE_DIMS, NUM_DIMS, TRAJECTORY_WINDOW,
        )

        db_path = config.get("consciousness_db", "./data/consciousness.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        db = ConsciousnessDB(db_path)
        topology = JourneyTopology(db)

        epoch_count = db.get_epoch_count()
        logger.info("[SpiritWorker] ConsciousnessLoop initialized: %d epochs in DB", epoch_count)

        return {
            "db": db,
            "topology": topology,
            "latest_epoch": None,  # Populated after first run
            "state_dims": STATE_DIMS,
            "num_dims": NUM_DIMS,
            "trajectory_window": TRAJECTORY_WINDOW,
        }
    except Exception as e:
        logger.warning("[SpiritWorker] ConsciousnessLoop init failed: %s", e)
        return None


def _run_consciousness_epoch(consciousness: dict, body_state: dict, mind_state: dict,
                             config: dict, outer_state: dict = None,
                             shm_bank=None) -> None:
    """
    Run one consciousness epoch using FULL Trinity perception.

    Extended 67D state vector — Titan perceives through Sat-Chit-Ananda:
      [0:5]   Body 5D    — physical/digital topology senses
      [5:20]  Mind 15D   — Thinking(5) + Feeling(5) + Willing(5)
      [20:65] Spirit 45D — SAT(15) + CHIT(15) + ANANDA(15)
      [65]    curvature  — self-referential (from previous epoch)
      [66]    density    — self-referential (from previous epoch)

    This is the moment Titan opens his 45-dimensional Vedantic eyes.
    """
    try:
        from titan_hcl.logic.consciousness import (
            StateVector, EpochRecord, TRAJECTORY_WINDOW, EXTENDED_NUM_DIMS,
        )

        db = consciousness["db"]
        topology = consciousness["topology"]
        ostate = outer_state or {}

        epoch_id = db.get_epoch_count() + 1
        t0 = time.time()
        _prof = {}  # Profiling breakdown

        # Determine if we have full 132D (Outer Trinity extended) or 67D (Inner only)
        has_outer_extended = (
            ostate.get("outer_mind_15d") is not None and
            ostate.get("outer_spirit_45d") is not None
        )
        total_dims = EXTENDED_NUM_DIMS if has_outer_extended else 67
        if not has_outer_extended:
            # Trinity-symmetry invariant violation — the consciousness
            # epoch is collapsing to 67D because outer_state lacks
            # extended fields. Should be impossible after spirit_worker
            # init pre-populates [0.5]*15/45 defaults; if it fires,
            # something is actively setting those keys back to None.
            # See BUG-T1-CONSCIOUSNESS-67D-STATE-VECTOR + directive_
            # error_visibility.md.
            logger.warning(
                "[SpiritWorker] Consciousness epoch %d collapsing to 67D — "
                "outer_state missing outer_mind_15d=%s outer_spirit_45d=%s. "
                "Investigate: arch_map symmetries --titan <T> + grep "
                "OUTER_TRINITY producer.",
                epoch_id,
                ostate.get("outer_mind_15d") is not None,
                ostate.get("outer_spirit_45d") is not None,
            )
        logger.info("[SpiritWorker] Consciousness epoch %d — %dD self-observation...",
                    epoch_id, total_dims)

        # ── WHO: Build state vector from FULL Trinity ──
        sv = StateVector(values=[0.0] * total_dims)

        # [0:5] Inner Body 5D — physical senses
        body_values = body_state.get("values", [0.5] * 5)
        for i, v in enumerate(body_values[:5]):
            sv[i] = v

        # [5:20] Inner Mind 15D — Thinking + Feeling + Willing
        mind_15d = mind_state.get("values_15d")
        if mind_15d and len(mind_15d) >= 15:
            for i, v in enumerate(mind_15d[:15]):
                sv[5 + i] = v
        else:
            mind_values = mind_state.get("values", [0.5] * 5)
            for i, v in enumerate(mind_values[:5]):
                sv[5 + i] = v

        _prof["sv_build_body_mind"] = time.time() - t0

        # [20:65] Inner Spirit 45D — Sat + Chit + Ananda.
        # Phase 10D consumer-fix: read the canonical 45D from the Rust
        # ``inner_spirit_45d`` SHM slot. Under l0_rust_enabled=true (fleet-wide)
        # the Rust trinity daemons own the full 45D computation (all SAT/CHIT/
        # ANANDA inputs), so this slot is the authoritative tensor — it
        # supersedes the retired Python path (``_collect_spirit_tensor`` +
        # ``collect_spirit_45d`` + ``_publish_spirit_state``'s ``_last_spirit_45d``
        # stash) which only ever produced PARTIAL dims from this scope.
        # Empirically verified live (non-flat, fresh) on T1 — audit §5.5/§10D.
        spirit_45d = None
        try:
            if shm_bank is not None:
                isp = shm_bank.read_inner_spirit_45d()
                if isp:
                    vals = isp.get("values") or []
                    if len(vals) >= 45:
                        spirit_45d = [float(x) for x in vals[:45]]
        except Exception as e:
            swallow_warn('[ConsciousnessEpoch] inner_spirit_45d SHM read', e,
                         key="logic.consciousness_epoch.inner_spirit_45d_shm_read", throttle=100)
        if spirit_45d is None:
            # SHM unavailable (cold boot / bank None) — neutral 45D so the epoch
            # still records a valid state vector. Rust populates the slot within
            # the first trinity pulses after boot.
            spirit_45d = [0.5] * 45
        for i, v in enumerate(spirit_45d[:45]):
            sv[20 + i] = v

        _prof["spirit_45d"] = time.time() - t0 - _prof["sv_build_body_mind"]

        # [65:130] Outer Trinity (when extended tensors available)
        if has_outer_extended:
            # [65:70] Outer Body 5D
            outer_body = ostate.get("outer_body", [0.5] * 5)
            for i, v in enumerate(outer_body[:5]):
                sv[65 + i] = v

            # [70:85] Outer Mind 15D
            outer_mind_15d = ostate.get("outer_mind_15d", [0.5] * 15)
            for i, v in enumerate(outer_mind_15d[:15]):
                sv[70 + i] = v

            # [85:130] Outer Spirit 45D
            outer_spirit_45d = ostate.get("outer_spirit_45d", [0.5] * 45)
            for i, v in enumerate(outer_spirit_45d[:45]):
                sv[85 + i] = v

            # [130:132] Self-referential
            meta_offset = 130
        else:
            # [65:67] Self-referential (67D mode)
            meta_offset = 65

        _prof["outer_trinity"] = time.time() - t0 - sum(_prof.values())

        # Self-referential: curvature and density from previous epoch
        _t_db0 = time.time()
        recent = db.get_recent_epochs(TRAJECTORY_WINDOW)
        _prof["db_get_recent"] = time.time() - _t_db0
        if recent:
            sv[meta_offset] = recent[-1].curvature
            sv[meta_offset + 1] = recent[-1].density
        else:
            sv[meta_offset] = 0.0
            sv[meta_offset + 1] = 0.0

        # ── WHY: Compute drift ──
        _t_drift0 = time.time()
        previous_sv = None
        if recent:
            prev_list = recent[-1].state_vector
            if isinstance(prev_list, str):
                prev_list = json.loads(prev_list)
            previous_sv = StateVector.from_list(prev_list)
        drift = sv - previous_sv if previous_sv else StateVector(values=[0.0] * total_dims)
        _prof["drift"] = time.time() - _t_drift0

        # ── WHAT: Compute trajectory (slope over rolling window) ──
        _t_traj0 = time.time()
        trajectory = _compute_trajectory(recent, num_dims=total_dims)
        _prof["trajectory"] = time.time() - _t_traj0

        # ── Journey topology ──
        _t_topo0 = time.time()
        journey_point = topology.compute_point(sv, epoch_id)
        curvature = topology.compute_curvature(journey_point)
        density = topology.compute_density(journey_point)
        _prof["topology"] = time.time() - _t_topo0

        # Feed back into state vector (self-referential loop)
        sv[meta_offset] = curvature
        sv[meta_offset + 1] = density

        # ── Store epoch ──
        _t_store0 = time.time()
        record = EpochRecord(
            epoch_id=epoch_id,
            timestamp=time.time(),
            block_hash="",  # No Solana access from Spirit process
            state_vector=sv.to_list(),
            drift_vector=drift.to_list(),
            trajectory_vector=trajectory.to_list(),
            journey_point=journey_point.to_tuple(),
            curvature=curvature,
            density=density,
            distillation="",  # LLM distillation runs separately if needed
            anchored_tx="",   # On-chain anchoring runs from Core
        )
        db.insert_epoch(record)
        _prof["db_insert"] = time.time() - _t_store0

        # Cache latest epoch for tensor computation
        inner_body_coh = sum(sv.values[0:5]) / 5.0
        inner_mind_coh = sum(sv.values[5:20]) / 15.0
        inner_spirit_coh = sum(sv.values[20:65]) / 45.0
        consciousness["latest_epoch"] = {
            "epoch_id": epoch_id,
            "state_vector": sv.to_list(),
            "drift_magnitude": drift.magnitude(),
            "trajectory_magnitude": trajectory.magnitude(),
            "curvature": curvature,
            "density": density,
            "journey_point": journey_point.to_tuple(),
            "body_coherence": inner_body_coh,
            "mind_coherence": inner_mind_coh,
            "spirit_coherence": inner_spirit_coh,
            "dims": total_dims,
        }
        if has_outer_extended:
            consciousness["latest_epoch"].update({
                "outer_body_coherence": sum(sv.values[65:70]) / 5.0,
                "outer_mind_coherence": sum(sv.values[70:85]) / 15.0,
                "outer_spirit_coherence": sum(sv.values[85:130]) / 45.0,
            })

        _prof["coherence_cache"] = time.time() - t0 - sum(_prof.values())
        elapsed = time.time() - t0
        # P6.3 PROFILING — log breakdown of epoch computation time
        _prof_str = " | ".join(f"{k}={v*1000:.0f}ms" for k, v in _prof.items())
        logger.info("[PROFILE] Epoch %d (%.0fms): %s", epoch_id, elapsed * 1000, _prof_str)
        if has_outer_extended:
            logger.info(
                "[SpiritWorker] Epoch %d complete (%.1fs). "
                "drift=%.4f trajectory=%.4f curvature=%.3f density=%.3f "
                "[132D: iB=%.3f iM=%.3f iS=%.3f | oB=%.3f oM=%.3f oS=%.3f]",
                epoch_id, elapsed, drift.magnitude(), trajectory.magnitude(),
                curvature, density,
                inner_body_coh, inner_mind_coh, inner_spirit_coh,
                sum(sv.values[65:70]) / 5.0,
                sum(sv.values[70:85]) / 15.0,
                sum(sv.values[85:130]) / 45.0,
            )
        else:
            logger.info(
                "[SpiritWorker] Epoch %d complete (%.1fs). "
                "drift=%.4f trajectory=%.4f curvature=%.3f density=%.3f "
                "[%dD: body=%.3f mind=%.3f spirit=%.3f]",
                epoch_id, elapsed, drift.magnitude(), trajectory.magnitude(),
                curvature, density, total_dims,
                inner_body_coh, inner_mind_coh, inner_spirit_coh,
            )

    except Exception as e:
        logger.error("[SpiritWorker] Consciousness epoch failed: %s", e, exc_info=True)




# ── Spirit Tensor Collection ────────────────────────────────────────

