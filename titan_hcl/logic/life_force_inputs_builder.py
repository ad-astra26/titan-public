"""
life_force_inputs_builder — pure helper aggregating 16 LifeForceEngine inputs.

Phase C v1.8.3 (D-SPEC-57) per rFP_titan_hcl_l2_separation_strategy §4.G.

EXTRACTED FROM `cognitive_worker.py:2370-2473` (the chi-evaluate block that
shipped under 2026-05-10 Track 1 drift). The extraction is BYTE-IDENTICAL to
the pre-§4.G compute path — every input collected here matched the kwargs
passed to `life_force_engine.evaluate(...)` at `cognitive_worker.py:2474`
pre-retirement. No logic changes; just relocation + an explicit pure
function signature that cognitive_worker now calls before writing
`life_force_inputs.bin` for life_force_worker to consume.

Why a builder helper (not just inline code):
  - testability: 16 input fields × per-field test (G14 test_life_force_inputs_builder.py)
  - parity guarantee: life_force_worker.evaluate uses these exact inputs
    (downstream byte-identical chi math vs. pre-extraction)
  - graceful degradation: each input has a fallback when its upstream source
    is cold/absent; the builder is the single place that policy lives

Known stub inputs (preserved verbatim from pre-extraction code):
  - sol_balance = 13.0 — hardcoded stub at `cognitive_worker.py:2434`
    (real SOL balance comes via SOLANA_BALANCE_UPDATED bus event, but
     pre-§4.G code never wired it; follow-up rFP target)
  - anchor_freshness = 0.5 — hardcoded stub at `cognitive_worker.py:2435`
    (real value comes from TimeChain anchor age delta; follow-up rFP target)
  - sovereignty_index = 0 — hardcoded stub at `cognitive_worker.py:2384`
    (real value comes from `sovereignty.get_post_mint_count()`; follow-up
     rFP target — listed in master plan §10 D17/D22)
  - infrastructure_health = 0.8 — life_force_engine.evaluate default
    (not passed in pre-extraction code either)

These are documented as known stubs so the follow-up rFP can wire real
values without touching this builder's signature. life_force_worker
receives them as-is.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from titan_hcl.logic.life_force import (
    compute_coherence_from_sv,
    compute_expression_fire_rate,
    compute_hormonal_vitality,
    compute_neuromodulator_homeostasis,
)

logger = logging.getLogger(__name__)


# ── Known-stub defaults (preserved from cognitive_worker.py pre-§4.G) ─────
_STUB_SOL_BALANCE: float = 13.0
_STUB_ANCHOR_FRESHNESS: float = 0.5
_STUB_SOVEREIGNTY_INDEX: int = 0
_STUB_INFRASTRUCTURE_HEALTH: float = 0.8


def compute_life_force_inputs(
    *,
    coordinator: Any,
    pi_monitor: Any,
    neural_nervous_system: Any,
    latest_epoch: dict[str, Any],
    consciousness: Optional[dict[str, Any]] = None,
    topology_snap: Optional[dict[str, Any]] = None,
    expression_state_reader: Any = None,
    vocab_db_path: str = "data/inner_memory.db",
) -> dict[str, Any]:
    """Aggregate the 16 LifeForceEngine.evaluate inputs into a single dict.

    Pure function — no side effects, no SHM writes. Caller (cognitive_worker)
    feeds result to `LifeForceInputsPublisher.publish(...)`.

    All inputs gracefully degrade to safe defaults on upstream cold-boot or
    missing sub-component. The pre-extraction code at
    `cognitive_worker.py:2370-2473` did exactly this — preserved verbatim.

    Returns dict with the 16 schema-v1 keys (see
    `life_force_inputs_specs.py` docstring for the full schema).
    """
    # ── Spirit inputs (4) ──────────────────────────────────────────────
    _sv = latest_epoch.get("state_vector", [])
    if hasattr(_sv, "to_list"):
        _sv = _sv.to_list()
    _sv = list(_sv) if _sv else []

    pi_heartbeat_ratio = (
        float(pi_monitor.heartbeat_ratio) if pi_monitor is not None else 0.0
    )
    developmental_age = (
        int(pi_monitor.developmental_age) if pi_monitor is not None else 0
    )
    sovereignty_index = _STUB_SOVEREIGNTY_INDEX  # follow-up rFP target

    spirit_coherence = 0.5
    if len(_sv) >= 130:
        try:
            _is_coh = compute_coherence_from_sv(_sv, 20, 65)
            _os_coh = compute_coherence_from_sv(_sv, 85, 130)
            spirit_coherence = (_is_coh + _os_coh) / 2.0
        except Exception:
            spirit_coherence = 0.5

    # ── Mind inputs (6) ────────────────────────────────────────────────
    vocabulary_size = 0
    try:
        from titan_hcl.utils.db import safe_connect as _safe_connect
        _vdb = _safe_connect(vocab_db_path)
        vocabulary_size = _vdb.execute(
            "SELECT COUNT(*) FROM vocabulary WHERE confidence > 0.3"
        ).fetchone()[0]
        _vdb.close()
    except Exception:
        vocabulary_size = 0

    _nm_sys = (
        getattr(coordinator, "neuromodulator_system", None)
        if coordinator is not None
        else None
    )

    learning_rate_gain = 1.0
    if _nm_sys is not None:
        try:
            learning_rate_gain = float(
                _nm_sys.get_modulation().get("learning_rate_gain", 1.0)
            )
        except Exception:
            learning_rate_gain = 1.0

    emotional_coherence = 0.5
    if _nm_sys is not None:
        try:
            emotional_coherence = float(
                getattr(_nm_sys, "_emotion_confidence", 0.5)
            )
        except Exception:
            emotional_coherence = 0.5

    neuromodulator_homeostasis = 0.5
    if _nm_sys is not None:
        try:
            neuromodulator_homeostasis = compute_neuromodulator_homeostasis(
                getattr(_nm_sys, "modulators", {})
            )
        except Exception:
            neuromodulator_homeostasis = 0.5

    mind_coherence = 0.5
    if len(_sv) >= 85:
        try:
            _im_coh = compute_coherence_from_sv(_sv, 5, 20)
            _om_coh = compute_coherence_from_sv(_sv, 70, 85)
            mind_coherence = (_im_coh + _om_coh) / 2.0
        except Exception:
            mind_coherence = 0.5

    # §4.B Track 3 — read composite stats from expression_state.bin SHM slot
    # (expression_worker is canonical writer under l0_rust=true).
    expression_fire_rate = 0.0
    _expr_stats: dict[str, Any] = {}
    if expression_state_reader is not None:
        try:
            _raw = expression_state_reader.read_variable()
            if _raw:
                import msgpack
                _expr_stats = msgpack.unpackb(_raw, raw=False) or {}
        except Exception:
            _expr_stats = {}
    try:
        expression_fire_rate = float(compute_expression_fire_rate(_expr_stats))
    except Exception:
        expression_fire_rate = 0.0

    # ── Body inputs (6) ────────────────────────────────────────────────
    sol_balance = _STUB_SOL_BALANCE          # follow-up rFP target
    anchor_freshness = _STUB_ANCHOR_FRESHNESS  # follow-up rFP target

    hormonal_vitality = 0.5
    if neural_nervous_system is not None:
        try:
            hormonal_vitality = float(
                compute_hormonal_vitality(
                    neural_nervous_system.get_stats().get("hormonal_system", {})
                )
            )
        except Exception:
            hormonal_vitality = 0.5

    body_coherence = 0.5
    if len(_sv) >= 70:
        try:
            _ib_coh = compute_coherence_from_sv(_sv, 0, 5)
            _ob_coh = compute_coherence_from_sv(_sv, 65, 70)
            body_coherence = (_ib_coh + _ob_coh) / 2.0
        except Exception:
            body_coherence = 0.5

    # Topology grounding via topology_30d.bin layout per
    # `titan-trinity-rs/src/topology.rs:337` assemble_topology_30d:
    #   [0:10]  outer_lower.topology_10d
    #   [10:20] inner_lower.topology_10d
    #   [20:30] whole_10d
    # Coherence = cosine similarity of inner_lower vs balanced ref [0.5]*10.
    topology_grounding = 0.5
    try:
        _topo_values = (
            topology_snap.get("values") if isinstance(topology_snap, dict) else None
        )
        if _topo_values and len(_topo_values) >= 20:
            from titan_hcl.logic.lower_topology import _cosine_sim
            _inner_lower_10d = list(_topo_values[10:20])
            topology_grounding = float(_cosine_sim(_inner_lower_10d, [0.5] * 10))
    except Exception:
        topology_grounding = 0.5

    infrastructure_health = _STUB_INFRASTRUCTURE_HEALTH

    return {
        # Spirit (4)
        "pi_heartbeat_ratio": pi_heartbeat_ratio,
        "developmental_age": developmental_age,
        "sovereignty_index": sovereignty_index,
        "spirit_coherence": spirit_coherence,
        # Mind (6)
        "vocabulary_size": vocabulary_size,
        "learning_rate_gain": learning_rate_gain,
        "emotional_coherence": emotional_coherence,
        "neuromodulator_homeostasis": neuromodulator_homeostasis,
        "mind_coherence": mind_coherence,
        "expression_fire_rate": expression_fire_rate,
        # Body (6)
        "sol_balance": sol_balance,
        "anchor_freshness": anchor_freshness,
        "hormonal_vitality": hormonal_vitality,
        "body_coherence": body_coherence,
        "topology_grounding": topology_grounding,
        "infrastructure_health": infrastructure_health,
    }
