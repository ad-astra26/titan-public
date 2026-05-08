"""
titan_plugin/logic/outer_body_tensor.py — 5D Outer Body Tensor (pure function).

Extracted from OuterTrinityCollector._collect_outer_body (outer_trinity.py)
as a module-level pure function for use by outer_body_worker subprocess.

Outer Body 5DT — V6 body-felt semantics:
  [0] interoception   — SOL balance (0.4) + block_delta (0.3) + anchor_fresh (0.3)
  [1] proprioception  — peer_entropy (0.5) + helper_health (0.3) + bus_module_div (0.2)
  [2] somatosensation — TX latency (0.4) + creation_nudge (0.3) + CPU spike rate (0.3)
  [3] entropy         — ping variance (0.4) + bus drop rate (0.3) + error rate (0.3)
  [4] thermal         — CPU thermal (0.35) + circadian (0.25) + hormonal_heat (0.40)
                        where hormonal_heat = mean(IMPULSE, VIGILANCE)

Thermal REDESIGNED 2026-05-06 per rFP_trinity_130d_awakening §12.1 + SPEC §23.7:
LLM-latency was the wrong signal (network/inference performance, not body
warmth). Hormonal heat (sympathetic-arousal — fight/flight pressure +
alertness) is the natural body-felt analog of warmth/cold.

All values normalized to [0.0, 1.0] where 0.5 = center (Middle Path).
"""
import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)

MAX_LATENCY_SECONDS = 30.0  # LLM response latency ceiling


def collect_outer_body_5d(
    sources: dict,
    last_outer_body: Optional[list] = None,
) -> list:
    """
    Collect Outer Body 5D tensor from sources dict.

    Args:
        sources: dict with keys matching OuterTrinityCollector.collect() contract:
            sol_balance, anchor_state, network_monitor_stats, system_sensor_stats,
            tx_latency_stats, block_delta_stats, agency_stats, helper_statuses,
            bus_stats, llm_avg_latency
        last_outer_body: previous outer_body [5 floats] for somatosensation
            creation_nudge proxy (self-reference with decay). None → 0.5 neutral.

    Returns:
        [5 floats] normalized to [0.0, 1.0].
    """
    agency = sources.get("agency_stats") or {}
    helper_statuses = sources.get("helper_statuses") or {}
    bus = sources.get("bus_stats") or {}
    sys_stats = sources.get("system_sensor_stats") or {}
    net_stats = sources.get("network_monitor_stats") or {}
    tx_lat = sources.get("tx_latency_stats") or {}
    blk_delta = sources.get("block_delta_stats") or {}
    anchor = sources.get("anchor_state") or {}

    # ── [0] interoception: energetic body-state ──────────────────────
    sol_balance = sources.get("sol_balance")
    if isinstance(sol_balance, (int, float)) and sol_balance >= 0:
        sol_norm = _clamp(math.log1p(sol_balance) / math.log1p(10.0))
    else:
        sol_norm = 0.5

    block_rate_norm = blk_delta.get("normalized", 0.5)

    anchor_fresh = 0.5
    if anchor.get("success") and anchor.get("last_anchor_time"):
        since = time.time() - anchor.get("last_anchor_time", time.time())
        anchor_fresh = max(0.05, 1.0 / (1.0 + since / 300.0))

    interoception = _clamp(0.4 * sol_norm + 0.3 * block_rate_norm + 0.3 * anchor_fresh)

    # ── [1] proprioception: body-in-network-space ────────────────────
    peer_entropy = net_stats.get("peer_entropy", 0.5)
    total_helpers = max(1, len(helper_statuses))
    available = sum(1 for s in helper_statuses.values() if s == "available")
    helper_health = available / total_helpers if total_helpers > 0 else 0.5
    bus_module_diversity = net_stats.get("bus_module_diversity", 0.5)
    proprioception = _clamp(0.5 * peer_entropy + 0.3 * helper_health + 0.2 * bus_module_diversity)

    # ── [2] somatosensation: touch + pressure ────────────────────────
    tx_lat_norm = tx_lat.get("normalized", 0.5)
    current_ob2 = 0.5
    if isinstance(last_outer_body, list) and len(last_outer_body) > 2:
        current_ob2 = float(last_outer_body[2])
    cpu_spikes = sys_stats.get("cpu_spike_rate", 0.0)
    somatosensation = _clamp(0.4 * tx_lat_norm + 0.3 * current_ob2 + 0.3 * cpu_spikes)

    # ── [3] entropy: disorder / unpredictability ─────────────────────
    ping_var = net_stats.get("ping_variance", 0.5)
    bus_drop_rate = net_stats.get("bus_drop_rate", 0.0)
    total_actions = agency.get("total_actions", 0)
    failed_actions = agency.get("failed_actions", 0)
    error_rate = failed_actions / total_actions if total_actions > 0 else 0.0
    entropy = _clamp(0.4 * ping_var + 0.3 * bus_drop_rate + 0.3 * error_rate)

    # ── [4] thermal: arousal / temperature (REDESIGNED 2026-05-06) ──
    # SPEC §23.7 — hormonal_heat = mean(IMPULSE, VIGILANCE). Pure
    # sympathetic-arousal signal. FOCUS intentionally excluded (deep
    # concentration is more cool-sustained than hot). LLM latency
    # dropped (it's not thermal — it's inference performance).
    cpu_thermal = sys_stats.get("cpu_thermal", 0.5)
    circadian = sys_stats.get("circadian_phase", 0.5)
    hormones = sources.get("hormone_levels") or {}
    impulse = float(hormones.get("IMPULSE", 0.5))
    vigilance = float(hormones.get("VIGILANCE", 0.5))
    hormonal_heat = (impulse + vigilance) / 2.0
    thermal = _clamp(0.35 * cpu_thermal + 0.25 * circadian + 0.40 * hormonal_heat)

    return [round(v, 4) for v in [
        interoception, proprioception, somatosensation, entropy, thermal
    ]]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi], handling NaN/None."""
    if value is None or not isinstance(value, (int, float)):
        return 0.5
    if math.isnan(value) or math.isinf(value):
        return 0.5
    return max(lo, min(hi, float(value)))
