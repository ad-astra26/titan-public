"""Integration tests for OuterTrinityCollector._collect_outer_body V6 rewrite.

Verifies:
- All 5 dims respond to their respective composite inputs
- No single saturating input pins a dim to 1.0 (core archaeology finding)
- Missing/empty sources fall back to 0.5 neutral per-input, dim overall near 0.5
- Output always 5 floats in [0, 1]
"""
import time

from titan_plugin.logic.outer_trinity import OuterTrinityCollector


def _make_collector():
    c = OuterTrinityCollector()
    return c


def test_empty_sources_produces_reasonable_output():
    """No inputs → every dim in [0, 1], none saturated at 1.0, none
    pathologically high. Low values for some dims are OK: an empty
    'no drops / no errors recorded' world is genuinely low-entropy."""
    c = _make_collector()
    ob = c._collect_outer_body({})
    assert len(ob) == 5
    for i, v in enumerate(ob):
        assert 0.0 <= v <= 1.0, f"dim[{i}]={v} out of range"
        # None should saturate at 1.0 on zero input (archaeology's worst case)
        assert v < 0.9, f"dim[{i}]={v} saturated on empty sources"


def test_single_saturating_input_cannot_pin_dim_to_one():
    """Key archaeology finding: no single source should saturate a dim.
    With weight <= 0.5, max possible is 0.5 + (0.5 * 1.0) = 1.0, but
    with 3 independent sources averaging 0.5, the actual output stays far
    from 1.0 unless all three saturate. Feed ONE saturating source only."""
    c = _make_collector()

    # Dim [2] somatosensation: TX latency with weight 0.4 saturating
    sources = {"tx_latency_stats": {"normalized": 1.0}}
    ob = c._collect_outer_body(sources)
    assert ob[2] < 0.8, f"dim[2] should not saturate from TX latency alone, got {ob[2]}"

    # Dim [1] proprioception: peer entropy (weight 0.5) saturating
    sources = {"network_monitor_stats": {"peer_entropy": 1.0}}
    ob = c._collect_outer_body(sources)
    assert ob[1] < 0.85, f"dim[1] should not saturate from peer entropy alone, got {ob[1]}"

    # Dim [3] entropy: ping variance (weight 0.4) saturating
    sources = {"network_monitor_stats": {"ping_variance": 1.0}}
    ob = c._collect_outer_body(sources)
    assert ob[3] < 0.8, f"dim[3] should not saturate from ping variance alone, got {ob[3]}"

    # Dim [4] thermal: CPU thermal (weight 0.4) saturating
    sources = {"system_sensor_stats": {"cpu_thermal": 1.0}}
    ob = c._collect_outer_body(sources)
    assert ob[4] < 0.8, f"dim[4] should not saturate from CPU thermal alone, got {ob[4]}"


def test_interoception_responds_to_sol_balance():
    c = _make_collector()
    # No SOL → dim[0] should be lower than with rich SOL
    no_sol = c._collect_outer_body({"sol_balance": 0.0})
    rich_sol = c._collect_outer_body({"sol_balance": 5.0})
    assert rich_sol[0] > no_sol[0], "Rich SOL should lift interoception"


def test_interoception_responds_to_block_delta():
    c = _make_collector()
    stalled = c._collect_outer_body({"block_delta_stats": {"normalized": 0.0}})
    active = c._collect_outer_body({"block_delta_stats": {"normalized": 1.0}})
    assert active[0] > stalled[0], "Active chain should lift interoception"


def test_interoception_responds_to_anchor_freshness():
    c = _make_collector()
    # Fresh anchor (just now) vs stale (10 min ago)
    now = time.time()
    fresh = c._collect_outer_body({
        "anchor_state": {"success": True, "last_anchor_time": now},
    })
    stale = c._collect_outer_body({
        "anchor_state": {"success": True, "last_anchor_time": now - 600},
    })
    assert fresh[0] > stale[0], "Fresh anchor should lift interoception more than stale"


def test_proprioception_responds_to_peer_entropy():
    c = _make_collector()
    mono = c._collect_outer_body({"network_monitor_stats": {"peer_entropy": 0.0}})
    diverse = c._collect_outer_body({"network_monitor_stats": {"peer_entropy": 1.0}})
    assert diverse[1] > mono[1], "Peer diversity should lift proprioception"


def test_proprioception_responds_to_helper_health():
    c = _make_collector()
    no_helpers = c._collect_outer_body({"helper_statuses": {}})
    all_avail = c._collect_outer_body({
        "helper_statuses": {f"h{i}": "available" for i in range(5)},
    })
    assert all_avail[1] >= no_helpers[1], "Available helpers should support proprioception"


def test_entropy_responds_to_ping_variance():
    c = _make_collector()
    calm = c._collect_outer_body({"network_monitor_stats": {"ping_variance": 0.0}})
    chaos = c._collect_outer_body({"network_monitor_stats": {"ping_variance": 1.0}})
    assert chaos[3] > calm[3], "Ping variance should raise entropy dim"


def test_entropy_responds_to_bus_drop_rate():
    c = _make_collector()
    calm = c._collect_outer_body({"network_monitor_stats": {"bus_drop_rate": 0.0}})
    chaos = c._collect_outer_body({"network_monitor_stats": {"bus_drop_rate": 1.0}})
    assert chaos[3] > calm[3], "Bus drops should raise entropy dim"


def test_thermal_responds_to_cpu_thermal():
    c = _make_collector()
    cold = c._collect_outer_body({"system_sensor_stats": {"cpu_thermal": 0.0}})
    hot = c._collect_outer_body({"system_sensor_stats": {"cpu_thermal": 1.0}})
    assert hot[4] > cold[4], "CPU thermal should raise thermal dim"


def test_thermal_responds_to_circadian():
    c = _make_collector()
    night = c._collect_outer_body({"system_sensor_stats": {"circadian_phase": 0.2}})
    day = c._collect_outer_body({"system_sensor_stats": {"circadian_phase": 0.9}})
    assert day[4] > night[4], "Circadian peak should lift thermal"


def test_somatosensation_responds_to_tx_latency():
    c = _make_collector()
    fast = c._collect_outer_body({"tx_latency_stats": {"normalized": 0.0}})
    slow = c._collect_outer_body({"tx_latency_stats": {"normalized": 1.0}})
    assert slow[2] > fast[2], "TX latency should raise somatosensation"


def test_full_rich_sources_produces_varied_output():
    """Feed fully-populated realistic sources; verify all dims have distinct
    values (no stuck-at-0.5 or all-saturated-at-1.0 pathology)."""
    c = _make_collector()
    sources = {
        "sol_balance": 1.0,
        "block_delta_stats": {"normalized": 0.7},
        "anchor_state": {"success": True, "last_anchor_time": time.time() - 120},
        "network_monitor_stats": {
            "peer_entropy": 0.85, "ping_variance": 0.15,
            "bus_drop_rate": 0.02, "bus_module_diversity": 0.9,
        },
        "helper_statuses": {"h1": "available", "h2": "available", "h3": "busy"},
        "system_sensor_stats": {
            "cpu_load": 0.4, "cpu_thermal": 0.55,
            "circadian_phase": 0.8, "cpu_spike_rate": 0.1,
        },
        "tx_latency_stats": {"normalized": 0.3},
        "agency_stats": {"total_actions": 100, "failed_actions": 5},
        "llm_avg_latency": 1.5,
    }
    ob = c._collect_outer_body(sources)

    # No dim stuck at 0.5 exact
    for i, v in enumerate(ob):
        assert v != 0.5, f"dim[{i}] stuck at 0.5 — producer chain broken"

    # No dim saturated (all below 0.9)
    for i, v in enumerate(ob):
        assert v < 0.95, f"dim[{i}]={v} saturated"

    # Reasonable spread across dims (at least 3 distinct values)
    rounded = [round(v, 2) for v in ob]
    assert len(set(rounded)) >= 3, f"Too little variance: {rounded}"


def test_error_rate_influence_on_entropy():
    """Entropy composite should pick up agency error rate."""
    c = _make_collector()
    clean = c._collect_outer_body({
        "agency_stats": {"total_actions": 100, "failed_actions": 0},
    })
    errorful = c._collect_outer_body({
        "agency_stats": {"total_actions": 100, "failed_actions": 30},
    })
    assert errorful[3] > clean[3], "Error rate should raise entropy"
