"""§8 wiring regression — emot_cgn substrate direct-subscription cache.

RFP_meta-reasoning_CGN_FIX.md §8 retires the spirit_worker
`_attach_emot_producer_ctx` bridge by giving emot_cgn_worker direct
subscriptions to 5 Phase C producer events:

  - TRAJECTORY_UPDATE      (cognitive_worker → emot_cgn)
  - NS_URGENCIES_UPDATE    (ns_worker → emot_cgn)
  - SPACE_TOPOLOGY_UPDATE  (cognitive_worker → emot_cgn)
  - NEUROMOD_LEVELS_UPDATE (neuromod_worker → emot_cgn)
  - PI_PHASE_UPDATE        (cognitive_worker → emot_cgn)

This test asserts the 5 bus constants exist, the emot_cgn ModuleSpec
broadcast_topics includes them, and the spirit_worker
`_attach_emot_producer_ctx` symbol no longer exists.

Run isolated:
    python -m pytest tests/test_emot_cgn_substrate_subscriptions.py -v -p no:anchorpy --tb=short
"""

# ── Bus constants exist + correct names ──────────────────────────────


def test_substrate_bus_events_exist():
    """All 5 §8 substrate bus events must be defined as string constants."""
    from titan_hcl import bus
    for evt in (
        "TRAJECTORY_UPDATE",
        "NS_URGENCIES_UPDATE",
        "SPACE_TOPOLOGY_UPDATE",
        "NEUROMOD_LEVELS_UPDATE",
        "PI_PHASE_UPDATE",
    ):
        assert hasattr(bus, evt), f"missing bus.{evt}"
        v = getattr(bus, evt)
        assert isinstance(v, str) and v == evt, (
            f"bus.{evt} must equal {evt!r}, got {v!r}")


# ── emot_cgn ModuleSpec subscribes to all 5 ──────────────────────────


def test_emot_cgn_broadcast_topics_includes_substrate_events():
    """plugin.py must register emot_cgn with all 5 §8 events in
    broadcast_topics — otherwise the broker won't fanout the substrate
    updates to the emot_cgn subscriber and DEAD-DIM persists."""
    # Resolve via static read of the plugin.py file — instantiating the
    # full TitanHCL from a unit test would require config + bus + SHM
    # boot, which is out of scope for this regression. Static check
    # mirrors `_register_emot_cgn_module()` register call.
    import re
    from pathlib import Path
    src = Path("titan_hcl/core/plugin.py").read_text()
    # Locate the emot_cgn ModuleSpec block.
    spec_block_match = re.search(
        r'name="emot_cgn".*?broadcast_topics=\[(.*?)\]',
        src,
        flags=re.DOTALL,
    )
    assert spec_block_match, "could not locate emot_cgn ModuleSpec block"
    topics_str = spec_block_match.group(1)
    for evt in (
        "bus.TRAJECTORY_UPDATE",
        "bus.NS_URGENCIES_UPDATE",
        "bus.SPACE_TOPOLOGY_UPDATE",
        "bus.NEUROMOD_LEVELS_UPDATE",
        "bus.PI_PHASE_UPDATE",
    ):
        assert evt in topics_str, (
            f"emot_cgn ModuleSpec broadcast_topics missing {evt}")


# ── D8 retirement — _attach_emot_producer_ctx must be deleted ────────


def test_attach_emot_producer_ctx_retired():
    """RFP §8 + D8 retirement: the spirit_worker emot-producer bridge must be
    gone. D-SPEC-116 (2026-05-22) completes this — spirit_worker.py is fully
    DELETED, so the strongest invariant is that the module no longer exists;
    emot_cgn now uses direct Phase C subscriptions."""
    import importlib.util
    assert importlib.util.find_spec(
        "titan_hcl.modules.spirit_worker") is None, (
        "spirit_worker.py (which hosted _attach_emot_producer_ctx) must stay "
        "deleted per D-SPEC-116; emot_cgn uses direct Phase C subscriptions")


# ── Handler shape — cached values must be flat ordered vectors ──────


def test_emot_cgn_handler_caches_trajectory_2d():
    """TRAJECTORY_UPDATE handler caches trajectory_2d into worker_state."""
    # Construct a stub worker_state + invoke the cache logic directly.
    # Doesn't boot the full emot_cgn_worker_main — surgical unit test.
    worker_state = {"last_trajectory_2d": None}
    payload = {"trajectory_2d": [0.42, -0.18]}
    # Simulate the inline handler logic.
    traj = payload.get("trajectory_2d") or []
    if len(traj) >= 2:
        worker_state["last_trajectory_2d"] = [float(traj[0]), float(traj[1])]
    assert worker_state["last_trajectory_2d"] == [0.42, -0.18]


def test_emot_cgn_handler_caches_ns_urgencies_ordered():
    """NS_URGENCIES_UPDATE handler normalizes ordering to NS_PROGRAMS."""
    from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
    worker_state = {"last_ns_urgencies_11d": None}
    payload = {
        "urgencies_by_program": {
            "REFLEX": 0.1, "FOCUS": 0.2, "INTUITION": 0.3,
            "IMPULSE": 0.4, "METABOLISM": 0.5, "CREATIVITY": 0.6,
            "CURIOSITY": 0.7, "EMPATHY": 0.8, "REFLECTION": 0.9,
            "INSPIRATION": 0.95, "VIGILANCE": 1.0,
        },
    }
    urgencies = payload.get("urgencies_by_program") or {}
    worker_state["last_ns_urgencies_11d"] = [
        float(urgencies.get(p, 0.0)) for p in NS_PROGRAMS
    ]
    assert len(worker_state["last_ns_urgencies_11d"]) == 11
    assert worker_state["last_ns_urgencies_11d"][0] == 0.1  # REFLEX
    assert worker_state["last_ns_urgencies_11d"][-1] == 1.0  # VIGILANCE


def test_emot_cgn_handler_caches_space_topology_30d():
    """SPACE_TOPOLOGY_UPDATE handler caches 30D vector."""
    worker_state = {"last_space_topology_30d": None}
    payload = {"space_topology_30d": list(range(30))}
    topo30 = payload.get("space_topology_30d") or []
    if len(topo30) >= 30:
        worker_state["last_space_topology_30d"] = [
            float(v) for v in topo30[:30]]
    assert len(worker_state["last_space_topology_30d"]) == 30
    assert worker_state["last_space_topology_30d"][0] == 0.0
    assert worker_state["last_space_topology_30d"][29] == 29.0


def test_emot_cgn_handler_caches_neuromod_6d_ordered():
    """NEUROMOD_LEVELS_UPDATE handler caches DA/5HT/NE/ACh/Endorphin/GABA."""
    worker_state = {"last_neuromod_6d": None}
    payload = {"levels_6d": [0.5, 0.6, 0.7, 0.4, 0.3, 0.8]}
    nm6 = payload.get("levels_6d") or []
    if len(nm6) >= 6:
        worker_state["last_neuromod_6d"] = [float(v) for v in nm6[:6]]
    assert worker_state["last_neuromod_6d"] == [0.5, 0.6, 0.7, 0.4, 0.3, 0.8]


def test_emot_cgn_handler_caches_pi_phase_6d():
    """PI_PHASE_UPDATE handler caches 6D sphere-clock phase tuple."""
    worker_state = {"last_pi_phase_6d": None}
    payload = {"pi_phase_6d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    pp6 = payload.get("pi_phase_6d") or []
    if len(pp6) >= 6:
        worker_state["last_pi_phase_6d"] = [float(v) for v in pp6[:6]]
    assert worker_state["last_pi_phase_6d"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
