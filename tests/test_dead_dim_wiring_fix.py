"""rFP_dead_dim_wiring_fix — regression tests for 4 dead-dim Python wiring bugs.

Each bug is a Python publisher-side wire-up gap left behind from the
PART B §8 deploy of RFP_meta-reasoning_CGN_FIX (D-SPEC-65 v1.9.6). The
pipes were correctly built; the producer-side reads used wrong schema
keys or read from stale in-process engines.

  Bug A — pi_phase   — cognitive_worker.py read wrong key path on
                       _sphere_clocks_snapshot (`phases` flat vs nested
                       under `clocks` with per-clock `.phase` field).
  Bug B — space_topology — cognitive_worker.py invented 3 keys
                       (`outer_lower_topology_10d`, etc.) that don't
                       exist in shm_bank.read_topology_30d() output;
                       canonical 30D payload is `values`.
  Bug C — ns_urgencies — ns_worker `last_urgencies` dict initialized
                       to zeros and never refreshed. Cross-process
                       wire-up gap from the ns_worker L2 carve-out:
                       canonical NS evaluator lives in cognitive_worker
                       (`coordinator._last_nervous_signals`).
                       NEW SHM slot `ns_program_urgencies_input.bin`
                       per SPEC §7.1 + D-SPEC-68 v1.13.0 (G18-pure —
                       NOT a bus event, per Maker greenlight). Consumer
                       applies peak-hold-decay 0.9.
  Bug D — neuromod_state — neuromod_worker called `get_modulation()`
                       which returns modulation factors (sensory_gain
                       etc.), NOT per-modulator levels. Correct source
                       is `neuromod_system.modulators[name].level`.

Run isolated:
    python -m pytest tests/test_dead_dim_wiring_fix.py -v -p no:anchorpy --tb=short
"""

import re
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Bug C — new SHM slot definition (state_registry + constants)
# ─────────────────────────────────────────────────────────────────────


def test_ns_program_urgencies_input_registry_spec_exists():
    """rFP §2.C / SPEC §7.1 + D-SPEC-68 — new RegistrySpec must be
    defined in state_registry, shape (11,) float32, single-writer."""
    from titan_hcl.core.state_registry import NS_PROGRAM_URGENCIES_INPUT
    import numpy as np
    assert NS_PROGRAM_URGENCIES_INPUT.name == "ns_program_urgencies_input"
    assert NS_PROGRAM_URGENCIES_INPUT.dtype == np.dtype("<f4")
    assert NS_PROGRAM_URGENCIES_INPUT.shape == (11,)
    assert NS_PROGRAM_URGENCIES_INPUT.schema_version == 1


def test_ns_program_urgencies_input_schema_version_constant():
    """Schema-version constant must be in _phase_c_constants.py."""
    from titan_hcl import _phase_c_constants as _c
    assert hasattr(_c, "NS_PROGRAM_URGENCIES_INPUT_SCHEMA_VERSION")
    assert _c.NS_PROGRAM_URGENCIES_INPUT_SCHEMA_VERSION == 1


def test_no_bus_event_for_ns_urgencies_cross_worker_transport():
    """SHM-pure path: there must NOT be a NS_PROGRAM_URGENCIES_TICK
    bus event constant (Maker rejected bus event in favor of SHM slot
    per G18 — see D-SPEC-68)."""
    from titan_hcl import bus
    assert not hasattr(bus, "NS_PROGRAM_URGENCIES_TICK"), (
        "bus.NS_PROGRAM_URGENCIES_TICK leaked into bus.py — SHM-pure "
        "path requires this constant NOT to exist (D-SPEC-68)")


# ─────────────────────────────────────────────────────────────────────
# Bug C — peak-hold-decay constant + formula (unchanged from bus-event
# version — peak-hold is an ns_worker-internal cache behavior)
# ─────────────────────────────────────────────────────────────────────


def test_urgency_peak_hold_decay_constant_exists():
    """ns_worker must export URGENCY_PEAK_HOLD_DECAY in (0, 1)."""
    from titan_hcl.modules import ns_worker
    assert hasattr(ns_worker, "URGENCY_PEAK_HOLD_DECAY")
    d = ns_worker.URGENCY_PEAK_HOLD_DECAY
    assert 0.0 < d < 1.0, f"decay must be in (0,1), got {d}"
    # Slight tolerance: spec lock at 0.9 ±0.01 — change should require rFP update.
    assert abs(d - 0.9) < 0.01, \
        f"URGENCY_PEAK_HOLD_DECAY drift from rFP-spec 0.9: {d}"


def test_peak_hold_formula_holds_peak_and_decays():
    """Simulate the peak-hold formula directly. After a peak of 0.7 +
    a snap to 0.0, the next stored value must be 0.7 × decay (not
    snapped to 0). Confirms transient peaks survive post-fire reset."""
    from titan_hcl.modules.ns_worker import URGENCY_PEAK_HOLD_DECAY
    last = 0.0
    last = max(0.7, last * URGENCY_PEAK_HOLD_DECAY)
    assert last == 0.7
    # Simulate the post-fire reset arriving from cognitive_worker.
    last = max(0.0, last * URGENCY_PEAK_HOLD_DECAY)
    assert abs(last - 0.7 * URGENCY_PEAK_HOLD_DECAY) < 1e-9
    # A second post-fire reset decays further (still nonzero).
    last = max(0.0, last * URGENCY_PEAK_HOLD_DECAY)
    expected = 0.7 * URGENCY_PEAK_HOLD_DECAY ** 2
    assert abs(last - expected) < 1e-9


def test_peak_hold_formula_snaps_up_on_new_peak():
    """A new value above the decayed previous must snap to current."""
    from titan_hcl.modules.ns_worker import URGENCY_PEAK_HOLD_DECAY
    last = 0.3
    last = max(0.8, last * URGENCY_PEAK_HOLD_DECAY)
    assert last == 0.8


# ─────────────────────────────────────────────────────────────────────
# Bug C — shm_reader_bank exposes read_ns_program_urgencies_input
# ─────────────────────────────────────────────────────────────────────


def test_shm_reader_bank_exposes_read_ns_program_urgencies_input():
    """ShmReaderBank must expose the new reader for ns_worker to use."""
    from titan_hcl.api.shm_reader_bank import ShmReaderBank
    assert hasattr(ShmReaderBank, "read_ns_program_urgencies_input"), \
        "ShmReaderBank.read_ns_program_urgencies_input missing"
    # Return shape must match the documented `urgencies_by_program`
    # dict key (same shape as legacy NS_URGENCIES_UPDATE bus payload so
    # ns_worker downstream emit path stays unchanged).
    import inspect
    src = inspect.getsource(ShmReaderBank.read_ns_program_urgencies_input)
    assert "urgencies_by_program" in src, \
        "reader must return `urgencies_by_program` key (matches downstream emit shape)"


# ─────────────────────────────────────────────────────────────────────
# Bug A — pi_phase fix (cognitive_worker.py): static check of fixed code
# ─────────────────────────────────────────────────────────────────────


def test_bug_a_pi_phase_reads_clocks_subdict():
    """The fixed publisher must read `_sphere_clocks_snapshot["clocks"]`
    and per-clock `.phase`. The old `.get("phases")` fallback path
    is the bug fingerprint."""
    src = Path("titan_hcl/modules/cognitive_worker.py").read_text()
    # New canonical access pattern.
    assert '("clocks")' in src, "fix missing — no clocks-subdict access"
    # Old bug fingerprint must be gone.
    assert "_phases = _sc_snap.get(\"phases\") or _sc_snap" not in src, \
        "old buggy 'phases' fallback path still present"
    # Per-clock .phase reads must be present.
    assert '(_clocks.get(name) or {}).get("phase"' in src, \
        "per-clock .phase access not found"


# ─────────────────────────────────────────────────────────────────────
# Bug B — space_topology fix: static check
# ─────────────────────────────────────────────────────────────────────


def test_bug_b_space_topology_reads_values():
    """The fixed publisher must read `_topology_snapshot["values"]`
    (the canonical 30D layout)."""
    src = Path("titan_hcl/modules/cognitive_worker.py").read_text()
    # Pull a wider window ending at the SPACE_TOPOLOGY_UPDATE emit.
    win_match = re.search(
        r"(.{2000}bus\.SPACE_TOPOLOGY_UPDATE.*?\}\))",
        src, flags=re.DOTALL)
    assert win_match, "could not locate SPACE_TOPOLOGY_UPDATE emit window"
    win = win_match.group(0)
    assert '.get("values")' in win, "fix missing — no `values` read in emit window"
    # Old bug fingerprints — must not appear as live code in the emit
    # block. (`outer_lower_topology_10d` may legitimately appear in
    # unrelated trinity-topology code in other parts of the file.)
    assert '.get("outer_lower_topology_10d")' not in win, \
        "old buggy outer_lower_topology_10d read still present in emit"
    assert '.get("inner_lower_topology_10d")' not in win, \
        "old buggy inner_lower_topology_10d read still present in emit"
    # The old gate condition — `if _outer10 or _inner10 or _whole10` —
    # must be gone (signature of the broken read pattern).
    assert "_outer10 or _inner10 or _whole10" not in win, \
        "old buggy 3-way gate condition still present in emit"


# ─────────────────────────────────────────────────────────────────────
# Bug C — producer (cognitive_worker writes SHM slot)
# ─────────────────────────────────────────────────────────────────────


def test_bug_c_producer_writes_shm_slot_in_cognitive_worker():
    """cognitive_worker must lazily attach a StateRegistryWriter for
    NS_PROGRAM_URGENCIES_INPUT and write per consciousness epoch from
    coordinator._last_nervous_signals.

    Source-of-truth note: inner_coordinator.tick() calls the LEGACY
    `self.nervous_system.evaluate(...)` (NervousSystem class) — its
    output is cached at `coordinator._last_nervous_signals`. NOT
    `state_refs["neural_nervous_system"]._all_urgencies` (V5 instance
    exists but is never evaluated by coordinator — verified 2026-05-17
    on T3 via diagnostic instrumentation)."""
    src = Path("titan_hcl/modules/cognitive_worker.py").read_text()
    # Must reference the new RegistrySpec import.
    assert "NS_PROGRAM_URGENCIES_INPUT" in src, \
        "cognitive_worker missing NS_PROGRAM_URGENCIES_INPUT import + writer"
    # Must source urgencies from coordinator._last_nervous_signals.
    assert "_last_nervous_signals" in src, \
        "cognitive_worker producer must source from coordinator._last_nervous_signals"
    # Must NOT use the rejected bus-event approach (NS_PROGRAM_URGENCIES_TICK
    # in any form).
    assert "NS_PROGRAM_URGENCIES_TICK" not in src, \
        "rejected bus-event constant leaked into cognitive_worker"


# ─────────────────────────────────────────────────────────────────────
# Bug C — consumer (ns_worker reads SHM slot with peak-hold-decay)
# ─────────────────────────────────────────────────────────────────────


def test_bug_c_consumer_reads_shm_slot_in_ns_worker():
    """ns_worker must read ns_program_urgencies_input via shm_bank and
    apply peak-hold-decay to last_urgencies."""
    src = Path("titan_hcl/modules/ns_worker.py").read_text()
    # Must call the new reader.
    assert "read_ns_program_urgencies_input" in src, \
        "ns_worker missing shm_bank.read_ns_program_urgencies_input call"
    # Must apply peak-hold-decay.
    assert "URGENCY_PEAK_HOLD_DECAY" in src, \
        "ns_worker consumer must apply URGENCY_PEAK_HOLD_DECAY"
    # Must NOT subscribe to or handle the rejected bus event.
    assert "NS_PROGRAM_URGENCIES_TICK" not in src, \
        "rejected bus-event handler leaked into ns_worker"


def test_bug_c_no_bus_subscription_in_plugin_py():
    """plugin.py must NOT include bus.NS_PROGRAM_URGENCIES_TICK in
    ns_worker broadcast_topics (rejected — SHM transport)."""
    src = Path("titan_hcl/core/plugin.py").read_text()
    assert "NS_PROGRAM_URGENCIES_TICK" not in src, \
        "plugin.py still references rejected bus event"


# ─────────────────────────────────────────────────────────────────────
# Bug D — neuromod_state reads from canonical modulator.level
# ─────────────────────────────────────────────────────────────────────


def test_bug_d_neuromod_reads_modulator_level():
    """The fixed neuromod_worker must read modulators[name].level
    (canonical source — same as _build_stats_payload)."""
    src = Path("titan_hcl/modules/neuromod_worker.py").read_text()
    needle = "bus.NEUROMOD_LEVELS_UPDATE"
    idx = src.find(needle)
    assert idx > 0, "could not locate bus.NEUROMOD_LEVELS_UPDATE"
    win_start = max(0, idx - 3500)
    win_end = min(len(src), idx + 500)
    win = src[win_start:win_end]
    assert 'getattr(neuromod_system, "modulators"' in win, \
        "fix missing — modulators dict not read in emit window"
    assert "_mod = neuromod_system.get_modulation()" not in win, \
        "old buggy get_modulation() path back in levels_6d emit window"


# ─────────────────────────────────────────────────────────────────────
# Bonus — shm_bank.read_neuromod TypeError fix
# ─────────────────────────────────────────────────────────────────────


def test_bonus_shm_bank_read_neuromod_uses_row_index_0():
    """Per state_registry.NEUROMOD_STATE shape=(6,4), payload[i] is a
    4-element row; level is field 0. Old `float(payload[i])` raised
    TypeError (0-dim conversion from 1-D array)."""
    src = Path("titan_hcl/api/shm_reader_bank.py").read_text()
    m = re.search(
        r"def read_neuromod\(self\)[^\n]*\n(.*?)(?=\n    def )",
        src, flags=re.DOTALL)
    assert m, "could not locate read_neuromod body"
    body = m.group(1)
    # Strip docstring (between triple quotes) so the bug-fingerprint check
    # doesn't trip on the explanatory comment we added.
    body_no_doc = re.sub(r'"""[\s\S]*?"""', '', body)
    assert "float(payload[i][0])" in body_no_doc, \
        "fix missing — must read payload[i][0] (level field)"
    assert not re.search(r"float\(payload\[i\]\)", body_no_doc), \
        "old buggy float(payload[i]) still present in code"


# ─────────────────────────────────────────────────────────────────────
# Schema sanity — NS_PROGRAMS / NS_PROGRAM_NAMES byte-identical
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Bug E — trajectory_state.bin SHM slot (D-SPEC-69 v1.14.0)
# ─────────────────────────────────────────────────────────────────────


def test_bug_e_trajectory_state_registry_spec_exists():
    """rFP §2.E / SPEC §7.1 + D-SPEC-69 — TRAJECTORY_STATE RegistrySpec
    must be defined: shape (2,) float32."""
    from titan_hcl.core.state_registry import TRAJECTORY_STATE
    import numpy as np
    assert TRAJECTORY_STATE.name == "trajectory_state"
    assert TRAJECTORY_STATE.dtype == np.dtype("<f4")
    assert TRAJECTORY_STATE.shape == (2,)
    assert TRAJECTORY_STATE.schema_version == 1


def test_bug_e_trajectory_state_schema_version_constant():
    from titan_hcl import _phase_c_constants as _c
    assert hasattr(_c, "TRAJECTORY_STATE_SCHEMA_VERSION")
    assert _c.TRAJECTORY_STATE_SCHEMA_VERSION == 1


def test_bug_e_shm_reader_bank_exposes_read_trajectory_state():
    from titan_hcl.api.shm_reader_bank import ShmReaderBank
    assert hasattr(ShmReaderBank, "read_trajectory_state")


def test_bug_e_cognitive_worker_writes_trajectory_state():
    """cognitive_worker must initialize TRAJECTORY_STATE writer + write
    [curvature, density] from latest_epoch per consciousness epoch."""
    src = Path("titan_hcl/modules/cognitive_worker.py").read_text()
    assert "TRAJECTORY_STATE" in src, "TRAJECTORY_STATE RegistrySpec not imported"
    assert "_trajectory_state_writer" in src, \
        "cognitive_worker missing _trajectory_state_writer state_refs key"
    # Must source from latest_epoch curvature + density (not the broken
    # state_vector dict path).
    assert '"curvature"' in src and '"density"' in src, \
        "cognitive_worker writer must read curvature + density from latest_epoch"


def test_bug_e_trajectory_update_bus_handler_retired():
    """emot_cgn must NOT subscribe to bus.TRAJECTORY_UPDATE for cache
    update; the bundle-assembly path reads from SHM instead."""
    src = Path("titan_hcl/modules/emot_cgn_worker.py").read_text()
    # The retirement notice must be present (preserves audit trail).
    assert "TRAJECTORY_UPDATE bus handler RETIRED" in src, \
        "retirement notice missing in emot_cgn_worker"
    # The active code must not have `elif msg_type == bus.TRAJECTORY_UPDATE`.
    assert "elif msg_type == bus.TRAJECTORY_UPDATE" not in src, \
        "emot_cgn still has active TRAJECTORY_UPDATE handler — should be retired"


def test_bug_e_meta_reasoning_trajectory_emit_retired():
    """meta_reasoning must NOT emit bus.TRAJECTORY_UPDATE — state flows
    via SHM slot now."""
    src = Path("titan_hcl/logic/meta_reasoning.py").read_text()
    assert "TRAJECTORY_UPDATE bus emit RETIRED" in src, \
        "retirement notice missing in meta_reasoning"
    # The active emit code must be gone (look for the type field).
    assert '_bus_mod.TRAJECTORY_UPDATE' not in src, \
        "meta_reasoning still has active TRAJECTORY_UPDATE emit"


# ─────────────────────────────────────────────────────────────────────
# Bug F — cgn_beta_state.bin SHM slot (D-SPEC-69 v1.14.0)
# ─────────────────────────────────────────────────────────────────────


def test_bug_f_cgn_beta_state_registry_spec_exists():
    """rFP §2.F / SPEC §7.1 + D-SPEC-69 — CGN_BETA_STATE RegistrySpec
    must be defined: shape (8,) float32."""
    from titan_hcl.core.state_registry import CGN_BETA_STATE
    import numpy as np
    assert CGN_BETA_STATE.name == "cgn_beta_state"
    assert CGN_BETA_STATE.dtype == np.dtype("<f4")
    assert CGN_BETA_STATE.shape == (8,)
    assert CGN_BETA_STATE.schema_version == 1


def test_bug_f_cgn_beta_state_schema_version_constant():
    from titan_hcl import _phase_c_constants as _c
    assert hasattr(_c, "CGN_BETA_STATE_SCHEMA_VERSION")
    assert _c.CGN_BETA_STATE_SCHEMA_VERSION == 1


def test_bug_f_shm_reader_bank_exposes_read_cgn_beta_state():
    from titan_hcl.api.shm_reader_bank import ShmReaderBank
    assert hasattr(ShmReaderBank, "read_cgn_beta_state")


def test_bug_f_cgn_worker_writes_cgn_beta_state():
    """cgn_worker must lazily attach a writer for CGN_BETA_STATE +
    write 8 floats in CGN_CONSUMERS order per beta snapshot."""
    src = Path("titan_hcl/modules/cgn_worker.py").read_text()
    assert "CGN_BETA_STATE" in src, "cgn_worker missing CGN_BETA_STATE import"
    assert "cgn_beta_state_writer" in src, \
        "cgn_worker missing cgn_beta_state_writer lazy-init"
    assert "_reward_ema" in src, \
        "cgn_worker writer must source from per-consumer _reward_ema"


def test_bug_f_cgn_beta_snapshot_bus_retired_in_cgn_worker():
    """cgn_worker must NOT emit bus.CGN_BETA_SNAPSHOT."""
    src = Path("titan_hcl/modules/cgn_worker.py").read_text()
    # Should not have `_send_msg(... bus.CGN_BETA_SNAPSHOT ...)` for the
    # beta snapshot emit (the slot-write replaces it).
    assert "bus.CGN_BETA_SNAPSHOT" not in src, \
        "cgn_worker still emits CGN_BETA_SNAPSHOT bus event — should be retired (SHM-only)"


def test_bug_f_cgn_beta_snapshot_handler_retired_in_emot_cgn():
    """emot_cgn must NOT subscribe to bus.CGN_BETA_SNAPSHOT; the
    bundle-assembly path reads from SHM instead."""
    src = Path("titan_hcl/modules/emot_cgn_worker.py").read_text()
    assert "CGN_BETA_SNAPSHOT bus handler RETIRED" in src, \
        "retirement notice missing in emot_cgn_worker"
    assert "elif msg_type == bus.CGN_BETA_SNAPSHOT" not in src, \
        "emot_cgn still has active CGN_BETA_SNAPSHOT handler"


def test_bug_e_f_plugin_broadcast_topics_retired():
    """plugin.py must NOT include bus.TRAJECTORY_UPDATE or
    bus.CGN_BETA_SNAPSHOT in emot_cgn ModuleSpec broadcast_topics (both
    retired in favor of SHM slots)."""
    src = Path("titan_hcl/core/plugin.py").read_text()
    # The emot_cgn ModuleSpec broadcast_topics block lives near the
    # 5 PART B §8 substrate events — check active references in that
    # window. We look for the retirement comments AND absence of active
    # list entries.
    assert "bus.TRAJECTORY_UPDATE RETIRED" in src or \
        "TRAJECTORY_UPDATE RETIRED" in src, \
        "retirement notice for TRAJECTORY_UPDATE missing in plugin.py"
    assert "CGN_BETA_SNAPSHOT RETIRED" in src, \
        "retirement notice for CGN_BETA_SNAPSHOT missing in plugin.py"


def test_emot_cgn_reads_both_shm_slots_in_bundle_path():
    """emot_cgn bundle-assembly path must call read_trajectory_state +
    read_cgn_beta_state before encode."""
    src = Path("titan_hcl/modules/emot_cgn_worker.py").read_text()
    assert "read_trajectory_state()" in src, \
        "emot_cgn must call shm_bank.read_trajectory_state in bundle path"
    assert "read_cgn_beta_state()" in src, \
        "emot_cgn must call shm_bank.read_cgn_beta_state in bundle path"


def test_ns_program_name_order_matches_protocol():
    """Bug C's slot payload uses NS_PROGRAMS row order from emot_bundle_protocol;
    ns_worker uses NS_PROGRAM_NAMES from shm_reader_bank. Orders MUST be
    byte-identical or the slot field gets permuted vs the protocol layout
    (and titanvm_registers.bin row order)."""
    from titan_hcl.api.shm_reader_bank import NS_PROGRAM_NAMES
    from titan_hcl.logic.emot_bundle_protocol import NS_PROGRAMS
    assert list(NS_PROGRAM_NAMES) == list(NS_PROGRAMS), (
        f"NS_PROGRAM_NAMES vs NS_PROGRAMS order mismatch: "
        f"{NS_PROGRAM_NAMES} vs {NS_PROGRAMS}")
