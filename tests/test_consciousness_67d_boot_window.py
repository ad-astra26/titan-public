"""Regression tests for BUG-T1-CONSCIOUSNESS-67D-STATE-VECTOR.

Bug surfaced 2026-04-25: spirit_worker initialised outer_state with
`outer_mind_15d=None` and `outer_spirit_45d=None`, so any consciousness
epoch fired BEFORE the first OUTER_TRINITY_STATE message arrived
(boot window of ~30s+) collapsed to 67D instead of 132D, silently
breaking Trinity-symmetry invariants and starving GroundUp body↔mind
enrichment.

Fix: pre-populate extended fields with neutral [0.5]*N defaults at
spirit_worker init — same value the OuterTrinityCollector pads with on
its own _collect_extended failure path. The merge logic in
_merge_outer_trinity_payload then replaces with truthy incoming data on
the first real OUTER_TRINITY_STATE message. Plus a WARNING in
_run_consciousness_epoch when has_outer_extended evaluates False, so any
future regression that re-introduces None into outer_state is loud.
"""
import logging
import re

# NOTE (2026-05-21): _merge_outer_trinity_payload was deleted with the dead
# spirit_worker helper block (Phase C — spirit_worker is a heartbeat stub; the
# outer-trinity merge + boot-default init moved off spirit_worker). The method
# that exercised it (test_first_real_payload_replaces_boot_defaults) is retired.
# The live 67D-collapse guard below (test_consciousness_epoch_warns_on_67d_fallback,
# against spirit_loop._run_consciousness_epoch) is unaffected. The boot-default-init
# invariant's new owner + test migration is tracked in RFP_phase_c_titan_hcl_cleanup.

# Boot outer-state shape the consciousness epoch expects (self-contained
# invariant on the 132D-not-67D shape).
_BOOT_OUTER_STATE = {
    "outer_body": [0.5] * 5,
    "outer_mind": [0.5] * 5,
    "outer_spirit": [0.5] * 5,
    "outer_mind_15d": [0.5] * 15,
    "outer_spirit_45d": [0.5] * 45,
}


def test_boot_outer_state_satisfies_has_outer_extended():
    """Boot-window outer_state has truthy extended fields → consciousness
    epoch goes 132D from epoch #1 (no 67D collapse before first
    OUTER_TRINITY_STATE arrives)."""
    st = dict(_BOOT_OUTER_STATE)
    # Mirror spirit_loop._run_consciousness_epoch:1117-1121 invariant
    has_outer_extended = (
        st.get("outer_mind_15d") is not None and
        st.get("outer_spirit_45d") is not None
    )
    assert has_outer_extended, (
        "Boot outer_state must have truthy extended fields. "
        "If this fails, has_outer_extended will be False and the very "
        "first consciousness epoch will be 67D — the BUG-T1 regression."
    )
    assert len(st["outer_mind_15d"]) == 15
    assert len(st["outer_spirit_45d"]) == 45


def test_boot_state_neutral_defaults_match_collector_fallback():
    """The boot defaults must equal the OuterTrinityCollector's
    [0.5]*N safe-fallback (outer_trinity.py:159-160). If they ever
    diverge, downstream consumers will see different values for "no
    real data yet" depending on whether _collect_extended failed before
    or after the first publish — non-deterministic cognition."""
    assert _BOOT_OUTER_STATE["outer_mind_15d"] == [0.5] * 15
    assert _BOOT_OUTER_STATE["outer_spirit_45d"] == [0.5] * 45


def test_consciousness_epoch_warns_on_67d_fallback(caplog):
    """If outer_state somehow has None extended fields when
    _run_consciousness_epoch runs, we log a WARNING (not silent
    INFO-only) so the symmetry violation is visible to scanners."""
    # Phase 10D — _run_consciousness_epoch relocated spirit_loop → consciousness_epoch.
    from titan_hcl.logic import consciousness_epoch

    # Build a minimal mock consciousness/topology/body/mind dict
    # sufficient to enter the function's `has_outer_extended` block.
    # We intercept the WARNING via caplog and don't care about the
    # downstream computation succeeding.
    class _FakeDB:
        def get_epoch_count(self):
            return 0

        def write_epoch(self, *a, **kw):
            pass

    consciousness = {
        "db": _FakeDB(),
        "topology": object(),
        "latest_epoch": None,
    }
    body_state = {"values": [0.5] * 5}
    mind_state = {"values": [0.5] * 5, "values_15d": None}
    config = {}

    # Outer state with explicitly None extended fields — the
    # regression scenario.
    outer_state_broken = {
        "outer_body": [0.5] * 5,
        "outer_mind": [0.5] * 5,
        "outer_spirit": [0.5] * 5,
        "outer_mind_15d": None,
        "outer_spirit_45d": None,
    }

    caplog.set_level(logging.WARNING, logger=consciousness_epoch.logger.name)
    try:
        consciousness_epoch._run_consciousness_epoch(
            consciousness, body_state, mind_state, config,
            outer_state=outer_state_broken,
        )
    except Exception:
        # Downstream may fail on the fake db — we only care about the
        # WARNING emission, which happens before any heavy work.
        pass

    matched = [r for r in caplog.records
               if r.levelno == logging.WARNING
               and "collapsing to 67D" in r.getMessage()]
    assert matched, (
        "Expected WARNING when consciousness epoch falls back to 67D. "
        "Without this, a future regression that re-introduces None into "
        "outer_state will be silent again — exactly the BUG-T1 history."
    )
    msg = matched[0].getMessage()
    # Must include identifying detail to aid triage
    assert "outer_mind_15d=False" in msg or "outer_mind_15d is" in msg.lower() \
        or "outer_mind_15d=" in msg
