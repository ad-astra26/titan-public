"""tests/test_metabolism_worker_extraction.py — SPEC v1.7.2 / D-SPEC-51 parity.

Per rFP_titan_hcl_l2_separation_strategy.md §4.J. Static assertions
that the SPEC documents what the code implements (and vice versa) for
the metabolism_worker extraction.

Covers four test categories per §21 D-SPEC-51 acceptance criteria #5:
  1. MetabolismShmReader cold-boot defaults + local feature lookup
  2. MetabolismProxy surface parity vs MetabolismController
  3. SPEC document drift (Changelog / glossary / §7.1 / §8.7 / §9.B / D-SPEC-51)
  4. Wiring drift (plugin.py / legacy_core.py / soul.py / kernel.py /
     phase_c_rpc_exemptions.yaml / bus.py / bus_specs.py / constants TOML)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


# ═════════════════════════════════════════════════════════════════════
# Category 1 — MetabolismShmReader
# ═════════════════════════════════════════════════════════════════════


def test_shm_reader_cold_boot_defaults():
    """Cold-boot (no SHM data yet) returns sensible defaults.

    Uses a synthetic titan_id so the test does not collide with live SHM
    state when run on a host where a real Titan (T1/T2/T3) is operating.
    """
    from titan_hcl.proxies.metabolism_proxy import MetabolismShmReader
    r = MetabolismShmReader(titan_id="T_COLD_BOOT_TEST")
    assert r.get_metabolic_tier() == "HEALTHY"
    assert r.get_gates_enforced() is False
    assert r.gates_enforced is False
    assert r.get_tier_info() == {}
    assert r.get_last_gate_decision_reason() == ""
    assert r.get_balance_pct() == 1.0


def test_shm_reader_can_use_feature_local_lookup():
    """can_use_feature does local TIER_FEATURES lookup; no bus needed."""
    from titan_hcl.proxies.metabolism_proxy import MetabolismShmReader
    from titan_hcl.core.metabolism import TIER_FEATURES
    r = MetabolismShmReader(titan_id="T1")
    # Cold tier=HEALTHY — features depend on the dict
    healthy_features = TIER_FEATURES.get("HEALTHY", {})
    if healthy_features:
        # Pick any feature defined in HEALTHY tier and assert parity
        for feat, allowed in healthy_features.items():
            assert r.can_use_feature(feat) == bool(allowed), (
                f"can_use_feature({feat}) mismatch vs TIER_FEATURES['HEALTHY']")


# ═════════════════════════════════════════════════════════════════════
# Category 2 — MetabolismProxy surface
# ═════════════════════════════════════════════════════════════════════


def test_proxy_exposes_controller_public_surface():
    """MetabolismProxy declares every public method MetabolismController has.

    Drop-in replacement contract: any call-site that used
    `_proxies["metabolism"].X` pre-extraction must continue to work
    post-extraction. We assert the proxy class has every public method
    name defined on MetabolismController.
    """
    from titan_hcl.core.metabolism import MetabolismController
    from titan_hcl.proxies.metabolism_proxy import MetabolismProxy

    # Public methods on the controller (exclude dunders + private _).
    controller_pub = {
        name for name in dir(MetabolismController)
        if not name.startswith("_") and callable(getattr(MetabolismController, name))
    }

    # Methods/properties on the proxy. The proxy has additional names
    # (work-RPC helpers, internal state) but must include every controller name.
    proxy_pub = {
        name for name in dir(MetabolismProxy)
        if not name.startswith("_")
    }

    # `get_social_gravity_score` is referenced in plugin.py:1629 comment but
    # NOT defined on MetabolismController today — exclude if missing.
    expected = controller_pub - {"get_social_gravity_score"}

    missing = expected - proxy_pub
    # Some controller methods were monkey-patched at runtime (the v3_*
    # growth getters). Those are exposed as async methods on the proxy.
    # We exclude the monkey-patch attribute names that aren't proper
    # methods on the unbound class.
    assert missing == set(), (
        f"MetabolismProxy is missing methods present on "
        f"MetabolismController: {sorted(missing)}")


def test_proxy_evaluate_gate_signature_preserved():
    """evaluate_gate(feature, caller="") returns (bool, float)."""
    from titan_hcl.proxies.metabolism_proxy import MetabolismProxy
    import inspect
    sig = inspect.signature(MetabolismProxy.evaluate_gate)
    params = list(sig.parameters.keys())
    assert params[0] == "self"
    assert params[1] == "feature"
    assert "caller" in params
    assert sig.parameters["caller"].default == ""


def test_proxy_async_methods_are_async():
    """can_afford / get_current_state / get_metabolic_health / growth
    methods stay async (matching MetabolismController async surface)."""
    from titan_hcl.proxies.metabolism_proxy import MetabolismProxy
    import inspect
    for name in (
        "evaluate_gate_async", "can_afford", "get_current_state",
        "get_metabolic_health", "get_learning_velocity",
        "get_directive_alignment", "get_social_density",
    ):
        method = getattr(MetabolismProxy, name)
        assert inspect.iscoroutinefunction(method), (
            f"MetabolismProxy.{name} should be async")


def test_proxy_sync_hot_reads_are_sync():
    """SHM-direct hot reads stay sync (sub-ms; called from kernel paths)."""
    from titan_hcl.proxies.metabolism_proxy import MetabolismProxy
    import inspect
    for name in (
        "get_metabolic_tier", "get_gates_enforced", "get_tier_info",
        "get_last_gate_decision_reason", "can_use_feature",
    ):
        method = getattr(MetabolismProxy, name)
        assert not inspect.iscoroutinefunction(method), (
            f"MetabolismProxy.{name} should be sync (hot path)")


# ═════════════════════════════════════════════════════════════════════
# Category 3 — SPEC document parity
# ═════════════════════════════════════════════════════════════════════


def _spec_text() -> str:
    return (ROOT / "titan-docs" / "SPEC_titan_architecture.md").read_text()


def test_spec_changelog_has_v172_metabolism_worker_row():
    spec = _spec_text()
    marker = "| 2026-05-14 | v1.7.2 (PATCH) |"
    assert marker in spec, (
        "Missing Changelog row for v1.7.2 PATCH bump (D-SPEC-51 / "
        "metabolism_worker extraction).")
    # Verify metabolism_worker appears in the v1.7.2 row itself. (Was a
    # brittle `changelog[:6000]` char-window slice — the changelog has since
    # grown past 50k chars so the v1.7.2 row sits well beyond char 6000;
    # the slice never reached it. Target the row directly instead.)
    v172_row = spec.split(marker, 1)[1].split("\n", 1)[0]
    assert "metabolism_worker" in v172_row


def test_spec_glossary_has_metabolism_worker_row():
    spec = _spec_text()
    glossary_block = spec.split("## §1 — Glossary")[1].split("## §2")[0]
    assert "**metabolism_worker**" in glossary_block, (
        "Missing §1 glossary row for metabolism_worker.")


def test_spec_71_has_metabolism_state_bin_row():
    spec = _spec_text()
    assert "`metabolism_state.bin`" in spec, (
        "Missing §7.1 slot row for metabolism_state.bin")
    assert "METABOLISM_STATE_SCHEMA_VERSION" in spec
    assert "METABOLISM_STATE_MAX_BYTES" in spec


def test_spec_87_has_metabolism_bus_event_rows():
    spec = _spec_text()
    section = spec.split("### §8.7")[1].split("### §8.8")[0]
    for event in (
        "METABOLIC_TIER_CHANGED",
        "GATE_DECISION_RECORDED",
        "METABOLIC_STATS_UPDATED",
    ):
        assert event in section, f"Missing §8.7 row for {event}"


def test_spec_9b_has_metabolism_worker_block():
    spec = _spec_text()
    assert "#### metabolism_worker (Python L2 module" in spec, (
        "Missing §9.B block for metabolism_worker.")
    # Take the full block until the next §9.B sub-heading or §9.C.
    block = spec.split("#### metabolism_worker")[1]
    # Stop at the next #### or §9.C header, whichever is first.
    stop_idx = len(block)
    for stop in ("\n#### ", "\n### §9.C"):
        idx = block.find(stop)
        if idx != -1 and idx < stop_idx:
            stop_idx = idx
    block = block[:stop_idx]
    # Sanity: declared owns / subs / pubs / shm slot / deps + narrative.
    for token in (
        "MetabolismController",
        "metabolism_state.bin",
        "bus.QUERY (dst=metabolism)",
        "SOLANA_BALANCE_UPDATED",
        "METABOLIC_TIER_CHANGED",
        "GATE_DECISION_RECORDED",
        "METABOLIC_STATS_UPDATED",
        "MetabolismShmReader",
    ):
        assert token in block, f"§9.B metabolism_worker block missing: {token}"


def test_spec_21_has_dspec_51_entry():
    spec = _spec_text()
    # Match the entry header specifically so later D-SPEC-NN entries that
    # cross-reference D-SPEC-51 (e.g. D-SPEC-57 sovereignty_worker citing the
    # metabolism_worker pattern) don't break this assertion.
    marker = "- **D-SPEC-51 "
    assert marker in spec, (
        f"Missing D-SPEC-51 header in §21 Decision Log "
        f"(expected literal {marker!r}).")
    # Decision-log block: take generous window after the entry header since
    # the entry spans multiple bullet sub-points covering Soul migration +
    # evaluate_gate authoritative path + implementation files + cascade
    # discipline + acceptance criteria. Cap at next "- **D-SPEC-" to scope
    # cleanly to this entry only.
    after = spec.split(marker, 1)[1]
    next_entry_idx = after.find("- **D-SPEC-")
    dspec_block = after if next_entry_idx == -1 else after[:next_entry_idx]
    for token in (
        "metabolism_worker",
        "evaluate_gate",
        "MetabolismShmReader",
        "Soul",
    ):
        assert token in dspec_block, (
            f"D-SPEC-51 entry missing required context: {token}")


# ═════════════════════════════════════════════════════════════════════
# Category 4 — Wiring drift
# ═════════════════════════════════════════════════════════════════════


def test_constants_toml_has_metabolism_state_bytes_and_schema():
    toml = (ROOT / "titan-docs" /
            "SPEC_titan_architecture_constants.toml").read_text()
    assert "METABOLISM_STATE_SCHEMA_VERSION" in toml
    assert "METABOLISM_STATE_MAX_BYTES" in toml
    # The constants were introduced in v1.7.2 (metabolism_worker SHIP);
    # the introduced_in attribution survives subsequent PATCH bumps.
    # Verify by checking the introduced_in row for the schema constant.
    assert 'introduced_in = "1.7.2"' in toml
    # SPEC version field exists (moves forward with each bump; not frozen here
    # per feedback_spec_changelog_mandatory.md + Changelog row check).
    assert "spec_version = " in toml


def test_kernel_proxy_aliases_includes_metabolism_proxy():
    from titan_hcl.core.kernel import KERNEL_PROXY_ALIASES
    assert "metabolism_proxy" in KERNEL_PROXY_ALIASES, (
        "metabolism_proxy missing from KERNEL_PROXY_ALIASES — "
        "RESPONSE messages won't route back to the proxy reply queue.")


def test_bus_has_three_new_event_constants():
    from titan_hcl import bus
    assert bus.METABOLIC_TIER_CHANGED == "METABOLIC_TIER_CHANGED"
    assert bus.GATE_DECISION_RECORDED == "GATE_DECISION_RECORDED"
    assert bus.METABOLIC_STATS_UPDATED == "METABOLIC_STATS_UPDATED"


def test_bus_specs_msg_specs_has_metabolism_events():
    from titan_hcl.bus_specs import MSG_SPECS
    assert "METABOLIC_TIER_CHANGED" in MSG_SPECS
    assert "GATE_DECISION_RECORDED" in MSG_SPECS
    assert "METABOLIC_STATS_UPDATED" in MSG_SPECS
    # Per SPEC §8.7 priority assignments
    assert MSG_SPECS["METABOLIC_TIER_CHANGED"].priority == 1
    assert MSG_SPECS["GATE_DECISION_RECORDED"].priority == 3
    assert MSG_SPECS["METABOLIC_STATS_UPDATED"].priority == 3
    # GATE_DECISION_RECORDED coalesces by feature
    assert MSG_SPECS["GATE_DECISION_RECORDED"].coalesce == ("feature",)


def test_phase_c_rpc_exemptions_has_metabolism_proxy_block():
    yaml_text = (ROOT / "titan-docs" /
                 "phase_c_rpc_exemptions.yaml").read_text()
    # Block header
    assert "metabolism_proxy — full MetabolismController surface" in yaml_text
    # Sites are formatted as `metabolism_proxy.py:<method>` in the site: field.
    assert "metabolism_proxy.py:evaluate_gate" in yaml_text
    assert "metabolism_proxy.py:get_current_state" in yaml_text
    assert "metabolism_proxy.py:can_afford" in yaml_text


def _wire_metabolism_body(src: str) -> str:
    """Extract just the body of def _wire_metabolism (until the next def)."""
    after = src.split("def _wire_metabolism")[1]
    # Body ends at the next top-level def (4-space indent + def).
    # Match the next `def ` at column 4 (method definition in the same class).
    idx = after.find("\n    def ")
    return after[:idx] if idx != -1 else after[:5000]


def test_plugin_py_no_longer_instantiates_metabolism_controller_inline():
    """No more `metabolism = MetabolismController(...)` in _wire_metabolism
    body. Only the proxy install + ModuleSpec registration remains."""
    src = (ROOT / "titan_hcl" / "core" / "plugin.py").read_text()
    wire_block = _wire_metabolism_body(src)
    assert "MetabolismController(" not in wire_block, (
        "plugin.py:_wire_metabolism still instantiates MetabolismController "
        "inline — extraction incomplete.")
    assert "MetabolismProxy" in wire_block, (
        "plugin.py:_wire_metabolism does not install MetabolismProxy.")


def test_soul_set_metabolism_is_noop_and_no_reverse_injection():
    """plugin.py no longer CALLs self.soul.set_metabolism.

    Match for the call shape (open paren after the name) so docstring
    mentions of the legacy injection don't trigger false positives.
    """
    import re
    # Match an actual code call: `self.soul.set_metabolism(...)` at the
    # start of a Python statement (preceded by whitespace + nothing else).
    # Excludes backtick-wrapped docstring references like
    # `` `self.soul.set_metabolism(metabolism)` reverse-injection``.
    plugin_src = (ROOT / "titan_hcl" / "core" / "plugin.py").read_text()
    for src_name, src in [("plugin.py", plugin_src)]:
        for line_no, line in enumerate(src.splitlines(), start=1):
            if "self.soul.set_metabolism(" not in line:
                continue
            stripped = line.lstrip()
            # Skip docstring / comment mentions wrapped in backticks.
            if "`self.soul.set_metabolism(" in line:
                continue
            if stripped.startswith("#"):
                continue
            pytest.fail(
                f"Soul.set_metabolism reverse-injection CALL still "
                f"present at {src_name}:{line_no}: {stripped!r}")

    # Soul's set_metabolism stub is still callable for back-compat
    # (returns None unconditionally).
    from titan_hcl.core.soul import SovereignSoul
    s = SovereignSoul.__new__(SovereignSoul)
    assert SovereignSoul.set_metabolism.__get__(s, SovereignSoul)(None) is None


def test_soul_uses_metabolism_shm_reader():
    """Soul's _check_nft_gate path uses MetabolismShmReader, not the
    legacy _metabolism reference."""
    src = (ROOT / "titan_hcl" / "core" / "soul.py").read_text()
    assert "MetabolismShmReader" in src
    assert "self._metabolism_reader" in src
    # The new gate check should reference can_use_feature on the reader.
    assert "can_use_feature(\"nfts\")" in src


# ═════════════════════════════════════════════════════════════════════
# Category 5 — Worker in-memory data structures
# ═════════════════════════════════════════════════════════════════════


def test_gate_decision_ring_bounded():
    """_GateDecisionRing is bounded at _RING_BUFFER_MAX entries."""
    from titan_hcl.modules.metabolism_worker import (
        _GateDecisionRing, _RING_BUFFER_MAX,
    )
    ring = _GateDecisionRing(max_size=_RING_BUFFER_MAX)
    for i in range(_RING_BUFFER_MAX + 50):
        ring.record({"i": i})
    assert len(ring) == _RING_BUFFER_MAX
    snap = ring.snapshot(limit=10)
    assert len(snap) == 10
    # Oldest evicted; newest preserved.
    assert snap[-1]["i"] == _RING_BUFFER_MAX + 49


def test_tier_history_window_gc():
    """_TierHistory drops entries older than the configured window."""
    from titan_hcl.modules.metabolism_worker import _TierHistory
    import time
    th = _TierHistory(window_s=0.05)  # 50ms window for the test
    th.record("HEALTHY", "CONSERVATIVE", time.time() - 1.0,
              balance_pct=0.5, gates_enforced=False)
    th.record("CONSERVATIVE", "SURVIVAL", time.time(),
              balance_pct=0.2, gates_enforced=True)
    snap = th.snapshot()
    # Stale (>50ms old) entry should be GC'd, recent one retained.
    assert len(snap) == 1
    assert snap[0]["tier_to"] == "SURVIVAL"


def test_last_balance_pct_is_property_not_method_regression_20260515():
    """Regression: MetabolismController._last_balance_pct is a @property —
    callsites must read it as `obj._last_balance_pct` (no parens). Calling
    `obj._last_balance_pct()` raises `'float' object is not callable`.

    Bug surfaced 2026-05-15 on T2+T3 after §4.J metabolism_worker shipped
    on 2026-05-14. Three production callsites had the trailing `()`:
      - titan_hcl/modules/metabolism_worker.py:568
      - titan_hcl/logic/metabolism_state_publisher.py:149
      - titan_hcl/core/kernel.py:2003
    Each fired `[MetabolismWorker] tier-refresh failed: 'float' object is
    not callable` every refresh cycle (~30s on T2, every boot on T3) until
    the parens were dropped. T1's logs were silent only because T1's
    metabolism_worker had not been reached during the observation window.
    """
    import re
    from pathlib import Path
    from titan_hcl.core.metabolism import MetabolismController

    # Invariant 1 — the descriptor on the class is a `property`.
    descriptor = MetabolismController.__dict__["_last_balance_pct"]
    assert isinstance(descriptor, property), (
        "_last_balance_pct must be a @property (not a method) — "
        "callers depend on the no-parens read form")

    # Invariant 2 — no production source file calls it with trailing `()`.
    repo_root = Path(__file__).resolve().parents[1]
    bad = re.compile(r"\._last_balance_pct\s*\(")
    offenders: list[str] = []
    for py in (repo_root / "titan_hcl").rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            if bad.search(line):
                offenders.append(
                    f"{py.relative_to(repo_root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Found callsites invoking _last_balance_pct as a method "
        "(call-with-parens). _last_balance_pct is a @property — drop the "
        "parens:\n  " + "\n  ".join(offenders))
