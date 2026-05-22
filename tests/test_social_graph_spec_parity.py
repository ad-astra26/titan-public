"""tests/test_social_graph_spec_parity.py — SPEC v1.7.1 parity assertions.

Per PLAN_microkernel_phase_c_social_graph_worker_extraction.md §7.5.

Static assertions that the SPEC documents what the code implements (and
vice versa). These are the regression gates for drift between SPEC,
TOML, RPC-exemption yaml, and the actual code surface — the kind of
silent drift that motivated `feedback_spec_changelog_mandatory.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


# ── SPEC v1.7.1 Changelog row ────────────────────────────────────────


def test_spec_changelog_has_v161_social_graph_worker_row():
    """Changelog row for v1.7.1 social_graph_worker landed."""
    spec = (ROOT / "titan-docs" / "SPEC_titan_architecture.md").read_text()
    assert "| 2026-05-14 | v1.7.1 (PATCH) |" in spec, (
        "Missing Changelog row for v1.7.1 PATCH bump (D-SPEC-50 / "
        "social_graph_worker extraction).")
    # Search the full changelog section — the v1.7.1 row ages downward as
    # newer rows are prepended (20-row cap), so a [:5000] slice misses it.
    chl = spec.split("## SPEC Changelog")[1].split("\n## ")[0]
    assert "social_graph_worker" in chl


# ── §1 Glossary entry ────────────────────────────────────────────────


def test_spec_glossary_has_social_graph_worker_row():
    spec = (ROOT / "titan-docs" / "SPEC_titan_architecture.md").read_text()
    glossary_block = spec.split("## §1 — Glossary")[1].split("## §2")[0]
    assert "**social_graph_worker**" in glossary_block, (
        "Missing §1 glossary row for social_graph_worker.")


# ── §7.1 SHM slot row ────────────────────────────────────────────────


def test_spec_71_has_social_graph_state_bin_row():
    spec = (ROOT / "titan-docs" / "SPEC_titan_architecture.md").read_text()
    assert "`social_graph_state.bin`" in spec, (
        "Missing §7.1 slot row for social_graph_state.bin")
    assert "SOCIAL_GRAPH_STATE_SCHEMA_VERSION" in spec
    assert "SOCIAL_GRAPH_STATE_MAX_BYTES" in spec


# ── §9.B Python tree block ───────────────────────────────────────────


def test_spec_9b_has_social_graph_worker_block():
    spec = (ROOT / "titan-docs" / "SPEC_titan_architecture.md").read_text()
    assert "#### social_graph_worker (Python L2 module" in spec, (
        "Missing §9.B social_graph_worker block.")


# ── §21 Decision Log entry ───────────────────────────────────────────


def test_spec_decision_log_has_d_spec_49():
    spec = (ROOT / "titan-docs" / "SPEC_titan_architecture.md").read_text()
    assert "**D-SPEC-50 " in spec, (
        "Missing §21 Decision Log entry D-SPEC-50 (social_graph_worker).")


# ── TOML constants ──────────────────────────────────────────────────


def test_toml_spec_version_is_current():
    """SPEC TOML version must monotonically advance.

    Originally checked == 1.6.1 (the interim version when social_graph_worker
    landed). At merge time D-SPEC-50 renumbered the bump to v1.7.1, and
    subsequent rFP §4.J extraction bumped to v1.7.2. The invariant the
    parity tests care about is that the TOML carries the SOCIAL_GRAPH_STATE
    constants — that is what test_toml_has_social_graph_state_constants
    asserts. The version-number assertion now just guards monotonicity.
    """
    toml = (
        ROOT / "titan-docs" / "SPEC_titan_architecture_constants.toml"
    ).read_text()
    # Must include a version line AT or AFTER the social_graph_worker
    # ship landing version (1.7.1).
    import re
    m = re.search(r'^spec_version = "([0-9]+\.[0-9]+\.[0-9]+)"',
                  toml, re.MULTILINE)
    assert m, "Missing spec_version = \"X.Y.Z\" line in TOML."
    ver = tuple(int(p) for p in m.group(1).split("."))
    assert ver >= (1, 7, 1), (
        f"TOML spec_version must be ≥ 1.7.1 (post-social_graph_worker "
        f"ship); got {m.group(1)}.")


def test_toml_has_social_graph_state_constants():
    toml = (
        ROOT / "titan-docs" / "SPEC_titan_architecture_constants.toml"
    ).read_text()
    assert "[constants.SOCIAL_GRAPH_STATE_SCHEMA_VERSION]" in toml
    assert "[constants.SOCIAL_GRAPH_STATE_MAX_BYTES]" in toml


def test_python_constants_regen_matches_toml():
    """Regen has been run — Python constants module has the 2 new constants."""
    from titan_hcl import _phase_c_constants as c
    assert hasattr(c, "SOCIAL_GRAPH_STATE_SCHEMA_VERSION")
    assert hasattr(c, "SOCIAL_GRAPH_STATE_MAX_BYTES")
    assert c.SOCIAL_GRAPH_STATE_SCHEMA_VERSION == 1
    assert c.SOCIAL_GRAPH_STATE_MAX_BYTES == 8192


# ── RPC exemptions YAML alignment ────────────────────────────────────


def test_rpc_exemptions_has_social_graph_proxy_entries():
    yaml = (
        ROOT / "titan-docs" / "phase_c_rpc_exemptions.yaml"
    ).read_text()
    # Spot-check: must have entries for the bug-fix method + a few core writes
    must_have = [
        "social_graph_proxy.py:record_interaction_async",
        "social_graph_proxy.py:get_or_create_user",
        "social_graph_proxy.py:should_engage",
        "social_graph_proxy.py:record_donation_async",
        "social_graph_proxy.py:record_inspiration_async",
        "social_graph_proxy.py:get_top_users_async",
    ]
    for site in must_have:
        assert site in yaml, (
            f"RPC exemptions missing required social_graph_proxy site: {site}")


def test_rpc_exemptions_g22_orphan_handlers_cleaned():
    """get_social_stats was previously orphan-allowlisted as deferred —
    must be removed now that mind_worker handler is deleted (G22
    violation closure).

    Note: get_or_create_user / should_engage / record_interaction REMAIN
    in orphan_handler_allowlist because the static G-RPC-4 scanner can't
    see dynamic-action calls via `SocialGraphProxy._work_rpc_sync` — same
    pattern as the historic mind_proxy entries. Each entry's rationale
    points to its corresponding social_graph_proxy work_rpc_sites entry.
    """
    yaml = (
        ROOT / "titan-docs" / "phase_c_rpc_exemptions.yaml"
    ).read_text()
    orphan_block = yaml.split("orphan_handler_allowlist:")[1] \
        if "orphan_handler_allowlist:" in yaml else ""
    # G22 closure: get_social_stats handler was REMOVED from mind_worker
    # in v1.7.1 (mind_worker._sense_taste reads SHM directly now).
    # The action MUST NOT appear in the allowlist anymore.
    assert '- action: "get_social_stats"' not in orphan_block, (
        "get_social_stats still in orphan_handler_allowlist — G22 "
        "violation should be CLOSED per D-SPEC-50 (mind_worker handler "
        "was REMOVED; stats now read via social_graph_state.bin SHM).")
    # get_or_create_user + should_engage entries SHOULD now point to
    # social_graph_proxy (not mind_proxy) — verify rationale text
    # references social_graph_proxy.
    for action in ("get_or_create_user", "should_engage"):
        entry_idx = orphan_block.find(f'- action: "{action}"')
        if entry_idx >= 0:
            # Look at the following ~200 chars for rationale
            tail = orphan_block[entry_idx:entry_idx + 400]
            assert "social_graph_proxy" in tail, (
                f"orphan allowlist entry for {action!r} must cite "
                f"social_graph_proxy as the dynamic-action caller "
                f"(was previously mind_proxy under the legacy alias).")


# ── Bus constants registration ───────────────────────────────────────


def test_bus_has_5_new_social_graph_events():
    from titan_hcl import bus
    for name in (
        "SOCIAL_GRAPH_READY",
        "SOCIAL_GRAPH_STATS_UPDATED",
        "SOCIAL_INTERACTION_RECORDED",
        "SOCIAL_DONATION_RECORDED",
        "SOCIAL_INSPIRATION_RECORDED",
    ):
        assert hasattr(bus, name), f"bus.py missing constant: {name}"
        assert getattr(bus, name) == name


def test_bus_specs_has_5_new_msg_specs():
    from titan_hcl.bus_specs import MSG_SPECS
    for name in (
        "SOCIAL_GRAPH_READY",
        "SOCIAL_GRAPH_STATS_UPDATED",
        "SOCIAL_INTERACTION_RECORDED",
        "SOCIAL_DONATION_RECORDED",
        "SOCIAL_INSPIRATION_RECORDED",
    ):
        assert name in MSG_SPECS, f"bus_specs.py missing BusMsgSpec: {name}"
    # READY is P1 (lifecycle, never drop)
    assert MSG_SPECS["SOCIAL_GRAPH_READY"].priority == 1
    # Stats notification is P3 with coalesce-by-type
    spec = MSG_SPECS["SOCIAL_GRAPH_STATS_UPDATED"]
    assert spec.priority == 3
    assert spec.coalesce == ("type",)


# ── Module registry ──────────────────────────────────────────────────


def test_kernel_proxy_aliases_includes_social_graph_proxy():
    from titan_hcl.core.kernel import KERNEL_PROXY_ALIASES
    assert "social_graph_proxy" in KERNEL_PROXY_ALIASES, (
        "social_graph_proxy must be in KERNEL_PROXY_ALIASES so the Rust "
        "broker routes RESPONSE messages back via SPEC v1.3.0 multi-name "
        "BUS_SUBSCRIBE.")


# ── Code-side alias deletion (G21 + clean architecture) ─────────────


def test_plugin_py_does_not_alias_social_graph_to_mind():
    """The legacy `_proxies['social_graph'] = _proxies['mind']` rot must be
    GONE from plugin.py (the sole registration site; legacy_core.py retired
    2026-05-21 / D-SPEC-106)."""
    plugin = (
        ROOT / "titan_hcl" / "core" / "plugin.py"
    ).read_text()
    import re
    pattern = re.compile(
        r"_proxies\[\"social_graph\"\]\s*=\s*self\._proxies\[\"mind\"\]")
    assert pattern.search(plugin) is None, (
        "Legacy MindProxy alias still present in plugin.py — must be "
        "DELETED per rFP §4.P + D-SPEC-50.")


def test_mind_worker_does_not_instantiate_social_graph():
    """SocialGraph(...) instantiation must be GONE from mind_worker.py."""
    mind = (
        ROOT / "titan_hcl" / "modules" / "mind_worker.py"
    ).read_text()
    import re
    # Match SocialGraph(db_path=...) instantiation (not the type-annotation
    # mentions, not the stats-reader class name)
    pattern = re.compile(r"\bSocialGraph\(db_path")
    assert pattern.search(mind) is None, (
        "mind_worker.py still instantiates SocialGraph — G21 single-writer "
        "violated (social_graph_worker now owns the DB).")


def test_plugin_py_does_not_instantiate_social_graph_in_wire_social():
    """Parent _wire_social must use proxy, not local SocialGraph."""
    plugin = (
        ROOT / "titan_hcl" / "core" / "plugin.py"
    ).read_text()
    # Locate _wire_social method body
    if "def _wire_social" in plugin:
        body = plugin.split("def _wire_social")[1].split("\n    def ")[0]
        assert "SocialGraph(db_path" not in body, (
            "plugin.py:_wire_social still instantiates SocialGraph — "
            "should pass self._proxies.get('social_graph') instead.")
