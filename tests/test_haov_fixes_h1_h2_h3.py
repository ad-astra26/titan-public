"""Tests for HAOV architectural fixes H.1+H.2+H.3 (2026-04-28).

H.1 — Confirmation-bias OR-cheat removal in language/social/self_model verifiers.
H.2 — _haov_dest map. UPDATED 2026-05-29 (rFP_haov_efficacy_closure F4 rewire):
       the map now holds ONLY live specialist verifiers (language/knowledge/
       emot_cgn); the 7 dead "spirit" routes are removed and impasse hypotheses
       verify in-process. The original "covers all 9 consumers" contract routed
       most verifies into a void after the spirit retirement.
H.3 — Periodic HAOV test pump in cgn_worker decouples test-trigger from
       per-consumer CGN_TRANSITION outcome events; expires stuck active_test.

Pure source-inspection tests — verifier branches and pump logic live in long
worker loops; runtime exercise is covered by the existing 76-test
test_meta_service_session2.py + integration soak.

2026-05-17 cleanup: spirit_worker.py was retired to a heartbeat stub in D8-3
(commit 72f95a6b). The 5 verifier-branch tests that searched spirit_worker.py
for code that has been deleted are removed. The emot_cgn check is updated to
match the actual coding convention (constants, not literal strings).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGE_WORKER = REPO_ROOT / "titan_hcl/modules/language_worker.py"
SPIRIT_WORKER = REPO_ROOT / "titan_hcl/modules/spirit_worker.py"
EMOT_CGN_WORKER = REPO_ROOT / "titan_hcl/modules/emot_cgn_worker.py"
CGN_WORKER = REPO_ROOT / "titan_hcl/modules/cgn_worker.py"


def _read(p: Path) -> str:
    return p.read_text()


def _read_code_only(p: Path) -> str:
    """Strip comment-only lines so tests can search for absence of cheat
    strings without false positives from documentation references to them."""
    lines = []
    for ln in p.read_text().splitlines():
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(ln)
    return "\n".join(lines)


# ─────────────────── H.1 OR-cheat removal ───────────────────


def test_h1a_language_verifier_qual_or_cheat_removed():
    src = _read_code_only(LANGUAGE_WORKER)
    assert "or _qual > 0.5" not in src, (
        "H.1.a regression — language verifier _qual > 0.5 OR-cheat returned")
    assert "(_conf_a > _conf_b + 0.01) or _prod_ok" in src, (
        "Honest per-word delta clause must remain")


# test_h1b_social_verifier_nm_delta_or_cheat_removed REMOVED 2026-05-17:
# spirit_worker.py's social verifier branch was retired in D8-3 (commit
# 72f95a6b). No code to defend against the OR-cheat regression because the
# whole verifier is gone. If a future migration re-introduces a social
# verifier in its new home, a new test should target that new home directly.

# test_h1c_self_model_verifier_depth_cheat_removed REMOVED 2026-05-17:
# Same reason as test_h1b — self_model verifier was retired in D8-3.


# ─────────────────── H.2 dest map + new verifiers ───────────────────


def test_h2_haov_dest_map_module_level_constant():
    src = _read(CGN_WORKER)
    assert "_HAOV_DEST_MAP = {" in src, (
        "Module-level _HAOV_DEST_MAP constant must be declared")


def test_h2_dest_map_only_live_specialist_verifiers():
    """Post rewire (rFP_haov_efficacy_closure, 2026-05-29): _HAOV_DEST_MAP holds
    ONLY consumers with a live specialist CGN_HAOV_VERIFY_REQ handler
    (language/knowledge/emot_cgn) for their concept-grounding hypotheses. The
    prior map routed 7 consumers to a dead ``"spirit"`` dst (no subscriber after
    the 2026-05-16 spirit retirement) — that black-holed verification (F4). Those
    routes are removed; impasse-sourced hypotheses (the dominant class, all
    consumers) are verified in-process via `_local_haov_verify`."""
    src = _read(CGN_WORKER)
    m = re.search(r"_HAOV_DEST_MAP\s*=\s*\{([^}]+)\}", src, re.DOTALL)
    assert m, "_HAOV_DEST_MAP literal must be parseable"
    body = m.group(1)
    # The only live specialist-verifier destinations.
    assert '"language": "language"' in body
    assert '"knowledge": "knowledge"' in body
    assert '"emotional": "emot_cgn"' in body
    # No dead "spirit" route, no phantom "dreaming" consumer.
    assert '"spirit"' not in body, "dead spirit route must be removed (F4)"
    assert '"dreaming"' not in body, "phantom dreaming entry must be removed"


def test_h2_impasse_hypotheses_verified_in_process():
    """The rewire routes impasse-sourced hypotheses to in-process verification
    (no bus, no dead dst). Both the pump and the outcome-driven path must call
    `_local_haov_verify` before any bus emit."""
    src = _read(CGN_WORKER)
    assert "_local_haov_verify(cgn, tracker, consumer_name)" in src, (
        "pump must verify impasse hypotheses in-process")
    assert "_local_haov_verify(\n                                    cgn, _haov_tracker, _haov_consumer)" in src \
        or "_local_haov_verify(cgn, _haov_tracker, _haov_consumer)" in src, (
        "outcome path must verify impasse hypotheses in-process")


def test_h2_emot_cgn_handles_emotional_verify():
    src = _read(EMOT_CGN_WORKER)
    # 2026-05-17 — pattern fixed: actual code uses the bus.* constant
    # (`msg_type == bus.CGN_HAOV_VERIFY_REQ`), not a literal-string
    # comparison. Adjust the assertion to match real codebase convention.
    assert "msg_type == bus.CGN_HAOV_VERIFY_REQ" in src, (
        "emot_cgn_worker must subscribe to CGN_HAOV_VERIFY_REQ")
    assert '_haov_consumer == "emotional"' in src, (
        "emot_cgn_worker must handle the emotional consumer specifically")
    # H.2 emotional uses cluster_conf + V_dominant deltas (no OR-cheat)
    assert "abs(_conf_a - _conf_b)" in src, (
        "Emotional verifier must use cluster_conf delta")
    assert "abs(_v_dom_a - _v_dom_b)" in src, (
        "Emotional verifier must use V_dominant delta")


# test_h2_spirit_handles_meta_verify REMOVED 2026-05-17:
# spirit_worker.py was retired to a heartbeat stub in D8-3 (commit
# 72f95a6b). Meta-verifier branch deleted with it. If a future migration
# re-homes meta-verification (e.g. inside cognitive_worker), a new test
# should target that new owner directly.


# test_h2_spirit_handles_reasoning_strategy_verify REMOVED 2026-05-17:
# Same reason — reasoning_strategy verifier branch was inside the now-
# retired spirit_worker.py body.


# test_h2_spirit_handles_dreaming_verify REMOVED 2026-05-17:
# Same reason — dreaming verifier branch was inside the now-retired
# spirit_worker.py body.


def test_h2_emot_cgn_verifier_sends_response():
    """The only verifier of these four (emot, meta, reasoning_strategy,
    dreaming) that survived D8-3 retirement is the emotional one in
    emot_cgn_worker. Reduced from the original test_h2_all_new_verifiers_
    send_response to assert only the still-extant case."""
    emot = _read(EMOT_CGN_WORKER)
    assert '"consumer": "emotional"' in emot, (
        "emot_cgn_worker must emit CGN_HAOV_VERIFY_RSP with consumer='emotional'")


# ─────────────────── H.3 periodic test pump ───────────────────


def test_h3_pump_function_exists():
    src = _read(CGN_WORKER)
    assert "def _run_haov_pump(" in src, (
        "_run_haov_pump module-level function must be declared")


def test_h3_pump_walks_all_trackers():
    src = _read(CGN_WORKER)
    # Pump iterates _haov_trackers
    assert "for consumer_name, tracker in cgn._haov_trackers.items()" in src, (
        "Pump must iterate all registered HAOV trackers")


def test_h3_pump_expires_stuck_active_tests():
    src = _read(CGN_WORKER)
    # Stuck active_test cleanup logic
    assert "stuck_timeout_s" in src, (
        "Pump must accept stuck_timeout_s parameter")
    assert "tracker._active_test = None" in src, (
        "Pump must clear stuck active_test entries")
    assert "HAOV_ACTIVE_TEST_TIMEOUT_S" in src, (
        "Per-config stuck-test timeout constant must be declared")


def test_h3_pump_calls_select_test_independent_of_inbound_messages():
    src = _read(CGN_WORKER)
    assert "tracker.select_test({" in src, (
        "Pump must call select_test")
    # Must run before recv_queue.get to fire on idle too
    while_idx = src.find("while True:")
    recv_idx = src.find("recv_queue.get(timeout=_heartbeat_interval)", while_idx)
    pump_idx = src.find("_last_haov_pump_ts >= HAOV_TEST_PUMP_INTERVAL_S",
                        while_idx)
    assert while_idx > 0 and pump_idx > 0 and recv_idx > 0, (
        "Loop structure must be parseable")
    assert pump_idx < recv_idx, (
        "Pump must fire BEFORE recv_queue.get so it runs even when idle")


def test_h3_pump_uses_shared_dest_map():
    src = _read(CGN_WORKER)
    # Extract pump body — bounded by `def _run_haov_pump(` on one side
    # and the next top-level `def ` on the other.
    m = re.search(
        r"def _run_haov_pump\([^)]*\):.*?(?=\ndef [a-zA-Z_])",
        src, re.DOTALL)
    assert m, "Pump function body must be parseable"
    body = m.group(0)
    assert "_HAOV_DEST_MAP" in body, (
        "Pump must use the shared _HAOV_DEST_MAP constant (DRY with handler)")


def test_h3_pump_timestamps_active_test_on_send():
    src = _read(CGN_WORKER)
    # Both the inline handler path and the pump must stamp ts on active_test
    # Look for at least 2 occurrences of timestamp assignment
    count = src.count('tracker._active_test["ts"]') + src.count(
        '_haov_tracker._active_test["ts"]')
    assert count >= 2, (
        "Both inline CGN_TRANSITION handler AND pump must stamp ts on "
        f"active_test for stuck-test recovery (count={count})")


def test_h3_pump_config_keys_present():
    src = _read(CGN_WORKER)
    assert 'config.get("haov_test_pump_interval_s"' in src
    assert 'config.get("haov_active_test_timeout_s"' in src


# ─────────────────── Cross-cutting (H.1+H.2+H.3) ───────────────────


def test_haov_no_or_cheats_remain_in_any_verifier():
    """Defect 2 cross-check — no system-wide baseline OR-clauses anywhere.
    (D-SPEC-116: spirit_worker.py deleted — it was a heartbeat stub with no
    verifiers; the OR-cheat guard now covers the live verifier hosts.)"""
    lang = _read_code_only(LANGUAGE_WORKER)
    forbidden = [
        "or _qual > 0.5",          # language baseline cheat
        "or _nm_delta > 0.02",     # social baseline cheat
        "or (_depth_a > _depth_b)",  # self_model monotonic-counter cheat
    ]
    for f in forbidden:
        assert f not in lang, (
            f"OR-cheat regression: {f!r} must not appear in any verifier")
