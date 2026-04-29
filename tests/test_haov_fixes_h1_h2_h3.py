"""Tests for HAOV architectural fixes H.1+H.2+H.3 (2026-04-28).

H.1 — Confirmation-bias OR-cheat removal in language/social/self_model verifiers.
H.2 — _haov_dest map covers all 9 registered consumers; 4 new verifier branches.
H.3 — Periodic HAOV test pump in cgn_worker decouples test-trigger from
       per-consumer CGN_TRANSITION outcome events; expires stuck active_test.

Pure source-inspection tests — verifier branches and pump logic live in long
worker loops; runtime exercise is covered by the existing 76-test
test_meta_service_session2.py + integration soak.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGE_WORKER = REPO_ROOT / "titan_plugin/modules/language_worker.py"
SPIRIT_WORKER = REPO_ROOT / "titan_plugin/modules/spirit_worker.py"
EMOT_CGN_WORKER = REPO_ROOT / "titan_plugin/modules/emot_cgn_worker.py"
CGN_WORKER = REPO_ROOT / "titan_plugin/modules/cgn_worker.py"


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


def test_h1b_social_verifier_nm_delta_or_cheat_removed():
    src = _read_code_only(SPIRIT_WORKER)
    assert "or _nm_delta > 0.02" not in src, (
        "H.1.b regression — social verifier nm_delta OR-cheat returned")
    assert "_confirmed = (_q_a > _q_b + 0.01)" in src, (
        "Honest social quality delta must remain")


def test_h1c_self_model_verifier_depth_cheat_removed():
    src = _read_code_only(SPIRIT_WORKER)
    assert "or (_depth_a > _depth_b)" not in src, (
        "H.1.c regression — self_model i_depth monotonic-counter cheat returned")
    assert "_confirmed = (_acc_a > _acc_b + 0.01)" in src, (
        "Honest accuracy delta must remain in self_model verifier")


# ─────────────────── H.2 dest map + new verifiers ───────────────────


def test_h2_haov_dest_map_module_level_constant():
    src = _read(CGN_WORKER)
    assert "_HAOV_DEST_MAP = {" in src, (
        "Module-level _HAOV_DEST_MAP constant must be declared")


def test_h2_dest_map_routes_all_9_registered_consumers():
    """All 9 consumers registered in cgn_worker pre-register block must
    have a routing entry."""
    src = _read(CGN_WORKER)
    # Extract the _HAOV_DEST_MAP literal
    m = re.search(r"_HAOV_DEST_MAP\s*=\s*\{([^}]+)\}", src, re.DOTALL)
    assert m, "_HAOV_DEST_MAP literal must be parseable"
    body = m.group(1)
    expected_consumers = [
        "language", "social", "reasoning", "knowledge", "coding",
        "self_model", "emotional", "meta", "reasoning_strategy", "dreaming",
    ]
    for c in expected_consumers:
        assert f'"{c}":' in body, (
            f"Consumer '{c}' missing from _HAOV_DEST_MAP")


def test_h2_emot_cgn_handles_emotional_verify():
    src = _read(EMOT_CGN_WORKER)
    assert 'msg_type == "CGN_HAOV_VERIFY_REQ"' in src, (
        "emot_cgn_worker must subscribe to CGN_HAOV_VERIFY_REQ")
    assert '_haov_consumer == "emotional"' in src, (
        "emot_cgn_worker must handle the emotional consumer specifically")
    # H.2 emotional uses cluster_conf + V_dominant deltas (no OR-cheat)
    assert "abs(_conf_a - _conf_b)" in src, (
        "Emotional verifier must use cluster_conf delta")
    assert "abs(_v_dom_a - _v_dom_b)" in src, (
        "Emotional verifier must use V_dominant delta")


def test_h2_spirit_handles_meta_verify():
    src = _read(SPIRIT_WORKER)
    assert '_haov_consumer == "meta"' in src, (
        "spirit_worker must add meta verifier branch")
    assert "meta_commit_rate" in src, (
        "Meta verifier must check commit_rate delta")


def test_h2_spirit_handles_reasoning_strategy_verify():
    src = _read(SPIRIT_WORKER)
    assert '_haov_consumer == "reasoning_strategy"' in src, (
        "spirit_worker must add reasoning_strategy verifier branch")
    assert "strategy_commit_rate" in src
    assert "strategy_total_chains" in src, (
        "reasoning_strategy verifier must check both rate AND new chains "
        "(AND, not OR — strict to avoid stale-data confirmation)")


def test_h2_spirit_handles_dreaming_verify():
    src = _read(SPIRIT_WORKER)
    assert '_haov_consumer == "dreaming"' in src, (
        "spirit_worker must add dreaming verifier branch")
    assert "dream_cycle_count" in src
    assert "epochs_since_dream" in src


def test_h2_all_new_verifiers_send_response():
    spirit = _read(SPIRIT_WORKER)
    emot = _read(EMOT_CGN_WORKER)
    # Each new verifier branch must emit CGN_HAOV_VERIFY_RSP back to cgn
    for consumer in ("meta", "reasoning_strategy", "dreaming"):
        # Look for "consumer": "<name>" inside CGN_HAOV_VERIFY_RSP payload
        assert f'"consumer": "{consumer}"' in spirit, (
            f"spirit_worker must emit CGN_HAOV_VERIFY_RSP with "
            f"consumer={consumer!r}")
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
    """Defect 2 cross-check — no system-wide baseline OR-clauses anywhere."""
    spirit = _read_code_only(SPIRIT_WORKER)
    lang = _read_code_only(LANGUAGE_WORKER)
    forbidden = [
        "or _qual > 0.5",          # language baseline cheat
        "or _nm_delta > 0.02",     # social baseline cheat
        "or (_depth_a > _depth_b)",  # self_model monotonic-counter cheat
    ]
    for f in forbidden:
        assert f not in lang and f not in spirit, (
            f"OR-cheat regression: {f!r} must not appear in any verifier")
