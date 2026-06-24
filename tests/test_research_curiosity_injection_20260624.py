"""RFP_titan_research_agent §1.4 step 2b — research-curiosity target injection.

Tests the two bounded new units: AgencyModule._pick_research_gap (anti-repeat
rotation over the gap pool) + agency_worker._make_research_gap_provider (throttled,
last-good snapshot read of `research_gaps`).
"""
from __future__ import annotations

import json
from collections import deque

from titan_hcl.logic.agency.module import AgencyModule
from titan_hcl.modules.agency_worker import _make_research_gap_provider


def _module_with(provider):
    m = AgencyModule.__new__(AgencyModule)
    m._research_gap_provider = provider
    m._last_research_gap = None
    return m


# ── _pick_research_gap ───────────────────────────────────────────────────────

def test_picks_top_gap_then_rotates_anti_repeat():
    gaps = [{"concept_id": "a"}, {"concept_id": "b"}, {"concept_id": "c"}]
    m = _module_with(lambda: gaps)
    first = m._pick_research_gap()
    assert first["concept_id"] == "a"          # highest salience first
    second = m._pick_research_gap()
    assert second["concept_id"] == "b"          # anti-repeat: skips 'a' just done


def test_single_gap_repeats_when_only_option():
    m = _module_with(lambda: [{"concept_id": "solo"}])
    assert m._pick_research_gap()["concept_id"] == "solo"
    assert m._pick_research_gap()["concept_id"] == "solo"  # only option → repeats


def test_empty_and_provider_error_return_none():
    assert _module_with(lambda: [])._pick_research_gap() is None

    def _boom():
        raise RuntimeError("snapshot gone")
    assert _module_with(_boom)._pick_research_gap() is None


# ── _make_research_gap_provider ──────────────────────────────────────────────

def test_provider_reads_research_gaps_from_snapshot(tmp_path):
    snap = tmp_path / "spine_snapshot.json"
    snap.write_text(json.dumps({
        "version": 1, "concepts": [],
        "research_gaps": [{"concept_id": "z", "name": "Zeta", "groundedness": 0.1}],
    }))
    provider = _make_research_gap_provider(str(snap), ttl_s=0.0)  # no throttle
    gaps = provider()
    assert len(gaps) == 1 and gaps[0]["concept_id"] == "z"
    assert gaps[0]["groundedness"] == 0.1


def test_provider_missing_file_returns_empty_last_good(tmp_path):
    provider = _make_research_gap_provider(str(tmp_path / "nope.json"), ttl_s=0.0)
    assert provider() == []   # missing → empty, no raise


def test_provider_throttle_caches_within_ttl(tmp_path):
    snap = tmp_path / "spine_snapshot.json"
    snap.write_text(json.dumps({"research_gaps": [{"concept_id": "a"}]}))
    provider = _make_research_gap_provider(str(snap), ttl_s=9999.0)
    assert provider()[0]["concept_id"] == "a"
    # overwrite the file; within TTL the provider must serve the cached value
    snap.write_text(json.dumps({"research_gaps": [{"concept_id": "b"}]}))
    assert provider()[0]["concept_id"] == "a"   # throttled → cached, not re-read


# ── allowlist survival (the bug the 100%-lock caught) ────────────────────────

def test_research_target_survives_build_result_allowlist():
    """_build_result's helper_params allowlist MUST keep `_research_target` — else
    2b's target never reaches the 3a scorer / 3c credit (silent drop)."""
    m = AgencyModule.__new__(AgencyModule)
    m._action_counter = 0
    m._history = deque()
    m._history_store = None
    m._action_timestamps = deque()
    m._retry_window_s = 60.0
    m._retry_history = deque()
    rt = {"concept_id": "z", "baseline_groundedness": 0.1, "domain_hint": "defi"}
    res = m._build_result(
        1, "research", "web_search",
        {"success": True, "result": "evidence"}, "reason", {},
        helper_params={"query": "q", "_research_target": rt, "code": "DROP"})
    hp = res["helper_params"]
    assert hp.get("_research_target") == rt   # survives the allowlist
    assert "code" not in hp                    # non-allowlisted still dropped
