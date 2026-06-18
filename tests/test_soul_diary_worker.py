"""Tests for the soul_diary_worker pipeline logic (RFP §1.0 P1).

Exercises the self-contained compose (own LLM + OVG), the gather, and the
full _author_cycle_entry / _maybe_close_cycle_and_author flow (authored persist ·
OVG soft-fail to minimal · circadian-cycle latch skip · CYCLE_CLOSED publish;
RFP presence §7.C INV-SD-5 swap) with fakes — no bus, no real LLM/chronicle write.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

import titan_hcl.bus as bus
from titan_hcl.core import soul_diary_chain
from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
from titan_hcl.modules import soul_diary_worker as sdw


@pytest.fixture(autouse=True)
def _no_art_render(monkeypatch):
    """P7 art render is a separate pipeline step (tested in test_soul_diary_p7_art);
    no-op it here so these P1/P2 pipeline tests don't write image files."""
    monkeypatch.setattr(sdw, "_render_art", lambda *a, **k: None)


class _FakeProvider:
    def __init__(self, text):
        self._text = text

    async def complete(self, prompt, *, system="", **kw):
        return self._text


class _FakeQueue:
    """Collects bus messages the worker would publish (no real bus)."""

    def __init__(self):
        self.msgs = []

    def put(self, msg):
        self.msgs.append(msg)


def _verifier(passed):
    v = MagicMock()
    v.verify_safety.return_value = MagicMock(passed=passed)
    return v


_PROMPTS = {"system_prompt": "s", "user_prompt": "u"}


def test_resolve_provider_name_reads_canonical_key():
    # BUGFIX 2026-06-10: the canonical config key is `inference_provider`; reading
    # `provider` (the old bug) always fell back to 402-dead venice → minimal entries.
    assert sdw._resolve_provider_name(
        {"inference_provider": "ollama_cloud"}) == "ollama_cloud"
    # inference_provider wins over a legacy `provider` alias.
    assert sdw._resolve_provider_name(
        {"inference_provider": "ollama_cloud", "provider": "venice"}) == "ollama_cloud"
    # legacy alias still honored when canonical absent.
    assert sdw._resolve_provider_name({"provider": "custom"}) == "custom"
    # last-resort default = the fleet default ollama_cloud (NOT venice).
    assert sdw._resolve_provider_name({}) == "ollama_cloud"


def test_compose_pass():
    text, ok = asyncio.run(sdw._compose_diary(_FakeProvider("authored entry"),
                                              _verifier(True), _PROMPTS))
    assert text == "authored entry" and ok is True


def test_compose_ovg_block():
    text, ok = asyncio.run(sdw._compose_diary(_FakeProvider("bad"),
                                              _verifier(False), _PROMPTS))
    assert text == "bad" and ok is False


def test_compose_empty_text():
    text, ok = asyncio.run(sdw._compose_diary(_FakeProvider("   "),
                                              _verifier(True), _PROMPTS))
    assert text == "" and ok is False


def test_compose_no_provider():
    text, ok = asyncio.run(sdw._compose_diary(None, _verifier(True), _PROMPTS))
    assert text == "" and ok is False


def _fake_shm():
    """A MagicMock ShmReaderBank returning the REAL payload shapes the gather
    helpers consume (so the test proves the real sources populate, not skeletons
    with empty inputs — DONE-contract / INV-NO-STUBS)."""
    shm = MagicMock()
    # neuromod_state.bin real shape: {"modulators": {name: {"level": x}}, ...}
    shm.read_neuromod.return_value = {
        "modulators": {"dopamine": {"level": 0.6}, "cortisol": {"level": 0.2},
                       "serotonin": {"level": 0.4}},
        "age_seconds": 1.0, "seq": 9}
    # mind_state.bin — where valence/mood actually live (NOT neuromod_state).
    shm.read_mind_state.return_value = {
        "mood_label": "Curious", "mood_valence": 0.62, "mood_intensity": 0.55}
    shm.read_memory_state.return_value = {
        "persistent_count": 412, "mempool_size": 8, "effective_nodes_24h": 73.4,
        "high_quality_count": 51, "learning_velocity": 0.83,
        "kg_node_count": 980, "kg_edge_count": 2304}
    shm.read_social_graph_state.return_value = {
        "users": 37, "edges": 64, "inspirations": 3, "donations": 1,
        "engagement_ledger_today": 12}
    shm.read_social_perception_state.return_value = {
        "sentiment_ema": 0.18, "interaction_rate": 0.4}
    shm.read_body_state.return_value = {
        "sol_balance": 0.00987, "sol_norm": 0.21, "anchor_fresh": 0.9}
    shm.read_network_state.return_value = {"balance_sol": 0.00987}
    shm.read_metabolism_state.return_value = {"tier": "HEALTHY", "balance_pct": 0.74}
    return shm


def test_gather_felt_reads_nested_modulators_and_mind():
    """BUGFIX 2026-06-10: felt reads read_neuromod()['modulators'][n]['level'] for
    the dominant + read_mind_state() for valence/mood — the old code read the wrong
    (flat) shape so felt NEVER surfaced live."""
    felt = sdw._gather_felt(_fake_shm())
    assert felt["dominant"] == "dopamine"          # highest modulator level
    assert felt["valence"] == 0.62                 # from mind_state, not neuromod
    assert felt["mood_label"] == "Curious"
    assert felt["intensity"] == 0.55
    assert felt["neuromod_levels"]["serotonin"] == 0.4
    # None-shm soft-fails to {} (never crashes the gather).
    assert sdw._gather_felt(None) == {}


def test_gather_memory_social_onchain_real_fields():
    """Each de-stubbed §1.1 source maps its REAL SHM field names (no empties)."""
    shm = _fake_shm()
    mem = sdw._gather_memory(shm)
    assert mem["persistent"] == 412 and mem["high_quality"] == 51
    assert mem["kg_nodes"] == 980 and mem["kg_edges"] == 2304
    soc = sdw._gather_social(shm)
    assert soc["users"] == 37 and soc["engagement_today"] == 12
    assert soc["sentiment_ema"] == 0.18
    oc = sdw._gather_onchain(shm)
    assert oc["sol_balance"] == 0.00987 and oc["metabolic_tier"] == "HEALTHY"
    assert oc["balance_pct"] == 0.74
    # all three soft-fail to {} when SHM is absent.
    assert (sdw._gather_memory(None), sdw._gather_social(None),
            sdw._gather_onchain(None)) == ({}, {}, {})


def test_gather_onchain_prefers_authoritative_sol_and_guards_sentinel():
    """SOL = the authoritative RPC balance (network_state.balance_sol), NOT the
    body_state cache that reads 0.0 in the boot-grace window; and the out-of-range
    balance_pct boot sentinel (-1.0) is dropped, never rendered as '-100% energy'.
    (Both surfaced live on the 2026-06-09 verify entry, 2026-06-10.)"""
    shm = MagicMock()
    shm.read_body_state.return_value = {"sol_balance": 0.0, "sol_norm": 0.0,
                                        "anchor_fresh": 0.0}            # lagging cache
    shm.read_network_state.return_value = {"balance_sol": 0.009860001}  # authoritative
    shm.read_metabolism_state.return_value = {"tier": "HEALTHY", "balance_pct": -1.0}
    oc = sdw._gather_onchain(shm)
    assert oc["sol_balance"] == 0.00986          # network_state wins over body 0.0
    assert "balance_pct" not in oc               # -1.0 sentinel guarded out
    assert oc["metabolic_tier"] == "HEALTHY"
    # when the RPC balance is genuinely absent, fall back to a non-zero body cache.
    shm.read_network_state.return_value = {}
    shm.read_body_state.return_value = {"sol_balance": 0.5, "sol_norm": 0.1}
    assert sdw._gather_onchain(shm)["sol_balance"] == 0.5
    # a sane balance_pct still renders.
    shm.read_metabolism_state.return_value = {"tier": "LOW", "balance_pct": 0.42}
    assert sdw._gather_onchain(shm)["balance_pct"] == 0.42


def test_gather_engrams_window_filters_cycle_span(tmp_path, monkeypatch):
    """engrams reads data/spine_snapshot.json and returns ONLY names whose latest
    version's created_at lies in the wall-clock window [start_ts, end_ts) — the
    closed circadian cycle's span (RFP §7.C; latest-version per concept_id,
    newest-first)."""
    import json
    import time
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    end = time.time()
    start = end - 86400.0           # the cycle's wall-clock span (≈ a day)
    mid = (start + end) / 2.0
    before = start - 86400          # prior cycle — excluded
    after = end + 3600              # after the cycle close — excluded
    snap = {"version": 1, "concepts": [
        {"concept_id": "c1", "version": 1, "name": "Old Idea", "created_at": before},
        {"concept_id": "c2", "version": 1, "name": "Glacier Ecosystems",
         "created_at": mid - 100},
        {"concept_id": "c2", "version": 2, "name": "Glacier Ecosystems v2",
         "created_at": mid},        # latest version in-window → this name wins
        {"concept_id": "c3", "version": 1, "name": "Future Idea", "created_at": after},
    ]}
    (tmp_path / "spine_snapshot.json").write_text(json.dumps(snap))
    names = sdw._gather_engrams_window(start, end)
    assert names == ["Glacier Ecosystems v2"]       # only the in-window latest version
    # missing snapshot soft-fails to [].
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path / "nope"))
    assert sdw._gather_engrams_window(start, end) == []


def test_gather_bundle_wires_all_sources_no_stubs(tmp_path, monkeypatch):
    """The whole gather: NO source is a hardcoded empty — every §1.1 element
    populates from its real surface (the exact stub this session removed)."""
    import json
    import time
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    end = time.time()
    start = end - 86400.0
    (tmp_path / "spine_snapshot.json").write_text(json.dumps({"concepts": [
        {"concept_id": "c1", "version": 1, "name": "Self-Refactor Patterns",
         "created_at": (start + end) / 2.0}]}))
    orch = SoulDiaryOrchestrator()
    bundle = sdw._gather_bundle({"promoted": 7, "pruned": 2, "epoch": 5},
                                _fake_shm(), orch, window_ts=(start, end))
    assert bundle["outcome"]["promoted"] == 7
    assert bundle["engrams_today"] == ["Self-Refactor Patterns"]
    assert bundle["memory"]["persistent"] == 412
    assert bundle["social"]["users"] == 37
    assert bundle["onchain"]["sol_balance"] == 0.00987
    assert bundle["felt"]["dominant"] == "dopamine"
    # the rendered grounded prompt actually carries these real facts.
    prompt = orch.build_compose_prompts(bundle)["user_prompt"]
    assert "Self-Refactor Patterns" in prompt
    assert "412 persistent" in prompt
    assert "0.0099 SOL" in prompt
    assert "dopamine" in prompt


def _orch(tmp_path):
    return SoulDiaryOrchestrator(state_path=str(tmp_path / "state.json"),
                                 ledger_path=str(tmp_path / "chain.json"))


_WINDOW = (1000.0, 2000.0)   # a closed cycle's wall-clock span (start_ts, end_ts)


def test_author_persists_authored_text(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    seen = {}
    monkeypatch.setattr(orch, "persist", lambda text, **kw: seen.update(text=text))
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o, **kw: o.build_bundle(
        sovereignty={"s": 0.58, "replies": 22}, outcome={"promoted": 7, "pruned": 2},
        felt={}, engrams_today=["Glacier"], memory={}, social={}, onchain={}))
    ok = sdw._author_cycle_entry({"promoted": 7}, cycle_id=4, window_ts=_WINDOW,
                                 orchestrator=orch,
                                 provider=_FakeProvider("Today I reflected."),
                                 verifier=_verifier(True), shm_reader=None,
                                 send_queue=_FakeQueue(), src="soul_diary")
    assert ok is True
    assert seen["text"] == "Today I reflected."          # authored text persisted
    assert len(soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))) == 1
    assert orch.should_author_cycle(4) is False           # cycle latch closed


def test_author_softfails_to_minimal_on_ovg_block(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    seen = {}
    monkeypatch.setattr(orch, "persist", lambda text, **kw: seen.update(text=text))
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o, **kw: o.build_bundle(
        sovereignty={"s": 0.5, "replies": 1}, outcome={"promoted": 3, "pruned": 0},
        felt={}, engrams_today=[], memory={}, social={}, onchain={}))
    ok = sdw._author_cycle_entry({}, cycle_id=1, window_ts=_WINDOW, orchestrator=orch,
                                 provider=_FakeProvider("hallucinated text"),
                                 verifier=_verifier(False), shm_reader=None,
                                 send_queue=_FakeQueue(), src="soul_diary")
    assert ok is True
    assert "hallucinated" not in seen["text"]   # OVG-blocked text NOT persisted
    assert "0.50" in seen["text"]               # minimal grounded entry instead


def test_author_latch_skips_second_same_cycle(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    orch.mark_authored_cycle(4)  # already authored this cycle
    called = {"persist": False}
    monkeypatch.setattr(orch, "persist", lambda *a, **k: called.update(persist=True))
    ok = sdw._author_cycle_entry({}, cycle_id=4, window_ts=_WINDOW, orchestrator=orch,
                                 provider=_FakeProvider("x"),
                                 verifier=_verifier(True), shm_reader=None,
                                 send_queue=_FakeQueue(), src="soul_diary")
    assert ok is True and called["persist"] is False  # latch closed → no second entry


def test_author_noop_cycle_latches_without_entry(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    called = {"persist": False}
    monkeypatch.setattr(orch, "persist", lambda *a, **k: called.update(persist=True))
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o, **kw: o.build_bundle(
        sovereignty={"replies": 0}, outcome={"promoted": 0, "pruned": 0},
        felt={}, engrams_today=[], memory={}, social={}, onchain={}))
    ok = sdw._author_cycle_entry({}, cycle_id=3, window_ts=_WINDOW, orchestrator=orch,
                                 provider=_FakeProvider("x"),
                                 verifier=_verifier(True), shm_reader=None,
                                 send_queue=_FakeQueue(), src="soul_diary")
    assert ok is True and called["persist"] is False      # no-op cycle → no entry
    assert orch.should_author_cycle(3) is False           # but latched (cycle)


class _FakeCounter:
    """Stands in for CircadianCycleCounter — `latch_if_trough` returns the new
    cycle_id (a latch fired) or None (no boundary)."""

    def __init__(self, fires_to=None, start_epoch=1000, start_ts=500.0):
        self._fires_to = fires_to
        self.cycle_start_epoch = start_epoch
        self.cycle_start_ts = start_ts

    def latch_if_trough(self, phase, age):
        return self._fires_to


class _FakeAge:
    def __init__(self, epochs):
        self._e = epochs

    def get_age_epochs(self):
        return self._e


def test_latch_fires_publishes_cycle_closed_and_authors(tmp_path, monkeypatch):
    """RFP §7.C — when the trough latch fires, the worker publishes CYCLE_CLOSED for
    the just-closed cycle (→ synthesis presence seal) AND authors that cycle's diary."""
    orch = _orch(tmp_path)
    monkeypatch.setattr(orch, "persist", lambda text, **kw: None)
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o, **kw: o.build_bundle(
        sovereignty={"s": 0.5, "replies": 5}, outcome={"promoted": 1, "pruned": 0},
        felt={}, engrams_today=["X"], memory={}, social={}, onchain={}))
    monkeypatch.setattr(sdw, "get_circadian_phase", lambda: 0.1)   # trough
    counter = _FakeCounter(fires_to=8, start_epoch=1000, start_ts=500.0)
    q = _FakeQueue()
    ok = sdw._maybe_close_cycle_and_author(
        {"promoted": 1}, counter=counter, age_reader=_FakeAge(2000),
        orchestrator=orch, provider=_FakeProvider("Reflecting."),
        verifier=_verifier(True), shm_reader=None, send_queue=q, src="soul_diary",
        clock=lambda: 9999.0)
    assert ok is True
    # CYCLE_CLOSED for the CLOSED cycle (new 8 → closed 7), Titan-time range + ts view
    cc = next(m for m in q.msgs if m["type"] == bus.CYCLE_CLOSED)
    assert cc["dst"] == "synthesis"
    assert cc["payload"] == {"cycle_id": 7, "start_epoch": 1000, "end_epoch": 2000,
                             "start_ts": 500.0, "ts_utc": 9999.0}
    # the diary authored the closed cycle (anchor TX emitted + cycle latched)
    assert any(m["type"] == bus.TIMECHAIN_COMMIT for m in q.msgs)
    assert orch.should_author_cycle(7) is False


def test_non_trough_meditation_noops(tmp_path, monkeypatch):
    """A non-boundary meditation: the latch does not fire → no CYCLE_CLOSED, no diary."""
    orch = _orch(tmp_path)
    called = {"persist": False}
    monkeypatch.setattr(orch, "persist", lambda *a, **k: called.update(persist=True))
    monkeypatch.setattr(sdw, "get_circadian_phase", lambda: 0.7)   # daytime
    q = _FakeQueue()
    ok = sdw._maybe_close_cycle_and_author(
        {}, counter=_FakeCounter(fires_to=None), age_reader=_FakeAge(2000),
        orchestrator=orch, provider=_FakeProvider("x"), verifier=_verifier(True),
        shm_reader=None, send_queue=q, src="soul_diary")
    assert ok is True
    assert called["persist"] is False                                  # no author
    assert not any(m["type"] == bus.CYCLE_CLOSED for m in q.msgs)       # no seal trigger


# ── P2 — ENRICH (⑥) + ANCHOR (⑦) ────────────────────────────────────────────


def test_p2_enrich_and_anchor_emitted(tmp_path, monkeypatch):
    """After persist+hash, the worker enriches synthesis (mempool add, self
    domain) and anchors the cumulative hash on the main chain — both as bus
    publishes whose shape matches the consuming workers (§1.0 ⑥⑦)."""
    orch = _orch(tmp_path)
    monkeypatch.setattr(orch, "persist", lambda text, **kw: None)
    monkeypatch.setattr(sdw, "_gather_bundle", lambda p, s, o, **kw: o.build_bundle(
        sovereignty={"s": 0.58, "replies": 22}, outcome={"promoted": 7, "pruned": 2},
        felt={}, engrams_today=["Glacier"], memory={}, social={}, onchain={}))
    q = _FakeQueue()
    ok = sdw._author_cycle_entry({"promoted": 7}, cycle_id=2, window_ts=_WINDOW,
                                 orchestrator=orch,
                                 provider=_FakeProvider("Today I reflected."),
                                 verifier=_verifier(True), shm_reader=None,
                                 send_queue=q, src="soul_diary")
    assert ok is True
    types = [m["type"] for m in q.msgs]
    assert bus.MEMORY_MEMPOOL_ADD in types        # ⑥ enrich
    assert bus.TIMECHAIN_COMMIT in types          # ⑦ anchor

    enrich = next(m for m in q.msgs if m["type"] == bus.MEMORY_MEMPOOL_ADD)
    assert enrich["dst"] == "memory"
    assert enrich["payload"]["source"] == "soul_diary"
    assert enrich["payload"]["agent_response"] == "Today I reflected."
    assert "domain:self" in enrich["payload"]["tags"]

    anchor = next(m for m in q.msgs if m["type"] == bus.TIMECHAIN_COMMIT)
    assert anchor["dst"] == "timechain"
    p = anchor["payload"]
    assert p["fork"] == "main"                    # FORK_MAIN — SELF journey, not ACT-R
    assert p["thought_type"] == "dailyDiary"
    assert p["source"] == "soul_diary"
    # the anchored hashes match the ledger head (triple-anchor parity, INV-SD-10)
    head = soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))[-1]
    assert p["content"]["cumulative_hash"] == head["cumulative_hash"]
    assert p["content"]["entry_hash"] == head["entry_hash"]


def test_p2_anchor_clears_main_fork_pot():
    """The anchor's significance/novelty/coherence + empty neuromods clear the
    main-fork PoT threshold (default 0.20) — the dailyDiary block is ADMITTED,
    not silently REJECTED. Guards the exact values _anchor_main_chain emits."""
    from titan_hcl.logic.proof_of_thought import PoTValidator
    pot = PoTValidator().create_pot(
        chi_available=0.5, metabolic_drain=0.0, attention=0.5, i_confidence=0.5,
        chi_coherence=0.3, neuromods={},          # empty → tonic 0.5 baseline
        novelty=0.5, significance=0.85, coherence=0.8,
        source="soul_diary", thought_type="dailyDiary", fork_name="main",
    )
    assert pot.valid is True, (
        f"main-fork PoT rejected the diary anchor: score={pot.pot_score} "
        f"threshold={pot.threshold} ({pot.rejection_reason})")


def test_p2_anchor_skipped_without_cumulative_hash():
    """No ledger row / empty hash → anchor no-ops (soft-fail), publishes nothing."""
    q = _FakeQueue()
    sdw._anchor_main_chain(q, "soul_diary", "2026-06-10", {})
    assert q.msgs == []
