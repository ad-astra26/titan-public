"""Write-liveness + emergent-significance tests for the ACT-R Episodic faculty
restoration (RFP_phase_c_actr_memory_rehoming Phase 1).

These tests close the exact blind spot that let the faculty die unnoticed: the old
boot-driver-parity test asserted episodic_mem was *referenced*, not that record_episode
was ever *called*. Here we assert the WRITE path is live end-to-end, and that
significance is EMERGENT (surprise that habituates), never a hardcoded constant.

Run isolated:  python -m pytest tests/test_episodic_rehoming.py -v -p no:anchorpy
"""
import os
import tempfile

from titan_hcl.logic.episodic_memory import (
    EpisodicMemory, compute_significance, SIGNIFICANCE_THRESHOLD)


# ── 1. Emergent significance = surprise, and it HABITUATES (no constants) ──────
def test_significance_is_emergent_surprise_and_habituates():
    """Repeated identical events produce a *decreasing* significance (the affective
    loop's surprise habituating), proving significance is earned per-event — not a
    hardcoded magnitude. RFP §6 acceptance."""
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "episodic_significance_state.json")

        # Warm the per-event baseline with a steady value (first fold seeds → None;
        # warming-up folds → None; on-baseline → None). No surprise is fabricated.
        warm = [compute_significance("pi_cluster", 1.0, sp) for _ in range(4)]
        assert all(s is None for s in warm), (
            "a steady, on-baseline signal must earn NO significance (got %r)" % warm)

        # Now a genuinely novel magnitude → real surprise. Repeat it: as the baseline
        # μ moves toward the new value, the SAME event becomes less surprising.
        s1 = compute_significance("pi_cluster", 12.0, sp)
        s2 = compute_significance("pi_cluster", 12.0, sp)
        s3 = compute_significance("pi_cluster", 12.0, sp)
        assert s1 is not None and s2 is not None and s3 is not None
        assert s1 > s2 > s3, (
            "repeated identical events must HABITUATE (decreasing significance), "
            "got %.4f, %.4f, %.4f" % (s1, s2, s3))
        # And the first novel event cleared the recording threshold (it is memorable).
        assert s1 >= SIGNIFICANCE_THRESHOLD


def test_significance_baselines_are_per_event_type():
    """Each event_type keys its OWN baseline — pi_cluster's history does not bleed
    into kin_exchange's surprise (RFP §4.1: separate _SignalBaseline per trigger)."""
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "sig.json")
        # Saturate pi_cluster at a high value.
        for _ in range(5):
            compute_significance("pi_cluster", 50.0, sp)
        # A brand-new event_type seeds independently (first observation → None,
        # NOT contaminated by pi_cluster's μ) and warms up on its OWN scale
        # (min_samples=2 → seed + one warm-up fold emit nothing).
        assert compute_significance("kin_exchange", 0.4, sp) is None   # seed
        assert compute_significance("kin_exchange", 0.4, sp) is None   # warming up
        # Now its own baseline is live; a deviation earns its own surprise — and the
        # value is on kin's 0..1 scale, proving it never inherited pi_cluster's μ=50.
        s = compute_significance("kin_exchange", 0.9, sp)
        assert s is not None and s > 0.0


# ── 2. The store writes (record_episode is a real, working append) ────────────
def test_record_episode_appends_and_recalls():
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        assert mem.count() == 0
        rid = mem.record_episode(
            event_type="pi_cluster", description="test pulse",
            felt_state=[0.1] * 130, hormonal_snapshot={"DA": 0.6},
            epoch_id=7, significance=0.8)
        assert rid is not None and mem.count() == 1
        # below-threshold significance is dropped (the gate works)
        assert mem.record_episode(event_type="kin_exchange", significance=0.1) is None
        assert mem.count() == 1
        # recall_by_feeling cosines over the SAME felt space we recorded
        hits = mem.recall_by_feeling([0.1] * 130, top_k=5)
        assert hits and hits[0]["event_type"] == "pi_cluster"


# ── 3. The owner dispatch is a LIVE caller (write-liveness, end-to-end) ────────
def test_dispatch_episode_record_is_a_live_writer():
    """_dispatch_episode_record (cognitive_worker) enriches + scores + persists a
    real row — the proof the faculty has a live writer, not a parity-anchor."""
    from titan_hcl.modules.cognitive_worker import _dispatch_episode_record
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        state_refs = {
            "episodic_mem": mem,
            # consciousness snapshot supplies the felt state_vector + epoch
            "consciousness": {"latest_epoch": {
                "state_vector": [0.2] * 130, "epoch_id": 11}},
            "neural_nervous_system": None,   # hormones optional → {}
        }

        def emit(metric):
            _dispatch_episode_record(state_refs, {
                "event_type": "pi_cluster",
                "description": "integration",
                "metric": metric,
                "epoch_id": 11,
            })

        # Warm-up (seed + on-baseline) writes nothing — honest "no earned surprise".
        for _ in range(4):
            emit(1.0)
        assert mem.count() == 0, "on-baseline events must not be recorded"

        # A surprising event clears the threshold and is appended with the FELT
        # state attached (so recall_by_feeling can find it later).
        emit(15.0)
        assert mem.count() == 1, "a surprising event must be recorded (live writer)"
        row = mem.get_autobiography(limit=1)[0]
        assert row["event_type"] == "pi_cluster"
        assert row["significance"] >= SIGNIFICANCE_THRESHOLD
        assert row["felt_state"] and len(row["felt_state"]) == 130


# ── 3b. action_completed gate inside _dispatch_experience_record ──────────────
def test_action_completed_episodic_gate():
    """A high-scoring autonomous ACTION (context.helper present + outcome_score>0.7)
    records an `action_completed` episode via _dispatch_experience_record; a
    non-action experience (no helper) or a low score does NOT. Faithful port of
    spirit_worker L7406 (RFP §4.1)."""
    from titan_hcl.modules.cognitive_worker import _dispatch_experience_record

    class _StubOrch:
        _plugins = {}

        def record_outcome(self, **kw):
            return None

    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        state_refs = {
            "exp_orchestrator": _StubOrch(),
            "episodic_mem": mem,
            "consciousness": {"latest_epoch": {
                "state_vector": [0.3] * 130, "epoch_id": 4}},
            "neural_nervous_system": None,
        }

        def exp(score, helper=True):
            ctx = {"helper": "research_helper"} if helper else {}
            _dispatch_experience_record(state_refs, {
                "domain": "research", "action_taken": "x",
                "outcome_score": score, "context": ctx, "epoch_id": 4})

        # A non-action experience (no helper) never records an episode.
        for _ in range(4):
            exp(0.95, helper=False)
        assert mem.count() == 0, "non-action experiences must not episode"
        # A low-scoring action does not clear the action gate.
        for _ in range(4):
            exp(0.3, helper=True)
        assert mem.count() == 0, "score<=0.7 actions must not episode"
        # High-scoring actions: warm the baseline, then a surprising score records.
        for _ in range(3):
            exp(0.75, helper=True)
        exp(0.99, helper=True)
        assert mem.count() >= 1, "a surprising high-score action must episode"
        assert mem.get_autobiography(limit=1)[0]["event_type"] == "action_completed"


# ── 4. The bus producer is targeted + coalesced (Phase-2 cross-process path) ──
def test_emit_episode_record_is_targeted_and_coalesced():
    import titan_hcl.bus as bus

    class _Q:
        def __init__(self):
            self.items = []

        def put_nowait(self, m):
            self.items.append(m)

    q = _Q()
    bus._episode_record_last_emit.clear()
    ok = bus.emit_episode_record(q, "synthesis", "x_engagement",
                                 description="engagement spike", metric=3.0,
                                 epoch_id=9)
    assert ok and len(q.items) == 1
    msg = q.items[0]
    assert msg["type"] == bus.EPISODE_RECORD
    assert msg["dst"] == "cognitive_worker"          # TARGETED, never dst="all"
    assert msg["payload"]["event_type"] == "x_engagement"
    # immediate re-emit on the same key is coalesced (bus hygiene)
    assert bus.emit_episode_record(q, "synthesis", "x_engagement", metric=4.0) is False
    assert len(q.items) == 1


# ── 5. Tier 2: WORD_LEARNED producer (language → cognitive) targeted + coalesced ──
def test_emit_word_learned_is_targeted_and_coalesced():
    import titan_hcl.bus as bus

    class _Q:
        def __init__(self):
            self.items = []

        def put_nowait(self, m):
            self.items.append(m)

    q = _Q()
    bus._word_learned_last_emit.clear()
    ok = bus.emit_word_learned(q, "language", "sovereignty", 0.62)
    assert ok and len(q.items) == 1
    msg = q.items[0]
    assert msg["type"] == bus.WORD_LEARNED
    assert msg["dst"] == "cognitive_worker"          # TARGETED, never dst="all"
    assert msg["payload"]["word"] == "sovereignty"
    assert msg["payload"]["confidence"] == 0.62
    # same word re-grounded within the interval is coalesced; a different word emits
    assert bus.emit_word_learned(q, "language", "sovereignty", 0.7) is False
    assert bus.emit_word_learned(q, "language", "metabolic", 0.55) is True
    assert len(q.items) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 (§7.2) — affective-signal + conversation episodes via the synthesis hook
# ══════════════════════════════════════════════════════════════════════════════

# ── P2.1 — schema migration: felt_impact + person_id persist + deserialize ─────
def test_record_episode_persists_felt_impact_and_person_id():
    import json
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        rid = mem.record_episode(
            event_type="conversation", description="with alice",
            felt_state=[0.4] * 65, hormonal_snapshot={"DA": 0.7},
            epoch_id=3, significance=1.2, felt_impact=0.08, person_id="alice")
        assert rid is not None
        row = mem.get_autobiography(limit=1)[0]
        assert abs(row["felt_impact"] - 0.08) < 1e-9
        assert row["person_id"] == "alice"
        # Phase-1-style call (no new fields) leaves them NULL — backward compatible.
        mem.record_episode(event_type="pi_cluster", significance=0.9,
                           felt_state=[0.1] * 130)
        pi = [r for r in mem.get_autobiography(limit=5)
              if r["event_type"] == "pi_cluster"][0]
        assert pi["felt_impact"] is None and pi["person_id"] is None
        _ = json  # silence lint; felt_state round-trips via _deserialize already


# ── P2.2 — backward-compat: a PRE-Phase-2 DB is migrated in place, rows survive ─
def test_old_schema_db_is_migrated_and_rows_survive():
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        dbp = os.path.join(d, "episodic_memory.db")
        # Build the OLD 8-column schema (no felt_impact/person_id) + one row.
        c = sqlite3.connect(dbp)
        c.execute(
            "CREATE TABLE episodic_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_type TEXT NOT NULL, description TEXT, felt_state TEXT, "
            "hormonal_snapshot TEXT, epoch_id INTEGER NOT NULL DEFAULT 0, "
            "significance REAL NOT NULL DEFAULT 0.0, created_at REAL NOT NULL)")
        c.execute("INSERT INTO episodic_memory "
                  "(event_type, significance, created_at) VALUES (?,?,?)",
                  ("great_pulse", 0.9, 1.0))
        c.commit()
        c.close()
        # Opening via EpisodicMemory must ALTER-add the new cols without data loss.
        mem = EpisodicMemory(db_path=dbp)
        assert mem.count() == 1, "the pre-existing row must survive migration"
        old = mem.get_autobiography(limit=1)[0]
        assert old["event_type"] == "great_pulse"
        assert old["felt_impact"] is None and old["person_id"] is None
        # And the migrated DB accepts the new fields.
        assert mem.record_episode(
            event_type="sol_receipt", significance=0.8,
            felt_impact=0.05, person_id="") is not None


# ── P2.3 — recorder PASSTHROUGH: synthesis-computed surprise is used verbatim ───
def test_dispatch_passthrough_significance_skips_recompute():
    """When the producer passes a pre-computed `significance` (the affective signals
    already folded Nudge.surprise), the recorder uses it DIRECTLY — it does NOT
    re-fold `metric` into the episodic EMA. Proven by recording on the FIRST-EVER
    event (the metric path would return None → no record), and by storing felt_impact
    + person_id (§7.2)."""
    from titan_hcl.modules.cognitive_worker import _dispatch_episode_record
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        state_refs = {
            "episodic_mem": mem,
            "consciousness": {"latest_epoch": {
                "state_vector": [0.2] * 130, "epoch_id": 5}},
            "neural_nervous_system": None,
        }
        # First-ever sol_receipt: the metric→EMA path would seed→None→no record.
        # The passthrough significance forces it through (proving no recompute).
        _dispatch_episode_record(state_refs, {
            "event_type": "sol_receipt",
            "description": "receipt",
            "metric": 0.05,
            "epoch_id": 5,
            "significance": 1.7,          # already-computed surprise
            "felt_impact": 0.09,          # the net's magnitude
            "person_id": "",
        })
        assert mem.count() == 1, "passthrough significance must record on first event"
        row = mem.get_autobiography(limit=1)[0]
        assert abs(row["significance"] - 1.7) < 1e-9, "exact passthrough value stored"
        assert abs(row["felt_impact"] - 0.09) < 1e-9
        # A passthrough below the 0.3 gate is still dropped (one currency, one gate).
        _dispatch_episode_record(state_refs, {
            "event_type": "x_engagement", "metric": 0.0,
            "epoch_id": 5, "significance": 0.1})
        assert mem.count() == 1, "below-threshold passthrough is gated like Phase 1"


# ── P2.4 — recorder still honours the Phase-1 metric path when no passthrough ───
def test_dispatch_metric_path_unchanged_without_passthrough():
    """A Phase-1-style payload (no significance/felt_impact/person_id) records
    exactly as before — the metric→EMA surprise path, byte-identical behaviour."""
    from titan_hcl.modules.cognitive_worker import _dispatch_episode_record
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        state_refs = {
            "episodic_mem": mem,
            "consciousness": {"latest_epoch": {
                "state_vector": [0.2] * 130, "epoch_id": 1}},
            "neural_nervous_system": None,
        }

        def emit(metric):
            _dispatch_episode_record(state_refs, {
                "event_type": "great_pulse", "metric": metric, "epoch_id": 1})

        for _ in range(4):
            emit(1.0)
        assert mem.count() == 0, "on-baseline metric path records nothing"
        emit(20.0)
        assert mem.count() == 1, "a surprising metric still records (Phase-1 path)"
        assert mem.get_autobiography(limit=1)[0]["felt_impact"] is None


# ── P2.5 — emit_episode_record carries the §7.2 passthrough fields ─────────────
def test_emit_episode_record_carries_passthrough_fields():
    import titan_hcl.bus as bus

    class _Q:
        def __init__(self):
            self.items = []

        def put_nowait(self, m):
            self.items.append(m)

    q = _Q()
    bus._episode_record_last_emit.clear()
    ok = bus.emit_episode_record(
        q, "synthesis", "maker_bond", metric=1.0, epoch_id=2,
        significance=2.4, felt_impact=0.11, person_id="maker")
    assert ok
    p = q.items[0]["payload"]
    assert p["significance"] == 2.4 and p["felt_impact"] == 0.11
    assert p["person_id"] == "maker"
    # A bare Phase-1 emit leaves the passthrough fields null/empty (backward compat).
    bus._episode_record_last_emit.clear()
    bus.emit_episode_record(q, "cognitive_worker", "kin_exchange", metric=0.5)
    p2 = q.items[1]["payload"]
    assert p2["significance"] is None and p2["felt_impact"] is None
    assert p2["person_id"] == ""


# ── P2.6 — emit_turn_context: targeted, per-interlocutor coalesced, skips anon ──
def test_emit_turn_context_targeted_coalesced_anon_skip():
    import titan_hcl.bus as bus

    class _Q:
        def __init__(self):
            self.items = []

        def put_nowait(self, m):
            self.items.append(m)

    q = _Q()
    bus._turn_context_last_emit.clear()
    # Anonymous / empty interlocutor → no-op (no WM attend for a faceless turn).
    assert bus.emit_turn_context(q, "synthesis", "") is False
    assert bus.emit_turn_context(q, "synthesis", "anonymous") is False
    assert len(q.items) == 0
    # A real interlocutor emits a TARGETED frame to cognitive_worker.
    assert bus.emit_turn_context(q, "synthesis", "alice", goal_class="chitchat") is True
    msg = q.items[0]
    assert msg["type"] == bus.TURN_CONTEXT and msg["dst"] == "cognitive_worker"
    assert msg["payload"]["user_id"] == "alice"
    # Same interlocutor within the interval coalesces; a different one emits.
    assert bus.emit_turn_context(q, "synthesis", "alice") is False
    assert bus.emit_turn_context(q, "synthesis", "bob") is True
    assert len(q.items) == 2


# ══════════════════════════════════════════════════════════════════════════════
# §5.2 — Recall wired into cognition (Leg 1 ambient + Leg 2 meta-reasoning RECALL)
# ══════════════════════════════════════════════════════════════════════════════

# ── R.1 — Leg 2: meta_reasoning RECALL dispatches to the REAL episodic faculty ──
def test_meta_reasoning_recall_dispatches_to_episodic_faculty():
    """`autobiographical_relevant` → get_autobiography (significance-ranked);
    `episodic_specific` → recall_by_feeling (resonant). Both leave the session-1
    stub (recruitment_resolved=True / session_1_stub=False). Without the faculty,
    they gracefully fall back to the exp_orchestrator stub."""
    from titan_hcl.logic.meta_reasoning import MetaReasoningEngine
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        for i, sig in enumerate([0.9, 1.5, 0.7]):
            mem.record_episode(event_type="conversation", description=f"talk {i}",
                               felt_state=[0.2] * 130, significance=sig,
                               person_id=f"person{i}")
        eng = MetaReasoningEngine()
        eng._episodic_mem = mem
        eng._cur_state_132d = [0.2] * 130
        eng.state.formulate_output = {"domain": "general"}

        # autobiographical_relevant → the life story (top by significance)
        out = eng._prim_recall("autobiographical_relevant", None, None, None)
        assert out["recruitment_resolved"] is True, "must resolve to the real faculty"
        assert out["session_1_stub"] is False, "no longer a session-1 stub"
        assert out["count"] >= 1 and out["best_match"] is True
        assert eng.state.recalled_data["results"][0]["significance"] == 1.5, \
            "get_autobiography ranks by significance (1.5 first)"

        # episodic_specific → resonant recall over the current felt-state
        out2 = eng._prim_recall("episodic_specific", None, None, None)
        assert out2["recruitment_resolved"] is True and out2["count"] >= 1
        assert "similarity" in eng.state.recalled_data["results"][0]

        # No faculty wired → graceful fallback (the old stub path, no crash)

        class _Orch:
            def recall_similar(self, domain, top_k=5):
                return []

        eng._episodic_mem = None
        out3 = eng._prim_recall("autobiographical_relevant", None, None, _Orch())
        assert out3["session_1_stub"] is True, "stub again without the faculty"
        assert out3["recruitment_resolved"] is False


# ── R.2 — Leg 1 condition: recall_by_feeling yields the (sig + similarity) the ──
#          ambient `episodic_echo` attend gates on ─────────────────────────────
def test_recall_by_feeling_yields_significant_resonant_echo():
    """The Leg-1 episodic_echo attend fires only for episodes that are BOTH
    significant (≥0.5) AND resonant (similarity ≥0.5). Assert recall_by_feeling
    surfaces such a row when one exists, carrying both fields."""
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        # a significant episode whose felt_state matches the probe exactly
        mem.record_episode(event_type="kin_exchange", description="kin",
                           felt_state=[0.5] * 130, significance=0.9)
        # a non-resonant low-sig one
        mem.record_episode(event_type="pi_cluster", description="pi",
                           felt_state=[-0.5] * 130, significance=0.4)
        hits = mem.recall_by_feeling([0.5] * 130, top_k=3)
        assert hits, "recall must return candidates"
        top = hits[0]
        assert top["event_type"] == "kin_exchange"
        assert top["similarity"] >= 0.5 and top["significance"] >= 0.5, \
            "the most-resonant hit clears both gates → episodic_echo would attend"
