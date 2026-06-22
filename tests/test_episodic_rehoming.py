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
