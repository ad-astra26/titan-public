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
        warm = [compute_significance("great_pulse", 1.0, sp) for _ in range(4)]
        assert all(s is None for s in warm), (
            "a steady, on-baseline signal must earn NO significance (got %r)" % warm)

        # Now a genuinely novel magnitude → real surprise. Repeat it: as the baseline
        # μ moves toward the new value, the SAME event becomes less surprising.
        s1 = compute_significance("great_pulse", 12.0, sp)
        s2 = compute_significance("great_pulse", 12.0, sp)
        s3 = compute_significance("great_pulse", 12.0, sp)
        assert s1 is not None and s2 is not None and s3 is not None
        assert s1 > s2 > s3, (
            "repeated identical events must HABITUATE (decreasing significance), "
            "got %.4f, %.4f, %.4f" % (s1, s2, s3))
        # And the first novel event cleared the recording threshold (it is memorable).
        assert s1 >= SIGNIFICANCE_THRESHOLD


def test_significance_baselines_are_per_event_type():
    """Each event_type keys its OWN baseline — great_pulse's history does not bleed
    into kin_exchange's surprise (RFP §4.1: separate _SignalBaseline per trigger)."""
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "sig.json")
        # Saturate great_pulse at a high value.
        for _ in range(5):
            compute_significance("great_pulse", 50.0, sp)
        # A brand-new event_type seeds independently (first observation → None,
        # NOT contaminated by great_pulse's μ) and warms up on its OWN scale
        # (min_samples=2 → seed + one warm-up fold emit nothing).
        assert compute_significance("kin_exchange", 0.4, sp) is None   # seed
        assert compute_significance("kin_exchange", 0.4, sp) is None   # warming up
        # Now its own baseline is live; a deviation earns its own surprise — and the
        # value is on kin's 0..1 scale, proving it never inherited great_pulse's μ=50.
        s = compute_significance("kin_exchange", 0.9, sp)
        assert s is not None and s > 0.0


# ── 2. The store writes (record_episode is a real, working append) ────────────
def test_record_episode_appends_and_recalls():
    with tempfile.TemporaryDirectory() as d:
        mem = EpisodicMemory(db_path=os.path.join(d, "episodic_memory.db"))
        assert mem.count() == 0
        rid = mem.record_episode(
            event_type="great_pulse", description="test pulse",
            felt_state=[0.1] * 130, hormonal_snapshot={"DA": 0.6},
            epoch_id=7, significance=0.8)
        assert rid is not None and mem.count() == 1
        # below-threshold significance is dropped (the gate works)
        assert mem.record_episode(event_type="kin_exchange", significance=0.1) is None
        assert mem.count() == 1
        # recall_by_feeling cosines over the SAME felt space we recorded
        hits = mem.recall_by_feeling([0.1] * 130, top_k=5)
        assert hits and hits[0]["event_type"] == "great_pulse"


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
                "event_type": "great_pulse",
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
        assert row["event_type"] == "great_pulse"
        assert row["significance"] >= SIGNIFICANCE_THRESHOLD
        assert row["felt_state"] and len(row["felt_state"]) == 130


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
