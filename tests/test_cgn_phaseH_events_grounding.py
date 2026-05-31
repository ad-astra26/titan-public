"""Phase H (RFP_cgn_enhancements §H / Inner Teacher Protocol) — events-teacher → CGN grounding.

The events-teacher is outer/narrative-oriented; the non-linguistic inner Titan (§11.4)
cannot read its prose. H translates recurring social concepts into the inner Titan's
native CGN modality across two of the three Inner-Teacher-Protocol channels:
  • FELT ("feel it")        — the window's felt signature rides each grounding.
  • SEMANTIC ("understand it") — co-occurring concepts become CGN associations.
(The PROCEDURAL channel — social.concept_grounded → SIGNAL_TO_PRIMITIVE — is the
consumer-side signal; here we assert the contract is registered.)

Covers the producer (the testable salience gate):
  1. recurrence gate — a concept grounds only after recurring ≥2 windows.
  2. relevance floor — low-relevance windows don't count toward recurrence.
  3. felt signature + associations attached to each grounding.
  4. per-window cap.
  5. recurrence persists across the cron's from_state/save_state lifecycle.
  6. social.concept_grounded is a registered (non-orphan) META-CGN signal.

Run: python -m pytest tests/test_cgn_phaseH_events_grounding.py -v -p no:anchorpy
"""
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

from titan_hcl.logic.events_teacher import (
    DistilledEvent, EventsTeacher,
    SOCIAL_GROUND_RECURRENCE_MIN, SOCIAL_GROUND_RELEVANCE_MIN,
)


def _evt(concepts, relevance=0.6, sentiment=0.4, arousal=0.5, felt="a warm exchange"):
    return DistilledEvent(
        source="mention", author="friend", topic="t", sentiment=sentiment,
        arousal=arousal, relevance=relevance, concept_signals=["YOU"],
        felt_summary=felt, contagion_type="warm", raw_text="", timestamp=0.0,
        semantic_concepts=list(concepts),
    )


def _teacher():
    t = EventsTeacher()
    return t


def test_recurrence_gate_requires_two_windows():
    t = _teacher()
    # Window 1: concept "music" appears once → below recurrence threshold → no grounding.
    g1 = t._compute_social_groundings([_evt(["music", "joy"])])
    assert g1 == [], "should not ground on first occurrence"
    assert t._concept_recurrence.get("music") == 1
    # Window 2: "music" recurs → now grounds.
    g2 = t._compute_social_groundings([_evt(["music", "calm"])])
    ids = {g["concept_id"] for g in g2}
    assert "music" in ids, "should ground after recurrence ≥ 2"
    assert SOCIAL_GROUND_RECURRENCE_MIN == 2


def test_low_relevance_does_not_count():
    t = _teacher()
    low = SOCIAL_GROUND_RELEVANCE_MIN - 0.05
    t._compute_social_groundings([_evt(["music"], relevance=low)])
    t._compute_social_groundings([_evt(["music"], relevance=low)])
    # Below the relevance floor → never counted → never grounds.
    assert t._concept_recurrence.get("music", 0) == 0


def test_felt_signature_and_associations():
    t = _teacher()
    t._compute_social_groundings([_evt(["music", "friend", "joy"], relevance=0.7,
                                       sentiment=0.6)])
    g = t._compute_social_groundings([_evt(["music", "friend", "joy"], relevance=0.8,
                                           sentiment=0.7)])
    music = next(x for x in g if x["concept_id"] == "music")
    # FELT channel — the strongest window's felt signature.
    assert music["felt"]["relevance"] == 0.8
    assert music["felt"]["sentiment"] == 0.7
    assert music["felt"]["contagion_type"] == "warm"
    # SEMANTIC channel — co-occurring concepts as associations (not itself).
    assert "friend" in music["associations"] and "joy" in music["associations"]
    assert "music" not in music["associations"]


def test_per_window_cap():
    from titan_hcl.logic.events_teacher import SOCIAL_GROUND_MAX_PER_WINDOW
    t = _teacher()
    many = [f"c{i}" for i in range(SOCIAL_GROUND_MAX_PER_WINDOW + 4)]
    # Recur all of them twice so all qualify, then assert the window cap holds.
    t._compute_social_groundings([_evt(many)])
    g = t._compute_social_groundings([_evt(many)])
    assert len(g) <= SOCIAL_GROUND_MAX_PER_WINDOW


def test_recurrence_persists_across_state():
    with tempfile.TemporaryDirectory() as tmp:
        state_path = os.path.join(tmp, "events_state.json")
        db_path = os.path.join(tmp, "events.db")
        t = EventsTeacher.from_state(state_path=state_path, db_path=db_path)
        t._compute_social_groundings([_evt(["music"])])
        assert t._concept_recurrence.get("music") == 1
        t.save_state(state_path)
        # New cron run reloads state → recurrence survives → second sighting grounds.
        t2 = EventsTeacher.from_state(state_path=state_path, db_path=db_path)
        assert t2._concept_recurrence.get("music") == 1
        g = t2._compute_social_groundings([_evt(["music"])])
        assert any(x["concept_id"] == "music" for x in g)


def test_social_concept_grounded_signal_registered():
    """The PROCEDURAL channel contract — social.concept_grounded must be a known
    (non-orphan) META-CGN signal, else handle_cross_consumer_signal drops it."""
    from titan_hcl.logic.meta_cgn import SIGNAL_TO_PRIMITIVE
    mapping = SIGNAL_TO_PRIMITIVE.get(("social", "concept_grounded"))
    assert mapping is not None
    # Monoculture-aware: FORMULATE biased away, integrative primitives up.
    assert mapping["FORMULATE"] < 0.5
    assert mapping["SYNTHESIZE"] >= 0.65
    assert "SPIRIT_SELF" in mapping


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("OK — all Phase H unit checks passed")
