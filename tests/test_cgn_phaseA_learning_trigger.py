"""Phase A (RFP_cgn_enhancements §9.1) — learning-event chain trigger.

Verifies the vertical slice end-to-end at the unit level:
  1. send_meta_request carries a grounding_payload in the bus message.
  2. MetaService.handle_request routes a grounding request to _grounding_sink
     (not the one-shot resolver) and acks the consumer.
  3. MetaReasoningEngine.enqueue_grounding appends to _pending_groundings.
  4. _start_chain seeds state.grounding_concept + entity_refs["current_topic"]
     + entry_primitive from _active_grounding (the §5.3 FORMULATE-collapse fix).

Run: python -m pytest tests/test_cgn_phaseA_learning_trigger.py -v -p no:anchorpy
"""
import warnings
warnings.filterwarnings("ignore")


class _FakeQueue:
    """Minimal stand-in for a worker send_queue."""
    def __init__(self):
        self.msgs = []

    def put_nowait(self, m):
        self.msgs.append(m)


def test_send_meta_request_carries_grounding_payload():
    from titan_hcl.logic.meta_service_client import send_meta_request
    from titan_hcl.logic.meta_consumer_contexts import (
        build_language_meta_context_30d)
    q = _FakeQueue()
    rid = send_meta_request(
        consumer_id="language",
        question_type="synthesize_insight",
        context_vector=build_language_meta_context_30d(),
        time_budget_ms=2000,
        send_queue=q,
        src="language",
        grounding_payload={"concept_id": "warmth"},
    )
    assert isinstance(rid, str) and rid
    assert len(q.msgs) == 1
    payload = q.msgs[0]["payload"]
    assert payload["grounding_payload"]["concept_id"] == "warmth"
    assert q.msgs[0]["dst"] == "cognitive_worker"
    assert payload["question_type"] == "synthesize_insight"


def test_meta_service_routes_grounding_to_sink():
    from titan_hcl.logic.meta_service import MetaService
    captured = {"sink": None, "responses": []}
    svc = MetaService(response_emitter=lambda m: captured["responses"].append(m))
    try:
        svc._grounding_sink = lambda entry: captured.__setitem__("sink", entry)
        msg = {
            "src": "language",
            "payload": {
                "consumer_id": "language",
                "question_type": "synthesize_insight",
                "request_id": "req-1",
                "context_vector": [0.0] * 30,
                "time_budget_ms": 2000,
                "grounding_payload": {"concept_id": "warmth"},
            },
        }
        out = svc.handle_request(msg)
        assert out is None  # accepted (not a sync rejection)
        # Sink received the grounding entry with the mapped entry_primitive.
        assert captured["sink"] is not None
        assert captured["sink"]["concept_id"] == "warmth"
        assert captured["sink"]["consumer"] == "language"
        assert captured["sink"]["entry_primitive"] == "SYNTHESIZE"
        # Consumer was acked, not dispatched to a one-shot resolver.
        assert any(
            r["payload"].get("reason") == "grounding_enqueued"
            for r in captured["responses"])
    finally:
        # Stop the dispatch loop thread so pytest exits cleanly.
        if hasattr(svc, "shutdown"):
            try:
                svc.shutdown()
            except Exception:
                pass


def _make_engine():
    from titan_hcl.logic.meta_reasoning import MetaReasoningEngine
    return MetaReasoningEngine(config={}, send_queue=None)


def test_enqueue_grounding_and_path0_seed():
    from collections import deque
    eng = _make_engine()
    assert isinstance(eng._pending_groundings, deque)
    assert eng._active_grounding is None

    # enqueue ignores payloads without a concept_id, accepts valid ones.
    eng.enqueue_grounding({"concept_id": ""})
    assert len(eng._pending_groundings) == 0
    eng.enqueue_grounding({
        "consumer": "language",
        "concept_id": "warmth",
        "entry_primitive": "SYNTHESIZE",
        "request_id": "req-42",
        "question_type": "synthesize_insight",
    })
    assert len(eng._pending_groundings) == 1

    # Simulate Path #0 (the tick() branch): pop → stash → _start_chain seeds.
    g = eng._pending_groundings.popleft()
    eng._active_grounding = g
    eng._start_chain("concept_grounding(language:warmth)", [0.0] * 132)

    assert eng.state.grounding_concept == "warmth"
    assert eng.state.grounding_consumer == "language"
    assert eng.state.entry_primitive == "SYNTHESIZE"
    # The concept seeds entity_refs so existing primitives walk it (§5.3 fix).
    assert eng.state.entity_refs.get("current_topic") == "warmth"
    # RFP_cgn_loop_closure §7.A (ARC-4) — request_id + question_type seeded so
    # _conclude_chain can emit a META_REASON_OUTCOME (handle_outcome requires
    # the request_id). The half that fed the emergent reward.
    assert eng.state.grounding_request_id == "req-42"
    assert eng.state.grounding_question_type == "synthesize_insight"
    # Consumed once.
    assert eng._active_grounding is None


def test_start_chain_without_grounding_is_unchanged():
    """Back-compat: a normal (mechanical-trigger) chain has empty grounding."""
    eng = _make_engine()
    eng._active_grounding = None
    eng._start_chain("periodic(50)", [0.0] * 132)
    assert eng.state.grounding_concept == ""
    assert eng.state.entry_primitive == ""
    assert eng.state.grounding_request_id == ""
    assert "current_topic" not in eng.state.entity_refs


# ── RFP_cgn_loop_closure §7.A (ARC-4) — outcome emit + multi-step credit ──

def test_send_meta_outcome_carries_primitive_sequence():
    """The emit helper threads the full chain (primitive_sequence) into the
    META_REASON_OUTCOME payload so the accumulator can do multi-step credit."""
    from titan_hcl.logic.meta_service_client import send_meta_outcome
    q = _FakeQueue()
    send_meta_outcome(
        request_id="req-42",
        consumer_id="knowledge",
        outcome_reward=0.36,
        actual_primitive_used="HYPOTHESIZE",
        primitive_sequence=["HYPOTHESIZE", "RECALL", "SYNTHESIZE"],
        send_queue=q,
        src="cognitive_worker",
    )
    assert len(q.msgs) == 1
    payload = q.msgs[0]["payload"]
    assert q.msgs[0]["type"] == "META_REASON_OUTCOME"
    assert q.msgs[0]["dst"] == "cognitive_worker"
    assert payload["request_id"] == "req-42"
    assert payload["outcome_reward"] == 0.36
    assert payload["primitive_sequence"] == ["HYPOTHESIZE", "RECALL", "SYNTHESIZE"]


def test_handle_outcome_feeds_accumulator_multistep():
    """End-to-end: MetaService.handle_outcome with a primitive_sequence routes
    through the internal outcome_sink → DynamicRewardAccumulator, crediting
    EVERY (consumer, primitive) tuple (the ARC-4 loop finally feeds α)."""
    from titan_hcl.logic.meta_service import MetaService
    svc = MetaService()
    try:
        before = svc._rewards._total_outcomes if svc._rewards else None
        assert svc._rewards is not None
        ok = svc.handle_outcome({
            "src": "cognitive_worker",
            "payload": {
                "consumer_id": "knowledge",
                "request_id": "req-42",
                "outcome_reward": 0.4,
                "primitive_sequence": ["HYPOTHESIZE", "RECALL", "DELEGATE"],
            },
        })
        assert ok is True
        # One chain → one outcome counted (drives α), three tuples credited.
        assert svc._rewards._total_outcomes == before + 1
        for p in ("HYPOTHESIZE", "RECALL", "DELEGATE"):
            assert svc._rewards._count[("knowledge", p, "_all")] == 1
            assert svc._rewards._rolling_mean[("knowledge", p, "_all")] == 0.4
    finally:
        if hasattr(svc, "shutdown"):
            try:
                svc.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    test_send_meta_request_carries_grounding_payload()
    test_meta_service_routes_grounding_to_sink()
    test_enqueue_grounding_and_path0_seed()
    test_start_chain_without_grounding_is_unchanged()
    test_send_meta_outcome_carries_primitive_sequence()
    test_handle_outcome_feeds_accumulator_multistep()
    print("OK — all Phase A unit checks passed")
