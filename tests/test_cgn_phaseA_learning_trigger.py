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
    # Consumed once.
    assert eng._active_grounding is None


def test_start_chain_without_grounding_is_unchanged():
    """Back-compat: a normal (mechanical-trigger) chain has empty grounding."""
    eng = _make_engine()
    eng._active_grounding = None
    eng._start_chain("periodic(50)", [0.0] * 132)
    assert eng.state.grounding_concept == ""
    assert eng.state.entry_primitive == ""
    assert "current_topic" not in eng.state.entity_refs


if __name__ == "__main__":
    test_send_meta_request_carries_grounding_payload()
    test_meta_service_routes_grounding_to_sink()
    test_enqueue_grounding_and_path0_seed()
    test_start_chain_without_grounding_is_unchanged()
    print("OK — all Phase A unit checks passed")
