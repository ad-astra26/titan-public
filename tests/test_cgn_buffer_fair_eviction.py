"""§7.D-A2 formation-layer — TransitionBuffer fair eviction.

Global-FIFO eviction let a high-rate consumer (meta) crowd low-rate ones
(emotional/language/self_model) out of the shared buffer, dropping them below
detect_impasse's >=10 threshold (dormancy no per-consumer wiring could fix).
Fair eviction drops the oldest transition of the MOST-represented consumer, so
every consumer keeps its recent history.
"""
import numpy as np

from titan_hcl.logic.cgn import TransitionBuffer
from titan_hcl.logic.cgn_types import CGNTransition


def _mk(consumer: str, cid: str = "c") -> CGNTransition:
    return CGNTransition(consumer=consumer, concept_id=cid,
                         state=np.zeros(30, dtype=np.float32), action=0,
                         action_params=np.zeros(4, dtype=np.float32))


def test_fair_eviction_protects_low_rate_consumer():
    buf = TransitionBuffer(max_size=100)
    for _ in range(10):           # 10 low-rate emotional transitions (oldest)
        buf.add(_mk("emotional"))
    for _ in range(500):          # flood with a high-rate consumer
        buf.add(_mk("meta"))
    # Global FIFO would have evicted all 10 emotional (they were oldest).
    emo = buf.get_consumer_transitions("emotional")
    assert len(emo) == 10, f"expected 10 emotional retained, got {len(emo)}"
    assert buf.size() == 100       # capped


def test_buffer_stays_capped():
    buf = TransitionBuffer(max_size=50)
    for _ in range(200):
        buf.add(_mk("meta"))
    assert buf.size() == 50


def test_single_consumer_is_still_fifo():
    buf = TransitionBuffer(max_size=5)
    for i in range(10):
        buf.add(_mk("solo", cid=str(i)))
    ids = [t.concept_id for t in buf.get_consumer_transitions("solo")]
    assert ids == ["5", "6", "7", "8", "9"]  # FIFO within one consumer


def test_chronological_order_preserved():
    buf = TransitionBuffer(max_size=4)
    buf.add(_mk("a", "a0"))
    buf.add(_mk("b", "b0"))
    buf.add(_mk("a", "a1"))
    buf.add(_mk("a", "a2"))
    buf.add(_mk("a", "a3"))  # over cap → evict oldest 'a' (a0); b0 survives
    assert [t.concept_id for t in buf.get_consumer_transitions("b")] == ["b0"]
    assert [t.concept_id for t in buf.get_consumer_transitions("a")] == [
        "a1", "a2", "a3"]
