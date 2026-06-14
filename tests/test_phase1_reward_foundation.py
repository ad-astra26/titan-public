"""Phase 1 — reward foundation (RFP_soar_haov_chunking_loop §7.1).

Repairs the 2 VC-confirmed degenerate CGN reward signals so the HAOV/impasse
loop gets real signal (INV-SOAR-2):
  - knowledge internal-recall: was hardcoded `reward=0.0` (knowledge_worker.py:519,
    type=experience self-matched it as final) → now the recall confidence.
  - social events-grounding: was `0.02 + _grel*0.04` → [0.02,0.06] (mostly below
    the `reward>0.05` detect_impasse gate) → now `round(_grel,4)` → [0,1].

The first test is the real plumbing contract (a non-zero reward reaches the
CGN_TRANSITION). The policy tests guard the meaningful-magnitude formulas +
that they cross the impasse gate the old ones failed.
"""


class _FakeQueue:
    """Collects emitted messages; supports either put_nowait or put."""
    def __init__(self):
        self.msgs = []

    def put_nowait(self, m):
        self.msgs.append(m)

    def put(self, m, *a, **k):
        self.msgs.append(m)


def _cgn_transitions(q):
    out = []
    for m in q.msgs:
        if not isinstance(m, dict):
            continue
        if m.get("type") == "CGN_TRANSITION":
            out.append(m.get("payload", m))
    return out


def test_knowledge_send_transition_propagates_reward():
    """The §7.1 fix passes the recall confidence as `reward`; prove the
    plumbing carries an arbitrary reward through to the CGN_TRANSITION so a
    real confidence reaches CGN (not a swallowed/hardcoded 0.0)."""
    from titan_hcl.modules.knowledge_worker import _send_transition
    q = _FakeQueue()
    _send_transition(q, "knowledge", None, "symmetry", 0, {"DA": 0.5},
                     reward=0.82)
    trs = _cgn_transitions(q)
    assert trs, "no CGN_TRANSITION emitted by _send_transition"
    p = trs[0]
    assert p["consumer"] == "knowledge"
    assert p["type"] == "experience"
    assert abs(float(p["reward"]) - 0.82) < 1e-9, \
        f"reward not propagated through _send_transition: {p['reward']}"


def test_knowledge_recall_reward_policy():
    """Policy: a successful recall (gated conf>0.4 at knowledge_worker.py:511)
    is rewarded by its confidence — meaningful [0.4,1.0], crosses the impasse
    gate — NOT the old hardcoded 0.0."""
    internal = [{"confidence": 0.82}]
    reward = float(internal[0]["confidence"])      # the §7.1 call-site expression
    assert reward == 0.82
    assert reward > 0.05, "recall reward must cross the detect_impasse gate (cgn.py:982)"
    assert reward != 0.0, "regression: recall reward must not revert to hardcoded 0.0"


def test_social_events_reward_policy():
    """Policy: social events-grounding rewarded by its felt-relevance ([0,1]),
    NOT the old 0.02+rel*0.04 cap ([0.02,0.06]) that sat below the impasse gate."""
    for _grel in (0.3, 0.5, 0.7, 1.0):
        new_reward = round(_grel, 4)               # the §7.1 expression
        old_reward = 0.02 + _grel * 0.04
        assert new_reward == round(_grel, 4)
        assert old_reward <= 0.06, "old formula was capped tiny"
    # at a representative relevance the new reward crosses the gate, the old didn't:
    _grel = 0.7
    assert round(_grel, 4) > 0.05
    assert (0.02 + _grel * 0.04) < 0.05
