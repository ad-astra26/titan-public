"""RFP_titan_research_agent §1.4 / §7.P3 step 3 — autonomous curiosity verify+credit.

Covers the three new hops:
  • 3a (agency_worker._maybe_emit_autonomous_experience): a research action that
    TARGETED a graph gap Z emits a Tier-1 routing reward + RESEARCH_CURIOSITY_GROUNDED
    to memory, and BYPASSES the LLM judge (no structural -0.5).
  • 3b (anchor-from-dict): build_promotion_tx makes a valid tx from a synthetic-id
    dict alone (no node store) — the resolved node-storage landmine.
  • 3c (research_wiki name_fn override): the override forces seed_research_concept to
    REFINE Z (bump) instead of minting a sibling — the keystone of Δ(Z)>0.
"""
from __future__ import annotations

from titan_hcl import bus
from titan_hcl.modules.agency_worker import _maybe_emit_autonomous_experience


# ── fakes (mirror tests/test_p8_autonomous_experience.py) ────────────────────
class _FakeJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, *, problem, action, evidence):
        self.calls += 1
        return {"solved": False, "verifiable": True, "correction": "", "confidence": 0.0}


class _FakeAgency:
    async def p8_rerun(self, *a, **k):
        return None


class _FakeSendQ:
    def __init__(self):
        self.items = []

    def put(self, msg):
        self.items.append(msg)


_RT = {"concept_id": "marinade_finance", "version": 2,
       "baseline_groundedness": 0.12, "domain_hint": "defi", "name": "Marinade Finance"}

_SUBSTANTIVE = ("Marinade Finance is a liquid-staking protocol on Solana that issues "
                "mSOL against staked SOL across a validator set, with delayed unstake.")


def _emit_curiosity(sq, judge, *, result, with_target=True):
    ar = {
        "helper": "web_search", "action_type": "research",
        "result": result, "reasoning": "ground concept",
        "helper_params": {"query": "what is Marinade Finance?",
                          "_research_target": _RT} if with_target else {"query": "q"},
    }
    _maybe_emit_autonomous_experience(
        sq, "agency", "test", {"posture": "research"}, ar, _FakeAgency(), judge)


def _by_type(sq, t):
    return [m for m in sq.items if m["type"] == t]


# ── 3a ───────────────────────────────────────────────────────────────────────
def test_3a_substantive_target_rewards_grounds_and_bypasses_judge():
    sq, judge = _FakeSendQ(), _FakeJudge()
    _emit_curiosity(sq, judge, result=_SUBSTANTIVE)

    rew = _by_type(sq, bus.SELF_LEARN_REWARD)
    assert len(rew) == 1, "exactly one Tier-1 routing reward"
    p = rew[0]["payload"]
    assert p["source"] == "curiosity"
    assert p["goal_class"] == "ground-concept:defi"
    assert p["reward"] > 0.0 and len(p["features"]) > 0

    grounded = _by_type(sq, bus.RESEARCH_CURIOSITY_GROUNDED)
    assert len(grounded) == 1 and grounded[0]["dst"] == "memory"
    gp = grounded[0]["payload"]
    assert gp["_research_target"] == _RT
    assert gp["content"] == _SUBSTANTIVE
    assert gp["query"] == "what is Marinade Finance?"

    assert judge.calls == 0, "the LLM judge MUST be bypassed (no structural -0.5)"


def test_3a_thin_evidence_zero_reward_no_ground_still_bypasses():
    sq, judge = _FakeSendQ(), _FakeJudge()
    _emit_curiosity(sq, judge, result="too short")   # < curiosity_min_evidence_chars

    rew = _by_type(sq, bus.SELF_LEARN_REWARD)
    assert len(rew) == 1 and rew[0]["payload"]["reward"] == 0.0
    assert _by_type(sq, bus.RESEARCH_CURIOSITY_GROUNDED) == []   # no anchor for thin evidence
    assert judge.calls == 0                                       # still bypassed


def test_3a_no_target_falls_through_to_judge():
    """A normal (non-targeted) autonomous research action is unaffected — it goes to
    the judge as before; no curiosity emits."""
    sq, judge = _FakeSendQ(), _FakeJudge()
    _emit_curiosity(sq, judge, result=_SUBSTANTIVE, with_target=False)
    assert _by_type(sq, bus.RESEARCH_CURIOSITY_GROUNDED) == []
    assert not any(m["payload"].get("source") == "curiosity"
                   for m in _by_type(sq, bus.SELF_LEARN_REWARD))
    assert judge.calls >= 1, "no target → normal judge path"


# ── 3b core: anchor-from-dict (no node store needed) ─────────────────────────
def test_3b_anchor_from_synthetic_dict_produces_valid_tx():
    import hashlib
    from titan_hcl.synthesis.promotion_anchor import build_promotion_tx
    content = _SUBSTANTIVE
    nid = -(int(hashlib.sha256(content.encode()).hexdigest()[:12], 16))
    node = {"id": nid, "user_prompt": "q", "agent_response": content,
            "tags": ["acquired:research"], "source_id": "research",
            "neuromod_context": {}}
    payload, tx_hash = build_promotion_tx(node, now=1782300000.0)
    assert isinstance(tx_hash, str) and len(tx_hash) >= 32   # real deterministic hash
    assert payload["content"]["node_id"] == nid
    # determinism: same dict + same now → same hash (idempotent re-anchor)
    _, tx2 = build_promotion_tx(node, now=1782300000.0)
    assert tx_hash == tx2


# ── 3c: name_fn override forces refine-Z (bump), not a sibling ────────────────
class _FakeEngramStore:
    def __init__(self, existing_ids):
        self._existing = set(existing_ids)
        self.created, self.bumped = [], []

    def latest_concept(self, concept_id):
        return {"concept_id": concept_id} if concept_id in self._existing else None

    def create_concept(self, *, concept_id, name, **k):
        self.created.append(concept_id)
        return type("CV", (), {"concept_id": concept_id, "version": 1})()

    def bump_version(self, *, concept_id, **k):
        self.bumped.append(concept_id)
        return type("CV", (), {"concept_id": concept_id, "version": 2})()

    def recompute_groundedness(self, *a, **k):
        pass


class _FakeCgnBridge:
    def register_spine_concept(self, *a, **k):
        pass


def test_3c_name_fn_override_refines_target_not_sibling():
    from titan_hcl.synthesis.research_wiki import seed_research_concept
    # Z already exists in the store → the override must BUMP Z, not create a sibling.
    store = _FakeEngramStore(existing_ids={"marinade_finance"})
    rt = _RT
    override = (lambda c, _t=rt: (_t["concept_id"], _t.get("name") or _t["concept_id"],
                                  _t.get("domain_hint", "") or ""))
    # a librarian name_fn that WOULD mint a sibling if used (proves the override matters)
    sibling = lambda c: ("liquid_staking_general", "Liquid Staking", "defi")

    cv = seed_research_concept(
        engram_store=store, cgn_bridge=_FakeCgnBridge(),
        tx_hash="deadbeef", content=_SUBSTANTIVE, name_fn=override,
        domain_hint="defi", felt_coverage=0.0, created_epoch=1.0)
    assert cv is not None and cv.concept_id == "marinade_finance"
    assert store.bumped == ["marinade_finance"]   # REFINED Z
    assert store.created == []                      # NOT a sibling

    # sanity: without the override the librarian would have created the sibling
    store2 = _FakeEngramStore(existing_ids={"marinade_finance"})
    seed_research_concept(
        engram_store=store2, cgn_bridge=_FakeCgnBridge(),
        tx_hash="deadbeef", content=_SUBSTANTIVE, name_fn=sibling,
        domain_hint="defi", felt_coverage=0.0, created_epoch=1.0)
    assert store2.created == ["liquid_staking_general"] and store2.bumped == []


class _FakeAgencyModule:
    def __init__(self, provider=None):
        self._research_gap_provider = provider
        self._last_research_gap = "stale"


def test_hot_apply_wires_provider_on_off_to_on():
    """RFP §1.4 — flipping research_curiosity_enabled hot-applies via the config-reload
    callback (NO agency restart → avoids the restart-module flap). OFF→ON wires the
    gap provider live."""
    from titan_hcl.modules.agency_worker import _wire_research_curiosity
    ag = _FakeAgencyModule(provider=None)
    on = _wire_research_curiosity(
        ag, {"self_learning": {"research_curiosity_enabled": True}}, "x.json")
    assert on is True
    assert callable(ag._research_gap_provider)   # wired live, no restart


def test_hot_apply_clears_provider_on_to_off():
    from titan_hcl.modules.agency_worker import _wire_research_curiosity
    ag = _FakeAgencyModule(provider=lambda: [])
    on = _wire_research_curiosity(
        ag, {"self_learning": {"research_curiosity_enabled": False}}, "x.json")
    assert on is False
    assert ag._research_gap_provider is None
    assert ag._last_research_gap is None          # anti-repeat state reset too


def test_hot_apply_idempotent_when_already_on():
    """Calling repeatedly with the flag ON must NOT recreate the provider (idempotent —
    the config-watch heartbeat fires on every version bump)."""
    from titan_hcl.modules.agency_worker import _wire_research_curiosity
    cfg = {"self_learning": {"research_curiosity_enabled": True}}
    ag = _FakeAgencyModule(provider=None)
    _wire_research_curiosity(ag, cfg, "x.json")
    first = ag._research_gap_provider
    _wire_research_curiosity(ag, cfg, "x.json")
    assert ag._research_gap_provider is first     # same instance, not recreated


def test_hot_apply_off_stays_off_noop():
    from titan_hcl.modules.agency_worker import _wire_research_curiosity
    ag = _FakeAgencyModule(provider=None)
    on = _wire_research_curiosity(ag, {"self_learning": {}}, "x.json")
    assert on is False and ag._research_gap_provider is None


def test_3c_override_lambda_returns_target_identity():
    rt = _RT
    override = (lambda c, _t=rt: (_t["concept_id"], _t.get("name") or _t["concept_id"],
                                  _t.get("domain_hint", "") or ""))
    cid, name, dom = override("any content")
    assert (cid, name, dom) == ("marinade_finance", "Marinade Finance", "defi")
