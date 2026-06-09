"""EEL Pillar A / Phase A2 — confirm → score → promote.

RFP_emergent_experience_learning §7.A2 / INV-EEL-2/6: the next user turn after a
researched answer is classified {confirm,dispute,neutral}; confirm `+δ` /
silence `+δ_weak` → the node's `confirmation_score` rises → meditation promotes
it; dispute `−δ` → prune (+ re-research). The pending window persists on the node
(survives restart). This suite covers the classifier, the reinforce/window
primitives, and the extracted meditation promote-gate.
"""
from __future__ import annotations

import pytest

from titan_hcl.synthesis.confirmation_intent import detect_confirmation
from titan_hcl.logic.meditation import classify_research_node
from titan_hcl.core.memory import TieredMemoryGraph


# ── the deterministic classifier (the research oracle, INV-EEL-2) ────────────

@pytest.mark.parametrize("turn,expected", [
    ("yep, thanks!", "confirm"),
    ("that's correct", "confirm"),
    ("perfect, that helps", "confirm"),
    ("yes", "confirm"),
    ("no, that's wrong", "dispute"),
    ("that's not right, search again", "dispute"),
    ("hmm that seems outdated", "dispute"),
    ("nope", "dispute"),
    ("no problem", "neutral"),          # benign "no" must NOT be a dispute
    ("what about Ethereum then?", "neutral"),
    ("", "neutral"),
])
def test_detect_confirmation(turn, expected):
    assert detect_confirmation(turn) == expected


# ── the extracted meditation promote-gate (A2 crux) ──────────────────────────

def test_gate_confirmed_promotes():
    node = {"confirmation_score": 1.0}
    decision, cs = classify_research_node(node, weak_delta=0.3)
    assert decision == "promote" and cs == 1.0


def test_gate_disputed_prunes():
    node = {"confirmation_score": -1.0}
    decision, cs = classify_research_node(node, weak_delta=0.3)
    assert decision == "prune" and cs == -1.0


def test_gate_silent_weak_confirms_in_place():
    """A node still unresolved at meditation (score 0) → weak-confirm → promote,
    and the mutation is written back (score=δ_weak, window cleared)."""
    node = {"confirmation_score": 0.0, "confirm_turns_left": 2}
    decision, cs = classify_research_node(node, weak_delta=0.3)
    assert decision == "promote" and cs == 0.3
    assert node["confirmation_score"] == 0.3
    assert node["confirm_turns_left"] is None


# ── reinforce(delta) + the pending-window primitives ─────────────────────────

@pytest.mark.asyncio
async def test_reinforce_delta_moves_confirmation_score_and_clears_window(tmp_path):
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    nid = await mem.add_to_mempool(
        "What is the TVL of Kamino?", "~$X", "user_a",
        tags=["acquired:research"], source="searxng:x")
    # research node opens a confirm window
    node = mem._node_store[nid]
    assert node["confirm_turns_left"] == 3       # [eel.confirm] confirm_window_turns
    assert node["confirmation_score"] == 0.0

    mem.reinforce_mempool_node(nid, 1.0)         # confirm +δ
    assert node["confirmation_score"] == 1.0
    assert node["confirm_turns_left"] is None    # window resolved


@pytest.mark.asyncio
async def test_reinforce_legacy_no_delta_unchanged(tmp_path):
    """The 3 legacy callers pass only node_id → survival bump, no confirmation."""
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    nid = await mem.add_to_mempool("hello", "hi", "user_a")
    before = mem._node_store[nid].get("mempool_reinforcements", 0)
    mem.reinforce_mempool_node(nid)              # delta=None (legacy)
    node = mem._node_store[nid]
    assert node["mempool_reinforcements"] == before + 1
    assert node["confirmation_score"] == 0.0     # untouched


@pytest.mark.asyncio
async def test_dispute_does_not_bump_survival(tmp_path):
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    nid = await mem.add_to_mempool(
        "TVL of Kamino?", "~$X", "user_a", tags=["acquired:research"], source="s")
    before = mem._node_store[nid].get("mempool_reinforcements", 0)
    mem.reinforce_mempool_node(nid, -1.0)        # dispute → must NOT extend life
    node = mem._node_store[nid]
    assert node["confirmation_score"] == -1.0
    assert node["mempool_reinforcements"] == before  # no survival bump


@pytest.mark.asyncio
async def test_plain_turn_has_no_window(tmp_path):
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    nid = await mem.add_to_mempool("just chatting", "ok", "user_a")
    assert mem._node_store[nid]["confirm_turns_left"] is None


@pytest.mark.asyncio
async def test_find_pending_scoped_to_user_and_resolves_on_confirm(tmp_path):
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    nid = await mem.add_to_mempool(
        "TVL of Kamino?", "~$X", "user_a", tags=["acquired:research"], source="s")
    assert [n["id"] for n in mem.find_pending_confirmation_nodes("user_a")] == [nid]
    assert mem.find_pending_confirmation_nodes("user_b") == []   # scoped to the user
    mem.reinforce_mempool_node(nid, 1.0)         # confirm resolves the window
    assert mem.find_pending_confirmation_nodes("user_a") == []


@pytest.mark.asyncio
async def test_tick_window_expires_to_weak_confirm(tmp_path):
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    nid = await mem.add_to_mempool(
        "TVL of Kamino?", "~$X", "user_a", tags=["acquired:research"], source="s")
    assert mem.tick_confirmation_window(nid, 0.3) == "pending"   # 3 → 2
    assert mem.tick_confirmation_window(nid, 0.3) == "pending"   # 2 → 1
    assert mem.tick_confirmation_window(nid, 0.3) == "expired_weak"  # 1 → 0 → weak
    node = mem._node_store[nid]
    assert node["confirmation_score"] == pytest.approx(0.3)
    assert node["confirm_turns_left"] is None


@pytest.mark.asyncio
async def test_pending_window_survives_reload(tmp_path):
    """Ad-1: the pending state is persisted on the node → survives a restart."""
    mem = TieredMemoryGraph(config={"data_dir": str(tmp_path)})
    nid = await mem.add_to_mempool(
        "TVL of Kamino?", "~$X", "user_a", tags=["acquired:research"], source="s")
    mem.reinforce_mempool_node(nid, 1.0)         # confirm (persists confirmation_score)
    mem._node_store.clear()
    mem._load_node_store()
    node = mem._node_store[nid]
    assert node["confirmation_score"] == 1.0     # survived reload
    assert node["confirm_turns_left"] is None
