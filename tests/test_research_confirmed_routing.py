"""BUG-RESEARCH-CONFIRMED-DST-ALL regression guard.

Commit 942e2652e (2026-06-21) flipped the agno PreHook RESEARCH_CONFIRMED emit
from targeted dst="memory" to dst="all" so that synthesis_worker would receive
it (P3 skill-cell path). But memory_worker is `reply_only=True`, and a reply_only
subscriber is SILENTLY SKIPPED from every dst="all" broadcast fan-out (D-SPEC-42).
The "fix" therefore gained synthesis and broke memory — killing the chat-path
research → memory-anchor → RESEARCH_CONCEPT_SEED → declarative-Engram concept seed
pipeline.

The correct emit is TWO targeted messages (dst="memory" + dst="synthesis"), which
reach both consumers (targeted delivery bypasses both the reply_only skip and the
broadcast filter) without flooding every other subscriber.

Two layers of guard here:
  1. the underlying DivineBus routing invariant (reply_only ↔ targeted-only);
  2. that `_emit_research_confirmed` actually emits two targeted msgs, never dst="all".
"""

from titan_hcl import bus as _bus_mod
from titan_hcl.bus import DivineBus, make_msg
from titan_hcl.modules.agno_hooks import _emit_research_confirmed


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:
            break
    return out


def test_reply_only_skipped_from_broadcast_but_receives_targeted():
    """The invariant that makes dst='all' wrong for a reply_only consumer: a
    reply_only subscriber gets NOTHING from a dst='all' broadcast, but DOES get a
    targeted dst=<name> message; a normal subscriber gets the broadcast."""
    bus = DivineBus()
    mem_q = bus.subscribe("memory", reply_only=True)   # mirrors memory_worker's spec
    syn_q = bus.subscribe("synthesis")                  # normal (wildcard) subscriber

    # A dst="all" broadcast: reply_only memory is skipped; synthesis receives it.
    bus.publish(make_msg(_bus_mod.RESEARCH_CONFIRMED, "pre_hook", "all", {"k": 1}))
    assert _drain(mem_q) == [], "reply_only subscriber must NOT receive dst='all'"
    syn_got = _drain(syn_q)
    assert len(syn_got) == 1 and syn_got[0]["type"] == _bus_mod.RESEARCH_CONFIRMED

    # A targeted dst="memory" message: reply_only memory DOES receive it.
    bus.publish(make_msg(_bus_mod.RESEARCH_CONFIRMED, "pre_hook", "memory", {"k": 2}))
    mem_got = _drain(mem_q)
    assert len(mem_got) == 1 and mem_got[0]["payload"]["k"] == 2
    assert _drain(syn_q) == [], "targeted dst='memory' must not reach synthesis"


class _RecordingBus:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)
        return 1


class _FakePlugin:
    def __init__(self, bus):
        self.bus = bus


def test_research_confirmed_emits_two_targeted_not_broadcast():
    """The fix: _emit_research_confirmed publishes exactly two RESEARCH_CONFIRMED
    messages, targeted to memory AND synthesis, never a dst='all' broadcast."""
    rb = _RecordingBus()
    node = {
        "id": 4242,
        "user_prompt": "what is a transformer model?",
        "agent_response": "an attention-based sequence architecture …",
        "acquired_source": "web_api_oracle",
        "neuromod_context": {"dopamine": 0.4},
    }
    _emit_research_confirmed(_FakePlugin(rb), node)

    rc = [m for m in rb.published if m.get("type") == _bus_mod.RESEARCH_CONFIRMED]
    assert len(rc) == 2, f"expected 2 targeted emits, got {len(rc)}"
    dsts = sorted(m.get("dst") for m in rc)
    assert dsts == ["memory", "synthesis"], dsts
    assert all(m.get("dst") != "all" for m in rc), "must NOT broadcast dst='all'"
    # both carry the same research payload
    for m in rc:
        p = m["payload"]
        assert p["node_id"] == 4242
        assert p["user_prompt"] == "what is a transformer model?"
        assert p["acquired_source"] == "web_api_oracle"
        assert p["felt"] == {"dopamine": 0.4}
