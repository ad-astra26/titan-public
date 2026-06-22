"""Concurrent multi-user chat — per-call agno agents, no route lock.

RFP_concurrent_multiuser_chat (umbrella RFP_chat_interface §2 #1). Verifies the
P1 mechanic:
  - `make_agent(ctx, tier)` mints a FRESH agent per call (INV-CC-2) sharing the
    heavy ctx state (INV-CC-3) with a FIXED per-tier model (INV-CC-4).
  - All THREE per-tier fields are baked at construction — model.id (ζ.5),
    model.max_tokens (ζ.6), instructions + reply-guidance (ζ.7).
  - tier=None reproduces the legacy default agent (INV-CC-7 parity).
  - The agno memory-bypass kwargs flow through per call (INV-CC-5).
  - Concurrent arun over distinct agents overlaps + never mixes responses
    (INV-CC-1; agno #3120 absent).
  - Worker flag/classify/per-call helpers behave (default ON kill-switch).

Run: python -m pytest tests/test_concurrent_chat.py -v -p no:anchorpy
"""
import asyncio
import sys
import time
import types
from dataclasses import dataclass

import pytest


# ── Lightweight fakes (no agno / TitanHCL stack needed) ──────────────────────

class _FakeModel:
    """Stand-in for agno's OpenAILike — fresh per get_agno_model() call."""
    def __init__(self, default_id="default-model"):
        self.id = default_id
        self.max_tokens = None


class _RecordingAgent:
    """Records the kwargs make_agent built it with; async arun echoes its id."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self.instructions = kwargs.get("instructions")

    async def arun(self, message, session_id=None, user_id=None):
        # Simulate an LLM call; return THIS agent's own model id so a mixing
        # bug (shared instance) would surface as a wrong/duplicated answer.
        await asyncio.sleep(0.05)
        out = types.SimpleNamespace()
        out.content = f"{message}|model={self.model.id}|max={self.model.max_tokens}"
        return out


class _FakeProvider:
    def __init__(self):
        self.models_built = 0

    def get_agno_model(self):
        self.models_built += 1
        return _FakeModel()

    def resolve_model_class(self, model_class):
        if not model_class:
            return None
        return f"model-{model_class}"


@dataclass
class _Tier:
    model_class: str
    max_tokens: int | None = None
    reply_guidance: str | None = None


def _install_fake_agno_agent(monkeypatch):
    """Patch `agno.agent.Agent` (imported lazily inside make_agent) with the
    recording fake so the bake-in is testable without the real framework."""
    mod = types.ModuleType("agno.agent")
    mod.Agent = _RecordingAgent
    monkeypatch.setitem(sys.modules, "agno.agent", mod)


def _make_ctx(provider):
    from titan_hcl.modules.agno_agent_factory import SharedChatCtx
    return SharedChatCtx(
        provider=provider,
        provider_name="fake",
        db="SHARED_DB",
        pre_hook="PRE",
        post_hook="POST",
        guardrail="GUARD",
        tools=["t1", "t2"],
        skill_context="SKILLS",
        description="desc",
        base_instructions=["base-1", "base-2"],
        # INV-CC-5: the memory-bypass + lean flags must flow through per call.
        agno_kwargs={"name": "Titan", "add_history_to_context": False,
                     "telemetry": False},
    )


# ── make_agent: 3-field bake-in ──────────────────────────────────────────────

def test_make_agent_bakes_all_three_tier_fields(monkeypatch):
    _install_fake_agno_agent(monkeypatch)
    from titan_hcl.modules.agno_agent_factory import make_agent

    ctx = _make_ctx(_FakeProvider())
    tier = _Tier(model_class="fast", max_tokens=80, reply_guidance="Keep it short.")
    agent = make_agent(ctx, tier)

    # ζ.5 model.id, ζ.6 max_tokens
    assert agent.model.id == "model-fast"
    assert agent.model.max_tokens == 80
    # ζ.7 guidance appended to base instructions (exact legacy text)
    assert agent.instructions[:2] == ["base-1", "base-2"]
    assert agent.instructions[-1] == (
        "RESPONSE LENGTH — Keep it short. Write a complete reply that "
        "finishes its thought within that length; never stop mid-sentence."
    )
    # base_instructions list not mutated in place (new list built)
    assert ctx.base_instructions == ["base-1", "base-2"]


def test_make_agent_shares_heavy_state_and_passes_kwargs(monkeypatch):
    _install_fake_agno_agent(monkeypatch)
    from titan_hcl.modules.agno_agent_factory import make_agent

    ctx = _make_ctx(_FakeProvider())
    agent = make_agent(ctx, _Tier(model_class="heavy"))
    k = agent.kwargs
    # INV-CC-3 — heavy state shared from ctx
    assert k["db"] == "SHARED_DB"
    assert k["tools"] == ["t1", "t2"]
    assert k["pre_hooks"] == ["GUARD", "PRE"]
    assert k["post_hooks"] == ["POST"]
    assert k["additional_context"] == "SKILLS"
    # INV-CC-5 — bypass / lean kwargs flow through every per-call agent
    assert k["add_history_to_context"] is False
    assert k["telemetry"] is False
    assert k["name"] == "Titan"


def test_make_agent_tier_none_is_legacy_default(monkeypatch):
    """INV-CC-7 — tier=None reproduces the default agent (no swap, no guidance)."""
    _install_fake_agno_agent(monkeypatch)
    from titan_hcl.modules.agno_agent_factory import make_agent

    ctx = _make_ctx(_FakeProvider())
    agent = make_agent(ctx, None)
    assert agent.model.id == "default-model"   # provider default, unmutated
    assert agent.model.max_tokens is None
    assert agent.instructions == ["base-1", "base-2"]   # no guidance line


def test_make_agent_max_tokens_omitted_when_tier_has_none(monkeypatch):
    _install_fake_agno_agent(monkeypatch)
    from titan_hcl.modules.agno_agent_factory import make_agent

    ctx = _make_ctx(_FakeProvider())
    agent = make_agent(ctx, _Tier(model_class="heavy", max_tokens=None))
    assert agent.model.id == "model-heavy"
    assert agent.model.max_tokens is None  # left at model default — not forced


def test_make_agent_returns_distinct_instances_and_models(monkeypatch):
    """INV-CC-2 / INV-CC-4 — own Agent + own model per call (no shared mutable)."""
    _install_fake_agno_agent(monkeypatch)
    from titan_hcl.modules.agno_agent_factory import make_agent

    provider = _FakeProvider()
    ctx = _make_ctx(provider)
    a1 = make_agent(ctx, _Tier(model_class="fast"))
    a2 = make_agent(ctx, _Tier(model_class="heavy"))
    assert a1 is not a2
    assert a1.model is not a2.model
    assert a1.model.id == "model-fast"
    assert a2.model.id == "model-heavy"
    assert provider.models_built == 2  # a fresh model per call


# ── Concurrency: no serialization, no response mixing ────────────────────────

def test_concurrent_arun_overlaps_and_does_not_mix(monkeypatch):
    """INV-CC-1 — N distinct agents arun concurrently: total ≈ max(single),
    and each request gets ITS OWN correct answer (agno #3120 absent)."""
    _install_fake_agno_agent(monkeypatch)
    from titan_hcl.modules.agno_agent_factory import make_agent

    ctx = _make_ctx(_FakeProvider())
    tiers = {
        "A": _Tier(model_class="fast", max_tokens=80),
        "B": _Tier(model_class="heavy"),
        "C": _Tier(model_class="light", max_tokens=200),
    }

    async def _drive():
        async def one(name):
            agent = make_agent(ctx, tiers[name])
            return await agent.arun(f"msg-{name}")
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(n) for n in ("A", "B", "C")))
        return results, time.perf_counter() - t0

    results, elapsed = asyncio.run(_drive())
    contents = [r.content for r in results]
    # Each request's own message + its own tier model — no cross-talk.
    assert contents[0] == "msg-A|model=model-fast|max=80"
    assert contents[1] == "msg-B|model=model-heavy|max=None"
    assert contents[2] == "msg-C|model=model-light|max=200"
    # 3 × 0.05s serial = 0.15s; concurrent should be well under serial.
    assert elapsed < 0.12, f"chats serialized (elapsed={elapsed:.3f}s)"


# ── Worker helpers: flag + classify + per-call build ─────────────────────────

def test_concurrent_chat_flag_default_on_and_killswitch():
    from titan_hcl.modules import agno_worker as w

    on = types.SimpleNamespace(_full_config={})                       # absent → ON
    off = types.SimpleNamespace(_full_config={"chat": {"concurrent_chat_enabled": False}})
    explicit_on = types.SimpleNamespace(_full_config={"chat": {"concurrent_chat_enabled": True}})
    broken = types.SimpleNamespace()  # no _full_config → fail-safe ON
    assert w._concurrent_chat_enabled(on) is True
    assert w._concurrent_chat_enabled(off) is False
    assert w._concurrent_chat_enabled(explicit_on) is True
    assert w._concurrent_chat_enabled(broken) is True


def test_classify_message_tier_none_without_provider():
    from titan_hcl.modules import agno_worker as w
    wp = types.SimpleNamespace(_inference_provider=None, _full_config={})
    assert w._classify_message_tier(wp, "hello") is None


def test_make_chat_agent_falls_back_to_shared_on_error():
    """No _chat_ctx / build failure → shared agent (chat keeps working)."""
    from titan_hcl.modules import agno_worker as w
    shared = object()
    wp = types.SimpleNamespace()  # no _chat_ctx attribute → make_agent raises
    got = w._make_chat_agent(wp, "hi", shared)
    assert got is shared


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:anchorpy"]))
