"""EEL Pillar B1 — goal_class deterministic labeler (RFP §7.B1 / §1.0 / INV-EEL-8).

The outcome key MUST be deterministic + reproducible (same goal → same class) and
the entity MUST ride as a parameter, never in the key — so one skill generalizes
across entities. These tests pin that contract.
"""
from titan_hcl.synthesis.goal_class import (
    goal_class, make_task_shape, task_shape_for_goal,
)


def test_deterministic_reproducible():
    assert goal_class("What's the TVL of Jupiter?") == goal_class("What's the TVL of Jupiter?")


def test_entity_generalization_one_skill_across_entities():
    # The §1.0 anti-pattern guard: entities are parameters, NOT part of the key.
    assert goal_class("TVL of Jupiter") == goal_class("TVL of Uniswap")
    assert goal_class("TVL of Aave on Solana") == goal_class("TVL of Jupiter")


def test_domain_action_buckets():
    assert goal_class("What's the current TVL of Jupiter on Solana?") == "defi-lookup"
    assert goal_class("what's the SOL price right now") == "market-lookup"
    assert goal_class("verify this python function returns 42") == "code-verify"
    assert goal_class("summarize the latest news on AI") == "web-summarize"
    assert goal_class("post a reply to that tweet") == "social-post"


def test_empty_and_unknown_fall_back_to_general():
    assert goal_class("") == "general-query"
    assert goal_class("tell me about your day") == "general-query"


def test_slug_is_clean():
    gc = goal_class("What's the TVL of Jupiter?!?")
    assert gc == gc.lower()
    assert " " not in gc and "?" not in gc
    assert not gc.startswith("-") and not gc.endswith("-")


def test_make_task_shape_signature():
    ts = make_task_shape("informational", "searxng-search", "defi")
    assert ts == "informational|searxng-search|defi"
    # empty parts collapse cleanly
    assert make_task_shape("computational", "coding_sandbox", "") == "computational|coding_sandbox"


def test_task_shape_for_goal_uses_same_domain_source():
    ts = task_shape_for_goal("informational", "defillama-api", "What's the TVL of Jupiter?")
    assert ts == "informational|defillama-api|defi"


def test_no_llm_no_embedding_pure_function():
    # INV-EEL-1: cheap, reproducible, no side effects — calling twice is identical
    # and never raises on odd input.
    for s in ("", "???", "🚀 moon", "a", "VERIFY THE HASH"):
        assert goal_class(s) == goal_class(s)
        assert isinstance(goal_class(s), str) and goal_class(s)
