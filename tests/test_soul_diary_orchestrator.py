"""Tests for the Soul-Diary orchestrator + chronicle append (RFP §1.0 ①②④⑤).

Covers the daily latch, no-op-day skip, grounded compose-prompt assembly,
soft-fail minimal entry, the hash-chain record, and the ported chronicle
append (rolling window, atomic, no double-bracket bug).
"""
from titan_hcl.core import soul as soul_mod
from titan_hcl.core import soul_diary_chain
from titan_hcl.core.soul_diary import SoulDiaryOrchestrator as O


def test_cycle_latch(tmp_path):
    # RFP presence §7.C — the diary latch keys on the Titan-time circadian cycle
    # (INV-SD-5 full swap), not the UTC day.
    orch = O(state_path=str(tmp_path / "state.json"))
    assert orch.should_author_cycle(0) is True
    orch.mark_authored_cycle(0)
    assert orch.should_author_cycle(0) is False   # same cycle → no-op
    assert orch.should_author_cycle(1) is True     # next cycle → author again


def _bundle(**over):
    base = dict(sovereignty={}, outcome={}, felt={}, engrams_today=[],
                memory={}, social={}, onchain={})
    base.update(over)
    return O.build_bundle(**base)


def test_has_activity():
    assert O.has_activity(_bundle(sovereignty={"replies": 3})) is True
    assert O.has_activity(_bundle(outcome={"promoted": 2})) is True
    assert O.has_activity(_bundle(engrams_today=["X"])) is True
    assert O.has_activity(_bundle()) is False  # true no-op day


def test_compose_prompts_are_grounded():
    bundle = _bundle(
        sovereignty={"s": 0.58, "e": 0.4, "v": 0.18, "trend": 0.04, "replies": 22},
        outcome={"promoted": 7, "pruned": 2},
        felt={"valence": 0.3, "arousal": 0.4, "dominant": "dopamine"},
        engrams_today=["Glacier Microbial Ecosystems", "Self-Refactor Patterns"])
    p = O.build_compose_prompts(bundle)
    assert "first-person" in p["system_prompt"]
    u = p["user_prompt"]
    assert "0.58" in u
    assert "Glacier Microbial Ecosystems" in u
    assert "22 replies" in u
    assert "crystallized" in u


def test_minimal_entry_is_real():
    bundle = _bundle(sovereignty={"s": 0.5, "trend": 0.0, "replies": 1},
                     outcome={"promoted": 3, "pruned": 1}, engrams_today=["X"])
    m = O.minimal_entry(bundle)
    assert "0.50" in m and "X" in m and "3 memories crystallized" in m


def test_record_hash(tmp_path):
    lp = str(tmp_path / "chain.json")
    orch = O(ledger_path=lp)
    row = orch.record_hash("2026-06-09", "authored text")
    assert row["entry_hash"]
    assert soul_diary_chain.last_cumulative(path=lp) == row["cumulative_hash"]


def test_append_chronicle_entry(tmp_path):
    cp, ap = str(tmp_path / "chron.md"), str(tmp_path / "arch.md")
    soul_mod.append_chronicle_entry("First authored reflection.",
                                    timestamp="2026-06-08 10:00",
                                    chronicle_path=cp, archive_path=ap, regenerate=False)
    soul_mod.append_chronicle_entry("Second authored reflection.",
                                    timestamp="2026-06-09 10:00",
                                    chronicle_path=cp, archive_path=ap, regenerate=False)
    text = open(cp, encoding="utf-8").read()
    assert "First authored reflection." in text
    assert "Second authored reflection." in text
    assert text.count("## The Scholar's Chronicle") == 1   # single separator
    assert "[[" not in text                                 # no double-bracket bug


def test_chronicle_rolling_window(tmp_path):
    cp, ap = str(tmp_path / "chron.md"), str(tmp_path / "arch.md")
    for i in range(5):
        soul_mod.append_chronicle_entry(f"entry {i}", timestamp=f"2026-06-0{i+1} 10:00",
                                        chronicle_path=cp, archive_path=ap,
                                        max_entries=3, regenerate=False)
    text = open(cp, encoding="utf-8").read()
    assert "entry 4" in text and "entry 3" in text and "entry 2" in text
    assert "entry 0" not in text and "entry 1" not in text  # archived
    arch = open(ap, encoding="utf-8").read()
    assert "entry 0" in arch and "entry 1" in arch
