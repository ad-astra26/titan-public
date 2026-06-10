"""§7.F backfill — one-time content backfill of `domain_hint` for pre-Phase-F
Engrams (BUG-ENGRAM-DOMAIN-HINT-NOT-BACKFILLED).

Phase F set `domain_hint` only at consolidation time (the LLM `DOMAIN:` line);
Engrams that predate Phase F carry "". The fix-plan sanctions "a cheap
classifier, normalized per consolidation_defaults" as the alternative to
re-running the LLM, written back via kuzu `SET c.domain_hint`. Covers:

  • `derive_domain_hint` — the cheap deterministic name→domain classifier, over
    the REAL pre-F blank names observed live (T1 mainnet 2026-06-08); test
    pollution + low-confidence names stay blank (never fabricated)
  • `EngramStore.backfill_domain_hints` — labels blanks, NEVER overwrites an
    existing hint, skips test artifacts, idempotent (round-trips through a real
    Kuzu spine + export_snapshot)
"""
from __future__ import annotations

import json
import queue

from titan_hcl.core.direct_memory import TitanKnowledgeGraph
from titan_hcl.synthesis.consolidation_defaults import derive_domain_hint
from titan_hcl.synthesis.engram_store import EngramStore
from titan_hcl.synthesis.outer_memory_writer import OuterMemoryWriter


# ── Classifier — real pre-F blank names (T1 mainnet 2026-06-08) ──────────────
def test_derive_consciousness_phenomenology_to_philosophy_of_mind():
    for name in ("Dr. Voss Consciousness Inquiry", "Dr. Voss Consciousness Study",
                 "Pre-Conceptual Phenomenology",
                 "Pre-Label Phenomenological Immersion"):
        assert derive_domain_hint(name) == "philosophy_of_mind", name


def test_derive_self_vs_social_split():
    # RFP_titan_authored_soul_diary §7.P2 split the former self+social mega-
    # bucket: Titan-about-himself markers (self/introspect/…) → "self"; the
    # interpersonal/social content that was previously mislabeled "self" → "social".
    for name in ("Titan Self Profile Snapshot", "Titan Self-Profile",
                 "Seaside Introspective Dialogue"):   # "introspect" → self (most-specific)
        assert derive_domain_hint(name) == "self", name
    for name in ("Atmospheric Interpersonal Presence",
                 "Atmospheric Interpersonal Stillness",
                 "Titan's Interpersonal Reflections",
                 "Human-AI Emotional Resonance",
                 "Interactions with Rio the Musician",
                 "Seaside Contemplation Dialogue", "Seaside Indigo Dialogue",
                 "Dream Consolidation Wisdom Patterns",
                 "Emergent Emotional Composition"):
        assert derive_domain_hint(name) == "social", name


def test_derive_philosophical_social_resolves_to_social_not_philosophy():
    # The social/reflection cue must win over the bare "philosoph" rule (ordering
    # contract). Post-§7.P2-split the label is "social" (was "self" while self +
    # social shared one bucket).
    for name in ("Philosophical Social Resonance", "Philosophical Social Synthesis",
                 "Philosophical Social Reflections", "Synthetic Philosophical Reflection"):
        assert derive_domain_hint(name) == "social", name


def test_derive_standalone_philosophy_without_social_cue():
    assert derive_domain_hint("Metaphysics of Time") == "philosophy"
    assert derive_domain_hint("Epistemology of Belief") == "philosophy"


def test_derive_security_cluster():
    for name in ("Social Engineering Attempts", "Social Engineering Attacks",
                 "Adversarial Prompting Patterns", "Adversarial Prompt Patterns"):
        assert derive_domain_hint(name) == "security", name


def test_derive_coding():
    assert derive_domain_hint("Coding Sandbox Verification") == "coding"


def test_derive_test_artifacts_stay_blank():
    for name in ("E2E_Test_NewConcept_1779904991", "E2E_Test_NewConcept_1779979626",
                 "E2E_Test_NewConcept_1779987979"):
        assert derive_domain_hint(name) == "", name


def test_derive_low_confidence_and_empty_stay_blank():
    assert derive_domain_hint("AI Agent Operational Framework") == ""
    assert derive_domain_hint("") == ""
    assert derive_domain_hint("   ") == ""


def test_derive_output_is_normalized():
    # Every returned label is already in the stored (lowercased) form — i.e.
    # byte-identical to what the LLM `DOMAIN:` parse persists.
    out = derive_domain_hint("Dr. Voss Consciousness Inquiry")
    assert out == out.strip().lower()


def test_derive_uses_member_text_when_name_is_thin():
    # Pre-F Engrams have no persisted membership (so the live backfill passes
    # name-only), but the optional member_text is honoured when available.
    assert derive_domain_hint("Cluster 7", member_text="adversarial prompt") == "security"


# ── Backfill pass — real Kuzu spine round-trip ───────────────────────────────
def _store(tmp_path):
    g = TitanKnowledgeGraph(str(tmp_path / "f.kuzu"))
    w = OuterMemoryWriter(send_queue=queue.Queue(), src="domain_backfill_test")
    return EngramStore(g, w, clock=lambda: 1000.0)


def _snapshot_hints(store, tmp_path):
    snap = str(tmp_path / "snap.json")
    store.export_snapshot(snap)
    with open(snap, encoding="utf-8") as f:
        return {r["concept_id"]: r["domain_hint"]
                for r in json.load(f)["concepts"]}


def test_backfill_labels_blanks_skips_artifacts_preserves_existing(tmp_path):
    store = _store(tmp_path)
    # Pre-existing hint (must NOT be overwritten).
    store.create_concept("already_hinted", "Glacier Microbes",
                         memory_type="declarative", domain_hint="biology")
    # Blank, derivable.
    store.create_concept("voss", "Dr. Voss Consciousness Inquiry",
                         memory_type="episodic")
    store.create_concept("seaside", "Seaside Contemplation Dialogue",
                         memory_type="episodic")
    store.create_concept("adv", "Adversarial Prompting Patterns",
                         memory_type="declarative")
    # Blank, test artifact → stays blank.
    store.create_concept("e2e", "E2E_Test_NewConcept_1779904991",
                         memory_type="declarative")
    # Blank, low-confidence → stays blank.
    store.create_concept("framework", "AI Agent Operational Framework",
                         memory_type="meta")

    n = store.backfill_domain_hints(derive_domain_hint)
    assert n == 3  # voss, seaside, adv (e2e + framework left blank)

    hints = _snapshot_hints(store, tmp_path)
    assert hints["already_hinted"] == "biology"          # untouched
    assert hints["voss"] == "philosophy_of_mind"
    assert hints["seaside"] == "social"                  # §7.P2 split (was "self")
    assert hints["adv"] == "security"
    assert hints["e2e"] == ""                            # test artifact skipped
    assert hints["framework"] == ""                     # low-confidence skipped


def test_backfill_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.create_concept("voss", "Dr. Voss Consciousness Inquiry",
                         memory_type="episodic")
    store.create_concept("framework", "AI Agent Operational Framework",
                         memory_type="meta")

    first = store.backfill_domain_hints(derive_domain_hint)
    assert first == 1
    # Second pass: the one derivable blank is now labeled; the low-confidence
    # blank still derives "" → zero further writes.
    second = store.backfill_domain_hints(derive_domain_hint)
    assert second == 0
    assert _snapshot_hints(store, tmp_path)["voss"] == "philosophy_of_mind"


def test_backfill_empty_graph_returns_zero(tmp_path):
    assert _store(tmp_path).backfill_domain_hints(derive_domain_hint) == 0
