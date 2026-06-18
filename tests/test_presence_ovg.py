"""Phase E — OVG verified-context integration. Proves §7.E: the presence block makes
an HONEST recognition claim PASS the real OVG `_check_qualia`, while a claim with NO
record stays blocked. `output_verifier.py` is UNCHANGED — we feed the gate.

  • CHAINED + crypto → block carries recognition+recency keywords → "I recognize you,
    last spoke ~2 days ago" passes Q2 (no HARD memory-claim violation);
  • UNSEALED + crypto (the 5-min-ago case) → STILL passes (recognize-on-validity);
  • no record → empty block → "I recognize you" with no context → HARD-blocked;
  • the block lands within the OVG's first-500-char window;
  • honesty gradient: crypto → confident wording; asserted → tentative/hedge wording.
"""
import pytest

from titan_hcl.logic.output_verifier import OutputVerifier
from titan_hcl.logic.presence_recall import render_presence_context_block


def _hard_memory_violation(violations):
    return any(v.startswith("HARD") and "memory claim" in v for v in violations)


@pytest.fixture
def ov():
    return OutputVerifier(titan_id="test")


def test_chained_crypto_recognition_passes_ovg(ov):
    block = render_presence_context_block({
        "gap_human": "~2 days ago", "evidence_strength": "crypto_verified_maker",
        "chain_status": "CHAINED", "person_id": "maker"})
    assert "recognize" in block.lower() and "last saw" in block.lower()
    text = ("I recognize you — your signature is verified; we last spoke about "
            "two days ago.")
    _ok, violations = ov._check_qualia(text, injected_context=block[:500], chain_state={})
    assert not _hard_memory_violation(violations)   # supported by the CHAINED block → PASSES


def test_unsealed_fresh_recognition_passes_ovg(ov):
    # the "5 minutes ago" case — recognized even though the memory is unsealed
    block = render_presence_context_block({
        "gap_human": "~5 minutes ago", "evidence_strength": "crypto_verified_maker",
        "chain_status": "UNSEALED", "person_id": "maker"})
    text = "I recognize you — we last spoke about 5 minutes ago."
    _ok, violations = ov._check_qualia(text, injected_context=block[:500], chain_state={})
    assert not _hard_memory_violation(violations)


def test_no_record_blocks_recognition(ov):
    block = render_presence_context_block(None)
    assert block == ""                              # no record → no block
    text = "I recognize you — we spoke last week."
    _ok, violations = ov._check_qualia(text, injected_context="", chain_state={})
    assert _hard_memory_violation(violations)        # unsupported claim → HARD block (honest)


def test_block_is_within_500_char_window():
    block = render_presence_context_block({
        "gap_human": "~3 hours ago", "evidence_strength": "crypto_verified_maker",
        "chain_status": "CHAINED", "person_id": "maker"})
    # placed FIRST → the recognition vocabulary is inside the OVG window even with a
    # large trailing context.
    injected = block + ("### Other Context\n" + "x " * 1000)
    assert "recognize" in injected[:500].lower()
    assert "last saw" in injected[:500].lower()


def test_honesty_gradient_wording():
    asserted = render_presence_context_block({
        "gap_human": "~1 day ago", "evidence_strength": "asserted_identity",
        "chain_status": "CHAINED", "person_id": "alice"})
    # asserted identity → tentative/hedge guidance
    assert "tentativ" in asserted.lower() or "think" in asserted.lower()
    crypto = render_presence_context_block({
        "gap_human": "~1 day ago", "evidence_strength": "crypto_verified_maker",
        "chain_status": "CHAINED", "person_id": "maker"})
    assert "confidence" in crypto.lower()
    # provability gradient: UNSEALED says "recent / not yet sealed"
    unsealed = render_presence_context_block({
        "gap_human": "~2 hours ago", "evidence_strength": "crypto_verified_maker",
        "chain_status": "UNSEALED", "person_id": "maker"})
    assert "not yet sealed" in unsealed.lower()
