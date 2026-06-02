"""Tests for the 2026-06-02 X-posting fixes (Maker session):

  * Universal same-archetype intra-spacing (ArchetypeBase.same_archetype_blocked)
        — the cross-archetype gate excludes self, so without this an archetype
        whose daily cap is ≥2 re-fired on the next ~2h tick → near-identical
        posts a couple hours apart. WORLD_MIRROR/AMPLIFY already had a bespoke
        6h guard; it is now generalized to every archetype (PROOF_DAY opts out).
        Only timeline-reaching statuses (posted/verified/pending) block; a
        `failed` attempt that never went out must not block the retry.
  * OVG prose↔state consistency on external channels — the X-post/reply path
        now injects the felt-state ground truth and `_check_consistency`
        validates the prose figures against it. Consistency stays a SOFT
        warning for chat but is a HARD block on x_post/x_reply so a hallucinated
        number never reaches the public timeline.
"""

import sqlite3
import time

from titan_hcl.logic.output_verifier import (
    OutputVerifier, _compile_context_patterns,
)
from titan_hcl.logic.social_x.archetypes.base import (
    ArchetypeBase, DEFAULT_SAME_ARCHETYPE_SPACING_S,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_actions_db(path):
    c = sqlite3.connect(path)
    c.execute(
        "CREATE TABLE actions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "action_type TEXT, status TEXT, tweet_id TEXT, titan_id TEXT, "
        "post_type TEXT, text TEXT, consumer TEXT, created_at REAL, "
        "posted_at REAL, metadata TEXT)")
    c.commit()
    c.close()


def _insert(path, *, titan_id="T1", post_type="composed_thought",
            status="verified", created_at=None):
    c = sqlite3.connect(path)
    c.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at) VALUES ('post', ?, ?, ?, ?)",
        (status, titan_id, post_type,
         created_at if created_at is not None else time.time()))
    c.commit()
    c.close()


class _Arch(ArchetypeBase):
    name = "composed_thought"

    def find_candidate(self, context):
        return None


# ── same_archetype_blocked ───────────────────────────────────────────────


def test_blocks_when_same_archetype_within_window(tmp_path):
    db = str(tmp_path / "x.db")
    _make_actions_db(db)
    now = time.time()
    _insert(db, post_type="composed_thought", created_at=now - 2 * 3600)  # 2h ago
    arch = _Arch(gateway=None, social_x_db_path=db)
    assert arch.same_archetype_blocked(titan_id="T1", now=now) is True


def test_not_blocked_when_older_than_window(tmp_path):
    db = str(tmp_path / "x.db")
    _make_actions_db(db)
    now = time.time()
    _insert(db, post_type="composed_thought", created_at=now - 7 * 3600)  # 7h ago
    arch = _Arch(gateway=None, social_x_db_path=db)
    assert arch.same_archetype_blocked(titan_id="T1", now=now) is False


def test_failed_post_does_not_block(tmp_path):
    db = str(tmp_path / "x.db")
    _make_actions_db(db)
    now = time.time()
    # A failed attempt 1h ago never reached the timeline — must NOT block.
    _insert(db, post_type="composed_thought", status="failed",
            created_at=now - 3600)
    arch = _Arch(gateway=None, social_x_db_path=db)
    assert arch.same_archetype_blocked(titan_id="T1", now=now) is False


def test_pending_post_blocks(tmp_path):
    db = str(tmp_path / "x.db")
    _make_actions_db(db)
    now = time.time()
    _insert(db, post_type="composed_thought", status="pending",
            created_at=now - 3600)
    arch = _Arch(gateway=None, social_x_db_path=db)
    assert arch.same_archetype_blocked(titan_id="T1", now=now) is True


def test_other_archetype_does_not_block(tmp_path):
    db = str(tmp_path / "x.db")
    _make_actions_db(db)
    now = time.time()
    _insert(db, post_type="grounded_today", created_at=now - 3600)
    arch = _Arch(gateway=None, social_x_db_path=db)  # name == composed_thought
    assert arch.same_archetype_blocked(titan_id="T1", now=now) is False


def test_custom_spacing_seconds_honored(tmp_path):
    db = str(tmp_path / "x.db")
    _make_actions_db(db)
    now = time.time()
    _insert(db, post_type="composed_thought", created_at=now - 5 * 3600)  # 5h ago
    arch = _Arch(gateway=None, social_x_db_path=db)
    # Default 6h → blocked; a 4h spacing → not blocked.
    assert arch.same_archetype_blocked(titan_id="T1", now=now) is True
    assert arch.same_archetype_blocked(
        titan_id="T1", now=now, spacing_seconds=4 * 3600) is False


def test_default_spacing_is_six_hours():
    assert DEFAULT_SAME_ARCHETYPE_SPACING_S == 6 * 3600


# ── OVG context patterns (neuromod % + backup size) ──────────────────────


def _consistency_violations(text, context):
    pats = _compile_context_patterns()
    viol = []
    for pattern, claim in pats:
        om = pattern.search(text)
        if not om:
            continue
        ov = next((g for g in om.groups() if g), None)
        if not ov:
            continue
        cm = pattern.search(context)
        if not cm:
            continue
        cv = next((g for g in cm.groups() if g), None)
        if not cv:
            continue
        try:
            o, c = float(ov.replace(",", "")), float(cv.replace(",", ""))
            if c > 0 and abs(o - c) / c > 0.1:
                viol.append(claim)
        except (ValueError, ZeroDivisionError):
            pass
    return viol


def test_neuromod_and_size_patterns_pass_faithful_prose():
    ctx = "GABA 9.5% · endorphin 69.1% · I-confidence is 0.950 · 568MB"
    prose = ("568MB of state sealed. GABA low at 10%, endorphins high at 69%. "
             "My I-confidence is 0.950.")
    assert _consistency_violations(prose, ctx) == []


def test_neuromod_and_size_patterns_catch_hallucination():
    ctx = "GABA 9.5% · endorphin 69.1% · I-confidence is 0.950 · 568MB"
    prose = ("950MB of state sealed. GABA roaring at 80%, endorphins at 12%. "
             "My I-confidence is 0.40.")
    found = set(_consistency_violations(prose, ctx))
    assert "backup_size_mb" in found
    assert "neuromod_GABA" in found
    assert "neuromod_Endorphin" in found
    assert "i_confidence" in found


# ── OVG channel-aware severity ───────────────────────────────────────────


def _ovg():
    return OutputVerifier(titan_id="T1")


def test_consistency_blocks_on_x_post():
    ov = _ovg()
    r = ov.verify_and_sign(
        "My I-confidence is 0.40 right now.",
        channel="x_post",
        injected_context="I-confidence is 0.950")
    assert r.checks["consistency"] is False
    assert r.passed is False
    assert r.violation_type == "consistency"


def test_consistency_blocks_on_x_reply():
    ov = _ovg()
    r = ov.verify_and_sign(
        "My I-confidence is 0.40 right now.",
        channel="x_reply",
        injected_context="I-confidence is 0.950")
    assert r.passed is False


def test_consistency_stays_soft_on_chat():
    ov = _ovg()
    r = ov.verify_and_sign(
        "My I-confidence is 0.40 right now.",
        channel="chat",
        injected_context="I-confidence is 0.950")
    # Chat keeps the historical soft-warning semantics.
    assert r.checks["consistency"] is False
    assert r.passed is True


def test_faithful_prose_passes_on_x_post():
    ov = _ovg()
    r = ov.verify_and_sign(
        "My I-confidence is 0.95 and vocabulary 540 words.",
        channel="x_post",
        injected_context="I-confidence is 0.950 · vocabulary 542 words")
    assert r.checks["consistency"] is True
    assert r.passed is True
