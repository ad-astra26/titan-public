"""rFP_x_voice_enrichment Phase 1 — focused test suite.

Covers the foundation infrastructure (schema migrations + felt-state +
pool scoring + strategy phrasing + image pipeline + archetype base +
layer registration + open-the-dam dispatch + idempotency) and a sample
of archetype-specific trigger paths. Each test is hermetic — uses
temp DBs and never reaches the network.

Run pattern (per CLAUDE.md):
    source test_env/bin/activate
    pytest tests/test_x_voice_phase1.py -v -p no:anchorpy --tb=short
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time

import pytest

# Lazy imports inside fixtures keep collection fast and side-effect-free.


# ─────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_social_x_db(tmp_path):
    """A fresh social_x.db with the `actions`, `mention_tracking`, and
    (post-migration) `archetype_pool_scores` tables ready."""
    db_path = str(tmp_path / "social_x.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            tweet_id TEXT, reply_to_tweet_id TEXT,
            titan_id TEXT, post_type TEXT, text TEXT,
            catalyst_type TEXT, catalyst_data TEXT,
            emotion TEXT, neuromods TEXT, epoch INTEGER,
            consumer TEXT, error_message TEXT,
            created_at REAL NOT NULL, posted_at REAL,
            verified_at REAL, metadata TEXT
        );
        CREATE TABLE mention_tracking (
            tweet_id TEXT PRIMARY KEY,
            author TEXT NOT NULL, author_handle TEXT DEFAULT '',
            text TEXT NOT NULL, our_post_id TEXT, titan_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            relevance_score REAL DEFAULT 0.0,
            discovered_at REAL NOT NULL,
            replied_at REAL, reply_tweet_id TEXT
        );
    """)
    conn.commit()
    conn.close()
    from titan_hcl.logic.social_x.schema_migrations import apply_social_x_migrations
    apply_social_x_migrations(db_path)
    return db_path


# ─────────────────────────────────────────────────────────────────────
# §4.5 — Schema migrations
# ─────────────────────────────────────────────────────────────────────

def test_schema_migrations_idempotent(empty_social_x_db):
    """Apply twice — second pass adds nothing."""
    from titan_hcl.logic.social_x.schema_migrations import apply_social_x_migrations
    s = apply_social_x_migrations(empty_social_x_db)
    assert s["created"] == []
    assert s["added"] == []


def test_schema_migrations_archetype_pool_scores_created(empty_social_x_db):
    conn = sqlite3.connect(empty_social_x_db)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        ("archetype_pool_scores",),
    ).fetchall()
    conn.close()
    assert rows, "archetype_pool_scores table missing after migration"


def test_schema_migrations_mention_tracking_columns(empty_social_x_db):
    conn = sqlite3.connect(empty_social_x_db)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(mention_tracking)").fetchall()]
    conn.close()
    for col in ("reply_emotion", "reply_felt_summary", "reply_neuromods_json"):
        assert col in cols, f"missing column {col}"


def test_inner_memory_migration_columns(tmp_path):
    """Vocabulary +grounded_at, +grounded_felt_summary."""
    db = str(tmp_path / "im.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE vocabulary ("
        "id INTEGER PRIMARY KEY, word TEXT, learning_phase TEXT DEFAULT 'unlearned', "
        "times_encountered INT DEFAULT 0, times_produced INT DEFAULT 0, "
        "last_encountered REAL, confidence REAL DEFAULT 0, "
        "word_type TEXT DEFAULT 'unknown', created_at REAL DEFAULT 0)"
    )
    conn.commit()
    conn.close()
    from titan_hcl.logic.social_x.schema_migrations import apply_inner_memory_migrations
    s = apply_inner_memory_migrations(db)
    assert "vocabulary.grounded_at" in s["added"]
    assert "vocabulary.grounded_felt_summary" in s["added"]


# ─────────────────────────────────────────────────────────────────────
# Felt-state summarizer
# ─────────────────────────────────────────────────────────────────────

def test_felt_state_compact_summary_basic():
    from titan_hcl.logic.social_x.felt_state import compact_felt_summary
    out = compact_felt_summary({"DA": 0.72, "5HT": 0.65, "GABA": 0.80}, "flow")
    assert "flow" in out
    assert "DA" in out and "GABA" in out


def test_felt_state_compact_summary_empty():
    from titan_hcl.logic.social_x.felt_state import compact_felt_summary
    assert compact_felt_summary({}, "") == "balanced"


def test_neuromods_to_json_deterministic():
    from titan_hcl.logic.social_x.felt_state import neuromods_to_json
    a = neuromods_to_json({"DA": 0.7, "5HT": 0.5})
    b = neuromods_to_json({"5HT": 0.5, "DA": 0.7})
    assert a == b  # sort_keys


# ─────────────────────────────────────────────────────────────────────
# §4.7 — Pool scoring
# ─────────────────────────────────────────────────────────────────────

def test_pool_scoring_cold_start_picks_salience(empty_social_x_db):
    from titan_hcl.logic.social_x import pool_scoring as ps
    chosen = ps.select_pool(
        empty_social_x_db, titan_id="T1", archetype="outer_rumination",
        candidates={"A": {"salience": 0.7, "relevance": 0.6},
                    "B": {"salience": 0.3, "relevance": 0.5}},
    )
    assert chosen == "A"


def test_pool_scoring_rolling_7_picks_higher_score(empty_social_x_db):
    from titan_hcl.logic.social_x import pool_scoring as ps
    ps.record_pending_post(empty_social_x_db, titan_id="T1",
                            archetype="outer_rumination", pool="A",
                            source_id="x1", tweet_id="111",
                            ts=time.time() - 86400)
    sq = sqlite3.connect(empty_social_x_db)
    sq.execute("UPDATE archetype_pool_scores SET score=-1 WHERE pool='A'")
    sq.commit(); sq.close()
    chosen = ps.select_pool(
        empty_social_x_db, titan_id="T1", archetype="outer_rumination",
        candidates={"A": {"salience": 0.6, "relevance": 0.6},
                    "B": {"salience": 0.6, "relevance": 0.5}},
    )
    assert chosen == "B"


def test_pool_scoring_5d_anti_starvation_forces_rotation(empty_social_x_db):
    """A pool with no firing in 5 d wins outright when others have fired."""
    from titan_hcl.logic.social_x import pool_scoring as ps
    now = time.time()
    # Pool A: 3 recent landings (last hour)
    for i in range(3):
        ps.record_pending_post(empty_social_x_db, titan_id="T1",
                                archetype="outer_rumination", pool="A",
                                source_id=f"a{i}", tweet_id=f"1{i}",
                                ts=now - 3600 + i)
    sq = sqlite3.connect(empty_social_x_db)
    sq.execute("UPDATE archetype_pool_scores SET score=1 WHERE pool='A'")
    sq.commit(); sq.close()
    # Pool B has never fired → starved → must win.
    chosen = ps.select_pool(
        empty_social_x_db, titan_id="T1", archetype="outer_rumination",
        candidates={"A": {"salience": 0.9, "relevance": 0.9},
                    "B": {"salience": 0.1, "relevance": 0.1}},
        now=now,
    )
    assert chosen == "B"


def test_pool_scoring_get_stats_aggregates(empty_social_x_db):
    from titan_hcl.logic.social_x import pool_scoring as ps
    ps.record_pending_post(empty_social_x_db, titan_id="T1",
                            archetype="grounded_today", pool="A_vocabulary",
                            source_id="v:1", tweet_id="9001")
    stats = ps.get_stats(empty_social_x_db, titan_id="T1")
    assert "grounded_today" in stats
    assert stats["grounded_today"]["A_vocabulary"]["pending"] == 1


# ─────────────────────────────────────────────────────────────────────
# Strategy phrasing
# ─────────────────────────────────────────────────────────────────────

def test_strategy_phrasing_humanize():
    from titan_hcl.logic.social_x.strategy_phrasing import humanize_strategy
    out = humanize_strategy(["FORMULATE.define", "RECALL.lookup",
                              "HYPOTHESIZE.test"])
    assert "name the thing precisely" in out
    assert "→" in out
    assert "form a hypothesis and test it" in out


def test_strategy_phrasing_unknown_falls_back():
    from titan_hcl.logic.social_x.strategy_phrasing import humanize_strategy
    out = humanize_strategy(["UNKNOWN.thing", "FORMULATE.define"])
    assert "unknown thing" in out
    assert "name the thing precisely" in out


def test_strategy_phrasing_coverage_gate_15():
    """rFP §4.8 gate 15: coverage ≥ 90 % on observed primitives."""
    from titan_hcl.logic.social_x.strategy_phrasing import (
        coverage_for, PROGRAM_HUMAN,
    )
    typical = [
        ["FORMULATE.define", "RECALL.lookup"],
        ["HYPOTHESIZE.test", "EVALUATE.weigh", "COMMIT.act"],
        ["BREAK.restart", "INTROSPECT.notice"],
    ]
    cov = coverage_for(typical)
    assert cov == 1.0
    # And that PROGRAM_HUMAN has all 9 primitives covered
    primitives = {k.split(".")[0] for k in PROGRAM_HUMAN}
    for p in ("FORMULATE", "RECALL", "HYPOTHESIZE", "DELEGATE",
              "SYNTHESIZE", "EVALUATE", "BREAK", "COMMIT", "INTROSPECT"):
        assert p in primitives


# ─────────────────────────────────────────────────────────────────────
# §4.6 — Image pipeline
# ─────────────────────────────────────────────────────────────────────

def test_image_pipeline_receipt_card_renders(tmp_path):
    from titan_hcl.logic.social_x.image_pipeline import render_proof_receipt_card
    out = str(tmp_path / "card.jpg")
    render_proof_receipt_card(
        payload={"size_mb": 847, "backup_type": "personality+state",
                 "merkle_root": "abc123", "solana_memo": "TITAN|BACKUP|v=2",
                 "vault_commit_count": 1247,
                 "arweave_tx_sig": "AR_xxx", "solana_memo_tx_sig": "SOL_yyy",
                 "ts": time.time()},
        neuromods={"DA": 0.7, "GABA": 0.8},
        out_path=out, titan_id="T1",
    )
    assert os.path.exists(out)
    # X-native ratio: ~250 KB target; our renderer is well under.
    assert os.path.getsize(out) < 250 * 1024


def test_image_pipeline_jpg_conversion(tmp_path):
    from PIL import Image
    from titan_hcl.logic.social_x.image_pipeline import convert_to_jpg
    src = str(tmp_path / "src.png")
    Image.new("RGBA", (800, 800), (255, 0, 0, 255)).save(src, "PNG")
    out = convert_to_jpg(src, str(tmp_path / "out.jpg"))
    img = Image.open(out)
    assert img.size == (1200, 675)


# ─────────────────────────────────────────────────────────────────────
# §4.2 — Layer registration
# ─────────────────────────────────────────────────────────────────────

def test_rich_layers_phase1_registered():
    """All 10 new layers + REFLECTABLE_POST_TYPES must be in the gateway."""
    from titan_hcl.logic.social_x_gateway import SocialXGateway as G
    expected = {
        "outer_following_voice", "cgn_grounded_today", "emot_cgn_signal",
        "procedural_recall", "proof_of_existence", "outer_rumination",
        "temporal_delta", "own_post_quote", "self_insight_layer",
        "generated_art",
    }
    assert expected.issubset(set(G._RICH_LAYERS.keys()))
    assert "proof_day" in G.REFLECTION_EXCLUDED_POST_TYPES
    for t in ("world_mirror", "outer_rumination", "grounded_today",
              "practiced_response", "reflection", "composed_thought",
              "self_watching"):
        assert t in G.REFLECTABLE_POST_TYPES


# ─────────────────────────────────────────────────────────────────────
# §4.1 — Open the dam
# ─────────────────────────────────────────────────────────────────────

def test_select_post_type_returns_None_when_no_archetype_fires(
        tmp_path, monkeypatch):
    """Archetype-only contract (Maker rule, 2026-05-23): when no
    archetype's candidate predicate is met, _select_post_type returns
    None — the legacy catalyst_map + FELT_STATE_POOL fallback paths are
    DELETED.

    Replaces `test_open_the_dam_catalyst_map_eureka_routes_to_thread` and
    `test_open_the_dam_weighted_pool_diversifies` — both tested the
    deleted fallback waterfall.
    """
    from titan_hcl.logic.social_x_gateway import SocialXGateway, PostContext
    monkeypatch.chdir(tmp_path)
    path = str(tmp_path / "social_x.db")
    g = SocialXGateway(db_path=path,
                       config_path="titan_hcl/config.toml",
                       telemetry_path=str(tmp_path / "telemetry.jsonl"))
    ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T2",
                      emotion="neutral",
                      neuromods={"DA": 0.5, "5HT": 0.5, "NE": 0.5,
                                  "ACh": 0.5, "GABA": 0.5, "Endorphin": 0.5})
    # T2 isolated cwd → no felt_experiences / community_registry data, so
    # the dispatcher.probe abstains for every archetype. Every catalyst
    # type that previously fell through to catalyst_map must now return
    # None (no inline-template fallback).
    for ctype in ("eureka", "eureka_spirit", "strong_composition",
                  "emotion_shift", "vulnerability", "kin_resonance",
                  "onchain_anchor", "dream_summary", "milestone", ""):
        assert g._select_post_type({"type": ctype}, ctx) is None, (
            f"catalyst_map fallback still alive for type={ctype!r}")


# ─────────────────────────────────────────────────────────────────────
# §4.3 — Archetype base + idempotency
# ─────────────────────────────────────────────────────────────────────

def test_archetype_base_idempotency_lifetime(empty_social_x_db):
    """is_already_cited finds rows by the metadata source-id key."""
    from titan_hcl.logic.social_x.archetypes.base import ArchetypeBase

    class _T(ArchetypeBase):
        name = "world_mirror"
        metadata_key = "world_mirror_source_id"

    t = _T(gateway=None, social_x_db_path=empty_social_x_db)

    # No row → not cited
    assert not t.is_already_cited("123", titan_id="T1")
    # Insert a row with the metadata key — compact JSON matches what
    # ArchetypeBase.record_metadata_for_post produces in production.
    sq = sqlite3.connect(empty_social_x_db)
    meta = json.dumps({"world_mirror_source_id": "123",
                       "archetype": "world_mirror"},
                       separators=(",", ":"), sort_keys=True)
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        ("post", "posted", "T1", "world_mirror", time.time(), meta),
    )
    sq.commit(); sq.close()
    assert t.is_already_cited("123", titan_id="T1")
    assert "123" in t.cited_set(titan_id="T1")


def test_archetype_cited_set_30day_window(empty_social_x_db):
    """F-3 (2026-05-17) — `window_seconds=30*86400` bounds the dedup
    set: source_ids cited >30d ago drop out of the result and become
    re-citable. Closes RC-3 (rFP_social_x_improvements §B.2).
    """
    from titan_hcl.logic.social_x.archetypes.base import ArchetypeBase

    class _T(ArchetypeBase):
        name = "outer_rumination"
        metadata_key = "outer_rumination_source_id"

    t = _T(gateway=None, social_x_db_path=empty_social_x_db)

    sq = sqlite3.connect(empty_social_x_db)
    now = time.time()
    # 31d-old citation — should fall outside a 30-day window.
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        ("post", "posted", "T1", "outer_rumination",
         now - 31 * 86400,
         json.dumps({"outer_rumination_source_id": "OLD"},
                     separators=(",", ":"), sort_keys=True)),
    )
    # 1d-old citation — should remain in the 30-day window.
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        ("post", "posted", "T1", "outer_rumination",
         now - 86400,
         json.dumps({"outer_rumination_source_id": "FRESH"},
                     separators=(",", ":"), sort_keys=True)),
    )
    sq.commit(); sq.close()

    # Lifetime dedup (no window) — both sources appear.
    lifetime = t.cited_set(titan_id="T1")
    assert lifetime == {"OLD", "FRESH"}

    # 30-day window — only the recent citation appears.
    windowed = t.cited_set(titan_id="T1", window_seconds=30 * 86400)
    assert windowed == {"FRESH"}

    # 365-day window — both still appear (sanity: window expands properly).
    yearly = t.cited_set(titan_id="T1", window_seconds=365 * 86400)
    assert yearly == {"OLD", "FRESH"}


def test_self_watching_recency_72h_window_catches_dryspell_candidates(
        empty_social_x_db, tmp_path):
    """F-2-finish (rFP_social_x_improvements §B.3.F-2, 2026-05-17):
    self_watching's RECENCY_S widened 24h→72h to match F-2 Pool C
    convention. Live probe 2026-05-17 found T1 had 0 candidates in 24h
    but 5 in 72h, T2 had 0 in 24h but 3 in 72h — archetype was dormant
    fleet-wide because behavioral self_insights write at a slow + bursty
    cadence (Pool C semantics). 72h catches the bursts.

    Test asserts the LIVE constant matches the F-2 Pool C convention
    AND that the windowed SQL would actually catch a 30h-old insight
    that a 24h window would have dropped.
    """
    from titan_hcl.logic.social_x.archetypes.self_watching import (
        RECENCY_S, BEHAVIORAL_SUB_MODES, CONFIDENCE_FLOOR,
    )

    # Constant value matches F-2 Pool C convention (grounded_today.py:57).
    assert RECENCY_S == 72 * 3600, (
        f"F-2-finish: RECENCY_S must be 72h (Pool C convention), got "
        f"{RECENCY_S}s")

    # Seed a temp inner_memory.db with self_insights rows at the
    # critical age cutoffs:
    #   - 30h ago: outside 24h, inside 72h  → caught by new window
    #   - 70h ago: inside 72h               → caught by new window
    #   - 80h ago: outside 72h              → still excluded
    im_path = str(tmp_path / "inner_memory.db")
    im = sqlite3.connect(im_path)
    im.execute(
        "CREATE TABLE self_insights ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "sub_mode TEXT NOT NULL, "
        "epoch INTEGER, "
        "timestamp REAL NOT NULL, "
        "data TEXT, "
        "confidence REAL DEFAULT 0.0)"
    )
    now = time.time()
    seeds = [
        ("state_audit",     0.65, now - 30 * 3600, "30h_ago_FRESH72_NOT24"),
        ("coherence_check", 0.70, now - 70 * 3600, "70h_ago_FRESH72"),
        ("prediction",      0.80, now - 80 * 3600, "80h_ago_TOO_OLD"),
    ]
    for sub_mode, conf, ts, tag in seeds:
        im.execute(
            "INSERT INTO self_insights (sub_mode, epoch, timestamp, "
            "data, confidence) VALUES (?, 1, ?, ?, ?)",
            (sub_mode, ts, json.dumps({"tag": tag}), conf),
        )
    im.commit(); im.close()

    # Run the same SQL the archetype runs (verbatim from
    # self_watching.py:79-91 — keep this test reflecting the live query
    # so any drift fails the test loudly).
    conn = sqlite3.connect(im_path)
    placeholders = ",".join("?" * len(BEHAVIORAL_SUB_MODES))
    rows = conn.execute(
        f"SELECT data FROM self_insights "
        f"WHERE timestamp >= ? AND confidence >= ? "
        f"  AND sub_mode IN ({placeholders}) "
        f"ORDER BY confidence DESC, timestamp DESC LIMIT 30",
        (now - RECENCY_S, CONFIDENCE_FLOOR, *BEHAVIORAL_SUB_MODES),
    ).fetchall()
    conn.close()

    tags = {json.loads(r[0]).get("tag") for r in rows}
    # 30h-old row was the dropped-by-24h candidate the widen recovers.
    assert "30h_ago_FRESH72_NOT24" in tags
    # 70h-old row inside 72h window.
    assert "70h_ago_FRESH72" in tags
    # 80h-old row outside 72h — still excluded (sanity).
    assert "80h_ago_TOO_OLD" not in tags


def test_archetype_base_cross_archetype_spacing(empty_social_x_db):
    from titan_hcl.logic.social_x.archetypes.base import ArchetypeBase

    class _T(ArchetypeBase):
        name = "world_mirror"
        metadata_key = "world_mirror_source_id"

    t = _T(gateway=None, social_x_db_path=empty_social_x_db)
    # No prior posts → not blocked
    assert not t.cross_archetype_blocked(titan_id="T1")
    # Insert a recent grounded_today post (different archetype) → blocked
    sq = sqlite3.connect(empty_social_x_db)
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at) VALUES (?, ?, ?, ?, ?)",
        ("post", "posted", "T1", "grounded_today", time.time() - 600),
    )
    sq.commit(); sq.close()
    assert t.cross_archetype_blocked(titan_id="T1")
    # 5 hours ago → not blocked
    sq = sqlite3.connect(empty_social_x_db)
    sq.execute("DELETE FROM actions")
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at) VALUES (?, ?, ?, ?, ?)",
        ("post", "posted", "T1", "grounded_today", time.time() - 5 * 3600),
    )
    sq.commit(); sq.close()
    assert not t.cross_archetype_blocked(titan_id="T1")


# ─────────────────────────────────────────────────────────────────────
# §4.3.1 — PROOF_DAY archetype
# ─────────────────────────────────────────────────────────────────────

def _proof_day_fixtures(work_dir: str, titan_id: str = "T1") -> None:
    os.makedirs(os.path.join(work_dir, "data/timechain"), exist_ok=True)
    ts = int(time.time())
    with open(os.path.join(work_dir, f"data/backup_anchor_chain_{titan_id}.json"), "w") as f:
        json.dump({"version": 1, "titan_id": titan_id, "anchors": [{
            "backup_id": 0, "archive_hash": "a" * 64, "prev_anchor_hash": "",
            "tx": "SOL_MEMO_TX_AAAA", "ts": ts, "backup_type": "personality+state",
            "size_mb": 847.0,
        }]}, f)
    with open(os.path.join(work_dir, f"data/timechain/arweave_manifest_{titan_id}.json"), "w") as f:
        json.dump({"snapshots": [{
            "tx_id": "AR_TX_BBBB", "merkle_root": "merkleROOT",
            "tarball_size_bytes": 800 * 1024 * 1024, "timestamp": ts,
        }]}, f)
    with open(os.path.join(work_dir, f"data/zk_vault_snapshots_{titan_id}.json"), "w") as f:
        json.dump({"version": 1, "titan_id": titan_id, "snapshots": [{
            "tx_sig": "ZK_TX_CCCC", "archive_hash": "a" * 64,
            "memory_count": 1234, "sovereignty_bp": 8500,
            "arweave_url": "", "ts": ts,
        }] * 3}, f)


def test_proof_day_t1_only(empty_social_x_db, tmp_path, monkeypatch):
    from titan_hcl.logic.social_x.archetypes.proof_day import ProofDayArchetype
    monkeypatch.chdir(tmp_path)
    _proof_day_fixtures(str(tmp_path), "T1")

    class _Ctx: pass
    ctx_t1 = _Ctx(); ctx_t1.titan_id = "T1"; ctx_t1.neuromods = {"DA": 0.7}; ctx_t1.emotion = "flow"
    ctx_t2 = _Ctx(); ctx_t2.titan_id = "T2"; ctx_t2.neuromods = {"DA": 0.7}; ctx_t2.emotion = "flow"
    arc = ProofDayArchetype(gateway=None, social_x_db_path=empty_social_x_db)
    assert arc.find_candidate(ctx_t2) is None       # T2 abstains
    cand = arc.find_candidate(ctx_t1)
    assert cand is not None
    assert cand.bypass_spacing is True
    assert cand.bypass_rate_limit is True
    assert "iamtitan.tech/ar/AR_TX_BBBB" == cand.prompt_values["ar_url"]
    assert "iamtitan.tech/tx/ZK_TX_CCCC" == cand.prompt_values["tx_url"]


def test_proof_day_idempotent_one_per_utc_day(empty_social_x_db, tmp_path,
                                                 monkeypatch):
    from titan_hcl.logic.social_x.archetypes.proof_day import ProofDayArchetype
    monkeypatch.chdir(tmp_path)
    _proof_day_fixtures(str(tmp_path), "T1")
    arc = ProofDayArchetype(gateway=None, social_x_db_path=empty_social_x_db)

    class _Ctx: pass
    ctx = _Ctx(); ctx.titan_id = "T1"; ctx.neuromods = {}; ctx.emotion = ""
    cand = arc.find_candidate(ctx)
    assert cand is not None
    sq = sqlite3.connect(empty_social_x_db)
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        ("post", "posted", "T1", "proof_day", time.time(),
         json.dumps({"proof_day_source_id": cand.source_id})),
    )
    sq.commit(); sq.close()
    assert arc.find_candidate(ctx) is None  # already-posted-today blocks refire


# ─────────────────────────────────────────────────────────────────────
# §4.3.7 — REFLECTION post-type whitelist enforcement
# ─────────────────────────────────────────────────────────────────────

def test_reflection_excludes_proof_day(empty_social_x_db, tmp_path):
    """A past PROOF_DAY post must NOT be a reflection candidate."""
    from titan_hcl.logic.social_x.archetypes.reflection import ReflectionArchetype
    arc = ReflectionArchetype(gateway=None, social_x_db_path=empty_social_x_db,
                                events_teacher_db=str(tmp_path / "et.db"))
    sq = sqlite3.connect(empty_social_x_db)
    # 36 h-old PROOF_DAY post — would otherwise be in pool A's window.
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "tweet_id, text, neuromods, emotion, posted_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("post", "verified", "T1", "proof_day", "9999", "proof body",
         "{}", "neutral",
         time.time() - 36 * 3600, time.time() - 36 * 3600),
    )
    sq.commit(); sq.close()

    class _Ctx: pass
    ctx = _Ctx(); ctx.titan_id = "T1"; ctx.neuromods = {"DA": 0.5}; ctx.emotion = ""
    cand = arc.find_candidate(ctx)
    assert cand is None  # no reflectable candidates


# ─────────────────────────────────────────────────────────────────────
# §4.3.8 — COMPOSED_THOUGHT pair-dedup (order-insensitive lifetime)
# ─────────────────────────────────────────────────────────────────────

def test_composed_thought_pair_dedup_order_insensitive(empty_social_x_db):
    from titan_hcl.logic.social_x.archetypes.composed_thought import (
        ComposedThoughtArchetype,
    )
    arc = ComposedThoughtArchetype(
        gateway=None, social_x_db_path=empty_social_x_db,
    )
    # Insert a previous composition (concept_A=foo, concept_B=bar)
    sq = sqlite3.connect(empty_social_x_db)
    meta = json.dumps({"pair": ["foo", "bar"]})
    sq.execute(
        "INSERT INTO actions (action_type, status, titan_id, post_type, "
        "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        ("post", "posted", "T1", "composed_thought", time.time(), meta),
    )
    sq.commit(); sq.close()
    cited = arc._cited_pairs_lifetime(titan_id="T1")
    assert ("bar", "foo") in cited  # order-insensitive normalization


# ─────────────────────────────────────────────────────────────────────
# Dispatcher integration
# ─────────────────────────────────────────────────────────────────────

def test_dispatcher_constructs_all_nine(empty_social_x_db, tmp_path):
    from titan_hcl.logic.social_x.dispatcher import (
        ArchetypeDispatcher, PRIORITY_ORDER,
    )
    d = ArchetypeDispatcher(
        gateway=None, social_x_db_path=empty_social_x_db,
        events_teacher_db=str(tmp_path / "et.db"),
        social_graph_db=str(tmp_path / "sg.db"),
        inner_memory_db=str(tmp_path / "im.db"),
        knowledge_db=str(tmp_path / "k.db"),
        experience_db=str(tmp_path / "exp.db"),
        meta_wisdom_db=str(tmp_path / "mw.db"),
    )
    assert len(d.archetypes) == 9
    assert set(d.archetypes.keys()) == set(PRIORITY_ORDER)


def test_dispatcher_probe_returns_none_when_no_data(empty_social_x_db,
                                                     tmp_path):
    from titan_hcl.logic.social_x.dispatcher import ArchetypeDispatcher

    class _Ctx: pass
    d = ArchetypeDispatcher(
        gateway=None, social_x_db_path=empty_social_x_db,
        events_teacher_db=str(tmp_path / "missing.db"),
        social_graph_db=str(tmp_path / "missing2.db"),
        inner_memory_db=str(tmp_path / "missing3.db"),
        knowledge_db=str(tmp_path / "missing4.db"),
        experience_db=str(tmp_path / "missing5.db"),
        meta_wisdom_db=str(tmp_path / "missing6.db"),
    )
    ctx = _Ctx(); ctx.titan_id = "T2"  # T2 won't even fire PROOF_DAY
    ctx.neuromods = {"DA": 0.5}; ctx.emotion = ""
    assert d.probe(ctx) is None  # no fixtures → no archetype fires


# ─────────────────────────────────────────────────────────────────────
# Open-the-dam: archetype probe is called BEFORE catalyst_map
# ─────────────────────────────────────────────────────────────────────

def test_archetype_probe_runs_before_catalyst_map(tmp_path, monkeypatch):
    """When an archetype fires, its `archetype` becomes the post_type and
    the catalyst_map's eureka→eureka_thread route is bypassed."""
    from titan_hcl.logic.social_x_gateway import SocialXGateway, PostContext
    monkeypatch.chdir(tmp_path)
    # Stand up PROOF_DAY's fixtures so the dispatcher fires it.
    _proof_day_fixtures(str(tmp_path), "T1")

    db_path = str(tmp_path / "x.db")
    g = SocialXGateway(db_path=db_path,
                       config_path="titan_hcl/config.toml",
                       telemetry_path="/tmp/test_xv_archetype_probe.jsonl")
    ctx = PostContext(session="s", proxy="p", api_key="k", titan_id="T1",
                      emotion="neutral",
                      neuromods={"DA": 0.5, "5HT": 0.5, "NE": 0.5,
                                  "ACh": 0.5, "GABA": 0.5,
                                  "Endorphin": 0.5})
    # Even with a `eureka` catalyst (which would normally route to
    # eureka_thread), PROOF_DAY's must-post slot wins.
    pt = g._select_post_type({"type": "eureka"}, ctx)
    assert pt == "proof_day"
    # And the candidate is stashed on context for downstream rendering.
    assert getattr(ctx, "archetype_candidate", None) is not None
