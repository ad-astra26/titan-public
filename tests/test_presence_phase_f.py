"""Phase F tests — RFP_verifiable_autobiographical_presence_memory §7.F.

Covers: F.1 Kuzu Person-node presence enrichment (migration idempotent + record_presence
upsert); F.2 hashed identity helping-signal (salted-HMAC, deterministic, IP-from-XFF) +
cross-handle merge keyed on the DID hash ONLY (shared-IP false-merge guard); F.3 sovereign
multi-person recall (rank/exclude/skip-opaque) + the honest-by-construction block render.
"""
import json
import os

import pytest

# ── F.2 — identity_hash util ────────────────────────────────────────────────
from titan_hcl.utils.identity_hash import (
    client_ip_from_xff, derive_salt, identity_hash,
)


def test_identity_hash_deterministic_salted_and_empty():
    salt = derive_salt("internal-key-abc")
    assert salt and len(salt) == 64
    h1 = identity_hash("did:privy:xyz", salt)
    assert h1 == identity_hash("did:privy:xyz", salt)      # deterministic
    assert len(h1) == 32                                   # 128-bit compact
    assert identity_hash("did:privy:xyz", derive_salt("other-key")) != h1  # per-salt
    # empty value OR empty salt → '' (a no-op helping-signal)
    assert identity_hash("", salt) == ""
    assert identity_hash("did:privy:xyz", "") == ""
    assert derive_salt("") == ""


def test_client_ip_from_xff_left_most_hop():
    assert client_ip_from_xff("203.0.113.10, 203.0.113.10", "127.0.0.1") == "203.0.113.10"
    assert client_ip_from_xff("  203.0.113.10 , x", "") == "203.0.113.10"
    assert client_ip_from_xff("", "203.0.113.10") == "203.0.113.10"   # no XFF → socket peer
    assert client_ip_from_xff("", "") == ""


# ── F.2 / F.3 — recall merge + multi-person (snapshot-driven, no DB) ─────────
from titan_hcl.logic.presence_recall import (  # noqa: E402
    PresenceRecall, render_recent_presence_block,
)


class _FakeTranslator:
    def to_human(self, gap):
        return f"~{gap}ep"


def _recall(tmp_path, persons):
    p = os.path.join(tmp_path, "snap.json")
    with open(p, "w") as f:
        json.dump({"persons": persons, "updated_ts": 0.0}, f)
    return PresenceRecall(snapshot_path=p, translator=_FakeTranslator(), age_reader=None)


def _rec(last_seen, ev="asserted_identity", chain="UNSEALED", did="", ip="", anchor=None):
    return {"last_seen_epoch": last_seen, "evidence_strength": ev,
            "chain_status": chain, "anchor": anchor, "did_hash": did, "ip_hash": ip}


def test_cross_handle_merge_by_did(tmp_path):
    # alice under @a (older, crypto+CHAINED) and a NEW handle @b sharing the DID hash
    r = _recall(tmp_path, {
        "@a": _rec(1000, "crypto_verified_device", "CHAINED", did="DID1", ip="IP1", anchor="x"),
        "@b": _rec(1200, "asserted_identity", "UNSEALED", did="DID1", ip="IP9"),
    })
    rec = r.recall("@b", now_age_epochs=1500, did_hash="DID1")
    assert rec is not None
    assert rec["last_seen_epoch"] == 1200          # most-recent across handles
    assert rec["evidence_strength"] == "crypto_verified_device"  # strongest carried
    assert rec["chain_status"] == "CHAINED"        # best provability carried
    assert rec["merged_handles"] == 2


def test_first_contact_new_handle_recognized_via_did(tmp_path):
    # @b_new has NO entry yet (first contact this session); @a shares the DID → recognized
    r = _recall(tmp_path, {
        "@a": _rec(1000, "crypto_verified_maker", "WIRED", did="DID1"),
    })
    rec = r.recall("@b_new", now_age_epochs=1100, did_hash="DID1")
    assert rec is not None
    assert rec["last_seen_epoch"] == 1000
    assert rec["evidence_strength"] == "crypto_verified_maker"


def test_ip_hash_is_not_a_merge_key(tmp_path):
    # two DIFFERENT humans behind one IP (NAT) — different DIDs → must NOT merge
    # (false recognition would be an anti-hallucination violation).
    r = _recall(tmp_path, {
        "alice": _rec(1000, did="DIDA", ip="SHARED"),
        "bob": _rec(1200, did="DIDB", ip="SHARED"),
    })
    rec = r.recall("bob", now_age_epochs=1300, did_hash="DIDB", ip_hash="SHARED")
    assert rec is not None
    assert rec["merged_handles"] == 1              # bob only — alice NOT merged in


def test_no_anchored_record_returns_none(tmp_path):
    r = _recall(tmp_path, {})
    assert r.recall("nobody", now_age_epochs=100) is None


def test_recall_recent_ranks_excludes_speaker_and_skips_opaque_did(tmp_path):
    r = _recall(tmp_path, {
        "maker": _rec(1400, "crypto_verified_maker", "CHAINED", did="DIDM", anchor="x"),
        "@alice": _rec(1300, did="DIDA"),
        "@bob": _rec(1000, chain="WIRED", did="DIDB"),
        "did:privy:opaque": _rec(1350, did="DIDX"),   # opaque id — not surfaceable
    })
    out = r.recall_recent(now_age_epochs=1500, exclude_person_id="maker",
                          exclude_did_hash="DIDM", top_k=3)
    ids = [b["person_id"] for b in out]
    assert "maker" not in ids                 # the speaker is excluded
    assert "did:privy:opaque" not in ids      # opaque DID skipped
    assert ids == ["@alice", "@bob"]          # ranked most-recent first (1300 < 1000 gap)
    assert out[0]["gap_human"] == "~200ep"    # translated only at the edge


def test_recall_recent_excludes_did_siblings_of_speaker(tmp_path):
    # the speaker also appears under a sibling handle (same DID) — must not self-surface
    r = _recall(tmp_path, {
        "@speaker_alt": _rec(1450, did="DIDM"),
        "@alice": _rec(1300, did="DIDA"),
    })
    out = r.recall_recent(now_age_epochs=1500, exclude_person_id="maker",
                          exclude_did_hash="DIDM", top_k=3)
    assert [b["person_id"] for b in out] == ["@alice"]


def test_render_recent_presence_block_only_anchored():
    block = render_recent_presence_block([{"person_id": "@alice", "gap_human": "~2 hours ago"}])
    assert "Recent Presence" in block and "@alice" in block and "~2 hours ago" in block
    assert render_recent_presence_block([]) == ""
    assert render_recent_presence_block(None) == ""


# ── F.1 — Kuzu Person-node presence enrichment ──────────────────────────────
def _fresh_graph(tmp_path, sub):
    kuzu = pytest.importorskip("kuzu")  # noqa: F841
    from titan_hcl.core.direct_memory import TitanKnowledgeGraph
    return TitanKnowledgeGraph(os.path.join(tmp_path, sub))


def test_person_presence_migration_is_idempotent(tmp_path):
    from titan_hcl.logic.social_x.schema_migrations import (
        apply_kuzu_person_presence_migrations,
    )
    g = _fresh_graph(tmp_path, "kg_mig.kuzu")
    # _init_schema already applied it on construction → a re-run adds nothing.
    assert apply_kuzu_person_presence_migrations(g)["added"] == []


def test_record_presence_upserts_newest_and_strongest(tmp_path):
    g = _fresh_graph(tmp_path, "kg_rec.kuzu")
    g.record_presence(name="alice", user_id="alice", age_epochs=1000,
                      evidence_strength="asserted_identity", chain_status="UNSEALED",
                      tx_hash="tx1", did_hash="DID1", ip_hash="IP1")
    g.record_presence(name="alice", user_id="alice", age_epochs=2000,
                      evidence_strength="crypto_verified_device", chain_status="WIRED",
                      tx_hash="tx2", did_hash="DID1", ip_hash="IP1")
    qr = g._conn.execute(
        "MATCH (p:Person {name: 'alice'}) RETURN p.presence_last_seen_epoch, "
        "p.presence_evidence_strength, p.presence_count, p.did_hash, "
        "p.presence_chain_status, p.presence_last_tx_hash")
    row = qr.get_next()
    assert int(row[0]) == 2000                       # newest epoch
    assert row[1] == "crypto_verified_device"        # strongest evidence retained
    assert int(row[2]) == 2                           # count incremented
    assert row[3] == "DID1"                           # identity hash stored
    assert row[4] == "WIRED" and row[5] == "tx2"      # newest interaction's chain/tx


def test_record_presence_keeps_strongest_when_older_is_stronger(tmp_path):
    g = _fresh_graph(tmp_path, "kg_rec2.kuzu")
    g.record_presence(name="bob", user_id="bob", age_epochs=2000,
                      evidence_strength="crypto_verified_maker", chain_status="CHAINED",
                      tx_hash="txA", did_hash="DIDB", ip_hash="")
    # a later but WEAKER interaction must not downgrade the recorded evidence
    g.record_presence(name="bob", user_id="bob", age_epochs=3000,
                      evidence_strength="asserted_identity", chain_status="UNSEALED",
                      tx_hash="txB", did_hash="DIDB", ip_hash="")
    qr = g._conn.execute(
        "MATCH (p:Person {name: 'bob'}) RETURN p.presence_evidence_strength, "
        "p.presence_last_seen_epoch")
    row = qr.get_next()
    assert row[0] == "crypto_verified_maker"   # strongest retained (honest gradient)
    assert int(row[1]) == 3000                 # last_seen still advances
