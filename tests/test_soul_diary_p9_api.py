"""P9 — public Soul-Diary archive API
(`RFP_titan_authored_soul_diary` §7.P9 / §6.8 / INV-SD-3).

Proves the read-only public surface serves the sanitized projection (never the
private entry), re-sanitizes at the edge (fail-closed backstop), exposes the
anchor/explorer links, and that the isolated router mounts at /v6/soul_diary.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from titan_hcl.core import soul_diary_chain
from titan_hcl.api import soul_diary_routes as sdr


def _seed_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_DATA_DIR", str(tmp_path))
    from titan_hcl.core.shadow_data_dir import resolve_data_path
    ledger = resolve_data_path("data/soul_diary_chain.json")
    # day 1 — clean public projection
    soul_diary_chain.append_entry(
        "2026-06-08", "private text one", distillation="Yesterday I rested.",
        public_entry="A quiet, settling day.", redactions=0, path=ledger)
    # day 2 — a deliberately DIRTY stored public_entry to exercise the edge backstop
    soul_diary_chain.append_entry(
        "2026-06-09", "private text two with /home/youruser/secret.py:12",
        distillation="Today I grew a little.",
        public_entry="I looked at /home/youruser/secret.py:12 on 203.0.113.10.",
        redactions=0, path=ledger)
    soul_diary_chain.update_refs("2026-06-09", nft_addr="NFTaddr123",
                                 arweave_uri="ar://realtx456", art_path="/x/art.jpg",
                                 path=ledger)
    return ledger


def test_entry_serves_sanitized_public_projection(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch)
    res = sdr.get_soul_diary_entry("2026-06-09")
    assert res["ok"] is True
    e = res["entry"]
    # the edge re-sanitize scrubbed the dirty stored public_entry (G9 backstop)
    assert "/home/youruser" not in e["entry"]
    assert "203.0.113.10" not in e["entry"]
    assert "secret.py:12" not in e["entry"]
    assert e["distillation"] == "Today I grew a little."
    assert e["nft_addr"] == "NFTaddr123"
    assert e["art"] == "art.jpg"                      # basename only, not the path
    assert e["links"]["solana_explorer"].endswith("NFTaddr123")
    assert e["cumulative_hash"]                        # verification hash present


def test_entry_never_exposes_private_or_full_path(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch)
    e = sdr.get_soul_diary_entry("2026-06-09")["entry"]
    # the private entry text is never in the ledger / never served; no raw fs path
    assert "private text" not in str(e)
    assert "/x/art.jpg" not in str(e)                 # only the basename is exposed


def test_missing_date_is_not_found(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch)
    res = sdr.get_soul_diary_entry("2099-01-01")
    assert res["ok"] is False and res["error"] == "not_found"


def test_index_is_newest_first(tmp_path, monkeypatch):
    _seed_ledger(tmp_path, monkeypatch)
    idx = sdr.get_soul_diary_index()
    assert idx["ok"] is True and idx["total"] == 2
    assert [it["date"] for it in idx["entries"]] == ["2026-06-09", "2026-06-08"]
    assert idx["entries"][0]["nft_addr"] == "NFTaddr123"
    assert idx["entries"][0]["has_art"] is True


def test_router_mounts_under_v6_soul_diary(tmp_path, monkeypatch):
    """The isolated APIRouter mounts at the right public paths."""
    _seed_ledger(tmp_path, monkeypatch)
    app = FastAPI()
    app.include_router(sdr.router)
    client = TestClient(app)
    r = client.get("/v6/soul_diary/2026-06-09")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert "/home/youruser" not in r.text          # sanitized over the wire
    idx = client.get("/v6/soul_diary")
    assert idx.status_code == 200 and idx.json()["total"] == 2
