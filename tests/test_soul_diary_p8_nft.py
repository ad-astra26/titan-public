"""P8 — Solana DailyNFT (gated) + triple-anchor parity
(`RFP_titan_authored_soul_diary` §7.P8 / §6.7 / INV-SD-10/11/13).

Hermetic — no real chain (FakeChainProvider + a monkeypatched mint). Proves:
  · the metadata JSON carries BOTH verification hashes + the sanitized excerpt;
  · the per-Titan mint gate (mainnet T1 OFF default; devnet T2/T3 on);
  · gated-on: art + metadata upload to Arweave → real uri → mint → ledger refs;
  · gated-off: zero chain I/O (no provider even constructed).
"""
import asyncio
import json
from unittest.mock import MagicMock

from titan_hcl.chain.fake import FakeChainProvider
from titan_hcl.core import soul_diary_chain
from titan_hcl.core.soul_diary import SoulDiaryOrchestrator
from titan_hcl.logic import daily_nft
from titan_hcl.modules import soul_diary_worker as sdw


# ── metadata builder (pure, §6.7 / INV-SD-10) ───────────────────────────────

def test_metadata_carries_both_hashes_excerpt_and_art():
    eh, ch = "a91f" * 16, "7c20" * 16
    meta = daily_nft.build_soul_diary_nft_metadata(
        date="2026-06-09", entry_hash=eh, cumulative_hash=ch,
        distillation="Today I learned more about myself.",
        sovereignty={"s": 0.58},
        felt={"mood_label": "Curious", "dominant": "dopamine", "valence": 0.6},
        art_uri="ar://fakeart")
    assert meta["name"] == "Titan Soul Diary — 2026-06-09"
    assert meta["symbol"] == "TSOUL"
    props = meta["properties"]
    assert props["entry_hash"] == eh and props["cumulative_hash"] == ch
    assert props["distillation"] == "Today I learned more about myself."
    assert meta["image"] == "ar://fakeart"
    traits = {a["trait_type"]: a["value"] for a in meta["attributes"]}
    assert traits["Entry Hash"] == eh and traits["Cumulative Hash"] == ch
    assert traits["Mood"] == "Curious" and traits["Dominant Neuromodulator"] == "dopamine"
    # no art → no image key (graceful)
    assert "image" not in daily_nft.build_soul_diary_nft_metadata(
        date="2026-06-09", entry_hash="x", cumulative_hash="y")


# ── the per-Titan mint gate (INV-SD-11) ─────────────────────────────────────

def test_mint_gate_default_off_and_per_titan_override():
    assert SoulDiaryOrchestrator.mint_enabled({}) is False                 # absent → OFF
    assert SoulDiaryOrchestrator.mint_enabled({"soul_diary": {}}) is False
    assert SoulDiaryOrchestrator.mint_enabled(
        {"soul_diary": {"mint_enabled": False}}) is False                  # T1 mainnet
    assert SoulDiaryOrchestrator.mint_enabled(
        {"soul_diary": {"mint_enabled": True}}) is True                    # T2/T3 devnet


def _orch(tmp_path):
    return SoulDiaryOrchestrator(state_path=str(tmp_path / "s.json"),
                                 ledger_path=str(tmp_path / "chain.json"))


# ── gated OFF (mainnet T1) → no chain I/O at all (INV-SD-11) ────────────────

def test_mint_gated_off_skips_all_chain_io(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    row = orch.record_hash("2026-06-09", "private entry", distillation="clean share")
    built = {"provider": False}
    monkeypatch.setattr(sdw, "_build_chain_provider",
                        lambda c: built.update(provider=True))
    sdw._mint_daily_nft(orch, row, {"sovereignty": {"s": 0.5}}, "2026-06-09",
                        config={"soul_diary": {"mint_enabled": False}}, titan_id="T1")
    updated = soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))[-1]
    assert updated["nft_addr"] is None         # nothing minted
    assert built["provider"] is False          # provider never even constructed


# ── gated ON (devnet T2/T3) → upload + mint + ledger refs ───────────────────

def test_mint_gated_on_uploads_mints_and_records(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    row = orch.record_hash("2026-06-09", "private entry text",
                           distillation="Today I grew a little.")
    art = tmp_path / "art.jpg"
    art.write_bytes(b"\xff\xd8\xff\x00 fake-jpeg-bytes")
    row["art_path"] = str(art)

    fake_provider = FakeChainProvider()
    mint_calls = []

    async def _fake_mint(network, *, epoch, sovereignty_idx, diary_entry,
                         total_nodes, art_path=None, permanent_url=None):
        mint_calls.append({"epoch": epoch, "sovereignty_idx": sovereignty_idx,
                           "diary_entry": diary_entry, "total_nodes": total_nodes,
                           "art_path": art_path, "permanent_url": permanent_url})
        return "FAKE_NFT_ADDR_123"

    monkeypatch.setattr(sdw, "_build_chain_provider", lambda c: fake_provider)
    monkeypatch.setattr(sdw, "_build_network_client", lambda c: MagicMock())
    monkeypatch.setattr(daily_nft, "mint_epoch_nft", _fake_mint)

    sdw._mint_daily_nft(
        orch, row,
        {"sovereignty": {"s": 0.58}, "felt": {"dominant": "dopamine"},
         "memory": {"kg_nodes": 980}},
        "2026-06-09",
        config={"soul_diary": {"mint_enabled": True}, "network": {}}, titan_id="T2")

    # art + metadata both uploaded to Arweave
    assert len(fake_provider.put_log) == 2
    updated = soul_diary_chain.load_chain(path=str(tmp_path / "chain.json"))[-1]
    assert updated["nft_addr"] == "FAKE_NFT_ADDR_123"
    assert updated["arweave_uri"].startswith("ar://fake_")

    call = mint_calls[-1]
    assert call["diary_entry"] == "Today I grew a little."   # SANITIZED distillation on-chain
    assert call["sovereignty_idx"] == 58.0                   # S 0.58 → 58%
    assert call["total_nodes"] == 980
    assert call["permanent_url"] == updated["arweave_uri"]   # real uri, not a stub

    # the uploaded metadata carries BOTH hashes → triple-anchor parity (INV-SD-10)
    meta_tx = updated["arweave_uri"].replace("ar://", "")
    meta = json.loads(asyncio.run(fake_provider.get_bytes(meta_tx)))
    assert meta["properties"]["entry_hash"] == row["entry_hash"]
    assert meta["properties"]["cumulative_hash"] == row["cumulative_hash"]
    assert meta["image"].startswith("ar://fake_")            # art image referenced


def test_mint_no_cumulative_hash_is_noop(tmp_path, monkeypatch):
    """No ledger row / empty cumulative_hash → mint no-ops even when enabled."""
    orch = _orch(tmp_path)
    monkeypatch.setattr(sdw, "_build_chain_provider",
                        lambda c: (_ for _ in ()).throw(AssertionError("should not build")))
    sdw._mint_daily_nft(orch, {}, {}, "2026-06-09",
                        config={"soul_diary": {"mint_enabled": True}}, titan_id="T2")
    assert soul_diary_chain.load_chain(path=str(tmp_path / "chain.json")) == []
