"""rFP X-post truthfulness & quality — PART A footer tests (2026-06-03).

Covers the per-Titan on-chain provenance footer built in
``SocialXGateway._assemble_final_text`` (Phases A+B+C):
  - Identity is config-sourced (no hardcoded literal) — T1 → GenesisNFT URL,
    T2/T3 → honest devnet marker (INV-XSEAL-1/7).
  - Epoch Seal links the latest real anchor, labelled with the lifetime seal
    count + ε ONLY when last_epoch_id>0 (never "ε 0") (INV-XSEAL-2/3).
  - Devnet seals carry ?cluster=devnet (INV-XSEAL-8); graceful omit when no sig.
  - The pre-2026-06 hardcoded literal `4o9HGwM47dy…` never appears.

Run: python -m pytest tests/test_x_provenance_footer_20260603.py -v -p no:anchorpy
"""
from __future__ import annotations

import pytest

HARDCODED_LITERAL = "4o9HGwM47dy"


@pytest.fixture
def gateway(tmp_path):
    from titan_hcl.logic.social_x_gateway import SocialXGateway
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[social_x]\nenabled = true\nmax_post_length = 4000\n"
        'url_domain = "https://example.com"\n'
    )
    gw = SocialXGateway(
        db_path=str(tmp_path / "x.db"),
        config_path=str(cfg),
        telemetry_path=str(tmp_path / "tel.jsonl"),
    )
    gw._boot_time = 0
    return gw


def _ctx(titan_id: str):
    from titan_hcl.logic.social_x_gateway import PostContext
    return PostContext(session="", proxy="", api_key="", titan_id=titan_id,
                       emotion="wonder", neuromods={"5HT": 0.71, "NE": 0.70})


def _cfg(genesis_url: str = "https://example.com/genesis", cluster: str = ""):
    return {"max_post_length": 4000, "url_domain": "https://example.com",
            "genesis_identity_url": genesis_url, "seal_cluster": cluster}


def test_t1_identity_and_seal_no_epoch_when_zero(gateway):
    """T1 (mainnet): /genesis Identity + Epoch Seal with seal#, NO ε when epoch=0."""
    gateway._latest_epoch_seal = lambda: ("TsZTNaE4FpDGsig", 0, 289)
    out = gateway._assemble_final_text(
        "Hello world.", "grounded_today", {}, _ctx("T1"), _cfg())
    assert "Identity: https://example.com/genesis" in out
    assert "Epoch Seal (seal #289): https://example.com/tx/TsZTNaE4FpDGsig" in out
    seal_tail = out.split("Epoch Seal")[1][:30]
    assert "ε" not in seal_tail, f"must not show ε when epoch=0: {seal_tail!r}"
    assert HARDCODED_LITERAL not in out


def test_t1_epoch_shown_when_known(gateway):
    """ε prefix lights up once last_epoch_id>0 (post anchor-BUG fix)."""
    gateway._latest_epoch_seal = lambda: ("SigABC", 58860, 289)
    out = gateway._assemble_final_text(
        "Hi.", "grounded_today", {}, _ctx("T1"), _cfg())
    assert "Epoch Seal (ε 58,860 · seal #289): https://example.com/tx/SigABC" in out
    assert HARDCODED_LITERAL not in out


def test_t2_devnet_marker_and_cluster_param(gateway):
    """T2/T3 (devnet): honest no-GenesisNFT marker + ?cluster=devnet on the seal."""
    gateway._latest_epoch_seal = lambda: ("DevSig123", 0, 12)
    out = gateway._assemble_final_text(
        "Hi.", "grounded_today", {}, _ctx("T2"), _cfg(genesis_url="", cluster="devnet"))
    assert "Identity: devnet — no mainnet GenesisNFT yet" in out
    assert "https://example.com/tx/DevSig123?cluster=devnet" in out
    assert "example.com/genesis" not in out
    assert HARDCODED_LITERAL not in out


def test_seal_omitted_when_no_anchor(gateway):
    """No valid anchor → omit the Epoch Seal line entirely (never fabricate)."""
    gateway._latest_epoch_seal = lambda: ("", 0, 0)
    out = gateway._assemble_final_text(
        "Hi.", "grounded_today", {}, _ctx("T1"), _cfg())
    assert "Identity: https://example.com/genesis" in out
    assert "Epoch Seal" not in out
    assert HARDCODED_LITERAL not in out


def test_proof_day_unchanged(gateway):
    """proof_day still builds exact Archive+Seal from archetype metadata."""
    ctx = _ctx("T1")
    class _Arc:
        metadata = {"arweave_tx": "ARW123", "zk_vault_tx": "VAULT456"}
    ctx.archetype_candidate = _Arc()
    out = gateway._assemble_final_text("Permanence.", "proof_day", {}, ctx, _cfg())
    assert "Archive: https://example.com/ar/ARW123" in out
    assert "Seal: https://example.com/tx/VAULT456" in out
    assert HARDCODED_LITERAL not in out
