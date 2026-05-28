"""Phase 10F (rFP §3G) — Trinity on-chain anchoring restoration.

The anchoring loop was dropped at the D8-3 spirit_worker gutting. It is restored
into timechain_worker, but its on-chain TX MUST stay gated by
config["anchor_enabled"] (default False) so restoring the loop spends NO SOL
unless explicitly enabled. These tests pin that safety gate + graceful no-ops.
The full should_anchor→Solana-TX path is unchanged from the original (verbatim
relocation) and requires a live wallet + RPC, so is not exercised here.
"""
import os

from titan_hcl.logic.trinity_anchor import maybe_anchor_trinity


def _consciousness(curvature=5.0, density=0.5, epoch_id=42):
    return {"latest_epoch": {
        "epoch_id": epoch_id, "curvature": curvature, "density": density}}


def test_disabled_is_absolute_noop(monkeypatch, tmp_path):
    """anchor_enabled missing/False → no data/ writes, no exception (no SOL)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    # default (no anchor_enabled key)
    maybe_anchor_trinity(_consciousness(), {}, [0.5] * 5, [0.5] * 5, [0.5] * 5)
    # explicit False
    maybe_anchor_trinity(_consciousness(), {"anchor_enabled": False},
                         [0.5] * 5, [0.5] * 5, [0.5] * 5)
    assert not os.path.exists(os.path.join("data", "anchor_state.json"))


def test_empty_consciousness_is_noop(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    maybe_anchor_trinity({}, {"anchor_enabled": True},
                         [0.5] * 5, [0.5] * 5, [0.5] * 5)
    maybe_anchor_trinity(None, {"anchor_enabled": True},
                         [0.5] * 5, [0.5] * 5, [0.5] * 5)
    assert not os.path.exists(os.path.join("data", "anchor_state.json"))


def test_enabled_low_curvature_no_anchor_but_ema_persists(monkeypatch, tmp_path):
    """Enabled + un-remarkable curvature + no block delta → no anchor TX, but
    the curvature EMA is persisted on the % 100 epoch tick (cheap tracking write).
    No Solana TX is attempted (no keypair present → would fail gracefully)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    # epoch_id=100 triggers EMA persistence; curvature≈EMA so not significant;
    # no timechain index.db → tc_delta=0 < 5000 → _enough_new_state False.
    maybe_anchor_trinity(_consciousness(curvature=2.0, density=0.5, epoch_id=100),
                         {"anchor_enabled": True}, [0.5] * 5, [0.5] * 5, [0.5] * 5)
    anchor_path = os.path.join("data", "anchor_state.json")
    assert os.path.exists(anchor_path), "EMA tracking should persist on %100 tick"
    import json
    state = json.load(open(anchor_path))
    # Only the EMA was written — no successful-anchor fields (no TX fired).
    assert "curvature_ema" in state
    assert "last_tx_sig" not in state
