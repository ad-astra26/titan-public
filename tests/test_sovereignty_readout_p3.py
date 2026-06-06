"""P3 shared rolling-sovereignty read-out (RFP_synthesis_decision_authority).

The ONE place the chronicle re-source, the meditation on-chain ZK-vault anchor,
and the backup persistence read the rolling sovereignty `S = 0.7·E + 0.3·V` from
the synthesis metrics snapshot (G18 file read) + convert it to on-chain basis
points. Pins the window selection, the bp conversion + clamp, and soft-fail.
"""
import json

from titan_hcl.synthesis.sovereignty_readout import (
    SOVEREIGNTY_BP_SCALE,
    read_rolling_sovereignty,
    rolling_sovereignty_bp,
)


def _snap(tmp_path, windows):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "synthesis_metrics_snapshot.json").write_text(
        json.dumps({"sovereignty": {"windows": windows}}))


# ── read_rolling_sovereignty ─────────────────────────────────────────────────

def test_prefers_7d_window(tmp_path):
    _snap(tmp_path, {
        "24h": {"replies": 5, "sovereignty": 0.50, "e": 0.6, "v": 0.2, "trend": 0.0},
        "7d": {"replies": 40, "sovereignty": 0.62, "e": 0.71, "v": 0.18, "trend": 0.04},
    })
    out = read_rolling_sovereignty(str(tmp_path))
    assert out["window"] == "7d" and out["s"] == 0.62
    assert out["e"] == 0.71 and out["v"] == 0.18


def test_falls_back_when_7d_empty(tmp_path):
    _snap(tmp_path, {
        "24h": {"replies": 3, "sovereignty": 0.4, "e": 0.5, "v": 0.1, "trend": 0.0},
        "7d": {"replies": 0, "sovereignty": 0.0, "e": 0.0, "v": 0.0, "trend": 0.0},
    })
    out = read_rolling_sovereignty(str(tmp_path))
    assert out["window"] == "24h" and out["replies"] == 3


def test_missing_snapshot_is_zero(tmp_path):
    out = read_rolling_sovereignty(str(tmp_path))  # no file
    assert out["replies"] == 0 and out["s"] == 0.0


# ── rolling_sovereignty_bp (the on-chain wire value) ─────────────────────────

def test_bp_conversion(tmp_path):
    _snap(tmp_path, {"7d": {"replies": 10, "sovereignty": 0.62,
                            "e": 0.7, "v": 0.2, "trend": 0.0}})
    assert rolling_sovereignty_bp(str(tmp_path)) == 6200  # 0.62 × 10000


def test_bp_clamped_to_scale(tmp_path):
    # Defensive: a corrupt > 1.0 snapshot value never exceeds the bp ceiling.
    _snap(tmp_path, {"7d": {"replies": 10, "sovereignty": 1.5,
                            "e": 0.0, "v": 0.0, "trend": 0.0}})
    assert rolling_sovereignty_bp(str(tmp_path)) == SOVEREIGNTY_BP_SCALE  # 10000


def test_bp_zero_when_missing(tmp_path):
    assert rolling_sovereignty_bp(str(tmp_path)) == 0


def test_bp_in_valid_range(tmp_path):
    _snap(tmp_path, {"7d": {"replies": 10, "sovereignty": 0.337,
                            "e": 0.0, "v": 0.0, "trend": 0.0}})
    bp = rolling_sovereignty_bp(str(tmp_path))
    assert 0 <= bp <= SOVEREIGNTY_BP_SCALE and bp == 3370
