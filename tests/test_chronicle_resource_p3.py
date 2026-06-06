"""P3 chronicle re-source (RFP_synthesis_decision_authority).

The soul Chronicle narrative is sourced from the ONE sovereignty score
`S = 0.7·E + 0.3·V` (rolling, with its E/V components, read from the synthesis
metrics snapshot — a G18 file read) folded with the meditation cycle's outcome,
re-pointed off the retired IQL `recorder.dream()` losses. These pin:
  - `_read_rolling_sovereignty` window selection + soft-fail,
  - `_append_to_chronicle` S/E/V → narrative mapping + the meditation fold-in,
  - that no IQL vocabulary leaks into the re-sourced entry.
"""
import json
from types import SimpleNamespace
from unittest.mock import patch

from titan_hcl.core.plugin import TitanHCL


def _self():
    return SimpleNamespace()


# ── _read_rolling_sovereignty — G18 snapshot file read ───────────────────────

def _write_snapshot(tmp_path, windows):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "synthesis_metrics_snapshot.json").write_text(
        json.dumps({"sovereignty": {"windows": windows}}))


def test_read_rolling_prefers_7d_with_data(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, {
        "24h": {"replies": 5, "sovereignty": 0.50, "e": 0.6, "v": 0.2, "trend": 0.0},
        "7d": {"replies": 40, "sovereignty": 0.62, "e": 0.71, "v": 0.18, "trend": 0.04},
        "all": {"replies": 100, "sovereignty": 0.55, "e": 0.6, "v": 0.2, "trend": None},
    })
    monkeypatch.chdir(tmp_path)
    out = TitanHCL._read_rolling_sovereignty(_self())
    assert out["window"] == "7d"
    assert out["replies"] == 40
    assert out["s"] == 0.62 and out["e"] == 0.71 and out["v"] == 0.18
    assert out["trend"] == 0.04


def test_read_rolling_falls_back_to_24h_when_7d_empty(tmp_path, monkeypatch):
    _write_snapshot(tmp_path, {
        "24h": {"replies": 3, "sovereignty": 0.4, "e": 0.5, "v": 0.1, "trend": 0.0},
        "7d": {"replies": 0, "sovereignty": 0.0, "e": 0.0, "v": 0.0, "trend": 0.0},
    })
    monkeypatch.chdir(tmp_path)
    out = TitanHCL._read_rolling_sovereignty(_self())
    assert out["window"] == "24h" and out["replies"] == 3


def test_read_rolling_missing_snapshot_returns_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no data/ snapshot at all
    out = TitanHCL._read_rolling_sovereignty(_self())
    assert out["replies"] == 0 and out["s"] == 0.0 and out["e"] == 0.0


# ── _append_to_chronicle — narrative mapping + meditation fold-in ─────────────

def _append(tmp_path, monkeypatch, sov, med):
    monkeypatch.chdir(tmp_path)
    chron = tmp_path / "titan_chronicles.md"
    with patch("titan_hcl.core.soul.CHRONICLE_PATH", str(chron)), \
            patch("titan_hcl.core.soul.regenerate_soul_md"):
        TitanHCL._append_to_chronicle(_self(), sov, med)
    return chron.read_text()


def test_high_s_renders_sovereign_clarity_and_folds_meditation(tmp_path, monkeypatch):
    text = _append(
        tmp_path, monkeypatch,
        {"window": "7d", "replies": 40, "s": 0.72, "e": 0.71, "v": 0.45, "trend": 0.04},
        {"promoted": 3, "pruned": 1})
    assert "Sovereign Clarity" in text
    assert "Sovereignty S: 0.72" in text
    assert "Knowledge (E=0.71)" in text
    assert "Insight (V=0.45)" in text
    assert "3 memories crystallized, 1 faded" in text
    assert "growing (+0.04)" in text
    # the legacy IQL source is fully gone from the entry
    assert "IQL" not in text and "Q-Value" not in text


def test_low_s_renders_foundational(tmp_path, monkeypatch):
    text = _append(
        tmp_path, monkeypatch,
        {"window": "24h", "replies": 2, "s": 0.1, "e": 0.05, "v": 0.0, "trend": -0.03},
        {"promoted": 0, "pruned": 0})
    assert "Foundational" in text
    assert "receding (-0.03)" in text


def test_rolling_window_archives_beyond_50(tmp_path, monkeypatch):
    sov = {"window": "7d", "replies": 10, "s": 0.5, "e": 0.5, "v": 0.2, "trend": 0.0}
    med = {"promoted": 1, "pruned": 0}
    for _ in range(55):
        _append(tmp_path, monkeypatch, sov, med)
    chron = (tmp_path / "titan_chronicles.md").read_text()
    # rolling window caps at 50 entries in the live file
    assert chron.count("] Meditation Cycle") <= 50
    # the overflow was archived, not lost
    assert (tmp_path / "data" / "history" / "soul_archive.md").exists()
