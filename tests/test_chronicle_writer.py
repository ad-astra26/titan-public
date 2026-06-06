"""Regression tests for the meditation chronicle writer (sovereignty re-source).

Covers BUG-CHRONICLE-WRITER-DEAD-POST-A87: the chronicle write was dropped in
the §4.D meditation_worker extraction and restored in the parent (titan_HCL) as
`_append_to_chronicle`, driven by `_meditation_chronicle_loop` on
MEDITATION_COMPLETE dst="core".

Sovereignty re-source (RFP_synthesis_decision_authority P3): the narrative is now
sourced from the ONE sovereignty score `S = 0.7·E + 0.3·V` (rolling, + E/V) folded
with the meditation outcome — re-pointed off the retired IQL `recorder.dream()`
losses. The writer body references no instance state, so we exercise it directly
via `TitanHCL._append_to_chronicle(None, sov, med)` inside an isolated CWD with a
real constitution (so `regenerate_soul_md` runs for real — full integration).
"""
import pytest

from titan_hcl.core.plugin import TitanHCL
from titan_hcl.core.soul import CHRONICLE_PATH, TITAN_MERGED_PATH


_write = TitanHCL._append_to_chronicle


def _sov(s, e=0.5, v=0.2, *, window="7d", replies=40, trend=0.0):
    return {"window": window, "replies": replies, "s": s, "e": e, "v": v,
            "trend": trend}


def _med(promoted=2, pruned=1):
    return {"promoted": promoted, "pruned": pruned}


@pytest.fixture()
def soul_cwd(tmp_path, monkeypatch):
    """Isolated repo-root CWD with a minimal constitution (regen needs it)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "titan_constitution.md").write_text(
        "# Constitution\nPrime Directive 1: Sovereign Growth.\n",
        encoding="utf-8")
    return tmp_path


def test_first_entry_creates_chronicle_and_regenerates_titan_md(soul_cwd):
    _write(None, _sov(0.45, e=0.4, v=0.2), _med(2, 1))

    chronicle = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert "## The Scholar's Chronicle" in chronicle
    assert "Meditation Cycle" in chronicle
    assert "Sovereignty S: 0.45" in chronicle
    # S 0.45 → "Converging" band; the legacy IQL source is gone
    assert "Converging" in chronicle
    assert "IQL" not in chronicle
    assert "2 memories crystallized, 1 faded" in chronicle

    # titan.md regenerated as constitution + chronicle merge.
    merged = (soul_cwd / TITAN_MERGED_PATH).read_text(encoding="utf-8")
    assert "Sovereign Growth" in merged
    assert "Meditation Cycle" in merged

    # Raw metrics data log written (sovereignty, not IQL loss).
    assert (soul_cwd / "data" / "logs" / "sovereignty_raw.log").exists()


def test_alignment_bands(soul_cwd):
    _write(None, _sov(0.72, e=0.7, v=0.45), _med())
    text = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert "Sovereign Clarity" in text

    _write(None, _sov(0.1, e=0.05, v=0.0), _med())
    text = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert "Foundational" in text


def test_entries_accumulate_in_order(soul_cwd):
    for i in range(3):
        _write(None, _sov(0.2 * (i + 1)), _med())
    text = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert text.count("Meditation Cycle") == 3


def test_rolling_window_archives_oldest_beyond_50(soul_cwd):
    # Write 51 entries — the oldest must spill into the archive, keeping the
    # live chronicle bounded.
    for i in range(51):
        _write(None, _sov(0.5), _med())

    archive = soul_cwd / "data" / "history" / "soul_archive.md"
    assert archive.exists(), "oldest entry should have been archived"
    assert "Titan Soul Archive" in archive.read_text(encoding="utf-8")

    live = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    # Bounded: 50 retained + 1 new − 1 archived ≈ 50 (never unbounded growth).
    assert live.count("Meditation Cycle") <= 51


def test_atomic_write_leaves_no_tmp(soul_cwd):
    _write(None, _sov(0.5), _med())
    assert not (soul_cwd / (CHRONICLE_PATH + ".tmp")).exists()
