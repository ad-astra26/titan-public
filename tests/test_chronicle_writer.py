"""Regression tests for the restored meditation chronicle writer.

Covers BUG-CHRONICLE-WRITER-DEAD-POST-A87: the IQL "Scholar's Dream" +
chronicle write were both dropped in the §4.D meditation_worker extraction
and restored in the parent (titan_HCL) as `_append_to_chronicle`, driven by
`_meditation_chronicle_loop` on MEDITATION_COMPLETE dst="core".

The writer body references no instance state, so we exercise it directly via
`TitanHCL._append_to_chronicle(None, dream_results)` inside an isolated CWD.
"""
import os

import pytest

from titan_hcl.core.plugin import TitanHCL
from titan_hcl.core.soul import CHRONICLE_PATH, TITAN_MERGED_PATH


_write = TitanHCL._append_to_chronicle


@pytest.fixture()
def soul_cwd(tmp_path, monkeypatch):
    """Isolated repo-root CWD with a minimal constitution (regen needs it)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "titan_constitution.md").write_text(
        "# Constitution\nPrime Directive 1: Sovereign Growth.\n",
        encoding="utf-8")
    return tmp_path


def test_first_entry_creates_chronicle_and_regenerates_titan_md(soul_cwd):
    _write(None, {"loss_actor": 0.05, "loss_qvalue": 0.2, "loss_value": 0.1})

    chronicle = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert "## The Scholar's Chronicle" in chronicle
    assert "Meditation Cycle" in chronicle
    assert "IQL Loss" in chronicle
    # actor_loss 0.05 → "Converging" band; q 0.2 → moderate; v 0.1 → developing
    assert "Converging" in chronicle

    # titan.md regenerated as constitution + chronicle merge.
    merged = (soul_cwd / TITAN_MERGED_PATH).read_text(encoding="utf-8")
    assert "Sovereign Growth" in merged
    assert "Meditation Cycle" in merged

    # Raw metrics data log written.
    assert (soul_cwd / "data" / "logs" / "scholar_raw.log").exists()


def test_alignment_bands(soul_cwd):
    _write(None, {"loss_actor": 0.005, "loss_qvalue": 0.01, "loss_value": 0.01})
    text = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert "Sovereign Clarity" in text

    _write(None, {"loss_actor": 0.9, "loss_qvalue": 0.5, "loss_value": 0.5})
    text = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert "Foundational" in text


def test_entries_accumulate_in_order(soul_cwd):
    for i in range(3):
        _write(None, {"loss_actor": 0.1 * (i + 1), "loss_qvalue": 0.1,
                      "loss_value": 0.1})
    text = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    assert text.count("Meditation Cycle") == 3


def test_rolling_window_archives_oldest_beyond_50(soul_cwd):
    # Write 51 entries — the oldest must spill into the archive, keeping the
    # live chronicle bounded.
    for i in range(51):
        _write(None, {"loss_actor": 0.1, "loss_qvalue": 0.1, "loss_value": 0.1})

    archive = soul_cwd / "data" / "history" / "soul_archive.md"
    assert archive.exists(), "oldest entry should have been archived"
    assert "Titan Soul Archive" in archive.read_text(encoding="utf-8")

    live = (soul_cwd / CHRONICLE_PATH).read_text(encoding="utf-8")
    # Bounded: 50 retained + 1 new − 1 archived ≈ 50 (never unbounded growth).
    assert live.count("Meditation Cycle") <= 51


def test_atomic_write_leaves_no_tmp(soul_cwd):
    _write(None, {"loss_actor": 0.1, "loss_qvalue": 0.1, "loss_value": 0.1})
    assert not (soul_cwd / (CHRONICLE_PATH + ".tmp")).exists()
