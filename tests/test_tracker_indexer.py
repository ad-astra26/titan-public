"""Tests for scripts/tracker_indexer.py — the classify() prose-keyword fix and
the --archive (graveyard relocation) mode.

Run: python -m pytest tests/test_tracker_indexer.py -v -p no:anchorpy --tb=short
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load scripts/tracker_indexer.py by file path. We deliberately do NOT add
# scripts/ to sys.path — scripts/titan_hcl.py would shadow the titan_hcl
# package and break the conftest autouse fixture. Register in sys.modules so
# dataclass annotation resolution (which looks up cls.__module__) works.
_spec = importlib.util.spec_from_file_location(
    "tracker_indexer", REPO_ROOT / "scripts" / "tracker_indexer.py")
ti = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ti
_spec.loader.exec_module(ti)


# ─── classify() — earliest-occurrence resolution ───────────────────────────

def test_classify_deferred_status_mentioning_shipped_dependency():
    """The regression that motivated the fix: a DEFERRED entry whose Status
    line mentions an upstream rFP was 'SHIPPED' must stay DEFERRED, not flip
    to SHIPPED (prose-keyword bleed)."""
    status, _, _ = ti.classify(
        "HIGH PRIORITY DEFERRED — blocker cleared. META-CGN consumer Phase 1 "
        "SHIPPED 2026-04-15. Ready for own session.")
    assert status == "DEFERRED"


def test_classify_leading_shipped_still_closes():
    status, _, _ = ti.classify("✅ SHIPPED 2026-05-12 — chunk 9Q migrated, was DEFERRED earlier")
    assert status == "SHIPPED"


def test_classify_open_beats_later_resolved_mention():
    status, _, _ = ti.classify("OPEN — will be RESOLVED once upstream lands")
    assert status == "OPEN"


def test_classify_strikethrough_defaults_closed():
    status, _, _ = ti.classify("~~BUG-FOO~~ — something")
    assert status == "CLOSED"


def test_classify_severity_and_date():
    status, sev, date = ti.classify("[HIGH, ✅ CLOSED 2026-05-29 — D-SPEC-143]")
    assert status == "CLOSED"
    assert sev == "HIGH"
    assert date == "2026-05-29"


# ─── _entry_block_bounds() ──────────────────────────────────────────────────

def test_entry_block_bounds_respects_section_and_entry_headers():
    lines = [
        "## Section A",        # 0
        "",                    # 1
        "### ENTRY-1 — t",     # 2
        "body line",           # 3
        "",                    # 4
        "---",                 # 5
        "",                    # 6
        "### ENTRY-2 — t",     # 7
        "more body",           # 8
        "## Section B",        # 9
    ]
    bounds = ti._entry_block_bounds(lines)
    assert bounds[2] == 7   # ENTRY-1 ends at ENTRY-2's header (trailing --- travels with it)
    assert bounds[7] == 9   # ENTRY-2 ends at the next `## ` header
    assert 0 not in bounds  # `## ` headers are not entry blocks


# ─── cmd_archive() end-to-end on a temp tracker ─────────────────────────────

@pytest.fixture
def temp_tracker(tmp_path, monkeypatch):
    # Real trackers always live under REPO_ROOT; point it at tmp_path so
    # render_index's `relative_to(REPO_ROOT)` resolves for the fixture paths.
    monkeypatch.setattr(ti, "REPO_ROOT", tmp_path)
    body = tmp_path / "TRACK.md"
    index = tmp_path / "TRACK_index.md"
    grave = tmp_path / "TRACK_FINISHED.md"
    body.write_text(
        "# Track\n\n"
        "## Active\n\n"
        "### ALPHA — an open one [HIGH, OPEN]\n\n"
        "Body of alpha.\n\n"
        "---\n\n"
        "### BETA — a shipped one [MEDIUM, ✅ SHIPPED 2026-05-01]\n\n"
        "Body of beta with closure evidence.\n\n"
        "---\n\n"
        "### GAMMA — deferred but mentions SHIPPED dep [LOW]\n\n"
        "**Status:** DEFERRED — upstream rFP SHIPPED 2026-04-01.\n\n"
        "---\n"
    )
    cfg = ti.TrackerConfig(
        name="track", body_path=body, index_path=index,
        slug_re=re.compile(r"^### (~~)?(?P<slug>[A-Z][A-Z0-9a-z _/\-]+?)(~~)?\s*(?:—|–|-) "),
        title="Track", archive_path=grave,
    )
    return cfg, body, index, grave


def test_cmd_archive_moves_only_closed(temp_tracker):
    cfg, body, index, grave = temp_tracker
    rc = ti.cmd_archive(cfg, date_str="2026-05-29")
    assert rc == 0

    src = body.read_text()
    # BETA (shipped) moved out; ALPHA (open) + GAMMA (deferred-with-prose) stay.
    assert "### BETA " not in src
    assert "### ALPHA " in src
    assert "### GAMMA " in src
    assert "Body of beta" not in src

    grave_text = grave.read_text()
    assert "### BETA — archived 2026-05-29 (was SHIPPED)" in grave_text
    assert "Body of beta with closure evidence." in grave_text
    assert "🪦 Archived 2026-05-29" in grave_text

    # Index regenerated + reflects 2 active (ALPHA, GAMMA), 1 closed implicitly gone.
    idx = index.read_text()
    assert "ALPHA" in idx and "GAMMA" in idx
    assert "BETA" not in idx


def test_cmd_archive_idempotent_second_run_noop(temp_tracker):
    cfg, body, index, grave = temp_tracker
    ti.cmd_archive(cfg, date_str="2026-05-29")
    grave_after_first = grave.read_text()
    rc = ti.cmd_archive(cfg, date_str="2026-05-29")
    assert rc == 0
    # Nothing closed remains → graveyard unchanged.
    assert grave.read_text() == grave_after_first


def test_cmd_archive_dry_run_writes_nothing(temp_tracker):
    cfg, body, index, grave = temp_tracker
    before = body.read_text()
    rc = ti.cmd_archive(cfg, dry_run=True, date_str="2026-05-29")
    assert rc == 0
    assert body.read_text() == before
    assert not grave.exists()


def test_cmd_archive_no_archive_path_errors(tmp_path):
    body = tmp_path / "X.md"
    body.write_text("### A — t [OPEN]\n")
    cfg = ti.TrackerConfig(
        name="x", body_path=body, index_path=tmp_path / "X_index.md",
        slug_re=re.compile(r"^### (?P<slug>[A-Z]+) — "), title="X", archive_path=None,
    )
    assert ti.cmd_archive(cfg) == 2
