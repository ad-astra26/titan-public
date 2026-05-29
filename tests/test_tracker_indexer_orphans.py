"""Regression guard for the tracker-indexer silent-drop drift class.

`scripts/tracker_indexer.py` uses table-row-first semantics: when a tracker
file (BUGS.md / OBSERVABLES.md) has a top index table, that table is the
canonical entry list. The original implementation returned ONLY table rows
and silently ignored any `### BUG-` body section that lacked a table row —
so an OPEN bug nobody added to the table vanished from the count entirely
(observed 2026-05-29: 3 BUGS + 19 OBSERVABLES body-only entries invisible).

The hardening folds orphan body sections back in from their `**Status:**`
line and warns. These tests pin that behavior so the silent-drop can't
regress.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "tracker_indexer",
    Path(__file__).resolve().parent.parent / "scripts" / "tracker_indexer.py",
)
ti = importlib.util.module_from_spec(_SPEC)
# Register before exec so the frozen dataclass can resolve its own module.
sys.modules["tracker_indexer"] = ti
_SPEC.loader.exec_module(ti)


BUGS_SLUG_RE = re.compile(
    r"^### (~~)?(?P<slug>(?:BUG-)?[A-Z][A-Z0-9_\-]+(?:-20\d{6})?)(~~)?\s*(?:—|–|-) "
)


def _cfg(tmp_path: Path) -> "ti.TrackerConfig":
    body = tmp_path / "BUGS.md"
    return ti.TrackerConfig(
        name="bugs",
        body_path=body,
        index_path=tmp_path / "BUGS_index.md",
        slug_re=BUGS_SLUG_RE,
        title="Test BUGS Index",
    )


def _write(cfg, text: str) -> None:
    cfg.body_path.write_text(text)


def test_orphan_body_entry_is_folded_in(tmp_path, capsys):
    """A body `### BUG-` section with no table row must still be counted,
    taking its status from the body **Status:** line, and emit a warning."""
    cfg = _cfg(tmp_path)
    _write(cfg, """# Bugs

| ID | Severity | Status | Title |
|---|---|---|---|
| [BUG-IN-TABLE-20260101](#bug-in-table-20260101) | 🟠 HIGH | OPEN | tracked |

### BUG-IN-TABLE-20260101 — tracked [HIGH, 2026-01-01]
- **Status:** OPEN

### BUG-ORPHAN-NO-ROW-20260202 — never added to the table [MEDIUM, 2026-02-02]
- **Status:** OPEN
""")
    entries = ti.parse_tracker(cfg)
    slugs = {e.slug for e in entries}
    assert "BUG-IN-TABLE-20260101" in slugs
    assert "BUG-ORPHAN-NO-ROW-20260202" in slugs, "orphan body entry was silently dropped"
    orphan = next(e for e in entries if e.slug == "BUG-ORPHAN-NO-ROW-20260202")
    assert orphan.status == "OPEN"
    # Warning must surface so a maintainer adds a real table row.
    assert "missing a top-table row" in capsys.readouterr().err


def test_graveyard_checkmark_header_is_not_an_orphan(tmp_path, capsys):
    """`### ✅ BUG-...` graveyard headers don't match slug_re, so they must
    NOT be surfaced as orphans (they're resolved + table-tracked)."""
    cfg = _cfg(tmp_path)
    _write(cfg, """# Bugs

| ID | Severity | Status | Title |
|---|---|---|---|
| [BUG-LIVE-20260101](#bug-live-20260101) | 🟠 HIGH | OPEN | live |

### BUG-LIVE-20260101 — live [HIGH, 2026-01-01]
- **Status:** OPEN

## Graveyard

### ✅ BUG-DONE-20259999 — FIXED 2025-99-99
- **Status:** FIXED
""")
    entries = ti.parse_tracker(cfg)
    slugs = {e.slug for e in entries}
    assert "BUG-DONE-20259999" not in slugs
    assert "missing a top-table row" not in capsys.readouterr().err


def test_orphan_status_classified_from_body(tmp_path):
    """An orphan whose body says DEFERRED must classify as DEFERRED, not OPEN."""
    cfg = _cfg(tmp_path)
    _write(cfg, """# Bugs

| ID | Severity | Status | Title |
|---|---|---|---|
| [BUG-ANCHOR-20260101](#bug-anchor-20260101) | 🟠 HIGH | OPEN | anchor |

### BUG-ANCHOR-20260101 — anchor [HIGH, 2026-01-01]
- **Status:** OPEN

### BUG-DEFERRED-ORPHAN-20260303 — deferred body-only [LOW, 2026-03-03]
- **Status:** DEFERRED — parked for a later session
""")
    entries = ti.parse_tracker(cfg)
    orphan = next(e for e in entries if e.slug == "BUG-DEFERRED-ORPHAN-20260303")
    assert orphan.status == "DEFERRED"
