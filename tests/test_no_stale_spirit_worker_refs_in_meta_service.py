"""
Grep guard: no STALE current-tense `spirit_worker` references in
meta_service.py or meta_resolvers.py post-D-SPEC-70 §13 cleanup.

The CODE was migrated to cognitive_worker correctly during D8-3
spirit_worker retirement (commit 72f95a6b 2026-05-16); only docstrings
drifted. This test locks the cleanup so it doesn't regress.

Historical references are ALLOWED (and required for archeology):
  - "migrated from spirit_worker" (lineage)
  - "Post-D8-3 spirit_worker retirement" (retirement marker)
  - "NOT routed through spirit_worker per D8 retirement" (invariant)
  - "spirit_worker-hosted kinds" + "P8 D8.4 legacy compat" (compat note)

Forbidden references are CURRENT-tense statements that imply
spirit_worker is the current host / sender / receiver:
  - "Lives inside spirit_worker's process"
  - "the spirit_worker bus loop forwards it"
  - "sync spirit_worker bus handler"
  - "Called from spirit_worker's sync bus loop"
  - "Called periodically from spirit_worker"
  - "spirit_worker's send_queue"

Reference: rFP_meta_reasoning_self_reasoning_resolver_migration §13
"""
from __future__ import annotations

import pathlib

import pytest


# Forbidden substrings — present-tense statements implying spirit_worker
# is currently active. Each match is a bug if it appears in either file.
FORBIDDEN_PRESENT_TENSE = [
    "Lives inside spirit_worker",
    "the spirit_worker bus loop forwards",
    "sync spirit_worker bus handler",
    "Called from spirit_worker",
    "Called periodically from spirit_worker",
    "spirit_worker's send_queue",
    "spirit_worker thread",
    "spirit_worker bus handler entry",
    "spirit_worker.meta_service",
]


def _check_file(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(line_no, line_text), ...] for forbidden matches."""
    if not path.exists():
        pytest.skip(f"{path} missing")
    bad: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8")
    for ln_no, line in enumerate(text.splitlines(), start=1):
        for needle in FORBIDDEN_PRESENT_TENSE:
            if needle in line:
                bad.append((ln_no, line.strip()))
                break
    return bad


def test_meta_service_no_present_tense_spirit_worker_refs():
    """meta_service.py: no current-tense spirit_worker references."""
    path = pathlib.Path("titan_hcl/logic/meta_service.py")
    bad = _check_file(path)
    assert not bad, (
        f"Stale present-tense spirit_worker refs in {path}:\n"
        + "\n".join(f"  L{n}: {t}" for n, t in bad)
        + "\n\nMigrate to cognitive_worker per D-SPEC-70 §13 cleanup."
    )


def test_meta_resolvers_no_present_tense_spirit_worker_refs():
    """meta_resolvers.py: no current-tense spirit_worker references."""
    path = pathlib.Path("titan_hcl/logic/meta_resolvers.py")
    bad = _check_file(path)
    assert not bad, (
        f"Stale present-tense spirit_worker refs in {path}:\n"
        + "\n".join(f"  L{n}: {t}" for n, t in bad)
        + "\n\nMigrate to cognitive_worker per D-SPEC-70 §13 cleanup."
    )


def test_historical_refs_still_present_in_meta_service():
    """meta_service.py SHOULD retain the migration marker (lineage)."""
    text = pathlib.Path("titan_hcl/logic/meta_service.py").read_text(
        encoding="utf-8")
    assert "migrated from spirit_worker" in text or \
           "Post-D8-3" in text or \
           "D8-3 retirement" in text, (
        "meta_service.py should retain a historical marker noting the "
        "spirit_worker → cognitive_worker migration (lineage discipline)."
    )


def test_historical_refs_still_present_in_meta_resolvers():
    """meta_resolvers.py SHOULD retain the D8-3 retirement markers."""
    text = pathlib.Path("titan_hcl/logic/meta_resolvers.py").read_text(
        encoding="utf-8")
    assert "D8 retirement" in text or "D8-3" in text or \
           "Post-D8-3" in text, (
        "meta_resolvers.py should retain the D8-3 retirement invariant "
        "markers (architectural lineage)."
    )
