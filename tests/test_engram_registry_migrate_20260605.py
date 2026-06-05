"""§7.G boot-migrate — legacy `synthesis_spine_concepts.json` → `synthesis_engrams.json`.

The external rename (concepts→engrams) is no-shim, so the CGN registry file is
renamed on first load (once) rather than read from the old path. Covers the
rename, the new-already-exists no-op, and the no-legacy cold-start.

Run: python -m pytest tests/test_engram_registry_migrate_20260605.py -v -p no:anchorpy
"""
import json

from titan_hcl.synthesis.cgn_bridge import CGNRegistrationBridge


def test_boot_migrate_renames_legacy_registry(tmp_path):
    legacy = tmp_path / "synthesis_spine_concepts.json"
    new = tmp_path / "synthesis_engrams.json"
    legacy.write_text(json.dumps(
        {"glacier_x": {"concept_id": "glacier_x", "name": "Glacier"}}))
    b = CGNRegistrationBridge(registry_path=str(new))
    assert b.is_registered("glacier_x") is True   # triggers _load → migrate
    assert new.exists() and not legacy.exists()   # renamed, legacy gone
    assert "glacier_x" in json.loads(new.read_text())  # data preserved


def test_no_migrate_when_new_already_exists(tmp_path):
    legacy = tmp_path / "synthesis_spine_concepts.json"
    new = tmp_path / "synthesis_engrams.json"
    legacy.write_text(json.dumps({"old": {"concept_id": "old"}}))
    new.write_text(json.dumps({"current": {"concept_id": "current"}}))
    b = CGNRegistrationBridge(registry_path=str(new))
    assert b.is_registered("current") is True
    assert b.is_registered("old") is False  # legacy NOT merged — new wins
    assert legacy.exists()                  # legacy untouched (new present)


def test_cold_start_no_legacy_is_noop(tmp_path):
    new = tmp_path / "synthesis_engrams.json"
    b = CGNRegistrationBridge(registry_path=str(new))
    assert b.is_registered("anything") is False  # empty, no crash


if __name__ == "__main__":
    import tempfile
    import pathlib
    for fn in (test_boot_migrate_renames_legacy_registry,
               test_no_migrate_when_new_already_exists,
               test_cold_start_no_legacy_is_noop):
        with tempfile.TemporaryDirectory() as td:
            fn(pathlib.Path(td))
    print("OK — §7.G registry migrate verified")
