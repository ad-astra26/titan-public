"""IntegrityKit — RFP_backup_redesign_spine Phase A.

The ONE shared integrity helper (build + restore). Verifies it WRAPS the
existing crypto verbatim (the §24.7 event-Merkle formula, the encoders' per-file
hash) and reimplements none of it (INV-BRS-8 / INV-BR-8).
"""
import hashlib

import pytest

from titan_hcl.logic.backup_integrity_kit import IntegrityKit
from titan_hcl.logic import diff_encoders
from titan_hcl.logic.backup_zk_commit import compute_event_merkle_root


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_sha256_file_matches_hashlib_and_encoders(tmp_path):
    p = tmp_path / "blob.bin"
    data = b"titan-sovereign-bytes" * 4096
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert IntegrityKit.sha256_file(str(p)) == expected
    # single source of truth = the encoders' file hash (no drift)
    assert IntegrityKit.sha256_file(str(p)) == diff_encoders.file_merkle_root(str(p))


def test_verify_tarball(tmp_path):
    p = tmp_path / "event.tar.zst"
    p.write_bytes(b"\x28\xb5\x2f\xfd" + b"payload" * 100)
    good = IntegrityKit.sha256_file(str(p))
    assert IntegrityKit.verify_tarball(str(p), good) is True
    assert IntegrityKit.verify_tarball(str(p), _h("wrong")) is False


def test_event_merkle_wraps_spec_formula():
    p, t, s = _h("personality"), _h("timechain"), _h("soul")
    # IntegrityKit.event_merkle must be byte-identical to the §24.7 function
    assert IntegrityKit.event_merkle(p, t, s) == compute_event_merkle_root(p, t, s)
    # soul=None path (no soul tier this event) also matches
    assert IntegrityKit.event_merkle(p, t) == compute_event_merkle_root(p, t, None)
    # malformed root → the wrapped function raises (not silently accepted)
    with pytest.raises(ValueError):
        IntegrityKit.event_merkle("deadbeef", t, s)


def test_verify_chain_link():
    assert IntegrityKit.verify_chain_link("evt_5", "evt_5") is True
    assert IntegrityKit.verify_chain_link(None, None) is True       # baseline: prev=None
    assert IntegrityKit.verify_chain_link("evt_5", "evt_4") is False  # break
    assert IntegrityKit.verify_chain_link(None, "evt_4") is False     # orphaned baseline
