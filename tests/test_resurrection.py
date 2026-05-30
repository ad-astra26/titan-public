"""W1.5 resurrection unit tests — the offline-testable pieces (steps 1-2 of
PLAN_w1_5_resurrection.md).

Covers:
  - backup_restore.build_arc_to_target — the production inverse-map that
    restores reconstructed component files back into a live install tree.
  - genesis_runner.write_bootable_identity — the shared kernel-boot identity
    writer (also exercised end-to-end through a real SSS 2-of-3 round-trip,
    proving a fresh-box resurrection reconstructs the SAME pubkey).

No network, no Solana, no Arweave. The Arweave manifest-chain walk itself is
covered by test_backup_restore.py; resurrection delegates to that engine, so
here we only test the two new restore-into-live-tree primitives.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
# Append (not insert-at-0): scripts/ contains a titan_hcl.py that would
# otherwise shadow the real titan_hcl *package*. Appending lets the package
# win while still exposing the setup_titan package under scripts/.
sys.path.append(str(_REPO / "scripts"))

from titan_hcl.logic.backup_restore import build_arc_to_target  # noqa: E402
from titan_hcl.utils import shamir  # noqa: E402
from setup_titan.genesis_runner import (  # noqa: E402
    identity_path,
    keypair_path,
    write_bootable_identity,
)


# ── build_arc_to_target — the inverse map ────────────────────────────────


def test_arc_to_target_personality_exact_file(tmp_path):
    arc = build_arc_to_target(str(tmp_path))
    # exact arc_name → its src_path (data/ + arc here)
    assert arc("personality", "inner_memory.db") == str(tmp_path / "data/inner_memory.db")


def test_arc_to_target_personality_renamed_dir_arc(tmp_path):
    """The arc 'neural_ns' maps to data/neural_nervous_system/ — proving the
    map inverts the real tuples, NOT a naive 'data/' + arc heuristic."""
    arc = build_arc_to_target(str(tmp_path))
    # Bare-dir exact match returns src verbatim (with its trailing slash) —
    # mirrors RebirthBackup._archive_name_to_path. Not a production input
    # (real tarballs carry per-file arc_names), but defined behaviour.
    assert arc("personality", "neural_ns") == str(tmp_path / "data/neural_nervous_system") + "/"
    # a file *inside* that directory arc resolves via prefix-match
    assert arc("personality", "neural_ns/buffer.pt") == \
        str(tmp_path / "data/neural_nervous_system/buffer.pt")


def test_arc_to_target_directory_prefix(tmp_path):
    arc = build_arc_to_target(str(tmp_path))
    # ("data/cgn/", "cgn") → cgn/<x> resolves under data/cgn/
    assert arc("personality", "cgn/state.bin") == str(tmp_path / "data/cgn/state.bin")


def test_arc_to_target_root_level_file(tmp_path):
    """titan_chronicles.md has NO data/ prefix — a repo-root file."""
    arc = build_arc_to_target(str(tmp_path))
    assert arc("personality", "titan_chronicles.md") == str(tmp_path / "titan_chronicles.md")


def test_arc_to_target_timechain_component(tmp_path):
    arc = build_arc_to_target(str(tmp_path))
    assert arc("timechain", "timechain/chain_main.bin") == \
        str(tmp_path / "data/timechain/chain_main.bin")


def test_arc_to_target_soul_component(tmp_path):
    arc = build_arc_to_target(str(tmp_path))
    assert arc("soul", "consciousness.db") == str(tmp_path / "data/consciousness.db")
    # exact file tuple in WEEKLY_EXTRA_PATHS wins (no shadowing by a dir arc)
    assert arc("soul", "cgn/affinity_history.jsonl") == \
        str(tmp_path / "data/cgn/affinity_history.jsonl")


def test_arc_to_target_unknown_component_raises(tmp_path):
    arc = build_arc_to_target(str(tmp_path))
    with pytest.raises(ValueError, match="unknown restore component"):
        arc("bogus", "inner_memory.db")


def test_arc_to_target_unknown_arc_raises(tmp_path):
    arc = build_arc_to_target(str(tmp_path))
    with pytest.raises(ValueError, match="not found in the personality path map"):
        arc("personality", "no_such_arc_xyz")


def test_arc_to_target_component_isolation(tmp_path):
    """An arc that lives in the soul map must NOT resolve under personality."""
    arc = build_arc_to_target(str(tmp_path))
    with pytest.raises(ValueError):
        arc("personality", "consciousness.db")  # consciousness.db is a soul path


# ── write_bootable_identity — shared kernel-boot writer ──────────────────


def _new_keypair_bytes() -> bytes:
    from solders.keypair import Keypair
    kp = Keypair()
    return bytes(kp), str(kp.pubkey())


def test_write_bootable_identity_files_and_perms(tmp_path):
    key_bytes, pubkey = _new_keypair_bytes()
    written = write_bootable_identity(tmp_path, key_bytes,
                                      titan_id="T1", titan_pubkey=pubkey)
    assert written == keypair_path(tmp_path)
    # keypair is a JSON array of 64 ints, 0600
    arr = json.loads(written.read_text())
    assert arr == list(key_bytes)
    assert oct(written.stat().st_mode & 0o777) == "0o600"
    # identity.json carries titan_id + pubkey
    ident = json.loads(identity_path(tmp_path).read_text())
    assert ident == {"titan_id": "T1", "titan_pubkey": pubkey}


def test_write_bootable_identity_roundtrips_to_same_pubkey(tmp_path):
    from solders.keypair import Keypair
    key_bytes, pubkey = _new_keypair_bytes()
    written = write_bootable_identity(tmp_path, key_bytes, titan_id="T1",
                                      titan_pubkey=pubkey)
    reloaded = Keypair.from_bytes(bytes(json.loads(written.read_text())))
    assert str(reloaded.pubkey()) == pubkey


def test_write_bootable_identity_rejects_wrong_length(tmp_path):
    with pytest.raises(ValueError, match="must be 64 bytes"):
        write_bootable_identity(tmp_path, b"\x00" * 32, titan_id="T1")


def test_resurrection_sss_2of3_reconstructs_bootable_identity(tmp_path):
    """End-to-end of the identity half of resurrection: split a real keypair
    3-of-which-2, then reconstruct from shards 1+3 (the fresh-box combo:
    Maker shard + on-chain shard) and materialize a bootable identity that
    loads back to the SAME pubkey."""
    from solders.keypair import Keypair
    key_bytes, pubkey = _new_keypair_bytes()

    shards = shamir.split_secret(key_bytes, n=3, t=2)
    # fresh box = Maker's Shard-1 + on-chain Shard-3 (indices 0 and 2)
    reconstructed = shamir.combine_shares([shards[0], shards[2]])
    assert reconstructed == key_bytes

    written = write_bootable_identity(tmp_path, reconstructed, titan_id="T1",
                                      titan_pubkey=pubkey)
    reloaded = Keypair.from_bytes(bytes(json.loads(written.read_text())))
    assert str(reloaded.pubkey()) == pubkey
