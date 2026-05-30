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


# ── end-to-end: restore_full + build_arc_to_target into a live tree ──────


def _sha256(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _full_diff(content: bytes) -> dict:
    return {"diff_mode": "full", "patch_bytes": content,
            "merkle_root": _sha256(content), "size_bytes": len(content),
            "encoder": "full_ship"}


class _DictArweave:
    def __init__(self):
        self._store: dict[str, bytes] = {}

    def put(self, data: bytes) -> str:
        import uuid
        tx = "ar_" + uuid.uuid4().hex[:16]
        self._store[tx] = data
        return tx

    async def fetch(self, tx_id: str) -> bytes:
        return self._store[tx_id]


def test_resurrection_restore_lands_real_arc_names_in_live_tree(tmp_path):
    """The novel restore-into-live-tree piece: restore_full + the production
    build_arc_to_target must write each component file to its REAL on-disk
    location under install_root (data/… and the root-level chronicle)."""
    import asyncio
    from titan_hcl.logic.backup_event_tarball import FileDiffSpec, pack_event_tarball
    from titan_hcl.logic.backup_restore import restore_full
    from titan_hcl.logic.backup_unified_manifest import UnifiedManifest, make_event
    from titan_hcl.logic.backup_zk_commit import compute_event_merkle_root

    arweave = _DictArweave()
    install_root = tmp_path / "fresh_box"
    install_root.mkdir()

    # Real arc_names spanning: data file, nested data file, root-level file,
    # timechain file. (Contents are arbitrary bytes for the round-trip.)
    personality_files = {
        "inner_memory.db": b"INNER-MEMORY-BYTES",
        "msl/msl_identity.json": b'{"I":0.9}',
        "titan_chronicles.md": b"# chronicle\nI remember.\n",
    }
    timechain_files = {"timechain/chain_main.bin": b"CHAIN-MAIN-BYTES"}

    def _pack(component: str, files: dict) -> dict:
        out = tmp_path / f"{component}.tar.gz"
        specs = [FileDiffSpec(arc, _full_diff(c)) for arc, c in files.items()]
        info = pack_event_tarball(event_id="evt1", event_type="baseline",
                                  component=component, file_specs=specs,
                                  output_path=str(out))
        return {"tx_id": arweave.put(out.read_bytes()),
                "merkle_root": info["tarball_sha256"],
                "size_bytes": info["size_bytes"], "diff_mode": "baseline"}

    p_sub = _pack("personality", personality_files)
    t_sub = _pack("timechain", timechain_files)
    compute_event_merkle_root(personality_merkle_root=p_sub["merkle_root"],
                              timechain_merkle_root=t_sub["merkle_root"],
                              soul_merkle_root=None)
    event = make_event(event_id="evt1", event_type="baseline",
                       prev_event_id=None, baseline_trigger="first_event",
                       personality=p_sub, timechain=t_sub, soul=None,
                       zk_commit_tx="sig1", zk_memo_prev_short="genesis")

    manifest = UnifiedManifest(titan_id="T1", base_dir=str(tmp_path / "mdir"))
    manifest._data["events"] = [event]
    manifest._data["current_baseline_event_id"] = "evt1"

    arc_to_target = build_arc_to_target(str(install_root))

    async def _no_memo(sig: str) -> str:
        raise AssertionError("memo_fetch must not be called with verify_zk=False")

    result = asyncio.run(restore_full(
        manifest=manifest, target_dir=str(install_root),
        arweave_fetch=arweave.fetch, memo_fetch=_no_memo,
        arc_to_target=arc_to_target, verify_zk_chain=False))

    assert result.status == "success", result.errors
    # Files landed at their REAL live-tree locations.
    assert (install_root / "data/inner_memory.db").read_bytes() == b"INNER-MEMORY-BYTES"
    assert (install_root / "data/msl/msl_identity.json").read_bytes() == b'{"I":0.9}'
    assert (install_root / "titan_chronicles.md").read_bytes() == b"# chronicle\nI remember.\n"
    assert (install_root / "data/timechain/chain_main.bin").read_bytes() == b"CHAIN-MAIN-BYTES"
