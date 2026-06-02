"""Tests for SPEC §24.5 per-format diff encoders.

Per rFP_backup_diff_baseline_unified_v1 §5.2 test-coverage requirements:
  - Round-trip property test: baseline + N incrementals → reconstructed_state
    == direct_snapshot_state (byte-exact) for N ∈ {1, 5, 30}
  - Merkle drift test: tamper a byte → apply halts at verify step
  - Encoder dispatch by file extension + format_hint

Three encoders tested:
  - timechain_tail (append-only .bin chains)
  - xdelta3 (SQLite/DuckDB/Kuzu DBs + large JSONs)
  - full_ship (FAISS / small JSONs)
"""

import hashlib
import json
import os
import random
from pathlib import Path

import pytest

from titan_hcl.logic.diff_encoders import (
    apply_diff,
    encode_diff,
    file_merkle_root,
    select_encoder,
    verify,
)
from titan_hcl.logic.diff_encoders import (
    full_ship,
    timechain_tail,
    xdelta3,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: str, data: bytes) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


# ── select_encoder dispatch ──────────────────────────────────────────────


@pytest.mark.parametrize("path,expected", [
    ("data/timechain/chain_episodic.bin", "timechain_tail"),
    ("data/timechain/chain_main.bin", "timechain_tail"),
    ("data/inner_memory.db", "xdelta3"),
    ("data/titan_memory.duckdb", "xdelta3"),
    ("data/knowledge_graph.kuzu", "xdelta3"),
    ("data/memory_vectors.faiss", "full_ship"),
])
def test_select_encoder_by_extension(path, expected):
    assert select_encoder(path) == expected


def test_select_encoder_small_json_full_ship(tmp_path):
    p = tmp_path / "tiny.json"
    p.write_text('{"x":1}')
    assert select_encoder(str(p)) == "full_ship"


def test_select_encoder_large_json_xdelta3(tmp_path):
    p = tmp_path / "big.json"
    p.write_text(json.dumps({"k": "v" * 20000}))
    assert select_encoder(str(p)) == "xdelta3"


def test_select_encoder_format_hint_overrides_extension(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"x" * 100)
    # Without hint, .bin without /timechain/chain_ prefix → full_ship default
    assert select_encoder(str(p)) == "full_ship"
    # With hint → forced
    assert select_encoder(str(p), format_hint="timechain_bin") == "timechain_tail"
    assert select_encoder(str(p), format_hint="db") == "xdelta3"


# ── timechain_tail encoder ───────────────────────────────────────────────


def test_timechain_tail_full_event_no_baseline(tmp_path):
    """First-ever event (no baseline) → full-ship the whole file."""
    chain = tmp_path / "chain.bin"
    chain.write_bytes(b"abc" * 1000)
    diff = timechain_tail.encode_diff(str(chain), baseline_path=None,
                                      block_range=(0, 999))
    assert diff["diff_mode"] == "full"
    # Phase 5 — bytes live on disk at patch_path (streaming refactor)
    with open(diff["patch_path"], "rb") as f:
        assert f.read() == b"abc" * 1000
    assert diff["prev_offset_bytes"] == 0
    assert diff["block_range"] == [0, 999]
    assert diff["merkle_root"] == _sha256_bytes(b"abc" * 1000)


def test_timechain_tail_incremental_diff(tmp_path):
    baseline = tmp_path / "baseline.bin"
    current = tmp_path / "current.bin"
    baseline.write_bytes(b"X" * 100)
    # Current = baseline + 50 new bytes (append-only)
    current.write_bytes(b"X" * 100 + b"Y" * 50)
    diff = timechain_tail.encode_diff(str(current), str(baseline),
                                      block_range=(100, 149))
    assert diff["diff_mode"] == "tail"
    # Phase 5 — tail bytes live on disk at patch_path (streamed)
    with open(diff["patch_path"], "rb") as f:
        assert f.read() == b"Y" * 50
    assert diff["prev_offset_bytes"] == 100
    assert diff["size_bytes"] == 150
    assert diff["block_range"] == [100, 149]


def test_timechain_tail_round_trip_byte_exact(tmp_path):
    """Round-trip: encode_diff → apply_diff → byte-identical to original."""
    baseline = tmp_path / "baseline.bin"
    current = tmp_path / "current.bin"
    restored = tmp_path / "restored.bin"
    baseline.write_bytes(b"BASE" * 50)
    current.write_bytes(b"BASE" * 50 + b"TAIL" * 25)
    diff = timechain_tail.encode_diff(str(current), str(baseline))
    timechain_tail.apply_diff(str(baseline), diff, str(restored))
    assert restored.read_bytes() == current.read_bytes()


def test_timechain_tail_truncation_falls_back_to_full(tmp_path):
    """If current is SMALLER than baseline (append-only violated) → full ship."""
    baseline = tmp_path / "baseline.bin"
    current = tmp_path / "current.bin"
    baseline.write_bytes(b"X" * 200)
    current.write_bytes(b"X" * 100)  # truncated
    diff = timechain_tail.encode_diff(str(current), str(baseline))
    assert diff["diff_mode"] == "full"
    # Phase 5 — full-ship returns source path (patch_owned=False)
    with open(diff["patch_path"], "rb") as f:
        assert f.read() == b"X" * 100


def test_timechain_tail_apply_detects_size_mismatch(tmp_path):
    """Restore-time size mismatch → ValueError (caller emits BACKUP_MERKLE_MISMATCH)."""
    baseline = tmp_path / "baseline.bin"
    baseline.write_bytes(b"X" * 50)
    diff = {
        "diff_mode": "tail",
        "patch_bytes": b"Y" * 25,
        "prev_offset_bytes": 50,
        "size_bytes": 999,  # WRONG — actual would be 75
        "merkle_root": "ff" * 32,
    }
    with pytest.raises(ValueError, match="size mismatch"):
        timechain_tail.apply_diff(str(baseline), diff, str(tmp_path / "out.bin"))


def test_timechain_tail_apply_detects_merkle_mismatch(tmp_path):
    """Restore-time content hash mismatch → ValueError."""
    baseline = tmp_path / "baseline.bin"
    baseline.write_bytes(b"X" * 50)
    diff = {
        "diff_mode": "tail",
        "patch_bytes": b"Y" * 25,
        "prev_offset_bytes": 50,
        "size_bytes": 75,
        "merkle_root": "ff" * 32,  # WRONG
    }
    with pytest.raises(ValueError, match="merkle_root mismatch"):
        timechain_tail.apply_diff(str(baseline), diff, str(tmp_path / "out.bin"))


def test_apply_diff_verify_output_false_downgrades_post_apply_to_advisory(tmp_path):
    """verify_output=False (source tarball already on-chain-arc-verified): a stale
    post-apply size/merkle_root mismatch is logged, NOT raised — the written bytes
    are still the authentic (arc-committed) content. Covers full_ship + tail; the
    incremental BASELINE check stays strict (tested separately)."""
    # full_ship: result == patch_bytes (the arc-authenticated member); stale merkle.
    out = tmp_path / "fs.bin"
    fs_diff = {"diff_mode": "full", "patch_bytes": b"AUTHENTIC", "encoder": "full_ship",
               "size_bytes": len(b"AUTHENTIC"), "merkle_root": "ee" * 32}  # stale
    with pytest.raises(ValueError, match="merkle_root mismatch"):
        full_ship.apply_diff(None, fs_diff, str(out))             # strict default
    full_ship.apply_diff(None, fs_diff, str(out), verify_output=False)  # advisory
    assert out.read_bytes() == b"AUTHENTIC"                       # authentic bytes written

    # timechain_tail: baseline[:offset] + tail; stale merkle → advisory proceeds.
    baseline = tmp_path / "tc_base.bin"; baseline.write_bytes(b"X" * 50)
    tc_diff = {"diff_mode": "tail", "patch_bytes": b"Y" * 25, "prev_offset_bytes": 50,
               "size_bytes": 75, "merkle_root": "ff" * 32}        # stale
    tc_out = tmp_path / "tc_out.bin"
    timechain_tail.apply_diff(str(baseline), tc_diff, str(tc_out), verify_output=False)
    assert tc_out.read_bytes() == b"X" * 50 + b"Y" * 25


# ── timechain_tail round-trip property test (N incrementals) ─────────────


@pytest.mark.parametrize("n_incrementals", [1, 5, 30])
def test_timechain_tail_n_incrementals_round_trip(tmp_path, n_incrementals):
    """rFP §5.2 round-trip property: baseline + N incrementals →
    reconstructed state byte-identical to direct snapshot."""
    chain = tmp_path / "chain.bin"
    snapshots = []  # path of file as it existed at each event
    # Baseline event: chain starts at 100 bytes
    chain.write_bytes(b"BASELINE_DATA_" * 20)
    snap = tmp_path / "snap_0.bin"
    snap.write_bytes(chain.read_bytes())
    snapshots.append(snap)
    events = []  # list of diff_dicts, in chronological order

    # Baseline diff. Phase 5: encoder returns patch_path = source-file path
    # (patch_owned=False), so the test must encode against the stable
    # snapshot rather than the live `chain` file (which will be mutated
    # below). In production this race doesn't exist because pack_event_tarball
    # streams from patch_path immediately after encode_diff — the source
    # file isn't mutated between encode and pack.
    events.append(timechain_tail.encode_diff(str(snap), baseline_path=None))

    # N incrementals — each appends a random-length chunk
    rng = random.Random(42)
    for i in range(n_incrementals):
        prev_snap = snapshots[-1]
        tail = bytes(rng.randint(0, 255) for _ in range(rng.randint(10, 100)))
        with open(chain, "ab") as f:
            f.write(tail)
        snap_i = tmp_path / f"snap_{i+1}.bin"
        snap_i.write_bytes(chain.read_bytes())
        snapshots.append(snap_i)
        diff = timechain_tail.encode_diff(str(chain), str(prev_snap))
        events.append(diff)

    # Reconstruct from events: apply baseline → apply each incremental
    reconstructed = tmp_path / "reconstructed.bin"
    timechain_tail.apply_diff(None, events[0], str(reconstructed))
    # Verify baseline reconstruction matches the original baseline snapshot
    assert reconstructed.read_bytes() == snapshots[0].read_bytes()

    for i in range(1, len(events)):
        # Apply patch — baseline is the previous reconstructed state
        new_out = tmp_path / f"recon_{i}.bin"
        timechain_tail.apply_diff(str(reconstructed), events[i], str(new_out))
        reconstructed = new_out
        # Each intermediate reconstruction must match the snapshot at that
        # event (byte-exact)
        assert reconstructed.read_bytes() == snapshots[i].read_bytes()

    # Final state byte-identical to direct snapshot
    assert reconstructed.read_bytes() == chain.read_bytes()


# ── xdelta3 encoder ──────────────────────────────────────────────────────


@pytest.mark.skipif(not xdelta3._xdelta3_available(),
                    reason="xdelta3 binary not installed (production hosts have it)")
def test_xdelta3_full_event_no_baseline(tmp_path):
    src = tmp_path / "db.sqlite"
    src.write_bytes(b"SQLITE_DATA" * 100)
    diff = xdelta3.encode_diff(str(src), baseline_path=None)
    assert diff["diff_mode"] == "full"
    # Phase 5 — bytes live on disk at patch_path (streaming refactor)
    with open(diff["patch_path"], "rb") as f:
        assert f.read() == b"SQLITE_DATA" * 100


@pytest.mark.skipif(not xdelta3._xdelta3_available(),
                    reason="xdelta3 binary not installed")
def test_xdelta3_round_trip_byte_exact(tmp_path):
    baseline = tmp_path / "baseline.db"
    current = tmp_path / "current.db"
    restored = tmp_path / "restored.db"
    baseline.write_bytes(b"BASELINE_DB_CONTENT" * 100)
    current.write_bytes(b"BASELINE_DB_CONTENT" * 90 + b"MUTATED_TAIL" * 50)
    diff = xdelta3.encode_diff(str(current), str(baseline))
    assert diff["diff_mode"] == "incremental"
    # Phase 5 — vcdiff lives on disk at patch_path; size via patch_size_bytes.
    # vcdiff should be smaller than the original (compression benefit).
    assert diff["patch_size_bytes"] < diff["size_bytes"]
    xdelta3.apply_diff(str(baseline), diff, str(restored))
    assert restored.read_bytes() == current.read_bytes()


@pytest.mark.skipif(not xdelta3._xdelta3_available(),
                    reason="xdelta3 binary not installed")
def test_xdelta3_apply_rejects_wrong_baseline(tmp_path):
    """rFP §11 risk #1 — xdelta3 patch refused on source hash mismatch."""
    baseline = tmp_path / "baseline.db"
    current = tmp_path / "current.db"
    baseline.write_bytes(b"A" * 1000)
    current.write_bytes(b"A" * 800 + b"B" * 200)
    diff = xdelta3.encode_diff(str(current), str(baseline))
    # Tamper the baseline before applying
    baseline.write_bytes(b"TAMPERED" * 100)
    with pytest.raises(ValueError, match="baseline merkle_root mismatch"):
        xdelta3.apply_diff(str(baseline), diff, str(tmp_path / "out.db"))


@pytest.mark.skipif(not xdelta3._xdelta3_available(),
                    reason="xdelta3 binary not installed")
def test_xdelta3_apply_detects_merkle_mismatch(tmp_path):
    """Post-apply sha256 verification."""
    src = tmp_path / "src.db"
    src.write_bytes(b"X" * 100)
    diff = xdelta3.encode_diff(str(src), baseline_path=None)
    diff["merkle_root"] = "ff" * 32  # tamper
    with pytest.raises(ValueError, match="merkle_root mismatch"):
        xdelta3.apply_diff(None, diff, str(tmp_path / "out.db"))


@pytest.mark.skipif(not xdelta3._xdelta3_available(),
                    reason="xdelta3 binary not installed")
@pytest.mark.parametrize("n_incrementals", [1, 5, 30])
def test_xdelta3_n_incrementals_round_trip(tmp_path, n_incrementals):
    """rFP §5.2 round-trip property test for xdelta3."""
    state_paths = []
    # Baseline state: 10 KB random
    rng = random.Random(7)
    base_data = bytes(rng.randint(0, 255) for _ in range(10 * 1024))
    p0 = tmp_path / "state_0.db"
    p0.write_bytes(base_data)
    state_paths.append(p0)
    events = [xdelta3.encode_diff(str(p0), baseline_path=None)]

    # Per-event mutation: flip a few bytes + append a chunk (typical DB churn)
    for i in range(n_incrementals):
        prev = state_paths[-1].read_bytes()
        # mutate ~5% of bytes randomly
        mutated = bytearray(prev)
        for _ in range(max(1, len(mutated) // 20)):
            idx = rng.randint(0, len(mutated) - 1)
            mutated[idx] = rng.randint(0, 255)
        # append a small chunk
        mutated.extend(bytes(rng.randint(0, 255) for _ in range(rng.randint(50, 500))))
        new_state = tmp_path / f"state_{i+1}.db"
        new_state.write_bytes(bytes(mutated))
        state_paths.append(new_state)
        diff = xdelta3.encode_diff(str(new_state), str(state_paths[-2]))
        events.append(diff)

    # Reconstruct
    reconstructed = tmp_path / "reconstructed.db"
    xdelta3.apply_diff(None, events[0], str(reconstructed))
    assert reconstructed.read_bytes() == state_paths[0].read_bytes()

    for i in range(1, len(events)):
        new_out = tmp_path / f"recon_{i}.db"
        xdelta3.apply_diff(str(reconstructed), events[i], str(new_out))
        reconstructed = new_out
        assert reconstructed.read_bytes() == state_paths[i].read_bytes(), (
            f"Mismatch at event {i}/{len(events) - 1}"
        )


# ── full_ship encoder ────────────────────────────────────────────────────


def test_full_ship_round_trip(tmp_path):
    src = tmp_path / "small.json"
    src.write_text('{"hello": "world", "x": 42}')
    out = tmp_path / "restored.json"
    diff = full_ship.encode_diff(str(src), baseline_path=None)
    assert diff["diff_mode"] == "full"
    full_ship.apply_diff(None, diff, str(out))
    assert out.read_bytes() == src.read_bytes()


def test_full_ship_rejects_non_full_mode(tmp_path):
    diff = {"diff_mode": "incremental", "patch_bytes": b"", "size_bytes": 0,
            "merkle_root": ""}
    with pytest.raises(ValueError, match="only supports diff_mode='full'"):
        full_ship.apply_diff(None, diff, str(tmp_path / "out"))


def test_full_ship_apply_detects_merkle_mismatch(tmp_path):
    diff = {"diff_mode": "full", "patch_bytes": b"hello",
            "size_bytes": 5, "merkle_root": "ff" * 32}
    with pytest.raises(ValueError, match="merkle_root mismatch"):
        full_ship.apply_diff(None, diff, str(tmp_path / "out"))


# ── top-level dispatch round-trip ────────────────────────────────────────


def test_top_level_encode_decode_records_encoder_name(tmp_path):
    p = tmp_path / "data/timechain/chain_episodic.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"chain_data" * 50)
    diff = encode_diff(str(p), baseline_path=None)
    assert diff["encoder"] == "timechain_tail"


def test_top_level_apply_routes_via_encoder_field(tmp_path):
    src = tmp_path / "small.json"
    src.write_text("{}")
    out = tmp_path / "out.json"
    diff = encode_diff(str(src), baseline_path=None)
    apply_diff(None, diff, str(out))
    assert out.read_bytes() == src.read_bytes()


def test_top_level_apply_rejects_unknown_encoder(tmp_path):
    bad = {"encoder": "garbage", "diff_mode": "full", "patch_bytes": b""}
    with pytest.raises(ValueError, match="Unknown encoder"):
        apply_diff(None, bad, str(tmp_path / "out"))


def test_file_merkle_root_matches_sha256(tmp_path):
    p = tmp_path / "test.bin"
    p.write_bytes(b"some content here")
    assert file_merkle_root(str(p)) == _sha256_bytes(b"some content here")


def test_verify_helper(tmp_path):
    p = tmp_path / "test.bin"
    p.write_bytes(b"abc")
    correct_root = _sha256_bytes(b"abc")
    assert verify(str(p), correct_root) is True
    assert verify(str(p), "ff" * 32) is False
