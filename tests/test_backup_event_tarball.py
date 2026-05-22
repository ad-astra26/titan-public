"""SPEC §24 — backup_event_tarball pack/unpack round-trip tests."""

from __future__ import annotations

import hashlib
import os
import tarfile

import pytest

from titan_hcl.logic.backup_event_tarball import (
    EVENT_METADATA_MEMBER,
    EVENT_TARBALL_EXT,
    EVENT_TARBALL_SCHEMA_VERSION,
    FILES_PREFIX,
    GZIP_MAGIC,
    ZSTD_MAGIC,
    FileDiffSpec,
    UnpackedEvent,
    pack_event_tarball,
    unpack_event_tarball,
)


def _full_diff_dict(content: bytes) -> dict:
    """Helper: synthesize the diff_dict shape that full_ship would produce."""
    return {
        "diff_mode": "full",
        "patch_bytes": content,
        "merkle_root": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "encoder": "full_ship",
    }


def _xdelta_incremental_dict(patch: bytes, baseline_hash: str,
                              post_hash: str, post_size: int) -> dict:
    return {
        "diff_mode": "incremental",
        "patch_bytes": patch,
        "merkle_root": post_hash,
        "size_bytes": post_size,
        "baseline_merkle_root": baseline_hash,
        "encoder": "xdelta3",
    }


def _tail_dict(tail: bytes, prev_offset: int, post_hash: str,
               post_size: int, block_range=None) -> dict:
    return {
        "diff_mode": "tail",
        "patch_bytes": tail,
        "merkle_root": post_hash,
        "size_bytes": post_size,
        "prev_offset_bytes": prev_offset,
        "block_range": list(block_range) if block_range else None,
        "encoder": "timechain_tail",
    }


def test_pack_creates_zstd_tar_with_metadata(tmp_path):
    """Phase 5 chunk 5F: pack now emits zstd-3 (was gzip-9). Verifies the
    magic bytes + that unpack_event_tarball can round-trip its own output.
    """
    out = tmp_path / f"event{EVENT_TARBALL_EXT}"
    specs = [
        FileDiffSpec("inner_memory.db", _full_diff_dict(b"a" * 100)),
        FileDiffSpec("titan_identity.json", _full_diff_dict(b"{}")),
    ]
    info = pack_event_tarball(
        event_id="evt_001",
        event_type="baseline",
        component="personality",
        file_specs=specs,
        output_path=str(out),
    )
    assert info["path"] == str(out)
    assert info["file_count"] == 2
    assert os.path.exists(out)
    assert info["size_bytes"] == os.path.getsize(out)
    # Magic bytes identify it as a zstd frame, not gzip.
    head = out.read_bytes()[:4]
    assert head == ZSTD_MAGIC, f"expected zstd magic, got {head.hex()}"
    # tarball_sha256 matches the on-disk compressed file.
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    assert info["tarball_sha256"] == h
    # All expected members present (verified via the unpack path which is
    # the production-canonical read API; raw tarfile.open(mode="r:gz")
    # would no longer apply).
    with unpack_event_tarball(str(out)) as ev:
        assert ev.event_id == "evt_001"
        arc_names = {f["arc_name"] for f in ev.files}
        assert "inner_memory.db" in arc_names
        assert "titan_identity.json" in arc_names


def test_pack_validates_event_type():
    with pytest.raises(ValueError, match="event_type"):
        pack_event_tarball(
            event_id="x", event_type="bogus", component="personality",
            file_specs=[], output_path="/tmp/should_not_create.tar.gz",
        )


def test_pack_validates_component():
    with pytest.raises(ValueError, match="component"):
        pack_event_tarball(
            event_id="x", event_type="baseline", component="weird",
            file_specs=[], output_path="/tmp/should_not_create.tar.gz",
        )


def test_pack_rejects_duplicate_arc_names(tmp_path):
    out = tmp_path / "event.tar.gz"
    specs = [
        FileDiffSpec("dup", _full_diff_dict(b"a")),
        FileDiffSpec("dup", _full_diff_dict(b"b")),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        pack_event_tarball(
            event_id="x", event_type="baseline", component="personality",
            file_specs=specs, output_path=str(out),
        )


def test_pack_rejects_empty_arc_name(tmp_path):
    out = tmp_path / "event.tar.gz"
    specs = [FileDiffSpec("", _full_diff_dict(b"a"))]
    with pytest.raises(ValueError, match="non-empty"):
        pack_event_tarball(
            event_id="x", event_type="baseline", component="personality",
            file_specs=specs, output_path=str(out),
        )


def test_pack_rejects_diff_missing_required_fields(tmp_path):
    out = tmp_path / "event.tar.gz"
    bad = {"patch_bytes": b"a", "diff_mode": "full"}  # missing merkle_root etc
    with pytest.raises(ValueError, match="missing required field"):
        pack_event_tarball(
            event_id="x", event_type="baseline", component="personality",
            file_specs=[FileDiffSpec("f", bad)],
            output_path=str(out),
        )


def test_pack_rejects_non_bytes_patch(tmp_path):
    """Phase 5 — pack_event_tarball requires patch_path (streaming) OR
    patch_bytes (legacy). A diff_dict carrying a `patch_bytes` value of
    the wrong type with no `patch_path` falls through to the legacy
    branch and is rejected there."""
    out = tmp_path / "event.tar.gz"
    bad = {
        "diff_mode": "full", "patch_bytes": "not bytes",
        "merkle_root": "0" * 64, "size_bytes": 0, "encoder": "full_ship",
    }
    with pytest.raises(ValueError, match="patch_path|patch_bytes"):
        pack_event_tarball(
            event_id="x", event_type="baseline", component="personality",
            file_specs=[FileDiffSpec("f", bad)],
            output_path=str(out),
        )


def test_unpack_round_trip_basic(tmp_path):
    out = tmp_path / "event.tar.gz"
    content_a = b"hello world" * 10
    content_b = b"{\"k\": 42}"
    specs = [
        FileDiffSpec("a.bin", _full_diff_dict(content_a)),
        FileDiffSpec("b.json", _full_diff_dict(content_b)),
    ]
    pack_event_tarball(
        event_id="evt_x", event_type="baseline", component="personality",
        file_specs=specs, output_path=str(out),
    )
    with unpack_event_tarball(str(out)) as ev:
        assert isinstance(ev, UnpackedEvent)
        assert ev.event_id == "evt_x"
        assert ev.event_type == "baseline"
        assert ev.component == "personality"
        assert ev.schema_version == EVENT_TARBALL_SCHEMA_VERSION
        assert len(ev.files) == 2
        assert ev.get_patch_bytes("a.bin") == content_a
        assert ev.get_patch_bytes("b.json") == content_b


def test_unpack_diff_dict_for_round_trips_all_metadata(tmp_path):
    out = tmp_path / "event.tar.gz"
    inc = _xdelta_incremental_dict(
        patch=b"xdelta_patch_bytes",
        baseline_hash="b" * 64,
        post_hash="a" * 64,
        post_size=1000,
    )
    pack_event_tarball(
        event_id="e", event_type="incremental", component="personality",
        file_specs=[FileDiffSpec("inner.db", inc)],
        output_path=str(out),
    )
    with unpack_event_tarball(str(out)) as ev:
        d = ev.diff_dict_for("inner.db")
        assert d["diff_mode"] == "incremental"
        assert d["patch_bytes"] == b"xdelta_patch_bytes"
        assert d["merkle_root"] == "a" * 64
        assert d["size_bytes"] == 1000
        assert d["baseline_merkle_root"] == "b" * 64
        assert d["encoder"] == "xdelta3"


def test_unpack_tail_encoder_round_trips_extra_fields(tmp_path):
    out = tmp_path / "event.tar.gz"
    tail = _tail_dict(
        tail=b"new_block_bytes",
        prev_offset=2048,
        post_hash="c" * 64,
        post_size=2063,
        block_range=(100, 105),
    )
    pack_event_tarball(
        event_id="e", event_type="incremental", component="timechain",
        file_specs=[FileDiffSpec("chain_episodic.bin", tail)],
        output_path=str(out),
    )
    with unpack_event_tarball(str(out)) as ev:
        d = ev.diff_dict_for("chain_episodic.bin")
        assert d["diff_mode"] == "tail"
        assert d["prev_offset_bytes"] == 2048
        assert d["block_range"] == [100, 105]
        assert d["patch_bytes"] == b"new_block_bytes"


def test_unpack_from_bytes(tmp_path):
    out = tmp_path / "event.tar.gz"
    pack_event_tarball(
        event_id="e", event_type="baseline", component="personality",
        file_specs=[FileDiffSpec("f", _full_diff_dict(b"data"))],
        output_path=str(out),
    )
    raw = out.read_bytes()
    with unpack_event_tarball(raw) as ev:
        assert ev.event_id == "e"
        assert ev.get_patch_bytes("f") == b"data"


def test_unpack_rejects_missing_metadata_member(tmp_path):
    # Build a tarball that has files/ but no __event_metadata.json.
    # Compressed with zstd to match the prod write path so unpack auto-detect
    # hits the zstd branch.
    import io as _io
    import zstandard as _zst

    out = tmp_path / f"bad{EVENT_TARBALL_EXT}"
    raw_tar = _io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w:") as tar:
        info = tarfile.TarInfo(name=FILES_PREFIX + "f")
        info.size = 4
        tar.addfile(info, _io.BytesIO(b"data"))
    cctx = _zst.ZstdCompressor(level=3)
    out.write_bytes(cctx.compress(raw_tar.getvalue()))
    with pytest.raises(ValueError, match="missing"):
        unpack_event_tarball(str(out))


def test_unpack_rejects_unknown_schema_version(tmp_path):
    """Surgically rewrite a tarball with a bogus schema_version. Uses gzip
    for the tampered output (cheaper to construct in-test); unpack_event_tarball
    auto-detects either compression so the path is exercised regardless."""
    import io as _io
    import json
    import zstandard as _zst

    # Build a tampered tarball from scratch (tar in memory → zstd-compress).
    tampered_meta = {
        "schema_version": 99,
        "event_id": "e", "event_type": "baseline", "component": "personality",
        "ts_unix": 0.0, "files": [],
    }
    meta_bytes = json.dumps(tampered_meta).encode("utf-8")
    raw_tar = _io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w:") as tar:
        info = tarfile.TarInfo(name=EVENT_METADATA_MEMBER)
        info.size = len(meta_bytes)
        info.mode = 0o644
        tar.addfile(info, _io.BytesIO(meta_bytes))
    out = tmp_path / f"evt{EVENT_TARBALL_EXT}"
    cctx = _zst.ZstdCompressor(level=3)
    out.write_bytes(cctx.compress(raw_tar.getvalue()))
    with pytest.raises(ValueError, match="schema_version"):
        unpack_event_tarball(str(out))


def test_unpack_get_patch_bytes_unknown_arc(tmp_path):
    out = tmp_path / "evt.tar.gz"
    pack_event_tarball(
        event_id="e", event_type="baseline", component="personality",
        file_specs=[FileDiffSpec("f", _full_diff_dict(b"x"))],
        output_path=str(out),
    )
    with unpack_event_tarball(str(out)) as ev:
        with pytest.raises(KeyError, match="not in event"):
            ev.get_patch_bytes("nope")


def test_unpack_detects_tarball_member_tamper(tmp_path):
    """If the patch_bytes_sha256 in metadata mismatches the actual file
    member's bytes, get_patch_bytes raises ValueError (tarball corruption)."""
    import io as _io
    import json
    import zstandard as _zst

    out = tmp_path / f"evt{EVENT_TARBALL_EXT}"
    pack_event_tarball(
        event_id="e", event_type="baseline", component="personality",
        file_specs=[FileDiffSpec("f", _full_diff_dict(b"original"))],
        output_path=str(out),
    )
    # Pull the metadata via the prod API so we don't reimplement the
    # zstd decompress dance just to read it.
    with unpack_event_tarball(str(out)) as ev:
        meta = {
            "schema_version": ev.schema_version,
            "event_id": ev.event_id,
            "event_type": ev.event_type,
            "component": ev.component,
            "ts_unix": ev.ts_unix,
            "files": list(ev.files),
            "extra": dict(ev.extra),
        }

    # Build a fresh tampered tar (in-memory) and re-compress with zstd.
    tampered = b"TAMPERED"
    raw_tar = _io.BytesIO()
    with tarfile.open(fileobj=raw_tar, mode="w:") as tar:
        info = tarfile.TarInfo(name=FILES_PREFIX + "f")
        info.size = len(tampered)
        info.mode = 0o644
        tar.addfile(info, _io.BytesIO(tampered))
        meta_bytes = json.dumps(meta).encode("utf-8")
        info = tarfile.TarInfo(name=EVENT_METADATA_MEMBER)
        info.size = len(meta_bytes)
        info.mode = 0o644
        tar.addfile(info, _io.BytesIO(meta_bytes))
    out.unlink()
    cctx = _zst.ZstdCompressor(level=3)
    out.write_bytes(cctx.compress(raw_tar.getvalue()))
    with unpack_event_tarball(str(out)) as ev:
        with pytest.raises(ValueError, match="sha256 mismatch"):
            ev.get_patch_bytes("f")


def test_unpack_reads_legacy_gzip_path(tmp_path):
    """Phase 5 chunk 5F backwards-compat: the 17 pre-flip Arweave entries
    (gzip-9) plus the Day 1 genesis anchor remain decodable through the same
    unpack API after the zstd swap on the write side."""
    import io as _io
    import json

    # Hand-built gzip-compressed event tarball that mirrors what
    # pack_event_tarball produced pre-chunk-5F.
    metadata = {
        "schema_version": EVENT_TARBALL_SCHEMA_VERSION,
        "event_id": "legacy_evt",
        "event_type": "baseline",
        "component": "personality",
        "ts_unix": 1712341234.0,
        "files": [{
            "arc_name": "f",
            "member": FILES_PREFIX + "f",
            "diff_mode": "full",
            "merkle_root": hashlib.sha256(b"legacy_payload").hexdigest(),
            "size_bytes": len(b"legacy_payload"),
            "encoder": "full_ship",
            "patch_bytes_sha256": hashlib.sha256(b"legacy_payload").hexdigest(),
        }],
    }
    out = tmp_path / "legacy.tar.gz"
    with tarfile.open(out, "w:gz", compresslevel=9) as tar:
        body = b"legacy_payload"
        info = tarfile.TarInfo(name=FILES_PREFIX + "f")
        info.size = len(body)
        info.mode = 0o644
        tar.addfile(info, _io.BytesIO(body))
        meta_bytes = json.dumps(metadata).encode("utf-8")
        info = tarfile.TarInfo(name=EVENT_METADATA_MEMBER)
        info.size = len(meta_bytes)
        info.mode = 0o644
        tar.addfile(info, _io.BytesIO(meta_bytes))
    head = out.read_bytes()[:2]
    assert head == GZIP_MAGIC

    # Both path and bytes inputs auto-detect gzip and decode.
    with unpack_event_tarball(str(out)) as ev:
        assert ev.event_id == "legacy_evt"
        assert ev.get_patch_bytes("f") == b"legacy_payload"
    with unpack_event_tarball(out.read_bytes()) as ev:
        assert ev.get_patch_bytes("f") == b"legacy_payload"


def test_unpack_rejects_unknown_magic_bytes(tmp_path):
    """A non-zstd, non-gzip payload is rejected upfront."""
    out = tmp_path / "junk.tar.zst"
    out.write_bytes(b"\x00\x00\x00\x00not a real tarball")
    with pytest.raises(ValueError, match="unknown tarball magic"):
        unpack_event_tarball(str(out))
    with pytest.raises(ValueError, match="unknown tarball magic"):
        unpack_event_tarball(b"\x00\x00\x00\x00not a real tarball")


def test_pack_extra_metadata_round_trip(tmp_path):
    out = tmp_path / "evt.tar.gz"
    extra = {"block_range": [100, 200], "free_form": "anything"}
    pack_event_tarball(
        event_id="e", event_type="incremental", component="timechain",
        file_specs=[FileDiffSpec("chain.bin", _full_diff_dict(b"x"))],
        output_path=str(out), extra_metadata=extra,
    )
    with unpack_event_tarball(str(out)) as ev:
        assert ev.extra == extra


def test_atomic_write_no_partial_on_error(tmp_path, monkeypatch):
    """If pack raises mid-way, the output_path is not left as a partial file."""
    out = tmp_path / "evt.tar.gz"
    # Inject a bad spec that fails AFTER tarball open but BEFORE finalization
    bad_specs = [
        FileDiffSpec("ok", _full_diff_dict(b"x")),
        FileDiffSpec("bad", {  # missing encoder → ValueError in _file_meta_from_diff
            "diff_mode": "full", "patch_bytes": b"y",
            "merkle_root": "0" * 64, "size_bytes": 1,
        }),
    ]
    with pytest.raises(ValueError, match="missing required field"):
        pack_event_tarball(
            event_id="e", event_type="baseline", component="personality",
            file_specs=bad_specs, output_path=str(out),
        )
    assert not out.exists()
    assert not (tmp_path / "evt.tar.gz.tmp").exists()
