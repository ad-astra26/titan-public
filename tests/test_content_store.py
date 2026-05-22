"""Tests for the content-addressed store (Synthesis Engine Phase 0 / 0B).

Run isolated per project convention:
    python -m pytest tests/test_content_store.py -v -p no:anchorpy --tb=short
"""
import hashlib

import pytest

from titan_hcl.synthesis.content_store import (
    BlobNotFound,
    ContentStore,
    CorruptBlob,
)


@pytest.fixture
def store(tmp_path):
    return ContentStore(root=tmp_path)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_put_get_roundtrip(store):
    data = b"the linux terminal is a text interface"
    h = store.put(data)
    assert h == _sha(data)
    assert store.get(h) == data


def test_put_returns_content_address(store):
    data = b"hello titan"
    assert store.put(data) == _sha(data)


def test_put_is_idempotent_single_blob(store):
    data = b"metaplex nft minting v6"
    h1 = store.put(data)
    h2 = store.put(data)
    assert h1 == h2
    # Exactly one blob file on disk (ignoring any stray tmp).
    blobs = [p for p in store.root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    assert len(blobs) == 1


def test_dedup_n_identical_one_file(store):
    data = b"identical payload" * 100
    for _ in range(25):
        store.put(data)
    blobs = [p for p in store.root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    assert len(blobs) == 1
    assert store.stat()["blob_count"] == 1


def test_distinct_content_distinct_blobs(store):
    a = store.put(b"alpha")
    b = store.put(b"beta")
    assert a != b
    assert store.stat()["blob_count"] == 2
    assert store.get(a) == b"alpha"
    assert store.get(b) == b"beta"


def test_missing_hash_raises(store):
    missing = _sha(b"never stored")
    assert not store.exists(missing)
    with pytest.raises(BlobNotFound):
        store.get(missing)


def test_invalid_hash_format_rejected(store):
    assert not store.exists("not-a-hash")
    with pytest.raises(BlobNotFound):
        store.get("xyz")
    with pytest.raises(BlobNotFound):
        store.get("ab" * 40)  # wrong length


def test_exists(store):
    data = b"exists check"
    assert not store.exists(_sha(data))
    h = store.put(data)
    assert store.exists(h)


def test_corruption_detected_on_read(store):
    data = b"will be tampered"
    h = store.put(data)
    # Tamper the stored blob in place.
    path = store._path_for(h)
    path.write_bytes(b"tampered bytes of different length")
    with pytest.raises(CorruptBlob):
        store.get(h)


def test_sharded_layout(store):
    data = b"shard me"
    h = store.put(data)
    expected = store.root / h[:2] / h[2:4] / h
    assert expected.exists()


def test_put_rejects_non_bytes(store):
    with pytest.raises(TypeError):
        store.put("a string, not bytes")  # type: ignore[arg-type]


def test_bytearray_accepted(store):
    data = bytearray(b"mutable buffer")
    h = store.put(data)
    assert store.get(h) == bytes(data)


def test_empty_blob(store):
    h = store.put(b"")
    assert h == _sha(b"")
    assert store.get(h) == b""


def test_stat_total_bytes(store):
    store.put(b"a" * 10)
    store.put(b"b" * 20)
    s = store.stat()
    assert s["blob_count"] == 2
    assert s["total_bytes"] == 30


def test_root_nesting_idempotent(tmp_path):
    # Passing a dir already named content_blobs must not double-nest.
    cb = tmp_path / "content_blobs"
    s = ContentStore(root=cb)
    assert s.root == cb
    # Passing a plain data dir nests content_blobs/ under it.
    s2 = ContentStore(root=tmp_path / "data")
    assert s2.root == tmp_path / "data" / "content_blobs"


def test_persistence_across_instances(tmp_path):
    h = ContentStore(root=tmp_path).put(b"durable")
    assert ContentStore(root=tmp_path).get(h) == b"durable"
