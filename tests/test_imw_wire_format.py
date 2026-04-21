"""Tests for IMW wire_format — frame encode/decode + schema roundtrip."""
import pytest

from titan_plugin.persistence.wire_format import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    WireFormatError,
    WriteRequest,
    WriteResponse,
    decode_length,
    encode_frame,
)


def test_request_roundtrip():
    req = WriteRequest.new_write("spirit:123", "INSERT INTO x VALUES (?)", (1,))
    payload = req.to_msgpack()
    back = WriteRequest.from_msgpack(payload)
    assert back.req_id == req.req_id
    assert back.sql == req.sql
    assert back.params == [1]
    assert back.sync is True
    assert back.target_db == "primary"
    assert back.op == "write"


def test_request_shadow_target():
    req = WriteRequest.new_write("main:1", "INSERT INTO y VALUES (?)", (2,),
                                    target_db="shadow")
    back = WriteRequest.from_msgpack(req.to_msgpack())
    assert back.target_db == "shadow"


def test_writemany_roundtrip():
    rows = [(1, "a"), (2, "b"), (3, "c")]
    req = WriteRequest.new_writemany("lang:42", "INSERT INTO t VALUES (?, ?)", rows)
    back = WriteRequest.from_msgpack(req.to_msgpack())
    assert back.op == "writemany"
    assert len(back.params) == 3
    assert back.params[0] == [1, "a"]


def test_ping():
    req = WriteRequest.new_ping("caller:1")
    back = WriteRequest.from_msgpack(req.to_msgpack())
    assert back.op == "ping"
    assert back.sync is True


def test_response_roundtrip():
    resp = WriteResponse(req_id="abc", ok=True, rowcount=1, last_row_id=42,
                           committed_at=100.0)
    back = WriteResponse.from_msgpack(resp.to_msgpack())
    assert back.ok is True
    assert back.rowcount == 1
    assert back.last_row_id == 42
    assert back.committed_at == 100.0


def test_response_error():
    resp = WriteResponse(req_id="abc", ok=False,
                           error={"type": "sqlite3.Error", "msg": "locked"})
    back = WriteResponse.from_msgpack(resp.to_msgpack())
    assert back.ok is False
    assert back.error["type"] == "sqlite3.Error"


def test_frame_length_header():
    payload = b"\x01" * 100
    frame = encode_frame(payload)
    assert len(frame) == 104
    assert decode_length(frame[:4]) == 100
    assert frame[4:] == payload


def test_frame_too_large():
    with pytest.raises(WireFormatError):
        encode_frame(b"\x00" * (MAX_FRAME_BYTES + 1))


def test_decode_bad_length():
    with pytest.raises(WireFormatError):
        decode_length(b"\xff\xff\xff\xff")  # 4 GB — exceeds MAX


def test_version_mismatch():
    import msgpack
    bad = msgpack.packb({
        "v": 999, "req_id": "x", "caller": "", "op": "write",
        "sql": "SELECT 1", "params": [], "sync": True, "ts": 0,
    }, use_bin_type=True)
    with pytest.raises(WireFormatError):
        WriteRequest.from_msgpack(bad)


def test_protocol_version_constant():
    assert PROTOCOL_VERSION == 1
