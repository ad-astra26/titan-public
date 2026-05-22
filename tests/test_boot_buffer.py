"""SPEC §8.0.bis boot-window buffer — Python↔Rust constant-parity guard.

The Python `BusSocketServer` (which owned the in-process boot-buffer) was
DELETED in D8-1 (commit 69035803) — the broker + its boot-buffer are now
Rust-owned (titan-kernel-rs / `titan-bus`). D8-1 retired 4 dependent tests but
MISSED this file; the BusSocketServer-behavior tests (publish/drain/overflow/
ttl/gc) were retired here on 2026-05-22 (Phase C close, D-SPEC-116) — that
behavior is exercised by the canonical Rust suite
`titan-rust/crates/titan-bus/tests/integration.rs::boot_buffer_*`.

What REMAINS live + worth guarding from Python: the BOOT_BUFFER_* constants in
`bus_socket.py` are a cross-language MIRROR of the Rust `boot_buffer::*` consts
(SPEC §8.10 byte-identical guarantee), so drift between them must surface.
"""
from __future__ import annotations

from titan_hcl.core.bus_socket import (
    BOOT_BUFFER_MAX_FRAMES_PER_DST,
    BOOT_BUFFER_TTL_S,
    BOOT_BUFFERED_TYPES,
)


def test_boot_buffered_types_constant_matches_spec():
    """Cross-language parity: BOOT_BUFFERED_TYPES + caps must match Rust
    `boot_buffer::{BOOT_BUFFERED_TYPES,BOOT_BUFFER_MAX_FRAMES_PER_DST,
    BOOT_BUFFER_TTL_S}` (SPEC §8.0.bis / §8.10)."""
    expected = {
        "MODULE_READY",
        "MODULE_HEARTBEAT",
        "MODULE_SHUTDOWN",
        "MODULE_CRASHED",
        "SUPERVISION_CHILD_DOWN",
        "SUPERVISION_CHILD_RESTARTED",
        "AGENCY_READY",
        "NS_READY",
        "MEMORY_READY",
        "MODULE_RELOAD_ACK",
    }
    assert BOOT_BUFFERED_TYPES == expected
    assert BOOT_BUFFER_MAX_FRAMES_PER_DST == 32
    assert BOOT_BUFFER_TTL_S == 60.0
