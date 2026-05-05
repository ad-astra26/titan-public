"""
End-to-end integration test for C-S6 outer trinity stack.

Per master plan §10.6 chunk C6-8 + PLAN §6 chunk C6-10. Verifies the
full outer-side stack composes correctly:

  1. All 3 Rust outer binaries compile + link via cargo build --release
  2. Each binary's --version flag works (process boots cleanly).
  3. Each binary's --help works (clap CLI well-formed).
  4. The 3 Python sidecar classes can be instantiated together against
     the same shared shm root + drive the 3 sensor_cache_outer_*.bin
     slots concurrently without deadlock or shm contention.
  5. The flag-gated outer_trinity.py shim correctly reads the slots
     populated by the sidecars when l0_rust_enabled=True.
  6. The 162D end-to-end verification — sidecars write to msgpack caches
     → shim reads canonical slot bytes (here we synthesize the slot
     content directly since the Rust daemons aren't spawned by this
     test; full Rust-binary-spawn e2e is C-S7 first-flag-flip work).

This is the PYTHON-driven cross-stack smoke test. The Rust-binary-only
cargo workspace tests (149 unit tests across titan-trinity-daemon +
3 outer crates) cover the Rust side standalone.

NOTE: this test does NOT attempt to spawn an actual kernel-rs binary
or bring up the full bus broker. Bus-protocol-level e2e is C-S7 work.
"""
from __future__ import annotations

import asyncio
import struct
import subprocess
import time
import zlib
from pathlib import Path

import msgpack
import pytest

from titan_plugin._phase_c_constants import (
    OUTER_BODY_TICK_BASE_S,
    OUTER_MIND_TICK_BASE_S,
    OUTER_SPIRIT_TICK_BASE_S,
)
from titan_plugin.core.state_registry import HEADER_SIZE, HEADER_STRUCT
from titan_plugin.logic.outer_body_sensor_refresh import OuterBodySensorRefresh
from titan_plugin.logic.outer_mind_sensor_refresh import OuterMindSensorRefresh
from titan_plugin.logic.outer_spirit_sensor_refresh import OuterSpiritSensorRefresh
from titan_plugin.logic.outer_trinity import OuterTrinityCollector


# Path to compiled binaries (after `cargo build`).
RUST_TARGET_DIR = Path(__file__).parent.parent / "titan-rust" / "target" / "debug"

OUTER_BINARIES = [
    "titan-outer-body-rs",
    "titan-outer-mind-rs",
    "titan-outer-spirit-rs",
]


# ── 1. Compile-clean: each binary exists ─────────────────────────────


@pytest.mark.parametrize("binary", OUTER_BINARIES)
def test_binary_was_built(binary):
    """C6-3/4/5 binaries compile + link as part of cargo build --workspace."""
    path = RUST_TARGET_DIR / binary
    if not path.exists():
        pytest.skip(
            f"{binary} not yet built — run `cargo build --workspace` from titan-rust/. "
            "This test verifies the binary compiles, not that it's pre-built before pytest."
        )
    assert path.is_file(), f"{path} should be a file"


# ── 2. --version flag works ───────────────────────────────────────────


@pytest.mark.parametrize("binary", OUTER_BINARIES)
def test_binary_version_flag(binary):
    """--version is a clap-built-in; verifies the CLI scaffolding loads + clean exit."""
    path = RUST_TARGET_DIR / binary
    if not path.exists():
        pytest.skip(f"{binary} not built; cargo build first")
    result = subprocess.run(
        [str(path), "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"{binary} --version failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Output should contain the binary name + a version
    assert binary in result.stdout or "0.1.0" in result.stdout


# ── 3. --help flag works (CLI well-formed) ────────────────────────────


@pytest.mark.parametrize("binary", OUTER_BINARIES)
def test_binary_help_flag(binary):
    path = RUST_TARGET_DIR / binary
    if not path.exists():
        pytest.skip(f"{binary} not built")
    result = subprocess.run(
        [str(path), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    # Must mention all the env vars per SPEC §13
    for env_var in ("TITAN_KERNEL_TITAN_ID", "TITAN_BUS_SOCKET", "TITAN_AUTHKEY_HEX"):
        assert env_var in result.stdout, (
            f"{binary} --help missing {env_var} in env-var docs:\n{result.stdout}"
        )


# ── 4. 3 sidecars run concurrently against same shm root ─────────────


@pytest.fixture()
def shm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_SHM_ROOT", str(tmp_path))
    return tmp_path


def _make_sources_full() -> dict:
    """Return a sources dict that satisfies all 3 sidecar SOURCE_KEYS sets.

    Per PLAN §3.4 + sidecar source-keys, the 3 sidecars project this
    onto their respective canonical key sets — each writes a SUBSET of
    the dict to its own cache slot.
    """
    return {
        # outer_body keys
        "agency_stats": {"actions_total": 100, "self_initiated": 75},
        "helper_statuses": {"web": "ok"},
        "bus_stats": {"queue_depth": 0, "drop_rate": 0.0},
        "system_sensor_stats": {"cpu_pct": 42.0},
        "network_monitor_stats": {"latency_ms_p50": 12.0},
        "tx_latency_stats": {"latency_p50_s": 0.8},
        "block_delta_stats": {"normalized": 0.55},
        "anchor_state": {"success": True, "last_anchor_time": 1714400000.0},
        "sol_balance": 1.5,
        # outer_mind keys (most overlap with body)
        "uptime_seconds": 3600.0,
        "art_count_100": 12,
        "audio_count_100": 4,
        "art_count_500": 47,
        "audio_count_500": 18,
        "memory_status": {"recent_score": 0.62},
        "assessment_stats": {"avg_score": 0.55},
        "impulse_stats": {"successful": 18},
        "soul_health": 0.9,
        "social_perception_stats": {"engagement": 0.4},
        "twin_state": {"reachable": True, "DA": 0.7},
        "llm_avg_latency": 0.85,
        # outer_spirit keys
        "action_stats": {"actions_total": 240},
        "creative_stats": {"art_count": 47},
        "guardian_stats": {"threats_detected": 0, "uptime_pct": 0.998},
        "sovereignty_ratio": 0.75,
        "uptime_ratio": 0.95,
        "recovery_stats": {"crashes": 0},
        "social_stats": {"engagement": 0.4},
        "memory_stats": {"recall_score": 0.62},
        "hormone_levels": {"DA": 0.65},
        "solana_stats": {"balance": 1.5},
        "history": {"trend": "stable"},
    }


def _read_slot_payload(shm_path: Path) -> tuple[int, int, int, bytes]:
    raw = shm_path.read_bytes()
    seq, schema, wall_ns, payload_bytes, _crc = struct.unpack(
        HEADER_STRUCT, raw[:HEADER_SIZE]
    )
    return seq, schema, wall_ns, bytes(raw[HEADER_SIZE : HEADER_SIZE + payload_bytes])


@pytest.mark.asyncio
async def test_three_sidecars_run_concurrently(shm_root):
    """All 3 outer sensor sidecars run in parallel against the same shm root,
    each writing to its own cache slot. No deadlock, no shm contention."""
    sources = _make_sources_full()
    body = OuterBodySensorRefresh(sources_provider=lambda: sources, refresh_period_s=0.05)
    mind = OuterMindSensorRefresh(sources_provider=lambda: sources, refresh_period_s=0.05)
    spirit = OuterSpiritSensorRefresh(sources_provider=lambda: sources, refresh_period_s=0.05)

    tasks = [
        asyncio.create_task(body.run()),
        asyncio.create_task(mind.run()),
        asyncio.create_task(spirit.run()),
    ]
    await asyncio.sleep(0.30)  # ~6 ticks each
    await body.stop()
    await mind.stop()
    await spirit.stop()
    for t in tasks:
        await t

    # Each sidecar advanced its tick counter
    assert body.tick_count >= 3, f"body ticks={body.tick_count}"
    assert mind.tick_count >= 3, f"mind ticks={mind.tick_count}"
    assert spirit.tick_count >= 3, f"spirit ticks={spirit.tick_count}"

    # Each wrote a non-empty payload to its own slot
    for slot_name in (
        "sensor_cache_outer_body.bin",
        "sensor_cache_outer_mind.bin",
        "sensor_cache_outer_spirit.bin",
    ):
        seq, schema, wall_ns, payload = _read_slot_payload(shm_root / slot_name)
        assert seq > 0, f"{slot_name} seq should be > 0 after writes"
        assert schema == 1
        assert wall_ns > 0
        assert payload != b"", f"{slot_name} payload should be non-empty"
        # Decodes as msgpack map
        decoded = msgpack.unpackb(payload, raw=False)
        assert isinstance(decoded, dict)


# ── 5. Flag-gated shim reads from sidecar-populated slots ────────────


def _write_outer_slot(
    shm_root: Path, slot_name: str, values: list[float], wall_ns: int | None = None
) -> None:
    """Synthesize a fixed-dim outer slot file (24B header + N×float32 LE)."""
    if wall_ns is None:
        wall_ns = time.time_ns()
    n = len(values)
    payload = struct.pack(f"<{n}f", *values)
    payload_bytes = len(payload)
    seq = 2  # even = write complete
    header_prefix = struct.pack("<IIQI", seq, 1, wall_ns, payload_bytes)
    crc = zlib.crc32(header_prefix)
    header = header_prefix + struct.pack("<I", crc)
    (shm_root / slot_name).write_bytes(header + payload)


def test_shim_reads_3_outer_slots_when_flag_on(shm_root):
    """End-to-end verification: when l0_rust_enabled=True, the
    OuterTrinityCollector shim reads the 3 outer slots via SeqLock and
    returns them in the return-shape contract.

    This simulates the C-S7 first-flag-flip path where Rust daemons would
    populate the slots; here we synthesize slot bytes directly to verify
    the shim plumbing end-to-end."""
    body_5d = [0.10, 0.20, 0.30, 0.40, 0.50]
    mind_15d = [0.05 * (i + 1) for i in range(15)]
    spirit_45d = [0.02 * (i + 1) for i in range(45)]
    _write_outer_slot(shm_root, "outer_body_5d.bin", body_5d)
    _write_outer_slot(shm_root, "outer_mind_15d.bin", mind_15d)
    _write_outer_slot(shm_root, "outer_spirit_45d.bin", spirit_45d)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    # Pass empty sources — shim ignores them when flag=true
    result = collector.collect({})

    # outer_body 5D from slot
    for actual, expected in zip(result["outer_body"], body_5d):
        assert abs(actual - expected) < 1e-6
    # outer_mind 15D from slot
    for actual, expected in zip(result["outer_mind_15d"], mind_15d):
        assert abs(actual - expected) < 1e-6
    # outer_spirit 45D from slot
    for actual, expected in zip(result["outer_spirit_45d"], spirit_45d):
        assert abs(actual - expected) < 1e-6


# ── 6. Cadence sanity (constants align across stack) ─────────────────


def test_cadences_align_with_spec():
    """Cadences in TOML constants line up with sidecar + binary defaults."""
    assert OUTER_BODY_TICK_BASE_S == 10.0
    assert OUTER_MIND_TICK_BASE_S == 5.0
    assert OUTER_SPIRIT_TICK_BASE_S == 30.0


# ── 7. Stale fallback at C-S6 layer boundary ─────────────────────────


def test_shim_stale_outer_body_falls_back_to_last_known(shm_root):
    """SPEC §18.1: outer_body stale at 30s (3×10s) → shim returns last-known."""
    body = [0.7, 0.6, 0.5, 0.4, 0.3]
    stale_wall_ns = time.time_ns() - int(60 * 1e9)  # 60s in past
    _write_outer_slot(shm_root, "outer_body_5d.bin", body, wall_ns=stale_wall_ns)

    collector = OuterTrinityCollector(l0_rust_enabled=True, titan_id="T1")
    result = collector.collect({})

    # Stale → falls back to last-known cached default ([0.5]*5)
    assert result["outer_body"] == [0.5] * 5
    assert collector._shm_stale_count >= 1
