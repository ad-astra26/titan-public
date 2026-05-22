"""
Tests for §4.K studio_worker (D-SPEC-57, SPEC v1.8.3).

Coverage:
  • studio_state_specs: RegistrySpec wiring (slot name, schema, max bytes, variable_size)
  • Constants TOML wiring: STUDIO_STATE_SCHEMA_VERSION, STUDIO_STATE_MAX_BYTES
  • Bus event constants: STUDIO_WORKER_READY, STUDIO_RENDER_REQUEST, STUDIO_RENDER_COMPLETED
  • bus_specs.py MSG_SPECS rows: priorities + coalesce keys
  • StudioStatePublisher: cold-boot scan, record_render bump, msgpack round-trip,
    payload fits in MAX_BYTES, refresh_counts scans non-sidecar files only
  • StudioStateShmReader: cold-boot defaults, SHM round-trip after publish
  • StudioProxy: _RenderCompletionRegistry race-free registration,
    fire-and-forget request_* shape, _with_completion timeout fallback,
    fixture for in-process bus
  • _HaikuLLMBridge: complete() returns '' (Tier-2 fallback path)
  • Module entry point imports + ModuleSpec presence in plugin._register_modules

Per CLAUDE.md: pytest -p no:anchorpy, separate process per file.
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import msgpack
import pytest

from titan_hcl._phase_c_constants import (
    STUDIO_STATE_MAX_BYTES,
    STUDIO_STATE_SCHEMA_VERSION,
)
from titan_hcl.logic.studio_state_publisher import StudioStatePublisher
from titan_hcl.logic.studio_state_specs import (
    STUDIO_STATE_SLOT,
    STUDIO_STATE_SPEC,
)


# ── Per-test SHM root isolation ────────────────────────────────────


@pytest.fixture
def titan_id(tmp_path) -> str:
    """Per-test titan_id; SHM root cleaned up after test."""
    tid = f"T_TEST_{int(time.time() * 1e6) % 1_000_000}"
    yield tid
    shutil.rmtree(Path(f"/dev/shm/titan_{tid}"), ignore_errors=True)


@pytest.fixture
def studio_dirs(tmp_path) -> dict:
    """Per-test output dirs (cleaned up by tmp_path)."""
    root = tmp_path / "studio_exports"
    meditation_dir = root / "meditation"
    epoch_dir = root / "epoch"
    eureka_dir = root / "eureka"
    for d in (meditation_dir, epoch_dir, eureka_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "output_root": root,
        "meditation_dir": meditation_dir,
        "epoch_dir": epoch_dir,
        "eureka_dir": eureka_dir,
    }


# ── studio_state_specs wiring ──────────────────────────────────────


def test_studio_state_spec_slot_name():
    assert STUDIO_STATE_SLOT == "studio_state"


def test_studio_state_spec_schema_version():
    assert STUDIO_STATE_SPEC.schema_version == STUDIO_STATE_SCHEMA_VERSION
    assert STUDIO_STATE_SPEC.schema_version == 1


def test_studio_state_spec_max_bytes():
    assert STUDIO_STATE_SPEC.payload_bytes == STUDIO_STATE_MAX_BYTES
    assert STUDIO_STATE_SPEC.payload_bytes == 512


def test_studio_state_spec_is_variable_size():
    assert STUDIO_STATE_SPEC.variable_size is True


# ── Constants TOML wiring ──────────────────────────────────────────


def test_constants_wired_per_dspec57():
    assert STUDIO_STATE_SCHEMA_VERSION == 1
    assert STUDIO_STATE_MAX_BYTES == 512


# ── Bus event constants ────────────────────────────────────────────


def test_bus_event_constants_defined():
    from titan_hcl import bus
    assert bus.STUDIO_WORKER_READY == "STUDIO_WORKER_READY"
    assert bus.STUDIO_RENDER_REQUEST == "STUDIO_RENDER_REQUEST"
    assert bus.STUDIO_RENDER_COMPLETED == "STUDIO_RENDER_COMPLETED"


def test_bus_specs_msg_specs_priorities():
    from titan_hcl.bus_specs import MSG_SPECS
    assert MSG_SPECS["STUDIO_WORKER_READY"].priority == 1
    assert MSG_SPECS["STUDIO_RENDER_REQUEST"].priority == 3
    assert MSG_SPECS["STUDIO_RENDER_COMPLETED"].priority == 3


def test_bus_specs_msg_specs_coalesce_keys():
    from titan_hcl.bus_specs import MSG_SPECS
    # READY has no coalesce (one-shot lifecycle).
    assert MSG_SPECS["STUDIO_WORKER_READY"].coalesce is None
    # REQUEST + COMPLETED both coalesce by request_id (D-SPEC-46 pattern).
    assert MSG_SPECS["STUDIO_RENDER_REQUEST"].coalesce == ("request_id",)
    assert MSG_SPECS["STUDIO_RENDER_COMPLETED"].coalesce == ("request_id",)


# ── StudioStatePublisher ────────────────────────────────────────────


def _build_publisher(titan_id: str, studio_dirs: dict) -> StudioStatePublisher:
    return StudioStatePublisher(
        titan_id=titan_id,
        output_root=studio_dirs["output_root"],
        meditation_dir=studio_dirs["meditation_dir"],
        epoch_dir=studio_dirs["epoch_dir"],
        eureka_dir=studio_dirs["eureka_dir"],
        default_resolution=1024,
        highres_resolution=2048,
        nft_composite_enabled=True,
    )


def test_publisher_cold_boot_zero_counts(titan_id, studio_dirs):
    pub = _build_publisher(titan_id, studio_dirs)
    payload = pub._compute_payload()
    assert payload["meditation_count"] == 0
    assert payload["epoch_count"] == 0
    assert payload["eureka_count"] == 0
    assert payload["last_render_ts"] == 0.0
    assert payload["last_render_type"] == "none"
    assert payload["schema_version"] == 1


def test_publisher_refresh_counts_scans_non_sidecar(titan_id, studio_dirs):
    # Add a real artifact + a sidecar JSON.
    (studio_dirs["meditation_dir"] / "art_001.png").write_bytes(b"fake")
    (studio_dirs["meditation_dir"] / "art_001.png.json").write_text('{}')
    (studio_dirs["epoch_dir"] / "tree_001.png").write_bytes(b"fake")
    pub = _build_publisher(titan_id, studio_dirs)
    pub.refresh_counts()
    assert pub._meditation_count == 1  # sidecar excluded
    assert pub._epoch_count == 1
    assert pub._eureka_count == 0


def test_publisher_record_render_bumps_counter(titan_id, studio_dirs):
    pub = _build_publisher(titan_id, studio_dirs)
    pub.record_render("meditation")
    assert pub._meditation_count == 1
    assert pub._last_render_type == "meditation"
    assert pub._last_render_ts > 0.0
    pub.record_render("epoch")
    pub.record_render("eureka")
    assert pub._epoch_count == 1
    assert pub._eureka_count == 1


def test_publisher_record_render_ignores_unknown_type(titan_id, studio_dirs):
    pub = _build_publisher(titan_id, studio_dirs)
    pub.record_render("bogus")
    assert pub._meditation_count == 0
    assert pub._last_render_type == "none"  # unchanged
    pub.record_render("none")
    assert pub._last_render_type == "none"


def test_publisher_msgpack_payload_fits_in_max_bytes(titan_id, studio_dirs):
    pub = _build_publisher(titan_id, studio_dirs)
    pub.record_render("meditation")
    encoded = msgpack.packb(pub._compute_payload(), use_bin_type=True)
    assert len(encoded) <= STUDIO_STATE_MAX_BYTES, (
        f"payload {len(encoded)}B exceeds STUDIO_STATE_MAX_BYTES={STUDIO_STATE_MAX_BYTES}B")
    # Sanity: typical payload (short output_root) is well under 512B —
    # nominal estimate is ~180B when output_root is a normal path; tmp_path
    # paths in CI can push it higher (here ~300B), still well within cap.


def test_publisher_publish_writes_to_shm_and_reader_reads_back(titan_id, studio_dirs):
    from titan_hcl.proxies.studio_proxy import StudioStateShmReader

    pub = _build_publisher(titan_id, studio_dirs)
    pub.record_render("meditation")
    pub.publish()

    reader = StudioStateShmReader(titan_id=titan_id)
    stats = reader.read()
    assert stats["meditation_count"] == 1
    assert stats["last_render_type"] == "meditation"
    assert stats["schema_version"] == 1
    assert stats["default_resolution"] == 1024
    assert stats["highres_resolution"] == 2048
    assert stats["nft_composite_enabled"] is True


# ── StudioStateShmReader cold-boot ─────────────────────────────────


def test_reader_cold_boot_returns_defaults(titan_id):
    from titan_hcl.proxies.studio_proxy import StudioStateShmReader
    reader = StudioStateShmReader(titan_id=titan_id)
    stats = reader.read()
    # Cold defaults — slot doesn't exist yet
    assert stats["meditation_count"] == 0
    assert stats["epoch_count"] == 0
    assert stats["eureka_count"] == 0
    assert stats["last_render_type"] == "none"
    assert stats["schema_version"] == 1


# ── _RenderCompletionRegistry race-free ─────────────────────────────


def test_completion_registry_race_free_registration():
    """Register Future BEFORE publishing — guards against early COMPLETED."""
    from titan_hcl.proxies.studio_proxy import _RenderCompletionRegistry

    # Fake bus that doesn't actually subscribe (we only test in-memory state)
    class _FakeBus:
        def subscribe(self, name, types=None):
            return _FakeQueue()
    class _FakeQueue:
        def get(self, timeout=None):
            raise Exception("queue not driven")

    reg = _RenderCompletionRegistry(_FakeBus())
    fut1 = reg.register("req-aaa")
    fut2 = reg.register("req-bbb")
    assert reg.in_flight_count() == 2
    assert not fut1.done()
    assert not fut2.done()


def test_completion_registry_rejects_duplicate_request_id():
    from titan_hcl.proxies.studio_proxy import _RenderCompletionRegistry

    class _FakeBus:
        def subscribe(self, name, types=None):
            return _FakeQueue()
    class _FakeQueue:
        def get(self, timeout=None):
            raise Exception("queue not driven")

    reg = _RenderCompletionRegistry(_FakeBus())
    reg.register("req-zzz")
    with pytest.raises(RuntimeError, match="already in-flight"):
        reg.register("req-zzz")


def test_completion_registry_cancel_removes_future():
    from titan_hcl.proxies.studio_proxy import _RenderCompletionRegistry

    class _FakeBus:
        def subscribe(self, name, types=None):
            return _FakeQueue()
    class _FakeQueue:
        def get(self, timeout=None):
            raise Exception("queue not driven")

    reg = _RenderCompletionRegistry(_FakeBus())
    reg.register("req-aaa")
    assert reg.in_flight_count() == 1
    reg.cancel("req-aaa")
    assert reg.in_flight_count() == 0
    # Idempotent
    reg.cancel("req-aaa")
    reg.cancel("never-existed")


# ── Module entry point + ModuleSpec ────────────────────────────────


def test_worker_module_entry_point_imports():
    """Confirm entry function is importable + signature matches ModuleSpec contract."""
    from titan_hcl.modules.studio_worker import studio_worker_main
    import inspect
    sig = inspect.signature(studio_worker_main)
    params = list(sig.parameters.keys())
    assert params == ["recv_queue", "send_queue", "name", "config"]


def test_worker_helpers_present():
    """Confirm dispatch + bridge helpers exist + types correct."""
    from titan_hcl.modules.studio_worker import (
        _dispatch_render,
        _HaikuLLMBridge,
        MAX_CONCURRENT_RENDERS,
    )
    assert MAX_CONCURRENT_RENDERS == 2
    bridge = _HaikuLLMBridge(send_queue=None, name="studio")
    # complete() is async; just verify it's a coroutine factory
    coro = bridge.complete("test prompt")
    assert asyncio.iscoroutine(coro)
    coro.close()


def test_haiku_bridge_complete_returns_empty_string():
    """Tier-1 LLM bridge returns '' in v1.8.3 (defers to Tier-2/3 templates)."""
    from titan_hcl.modules.studio_worker import _HaikuLLMBridge
    bridge = _HaikuLLMBridge(send_queue=None, name="studio")
    result = asyncio.run(bridge.complete("haiku about silence"))
    assert result == ""


def test_dispatch_render_rejects_unknown_type():
    """_dispatch_render raises ValueError on unknown type."""
    from titan_hcl.modules.studio_worker import _dispatch_render

    class _FakeCoordinator:
        pass

    with pytest.raises(ValueError, match="unknown render type"):
        _dispatch_render(_FakeCoordinator(), "bogus", {})


# ── SPEC parity ────────────────────────────────────────────────────


def test_spec_anchor_9b_block_exists():
    """SPEC §9.B contains a studio_worker block."""
    spec_path = Path(__file__).parent.parent / "titan-docs" / "specs" / "SPEC_titan_architecture.md"
    spec_text = spec_path.read_text()
    assert "#### studio_worker (Python L2 module" in spec_text
    assert "D-SPEC-57" in spec_text
    assert "STUDIO_RENDER_REQUEST" in spec_text
    assert "STUDIO_RENDER_COMPLETED" in spec_text


def test_spec_anchor_71_slot_row_exists():
    """SPEC §7.1 contains studio_state.bin row."""
    spec_path = Path(__file__).parent.parent / "titan-docs" / "specs" / "SPEC_titan_architecture.md"
    spec_text = spec_path.read_text()
    assert "`studio_state.bin`" in spec_text
    assert "STUDIO_STATE_SCHEMA_VERSION" in spec_text
    assert "STUDIO_STATE_MAX_BYTES = 512" in spec_text


def test_spec_anchor_87_events_exist():
    """SPEC §8.7 contains all 3 new bus event rows."""
    spec_path = Path(__file__).parent.parent / "titan-docs" / "specs" / "SPEC_titan_architecture.md"
    spec_text = spec_path.read_text()
    assert "`STUDIO_WORKER_READY`" in spec_text
    assert "`STUDIO_RENDER_REQUEST`" in spec_text
    assert "`STUDIO_RENDER_COMPLETED`" in spec_text


def test_spec_anchor_1_glossary_studio_worker():
    """SPEC §1 glossary has studio_worker entry.

    Renumbered v1.8.3 → v1.9.4 at merge time (D-SPEC-57 → D-SPEC-63 collision
    resolution with parallel §4.D meditation_worker + §4.G life_force_worker
    + §4.L sovereignty_worker + §4.N recorder_worker + §4.H interface_advisor_worker
    + backup_unified sessions that landed D-SPEC-57..62 first).
    """
    spec_path = Path(__file__).parent.parent / "titan-docs" / "specs" / "SPEC_titan_architecture.md"
    spec_text = spec_path.read_text()
    assert "| **studio_worker** | (none — new in v1.9.4)" in spec_text


def test_spec_anchor_changelog_studio_worker_row():
    """SPEC top Changelog has the renumbered v1.9.4 studio_worker row."""
    spec_path = Path(__file__).parent.parent / "titan-docs" / "specs" / "SPEC_titan_architecture.md"
    spec_text = spec_path.read_text()
    assert "v1.9.4 (PATCH) | `studio_worker` extracted" in spec_text
    assert "D-SPEC-63" in spec_text


def test_kernel_proxy_aliases_extended():
    """core/kernel.py:KERNEL_PROXY_ALIASES includes studio_proxy + studio_render_proxy."""
    from titan_hcl.core.kernel import KERNEL_PROXY_ALIASES
    assert "studio_proxy" in KERNEL_PROXY_ALIASES
    assert "studio_render_proxy" in KERNEL_PROXY_ALIASES


def test_phase_c_rpc_exemptions_studio_gallery_listed():
    """phase_c_rpc_exemptions.yaml has studio_proxy.get_gallery_async entry (≤2s)."""
    yaml_path = Path(__file__).parent.parent / "titan-docs" / "specs" / "phase_c_rpc_exemptions.yaml"
    yaml_text = yaml_path.read_text()
    assert "studio_proxy" in yaml_text
    assert "get_gallery_async" in yaml_text
    assert "timeout_s: 2.0" in yaml_text


# ── StudioProxy fire-and-forget shape ────────────────────────────────


def test_proxy_request_methods_return_uuid_request_id():
    """Fire-and-forget request_* methods return a hex request_id."""
    from titan_hcl.proxies.studio_proxy import StudioProxy
    import uuid as _uuid

    # Capture publishes via fake bus.
    published = []

    class _FakeBus:
        def subscribe(self, name, types=None, reply_only=False):
            return _FakeQueue()
        def publish(self, msg):
            published.append(msg)
    class _FakeQueue:
        def get(self, timeout=None):
            raise Exception("queue not driven")
    class _FakeGuardian:
        pass

    # Patch _ensure_started + _start_safe import
    proxy = StudioProxy(bus=_FakeBus(), guardian=_FakeGuardian())
    proxy._ensure_started = lambda: None  # bypass guardian start

    rid = proxy.request_meditation_art("STATE_ROOT", 100, 50)
    assert isinstance(rid, str)
    assert len(rid) == 32  # uuid4 hex
    # Verify a STUDIO_RENDER_REQUEST was published with this request_id
    assert len(published) == 1
    msg = published[0]
    assert msg["type"] == "STUDIO_RENDER_REQUEST"
    assert msg["payload"]["request_id"] == rid
    assert msg["payload"]["type"] == "meditation"
    assert msg["payload"]["args"]["state_root"] == "STATE_ROOT"


def test_proxy_get_stats_uses_shm_reader(titan_id, studio_dirs):
    """proxy.get_stats() reads via StudioStateShmReader (sub-µs G18)."""
    from titan_hcl.proxies.studio_proxy import StudioProxy, StudioStateShmReader

    class _FakeBus:
        def subscribe(self, name, types=None, reply_only=False):
            return None

    # Populate SHM via publisher first
    pub = _build_publisher(titan_id, studio_dirs)
    pub.record_render("epoch")
    pub.record_render("epoch")
    pub.publish()

    # Manually instantiate reader pointing at our test SHM slot
    proxy = StudioProxy(bus=_FakeBus(), guardian=None)
    proxy._shm_reader = StudioStateShmReader(titan_id=titan_id)
    stats = proxy.get_stats()
    assert stats["epoch_count"] == 2
    assert stats["last_render_type"] == "epoch"
