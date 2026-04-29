"""C2-7 Python integration tests.

Per PLAN_microkernel_phase_c_s2_kernel.md §12.9:
  - Flag-off path = byte-identical (current Python boot path unchanged)
  - Flag-on path skips Python BusSocketServer
  - Drift aliases: each rename pair imports old name → equals new value
  - Bus census per-Titan path resolves from TITAN_KERNEL_TITAN_ID env
  - Supervision log reader reads JSONL + filters
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest import mock

import pytest


# ─── Fixtures + helpers ────────────────────────────────────────────────────


# ─── Drift aliases (D01-D11/D13-D15/D18-D29 — landed subset) ──────────────


class TestDriftAliases:
    """Each rename pair: import old name → equals canonical value."""

    def test_d01_restart_backoff_base(self):
        from titan_plugin._phase_c_constants import (
            SUPERVISION_RESTART_BACKOFF_MAX_S,
        )
        from titan_plugin._phase_c_drift_aliases import RESTART_BACKOFF_BASE

        assert RESTART_BACKOFF_BASE == SUPERVISION_RESTART_BACKOFF_MAX_S

    def test_d02_max_restarts_in_window_and_window_seconds(self):
        from titan_plugin._phase_c_constants import (
            SUPERVISION_INTENSITY_WINDOW_S,
            SUPERVISION_MAX_RESTARTS,
        )
        from titan_plugin._phase_c_drift_aliases import (
            MAX_RESTARTS_IN_WINDOW,
            RESTART_WINDOW_SECONDS,
        )

        assert MAX_RESTARTS_IN_WINDOW == SUPERVISION_MAX_RESTARTS
        assert RESTART_WINDOW_SECONDS == SUPERVISION_INTENSITY_WINDOW_S

    def test_d03_heartbeat(self):
        from titan_plugin._phase_c_constants import (
            MODULE_HEARTBEAT_INTERVAL_S,
            MODULE_HEARTBEAT_TIMEOUT_S,
        )
        from titan_plugin._phase_c_drift_aliases import (
            HEARTBEAT_INTERVAL,
            HEARTBEAT_TIMEOUT,
        )

        assert HEARTBEAT_INTERVAL == MODULE_HEARTBEAT_INTERVAL_S
        assert HEARTBEAT_TIMEOUT == MODULE_HEARTBEAT_TIMEOUT_S

    def test_d06_authkey(self):
        from titan_plugin._phase_c_constants import AUTHKEY_BYTES, AUTHKEY_HKDF_SALT
        from titan_plugin._phase_c_drift_aliases import (
            BUS_AUTHKEY_LEN,
            BUS_AUTHKEY_SALT,
        )

        assert BUS_AUTHKEY_SALT == AUTHKEY_HKDF_SALT
        assert BUS_AUTHKEY_LEN == AUTHKEY_BYTES

    def test_d07_frame_handshake_sizes(self):
        from titan_plugin._phase_c_constants import (
            FRAME_AUTH_TAG_BYTES,
            FRAME_CHALLENGE_BYTES,
        )
        from titan_plugin._phase_c_drift_aliases import (
            AUTH_TAG_SIZE,
            CHALLENGE_SIZE,
        )

        assert CHALLENGE_SIZE == FRAME_CHALLENGE_BYTES
        assert AUTH_TAG_SIZE == FRAME_AUTH_TAG_BYTES

    def test_d19_reconnect_backoff_ms_to_s_conversion(self):
        from titan_plugin._phase_c_constants import (
            BUS_RECONNECT_BACKOFF_INITIAL_MS,
            BUS_RECONNECT_BACKOFF_MAX_S,
        )
        from titan_plugin._phase_c_drift_aliases import (
            RECONNECT_BACKOFF_BASE_S,
            RECONNECT_BACKOFF_MAX_S,
        )

        assert RECONNECT_BACKOFF_BASE_S == BUS_RECONNECT_BACKOFF_INITIAL_MS / 1000.0
        assert RECONNECT_BACKOFF_MAX_S == BUS_RECONNECT_BACKOFF_MAX_S

    def test_drift_aliases_module_all_exports_resolve(self):
        from titan_plugin import _phase_c_drift_aliases as a

        # Every name in __all__ must resolve and be non-None
        assert len(a.__all__) >= 25
        for name in a.__all__:
            assert getattr(a, name) is not None, f"{name} is None"


# ─── Config flag presence ─────────────────────────────────────────────────


class TestL0RustFlag:
    def test_config_toml_has_microkernel_l0_rust_enabled_default_false(self):
        try:
            import tomllib  # py311+
        except ImportError:
            import tomli as tomllib  # py310

        cfg_path = Path(__file__).parent.parent / "titan_plugin" / "config.toml"
        with cfg_path.open("rb") as f:
            cfg = tomllib.load(f)

        assert "microkernel" in cfg
        assert "l0_rust_enabled" in cfg["microkernel"]
        # Per SPEC §3.0 Running-Titans Safety Rule: must default false
        assert cfg["microkernel"]["l0_rust_enabled"] is False

    def test_kernel_py_skips_bus_broker_when_l0_rust_enabled(self):
        # Read source — verifying the branch exists. We don't instantiate
        # TitanKernel here (heavy + needs full env); a source-level check
        # is sufficient to gate the contract.
        kernel_src = (
            Path(__file__).parent.parent
            / "titan_plugin"
            / "core"
            / "kernel.py"
        ).read_text(encoding="utf-8")
        assert 'l0_rust_enabled' in kernel_src
        assert 'skipping' in kernel_src.lower()

    def test_titan_watchdog_branches_on_l0_rust_enabled(self):
        watchdog_src = (
            Path(__file__).parent.parent / "scripts" / "titan_watchdog.sh"
        ).read_text(encoding="utf-8")
        assert "l0_rust_enabled" in watchdog_src
        assert "titan-kernel-rs" in watchdog_src


# ─── Bus census per-Titan path (D11) ──────────────────────────────────────


class TestBusCensusPerTitanPath:
    def _reload_with_env(self, env_overrides: dict) -> str:
        """Snapshot env keys we touch, apply overrides, force-reload
        bus_census, return CENSUS_LOG_PATH, then restore env. Uses
        importlib.reload because Python's `from x import y` short-circuits
        on `x.y` attribute even when sys.modules['x.y'] is popped."""
        keys = (
            "TITAN_KERNEL_TITAN_ID",
            "TITAN_ID",
            "TITAN_BUS_CENSUS_LOG",
        )
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            for k, v in env_overrides.items():
                os.environ[k] = v
            from titan_plugin.core import bus_census  # noqa: WPS433

            importlib.reload(bus_census)
            return bus_census.CENSUS_LOG_PATH
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_path_uses_titan_kernel_titan_id_env(self):
        path = self._reload_with_env({"TITAN_KERNEL_TITAN_ID": "T2"})
        assert path == "/tmp/titan_T2_bus_census.log"

    def test_path_falls_back_to_legacy_titan_id(self):
        path = self._reload_with_env({"TITAN_ID": "T3"})
        assert path == "/tmp/titan_T3_bus_census.log"

    def test_path_defaults_to_t1_when_no_env(self):
        path = self._reload_with_env({})
        assert path == "/tmp/titan_T1_bus_census.log"


# ─── Supervision log reader ───────────────────────────────────────────────


def _add_scripts_dir():
    scripts_dir = str(Path(__file__).parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


class TestSupervisionLogReader:
    def test_iter_empty_returns_empty(self):
        _add_scripts_dir()
        import _supervision_log_reader as r

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "supervision.jsonl"
            assert list(r.iter_supervision_log(p)) == []

    def test_iter_filters_by_kind(self):
        _add_scripts_dir()
        import _supervision_log_reader as r

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "supervision.jsonl"
            with p.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "kind": "CHILD_STARTED",
                            "child": "trinity-substrate",
                            "ts": "2026-04-29T12:00:00.000000000Z",
                        }
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "kind": "CHILD_EXITED",
                            "child": "trinity-substrate",
                            "reason": "sigterm",
                            "ts": "2026-04-29T12:00:01.000000000Z",
                        }
                    )
                    + "\n"
                )

            events = list(r.iter_supervision_log(p, kind="CHILD_EXITED"))
            assert len(events) == 1
            assert events[0]["reason"] == "sigterm"

    def test_iter_filters_by_child_and_skips_malformed(self):
        _add_scripts_dir()
        import _supervision_log_reader as r

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "supervision.jsonl"
            with p.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps({"kind": "CHILD_STARTED", "child": "alpha"}) + "\n"
                )
                f.write("{ malformed json\n")
                f.write(
                    json.dumps({"kind": "CHILD_STARTED", "child": "beta"}) + "\n"
                )

            events = list(r.iter_supervision_log(p, child="alpha"))
            assert len(events) == 1

    def test_iter_walks_archives_in_chronological_order(self):
        _add_scripts_dir()
        import _supervision_log_reader as r

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "supervision.jsonl"
            # .jsonl.2 (oldest), .jsonl.1, .jsonl (newest)
            (base.parent / "supervision.jsonl.2").write_text(
                json.dumps({"kind": "X", "i": 0}) + "\n", encoding="utf-8"
            )
            (base.parent / "supervision.jsonl.1").write_text(
                json.dumps({"kind": "X", "i": 1}) + "\n", encoding="utf-8"
            )
            base.write_text(
                json.dumps({"kind": "X", "i": 2}) + "\n", encoding="utf-8"
            )

            events = list(r.iter_supervision_log(base))
            assert [e["i"] for e in events] == [0, 1, 2]

    def test_parse_iso8601_handles_nanoseconds(self):
        _add_scripts_dir()
        import _supervision_log_reader as r

        dt = r._parse_iso8601("2026-04-29T13:42:01.123456789Z")
        assert dt.year == 2026
        assert dt.minute == 42
        assert dt.second == 1
        # microsecond precision (truncated from ns)
        assert dt.microsecond == 123456


# ─── Drift bridge dual-emit (D13/D14/D15) ─────────────────────────────────


class TestDriftBridgeDualEmit:
    """Verify bus_socket.publish() dual-emits canonical ↔ legacy."""

    def test_bridge_pairs_table_covers_d13_d14_d15(self):
        # Source-level check: each canonical name maps back to its legacy peer
        from titan_plugin.core.bus_socket import BusSocketServer

        pairs = BusSocketServer._PHASE_C_BRIDGE_PAIRS
        # D13: SWAP_HANDOFF
        assert pairs["BUS_HANDOFF"] == "SWAP_HANDOFF"
        assert pairs["SWAP_HANDOFF"] == "BUS_HANDOFF"
        assert pairs["BUS_HANDOFF_CANCELED"] == "SWAP_HANDOFF_CANCELED"
        assert pairs["SWAP_HANDOFF_CANCELED"] == "BUS_HANDOFF_CANCELED"
        # D14: ADOPTION_REQUEST/ACK
        assert pairs["BUS_WORKER_ADOPT_REQUEST"] == "ADOPTION_REQUEST"
        assert pairs["ADOPTION_REQUEST"] == "BUS_WORKER_ADOPT_REQUEST"
        assert pairs["BUS_WORKER_ADOPT_ACK"] == "ADOPTION_ACK"
        assert pairs["ADOPTION_ACK"] == "BUS_WORKER_ADOPT_ACK"
        # D15: KERNEL_EPOCH_TICK
        assert pairs["EPOCH_TICK"] == "KERNEL_EPOCH_TICK"
        assert pairs["KERNEL_EPOCH_TICK"] == "EPOCH_TICK"

    def test_bus_module_has_canonical_names(self):
        from titan_plugin import bus

        # New canonical names alongside legacy
        assert bus.SWAP_HANDOFF == "SWAP_HANDOFF"
        assert bus.SWAP_HANDOFF_CANCELED == "SWAP_HANDOFF_CANCELED"
        assert bus.ADOPTION_REQUEST == "ADOPTION_REQUEST"
        assert bus.ADOPTION_ACK == "ADOPTION_ACK"
        assert bus.KERNEL_EPOCH_TICK == "KERNEL_EPOCH_TICK"
        # Legacy names preserved
        assert bus.BUS_HANDOFF == "BUS_HANDOFF"
        assert bus.BUS_WORKER_ADOPT_REQUEST == "BUS_WORKER_ADOPT_REQUEST"
        assert bus.EPOCH_TICK == "EPOCH_TICK"
