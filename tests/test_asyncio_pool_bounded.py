"""Tests for the bounded asyncio default-executor pool sizing.

Closes a load-bearing contributor to BUG-PARENT-MEMORY-LEAK-HOST-OOM-20260428:
pre-V6 hardcoded max_workers=64 → V6 microkernel-aware max_workers=16.

See `_resolve_asyncio_pool_size` in scripts/titan_main.py and
rFP_microkernel_phase_a8_l2_l3_residency_completion §A.8.2 §3.4.
"""
import os
import sys

# Make scripts/ importable so we can pull `_resolve_asyncio_pool_size` from titan_main.py
_SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from titan_main import _resolve_asyncio_pool_size  # noqa: E402


def test_default_returns_bounded_16_when_no_config():
    assert _resolve_asyncio_pool_size({}) == 16


def test_default_returns_bounded_16_when_no_microkernel_section():
    assert _resolve_asyncio_pool_size({"network": {}}) == 16


def test_explicit_max_workers_honored():
    cfg = {"microkernel": {"asyncio_pool_max_workers": 24}}
    assert _resolve_asyncio_pool_size(cfg) == 24


def test_bounded_disabled_returns_legacy_64():
    cfg = {"microkernel": {"asyncio_pool_bounded_enabled": False}}
    assert _resolve_asyncio_pool_size(cfg) == 64


def test_bounded_disabled_ignores_max_workers():
    cfg = {
        "microkernel": {
            "asyncio_pool_bounded_enabled": False,
            "asyncio_pool_max_workers": 8,
        }
    }
    assert _resolve_asyncio_pool_size(cfg) == 64


def test_bounded_enabled_explicit_true():
    cfg = {"microkernel": {"asyncio_pool_bounded_enabled": True}}
    assert _resolve_asyncio_pool_size(cfg) == 16


def test_bounded_with_explicit_size():
    cfg = {
        "microkernel": {
            "asyncio_pool_bounded_enabled": True,
            "asyncio_pool_max_workers": 12,
        }
    }
    assert _resolve_asyncio_pool_size(cfg) == 12


def test_malformed_max_workers_falls_back_to_16():
    cfg = {"microkernel": {"asyncio_pool_max_workers": "not-a-number"}}
    assert _resolve_asyncio_pool_size(cfg) == 16


def test_microkernel_not_a_dict_returns_default():
    cfg = {"microkernel": "broken"}
    assert _resolve_asyncio_pool_size(cfg) == 16


def test_no_arg_loads_runtime_config():
    """Calling with no arg → loads from titan_params.toml live."""
    size = _resolve_asyncio_pool_size()
    assert 1 <= size <= 128, f"unexpected pool size: {size}"


def test_string_int_max_workers_coerces():
    cfg = {"microkernel": {"asyncio_pool_max_workers": "20"}}
    assert _resolve_asyncio_pool_size(cfg) == 20
