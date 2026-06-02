"""
test_guardian_hcl_module_catalog_importable — smoke test for the carved
module_catalog.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0. We don't actually call
build_catalog (that requires worker imports + Solana keypair env); we just
verify the module imports cleanly + the signature is the documented one
+ the api section references api_main not api_subprocess directly.
"""
import inspect

import pytest

import titan_hcl.module_catalog as mc


def test_module_catalog_imports():
    assert hasattr(mc, "build_catalog")
    assert callable(mc.build_catalog)


def test_build_catalog_signature_is_phase6_canonical():
    sig = inspect.signature(mc.build_catalog)
    params = list(sig.parameters.keys())
    assert params[:3] == ["bus", "guardian", "config"], (
        f"Phase 6 contract: build_catalog(bus, guardian, config, *, titan_id, kernel=None). "
        f"Got: {params}")
    assert "titan_id" in sig.parameters
    assert "kernel" in sig.parameters
    # kernel must have a default (callers like guardian_hcl pass kernel=None)
    assert sig.parameters["kernel"].default is None


def test_module_catalog_uses_api_main_entry_not_api_subprocess_main():
    """The carved api ModuleSpec registers entry_fn=api_main.entry, NOT
    api_subprocess.api_subprocess_main directly — so setproctitle('titan_hcl_api')
    runs first I/O per INV-PROC-1."""
    src = inspect.getsource(mc)
    assert "from titan_hcl.api.api_main import entry as api_main_entry" in src, (
        "Phase 6 catalog must import api_main.entry (the setproctitle "
        "wrapper), not api_subprocess_main directly")
    assert "entry_fn=api_main_entry" in src


def test_api_modulespec_is_kernel_supervised():
    """SPEC §11.B.4 INV-PROC-5 / §11.B.5 — the api ModuleSpec must carry
    kernel_supervised=True so guardian_hcl stands down (the kernel is the sole
    spawner/respawner). Without it, the L1 Supervisor's shm_pid_dead check
    races the kernel's zero-downtime swap → a doomed orchestrator spawn that
    zombies. Source-text assertion (mirrors the entry_fn check above; building
    the full catalog needs worker imports + a Solana keypair)."""
    src = inspect.getsource(mc)
    assert "kernel_supervised=True" in src, (
        "the api ModuleSpec must set kernel_supervised=True — else guardian_hcl "
        "will race the kernel's api lifecycle (zombie on every swap/crash)")
    # And the ModuleSpec field exists with a safe default.
    from titan_hcl.orchestrator.module_registry import ModuleSpec
    assert ModuleSpec(name="x", entry_fn=lambda *a: None).kernel_supervised is False


def test_api_main_entry_imports_and_is_callable():
    from titan_hcl.api import api_main
    assert callable(api_main.entry)
    # Signature must match Guardian's module_wrapper expectation:
    # entry(recv_queue, send_queue, name, config)
    sig = inspect.signature(api_main.entry)
    assert list(sig.parameters.keys()) == ["recv_queue", "send_queue", "name", "config"]
