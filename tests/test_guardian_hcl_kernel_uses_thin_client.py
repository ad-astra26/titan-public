"""
test_guardian_hcl_kernel_uses_thin_client — verify TitanKernel constructs
GuardianHCLClient (NOT the in-process Guardian) under Phase 6.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0. Closes the PURE CUTOVER
invariant: NO in-process Guardian construction left behind in the kernel.
"""
import inspect

from titan_hcl.core import kernel as kernel_mod
from titan_hcl.guardian_hcl_client import GuardianHCLClient


def test_kernel_imports_guardianhclclient_not_guardian():
    """The kernel module must import GuardianHCLClient (the Phase 6 thin
    client) and NOT the legacy in-process Guardian class. Source-level
    inspection — covers the import statement before any TitanKernel
    instance is created."""
    src = inspect.getsource(kernel_mod)
    assert "from titan_hcl.guardian_hcl_client import GuardianHCLClient" in src, (
        "Phase 6 PURE CUTOVER: kernel.py must import GuardianHCLClient, not "
        "Guardian (the in-process L1 supervisor)")
    assert "from titan_hcl.guardian_hcl import Guardian\n" not in src, (
        "Phase 6 PURE CUTOVER: kernel.py must NOT import the in-process "
        "Guardian — it now lives in guardian_hcl process exclusively")


def test_kernel_does_not_construct_guardian_in_init():
    """Source-level check that __init__ assigns GuardianHCLClient(self.bus)
    to self.guardian, never Guardian(...)"""
    src = inspect.getsource(kernel_mod)
    assert "GuardianHCLClient(self.bus)" in src, (
        "Phase 6: self.guardian must be a GuardianHCLClient instance")
    # The legacy `Guardian(self.bus, config=...)` construction must not
    # appear as live code. (A single historical mention in a comment that
    # explicitly says "Pre-Phase-6" is allowed — that's documentation.)
    code_lines = [ln for ln in src.splitlines()
                  if "Guardian(self.bus, config=" in ln
                  and not ln.lstrip().startswith("#")]
    assert not code_lines, (
        f"Phase 6 PURE CUTOVER: live code must not construct in-process "
        f"Guardian. Offending lines: {code_lines}")


def test_kernel_guardian_loop_carved_to_state_publish_loop():
    """Phase 6 renamed _guardian_loop → _l0_state_publish_loop (drops
    Guardian.monitor_tick + drain_send_queues + guardian_state.bin publish;
    keeps soul_state.bin + network_state.bin publish)."""
    src = inspect.getsource(kernel_mod)
    assert "async def _l0_state_publish_loop" in src, (
        "Phase 6 renamed the loop to _l0_state_publish_loop")
    assert "async def _guardian_loop" not in src, (
        "Phase 6 PURE CUTOVER: _guardian_loop deleted, not preserved")


def test_guardian_hcl_client_provides_full_legacy_surface():
    """Plugin callsites use self.guardian.{start,stop,is_running,_modules,
    get_status,reload_module,restart_module,stop_all,start_all,enable,
    drain_send_queues,monitor_tick}. All must exist on the client."""
    required = {
        "start", "stop", "is_running", "_modules", "get_status",
        "reload_module", "restart_module", "stop_all", "start_all",
        "enable", "drain_send_queues", "monitor_tick",
        "register",  # raises — but must be present
        "get_modules_by_layer", "layer_stats",
    }
    surface = set(dir(GuardianHCLClient))
    missing = required - surface
    assert not missing, f"GuardianHCLClient missing surface: {missing}"
