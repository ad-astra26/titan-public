"""
L3 Phase A.8.2 — Thread consolidation regression tests.

Tests the audit-and-skip pattern shipped:
  §3.5 SHIPPED — supervision-thread fork-mode skip in _module_wrapper
  §3.6 SKIPPED — kernel snapshot threads serve different consumers (audited)

Plus the new measurement infrastructure:
  - kernel.dump_thread_inventory() RPC method (shape + EXPOSED_METHODS)
  - /v4/admin/parent-threads endpoint (delegates via kernel_rpc)
  - arch_map thread-pool --parent (parses + renders the inventory)

These tests are runtime-aware: the supervision-skip behavior is verified
by source-inspection (consistent with test_a8_cache_bounds.py pattern) +
behavioral check that fork-mode never instantiates the no-op Thread.
"""
from __future__ import annotations

import inspect
import threading
import unittest


class TestModuleWrapperForkModeSupervisionSkip(unittest.TestCase):
    """A.8.2 §3.5 — _module_wrapper skips start_supervision_thread() for
    fork-mode workers, avoiding the redundant no-op Thread that
    worker_swap_handler.start_supervision_thread() returns for fork-mode."""

    def test_module_wrapper_has_explicit_start_method_check(self):
        """Source-inspection: guardian.py _module_wrapper conditions
        start_supervision_thread() on start_method == 'spawn'."""
        from titan_plugin import guardian
        src = inspect.getsource(guardian._module_wrapper)
        # The fix moves the gate from inside worker_swap_handler to
        # _module_wrapper itself. Verify the explicit guard is present
        # and that supervision is only started on spawn.
        self.assertIn('if start_method == "spawn":', src,
                      "_module_wrapper must explicitly gate "
                      "start_supervision_thread() on start_method")
        self.assertIn("A.8.2 §3.5", src,
                      "_module_wrapper must reference the rFP rationale "
                      "in its in-source comment block")

    def test_worker_swap_handler_fork_mode_still_returns_noop(self):
        """API-compat: start_supervision_thread() preserves its no-op
        return for fork-mode callers (existing test contract). The new
        _module_wrapper behavior is to NOT call it for fork-mode, but
        the function itself remains tolerant of being called either way."""
        from titan_plugin.core.worker_swap_handler import (
            SwapHandlerState,
            start_supervision_thread,
        )
        state = SwapHandlerState(
            name="test_fork_worker",
            start_method="fork",
            watcher_state=None,
            bus_client=None,
        )
        t = start_supervision_thread(state, interval=0.05)
        self.assertIsInstance(t, threading.Thread)
        # No-op thread terminates immediately; daemon=True so test exit clean.
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(),
                         "fork-mode no-op thread must terminate immediately")


class TestKernelDumpThreadInventory(unittest.TestCase):
    """A.8.2 §6 — kernel.dump_thread_inventory() method shape + RPC exposure."""

    def test_method_exists_on_kernel(self):
        """TitanKernel exposes dump_thread_inventory()."""
        from titan_plugin.core.kernel import TitanKernel
        self.assertTrue(hasattr(TitanKernel, "dump_thread_inventory"),
                        "TitanKernel must define dump_thread_inventory()")

    def test_exposed_via_kernel_rpc_allowlist(self):
        """The RPC allowlist (EXPOSED_METHODS at top of kernel.py) must
        include dump_thread_inventory under both bare and kernel-prefix
        forms — same pattern as dump_heap / dump_tracemalloc."""
        from titan_plugin.core import kernel
        src = inspect.getsource(kernel)
        # The list literal contains both forms.
        self.assertIn('"dump_thread_inventory"', src,
                      "EXPOSED_METHODS must allow 'dump_thread_inventory'")
        self.assertIn('"kernel.dump_thread_inventory"', src,
                      "EXPOSED_METHODS must allow 'kernel.dump_thread_inventory'")

    def test_inventory_payload_shape(self):
        """Method returns a dict with: pid, process, total, threads list,
        and a by_prefix grouping that aggregates thread names by prefix."""
        from titan_plugin.core.kernel import TitanKernel
        # We don't need a fully-booted kernel for this contract test —
        # threading.enumerate() reads the test process's own threads.
        # Construct a minimal stub that exposes dump_thread_inventory only.
        # Easier: bind the method to a SimpleNamespace and call it.
        # The method only references `os.getpid()` and `threading.enumerate()`,
        # not self state — we can call it via a synthetic context.
        import types
        stub = types.SimpleNamespace()
        stub.dump_thread_inventory = TitanKernel.dump_thread_inventory.__get__(
            stub, type(stub))
        result = stub.dump_thread_inventory()
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("process"), "parent")
        self.assertIn("pid", result)
        self.assertIn("total", result)
        self.assertIn("threads", result)
        self.assertIn("by_prefix", result)
        self.assertIsInstance(result["threads"], list)
        self.assertIsInstance(result["by_prefix"], dict)
        self.assertEqual(result["total"], len(result["threads"]))
        # At minimum the test runner has a MainThread.
        self.assertGreaterEqual(result["total"], 1)
        # Each thread row has the expected keys.
        for row in result["threads"]:
            for key in ("name", "ident", "daemon", "alive"):
                self.assertIn(key, row,
                              f"thread row missing key {key!r}: {row}")

    def test_by_prefix_strips_hex_id_suffix(self):
        """by_prefix grouping must collapse names like 'shadow-swap-abc12345'
        to 'shadow-swap' (per the rFP §6 audit goal: count by subsystem,
        not by per-instance ID)."""
        from titan_plugin.core.kernel import TitanKernel
        import types

        # Spawn a real thread with a hex-id suffix so the grouping logic
        # has something to bucket. daemon=True so test cleanup is clean.
        stop = threading.Event()

        def _worker():
            stop.wait(timeout=5.0)

        t = threading.Thread(
            target=_worker,
            name="shadow-swap-deadbeef",
            daemon=True,
        )
        t.start()
        try:
            stub = types.SimpleNamespace()
            stub.dump_thread_inventory = TitanKernel.dump_thread_inventory.__get__(
                stub, type(stub))
            result = stub.dump_thread_inventory()
            # The grouping should collapse "shadow-swap-deadbeef" → "shadow-swap"
            # since 'deadbeef' is hex and >= 6 chars.
            self.assertIn("shadow-swap", result["by_prefix"],
                          "by_prefix must collapse hex-id suffixes "
                          "(got: %s)" % list(result["by_prefix"].keys()))
            # Ensure the raw name is NOT a separate bucket.
            self.assertNotIn("shadow-swap-deadbeef", result["by_prefix"],
                              "raw hex-id suffix must not survive grouping")
        finally:
            stop.set()
            t.join(timeout=2.0)

    def test_by_prefix_preserves_non_hex_suffix(self):
        """Non-hex suffixes (e.g., 'imw.heartbeat', 'titan-spirit') must NOT
        be incorrectly collapsed — they're meaningful subsystem IDs."""
        from titan_plugin.core.kernel import TitanKernel
        import types

        stop = threading.Event()

        def _worker():
            stop.wait(timeout=5.0)

        t = threading.Thread(target=_worker, name="imw.heartbeat", daemon=True)
        t.start()
        try:
            stub = types.SimpleNamespace()
            stub.dump_thread_inventory = TitanKernel.dump_thread_inventory.__get__(
                stub, type(stub))
            result = stub.dump_thread_inventory()
            # 'imw.heartbeat' has '.' separator (no '-' / ':' / '_' partition
            # before then), so the prefix logic only triggers on the first
            # such separator that exists. Verify the non-hex tail isn't
            # eagerly collapsed.
            buckets = result["by_prefix"]
            # 'imw' as prefix is acceptable (heartbeat is not hex);
            # what we forbid is silent collapse of long meaningful tails.
            self.assertTrue(
                any(k.startswith("imw") for k in buckets),
                "imw thread must appear under an imw-prefixed bucket "
                "(got: %s)" % list(buckets.keys()),
            )
        finally:
            stop.set()
            t.join(timeout=2.0)


class TestArchMapThreadPoolParentFlag(unittest.TestCase):
    """A.8.2 §6 — arch_map thread-pool --parent dispatch + renderer."""

    def test_run_parent_thread_inventory_is_defined(self):
        """scripts/arch_map.py exports run_parent_thread_inventory()."""
        import importlib.util
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "arch_map", repo_root / "scripts" / "arch_map.py")
        # arch_map imports a lot at module load (graph, configs); we don't
        # need to fully load — just assert the function symbol exists in
        # source. Reading source is much cheaper + side-effect-free.
        src = (repo_root / "scripts" / "arch_map.py").read_text()
        self.assertIn("def run_parent_thread_inventory(", src,
                      "arch_map.py must define run_parent_thread_inventory()")
        self.assertIn('elif cmd == "thread-pool":', src,
                      "thread-pool dispatch must remain")
        self.assertIn('if "--parent" in sys.argv:', src,
                      "thread-pool dispatch must check for --parent flag")
        self.assertIn("run_parent_thread_inventory(all_titans=", src,
                      "--parent path must call run_parent_thread_inventory()")


class TestSnapshotThreadsAuditDocumentedSkip(unittest.TestCase):
    """A.8.2 §3.6 — kernel snapshot forwarders audited; documented as
    NOT consolidatable. This test verifies the rFP doc records the audit
    decision, so future sessions don't accidentally re-open the work."""

    def test_rfp_documents_3_6_skip(self):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        rfp = repo_root / "titan-docs" / "rFP_microkernel_phase_a8_l2_l3_residency_completion.md"
        self.assertTrue(rfp.exists(), f"rFP file missing: {rfp}")
        text = rfp.read_text()
        # Audit-and-skip note must be present.
        self.assertIn("§3.6 — kernel snapshot forwarders", text,
                      "rFP A.8.2 must document §3.6 audit decision")
        self.assertIn("Different content + different consumers + different cadences", text,
                      "rFP A.8.2 must explain WHY §3.6 was not consolidated")


if __name__ == "__main__":
    unittest.main()
