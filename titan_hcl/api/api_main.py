"""
titan_hcl/api/api_main.py — Standalone entry for the titan_hcl_api L3 process.

Phase 6 / SPEC §11.B.4 / D-SPEC-135 / v1.62.0.

INV-PROC-1: ps identity = `titan_hcl_api` (setproctitle first I/O).
INV-PROC-5: independent crash domain — kill -9 titan_hcl does NOT take down
            titan_hcl_api. Mutating endpoints transiently fail until L2
            (titan_hcl) returns; state reads stay 200 via SHM-direct (G18).

Spawned by guardian_hcl as a normal Guardian-supervised L3 module per the
module_catalog entry (see titan_hcl/module_catalog.py). The historical
`api_subprocess.api_subprocess_main(recv_queue, send_queue, name, config)`
function is preserved as the body — this file is a thin entry that:
  1. Sets `setproctitle('titan_hcl_api')` (INV-PROC-1)
  2. Bootstraps the worker bus contract (recv/send queues via the
     existing Guardian._module_wrapper protocol — guardian_hcl calls this
     module's entry via mp.Process, passing the recv/send queues + name +
     config like every other worker)
  3. Delegates to api_subprocess_main for the unchanged body
"""
from __future__ import annotations


def entry(recv_queue, send_queue, name: str, config: dict) -> None:
    """Module entry function called by Guardian via mp.Process.

    Per Guardian._module_wrapper (titan_hcl/guardian_hcl/core.py): worker
    entry functions are invoked with (recv_queue, send_queue, name, config).
    This entry is what guardian_hcl's module_catalog registers as
    ModuleSpec.entry_fn for the api module (name="titan_hcl_api").
    """
    # INV-PROC-1 — set ps identity as first I/O. Soft-fallback matches
    # scripts/titan_hcl.py + scripts/guardian_hcl.py.
    try:
        import setproctitle as _spt
        _spt.setproctitle("titan_hcl_api")
    except ImportError:
        pass

    # Phase 11 §11.I.5 — populate the api module's SHM state slot so
    # /v6/readiness counts it toward mandatory_ready (W3 sweep / commit
    # 58761482 skipped this entry because api_main.py lives under
    # titan_hcl/api/ rather than titan_hcl/modules/). Same pattern as
    # the 39 worker entries. Logged-and-tolerated init failure so a
    # missing SHM root doesn't take down the api process.
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name="api",
            layer="L3",
            boot_priority=BootPriority.MANDATORY,
        )
        _state_writer.write_state("starting")
    except Exception as _sw_err:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[titan_hcl_api] Phase 11 ModuleStateWriter init failed "
            "(continuing on legacy MODULE_READY path): %s", _sw_err)
        _state_writer = None

    from titan_hcl.api.api_subprocess import api_subprocess_main
    # The api_subprocess_main run loop reaches "booted" after KernelRPCClient
    # is wired (~5s into entry) and "running" after uvicorn binds the port
    # (~15-30s further). Surface "booted" pre-delegate so the SHM slot
    # transitions out of "starting" early; api_subprocess_main itself emits
    # MODULE_READY on uvicorn-up which (under legacy bus-only liveness) is
    # the canonical readiness signal — Supervisor uses the MAX(SHM, bus)
    # heartbeat per §11.I.5.
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception:
            pass
    api_subprocess_main(recv_queue, send_queue, name, config)


def main() -> None:
    """Manual invocation entry — supports `python -m titan_hcl.api.api_main`.

    Useful for local diagnostics. In the production fleet, guardian_hcl
    spawns this module via the Guardian module-spec entry_fn (which calls
    `entry(...)` above with proper queues). When invoked via `-m`, the
    queues are not provided — we emit a friendly diagnostic and exit.
    """
    import sys
    print(
        "titan_hcl_api is normally spawned by guardian_hcl as a "
        "Guardian-supervised L3 module. To run standalone for diagnostics, "
        "wire mp.Queue handles to entry(recv_queue, send_queue, name, config) "
        "directly. See titan_hcl/api/api_subprocess.py for the body.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
