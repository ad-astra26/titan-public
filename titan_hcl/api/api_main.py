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

    from titan_hcl.api.api_subprocess import api_subprocess_main
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
