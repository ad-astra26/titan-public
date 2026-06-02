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

    # Native-crash visibility (SPEC §11.I.4) — dump a C+Python traceback to
    # stderr→journal on a fatal native signal; the @with_error_envelope cascade
    # only catches Python exceptions, not signals.
    try:
        import faulthandler as _faulthandler
        _faulthandler.enable()
    except Exception:
        pass

    # SPEC §11.B.5 / rFP_kernel_zero_downtime_api_reload P3 — reload-child
    # detection. Gated on TITAN_API_RELOAD_CHILD (NOT TITAN_API_REUSEPORT —
    # the kernel sets REUSEPORT=1 on EVERY api spawn so the running api is
    # always swap-ready, whereas RELOAD_CHILD=1 marks ONLY the NEW process of
    # an actual zero-downtime swap). When set, this process is a NEW api
    # co-bound with the OLD api via SO_REUSEPORT during a kernel-driven reload.
    # The per-module SHM state slot is SINGLE-WRITER (state_registry), so a
    # reload child must NOT write the canonical `module_api_state.bin` while
    # OLD still owns it. Instead it writes a DEDICATED
    # `module_api_reload_state.bin` slot (the kernel health-gates on THAT —
    # pid-specific), and only PROMOTES to the canonical slot once OLD has
    # exited (self-promote on OLD-pid-death, below). Normal boots
    # (RELOAD_CHILD unset) write the canonical slot directly, unchanged —
    # single-writer everywhere.
    import os as _os
    _is_reload_child = _os.environ.get("TITAN_API_RELOAD_CHILD") == "1"
    _state_module_name = "api_reload" if _is_reload_child else "api"

    # Phase 11 §11.I.5 — populate the api module's SHM state slot so
    # /v6/readiness counts it toward mandatory_ready (W3 sweep / commit
    # 58761482 skipped this entry because api_main.py lives under
    # titan_hcl/api/ rather than titan_hcl/modules/). Same pattern as
    # the 39 worker entries. Logged-and-tolerated init failure so a
    # missing SHM root doesn't take down the api process.
    try:
        from titan_hcl.core.module_state import BootPriority, ModuleStateWriter
        _state_writer = ModuleStateWriter(
            module_name=_state_module_name,
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
    # Surface "booted" pre-delegate so the SHM slot transitions out of
    # "starting" early.
    if _state_writer is not None:
        try:
            _state_writer.write_state("booted")
        except Exception:
            pass
        # Self-attest "running" once uvicorn is actually serving. The api is a
        # kernel-spawned PEER, not an orchestrator-probed worker — nothing
        # dispatches MODULE_PROBE_REQUEST to it, and the legacy MODULE_READY
        # readiness signal this used to rely on is DELETED (Phase 11 D1/D2). So
        # without this it sits at "booted" forever (functional, but
        # /v6/readiness under-counts it → 39/40). A daemon thread polls the
        # api's OWN /health until 200 (uvicorn bound + app up) then writes
        # "running" to its slot, mirroring how the Rust L0/L1 daemons
        # self-attest. Daemon thread because api_subprocess_main blocks on
        # uvicorn; the ModuleStateWriter heartbeat daemon then keeps it fresh.
        import threading as _threading

        def _self_attest_running() -> None:
            import os as _os
            import time as _time
            import urllib.request as _url
            try:
                from titan_hcl.config_loader import load_titan_config as _ltc
                _port = int(_os.environ.get("TITAN_API_PORT")
                            or _ltc().get("api", {}).get("port", 7777))
            except Exception:  # noqa: BLE001
                _port = int(_os.environ.get("TITAN_API_PORT", "7777"))
            _hc = f"http://127.0.0.1:{_port}/health"
            _deadline = _time.time() + 180.0  # uvicorn bind + kernel_rpc connect
            while _time.time() < _deadline:
                try:
                    with _url.urlopen(_hc, timeout=3) as _r:
                        if getattr(_r, "status", None) == 200 or _r.getcode() == 200:
                            _state_writer.write_state("running")
                            return
                except Exception:  # noqa: BLE001
                    pass
                _time.sleep(2.0)

        _threading.Thread(
            target=_self_attest_running,
            name="api-self-attest-running", daemon=True).start()

        # SPEC §11.B.5 P3 — self-promote on OLD-pid-death. A reload child
        # gates on its dedicated `module_api_reload_state.bin`; once the OLD
        # api process (the current owner of the canonical `module_api_state.bin`
        # slot) has exited, NEW takes over the canonical slot as its SOLE
        # writer — so the slot never has two writers AND /v6/readiness keeps a
        # fresh api entry after the swap (the canonical slot would otherwise go
        # stale in 60-180s once OLD's heartbeat stops). Decoupled + IPC-free.
        if _is_reload_child:
            def _promote_to_canonical() -> None:
                import time as _time
                import logging as _plog
                _lg = _plog.getLogger(__name__)
                try:
                    from titan_hcl.core.module_state import (
                        BootPriority as _BP, ModuleStateReader as _MSR,
                        ModuleStateWriter as _MSW)
                except Exception as _imp_err:  # noqa: BLE001
                    _lg.warning("[titan_hcl_api] promote import failed: %s", _imp_err)
                    return
                _my_pid = _os.getpid()
                try:
                    _canon_reader = _MSR(module_name="api")
                except Exception:  # noqa: BLE001
                    _canon_reader = None

                def _old_alive() -> bool:
                    # OLD owns the canonical slot. Alive iff its pid is a live,
                    # distinct process. Missing slot / dead pid / our-own-pid
                    # all mean "no OLD to wait for → promote now".
                    if _canon_reader is None:
                        return False
                    try:
                        _entry = _canon_reader.read()
                    except Exception:  # noqa: BLE001
                        return False
                    if _entry is None:
                        return False
                    _old_pid = int(getattr(_entry, "pid", 0) or 0)
                    if _old_pid <= 0 or _old_pid == _my_pid:
                        return False
                    try:
                        _os.kill(_old_pid, 0)
                        return True            # live, distinct OLD still up
                    except ProcessLookupError:
                        return False           # OLD gone → promote
                    except PermissionError:
                        return True            # exists (not ours) → still up
                    except Exception:  # noqa: BLE001
                        return False

                # Wait for OLD to exit. Bounded by API_RELOAD_HEALTH_TIMEOUT +
                # drain (matches §11.B.5); if OLD lingers past that, promote
                # anyway so readiness never stalls (kernel SIGKILLs OLD on
                # drain-timeout, so this is a safety net, not the normal path).
                _deadline = _time.time() + 60.0
                while _old_alive() and _time.time() < _deadline:
                    _time.sleep(1.0)
                try:
                    _canon = _MSW(
                        module_name="api", layer="L3",
                        boot_priority=_BP.MANDATORY)
                    _canon.write_state("running")
                    # Keep a process-lifetime ref so its heartbeat daemon
                    # (which keeps the canonical slot fresh) is not GC'd.
                    globals()["_api_canonical_writer"] = _canon
                    _lg.info(
                        "[titan_hcl_api] §11.B.5 promoted to canonical "
                        "module_api_state.bin (pid=%d) after OLD exit", _my_pid)
                except Exception as _pe:  # noqa: BLE001
                    _lg.warning("[titan_hcl_api] canonical promotion failed: %s", _pe)

            _threading.Thread(
                target=_promote_to_canonical,
                name="api-reload-promote", daemon=True).start()
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
