"""Guardian entry function for the IMW daemon module.

Registered in titan_plugin/v5_core.py as a ModuleSpec named 'imw'. Guardian
spawns a subprocess that calls imw_main(recv_queue, send_queue, name, config).

The IMW daemon is mostly independent of the DivineBus: it communicates with
callers via a unix domain socket, not bus messages. But it DOES participate
in the bus for heartbeat + lifecycle (MODULE_READY/CRASHED) so Guardian can
supervise it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


def imw_main(recv_queue, send_queue, name: str, config: dict) -> None:
    """Main entry — runs until SIGTERM / SHUTDOWN message / crash.

    Args:
        recv_queue: DivineBus → this module (we only watch for SHUTDOWN)
        send_queue: this module → DivineBus (heartbeats, ready, crashed)
        name: module name ("imw" or "observatory_writer" — multi-instance safe
              since 2026-04-21)
        config: merged [persistence] dict from v5_core.py
    """
    # Defer heavy imports to inside child process (lazy import rule)
    from queue import Empty
    from titan_plugin.persistence.config import IMWConfig
    from titan_plugin.persistence.writer_service import IMWDaemon
    from titan_plugin.bus import MODULE_CRASHED, MODULE_HEARTBEAT, MODULE_READY, make_msg

    # Set process title so `ps aux` shows e.g. [imw_writer] / [observatory_writer]
    # instead of all writers blending into "python -u titan_main". Optional —
    # if setproctitle isn't installed, no-op (graceful fallback).
    try:
        import setproctitle as _spt
        _spt.setproctitle(f"titan-{name}")
    except ImportError:
        pass  # setproctitle is optional; ps clarity is nice-to-have, not required

    cfg = IMWConfig.from_dict(config)

    # Spin up asyncio loop in the main thread of this subprocess
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()
    daemon = IMWDaemon(cfg)

    def _signal(_signum, _frame):
        loop.call_soon_threadsafe(stop_event.set)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal)
        except (ValueError, OSError):
            pass

    # Heartbeat + bus watcher in a background thread. Also dump metrics
    # snapshot to a JSON file so the dashboard /v4/imw-health endpoint
    # can read them without adding an IPC protocol op for metrics.
    # 2026-04-21: filename derived from module name so multiple writer
    # instances (e.g. imw + observatory_writer) don't collide on the same
    # JSON file. IMW's name="imw" → "imw_metrics.json" (unchanged path
    # for backwards compatibility with the existing /v4/imw-health
    # endpoint). New observatory_writer name → "observatory_writer_metrics.json".
    import json as _json_mod
    import pathlib as _pathlib_mod
    METRICS_FILE = _pathlib_mod.Path(cfg.journal_dir) / f"{name}_metrics.json"

    def _heartbeat_thread():
        while not stop_event.is_set():
            try:
                snap = daemon.metrics_snapshot()
                send_queue.put(make_msg(MODULE_HEARTBEAT, name, "guardian", {
                    "ts": time.time(),
                    "metrics": snap,
                }))
                # Atomic file write for endpoint to read
                tmp = METRICS_FILE.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    _json_mod.dump(snap, f)
                os.replace(tmp, METRICS_FILE)
            except Exception:
                pass
            time.sleep(10.0)

    def _bus_watcher_thread():
        while not stop_event.is_set():
            try:
                msg = recv_queue.get(timeout=1.0)
            except Empty:
                continue
            except Exception:
                return
            if isinstance(msg, dict) and msg.get("type") == "MODULE_SHUTDOWN":
                logger.info("[imw] shutdown from bus")
                loop.call_soon_threadsafe(stop_event.set)
                return

    threading.Thread(target=_heartbeat_thread, name="imw.heartbeat", daemon=True).start()
    threading.Thread(target=_bus_watcher_thread, name="imw.bus", daemon=True).start()

    async def _run():
        await daemon.start()
        try:
            send_queue.put(make_msg(MODULE_READY, name, "guardian", {}))
        except Exception:
            pass
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.5)
        finally:
            await daemon.stop()

    try:
        loop.run_until_complete(_run())
    except Exception as e:
        logger.error("[imw] fatal: %s", e, exc_info=True)
        try:
            send_queue.put(make_msg(MODULE_CRASHED, name, "guardian",
                                       {"error": str(e)}))
        except Exception:
            pass
        raise
    finally:
        loop.close()
