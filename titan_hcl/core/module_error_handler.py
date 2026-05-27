"""titan_hcl/core/module_error_handler.py — Phase 11 Chunk 11C decorator.

Per SPEC §11.I.4 (D-SPEC-141 / v1.65.0):

  > Every worker, helper, provider, tool that raises must either
  >   (a) be wrapped in `@with_error_envelope(...)` which auto-publishes
  >       `ModuleError(severity=ERROR, ...)` before re-raising, OR
  >   (b) explicitly call `bus.publish_module_error(ModuleError(...))` before
  >       raising / before returning a soft failure.

This module provides (a). Adoption is per-function and additive: a worker
that hasn't yet been migrated is unaffected; once a function is decorated
its exceptions surface on the MODULE_ERROR bus topic before propagating.

Design points:

- Sync and async functions are both supported (separate wrappers chosen
  via `inspect.iscoroutinefunction`).
- KeyboardInterrupt and SystemExit are NOT caught — they propagate to the
  runtime exactly as they would without the decorator. We catch the
  `Exception` hierarchy only.
- The bus sender is discovered at call-time by inspecting the wrapped
  function's actual arguments for a parameter named `sender_arg`
  (default `"send_queue"`). If the wrapped function takes a `bus` arg
  instead, pass `sender_arg="bus"` to the decorator. If no sender is
  discoverable, the envelope publish is skipped — the exception is still
  re-raised so the spawn-side supervision watcher (titan_hcl, per
  SPEC §11.I.4) gets the chance to convert it to a FATAL
  UNCAUGHT_EXCEPTION envelope.
- Re-raise uses bare `raise` so the original traceback is preserved.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from ..bus import publish_module_error
from ..errors import ModuleError, ModuleErrorCode, Severity

logger = logging.getLogger(__name__)


def _resolve_sender(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    sender_arg: str,
) -> Optional[Any]:
    """Find the sender object in the wrapped fn's call arguments.

    Resolution order:
      1. kwargs[sender_arg] if present
      2. positional arg at the index matching sender_arg in fn's signature
      3. None (decorator will skip the envelope publish — exception still raised)

    Returns the sender object (which `publish_module_error` will duck-type
    as either a worker `send_queue` (`put_nowait`) or a `DivineBus`-style
    client (`publish`)).
    """
    if sender_arg in kwargs:
        return kwargs[sender_arg]
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters)
    if sender_arg in params:
        idx = params.index(sender_arg)
        if idx < len(args):
            return args[idx]
    return None


def with_error_envelope(
    *,
    module_name: str,
    subsystem: str,
    severity: Severity = Severity.ERROR,
    error_code: str | ModuleErrorCode = ModuleErrorCode.UNCAUGHT_EXCEPTION,
    sender_arg: str = "send_queue",
    suggested_remediation: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a worker entry-fn or helper so its exceptions surface a typed
    ModuleError envelope on the bus before re-raising.

    Args:
        module_name: producer module name (e.g. "agno_worker"); becomes
                     `ModuleError.module_name`.
        subsystem:   sub-area within the module (e.g. "ovg.warmup",
                     "llm.client"); becomes `ModuleError.subsystem`.
        severity:    Severity for caught exceptions (default ERROR).
                     For FATAL boot-paths consider `Severity.FATAL`.
        error_code:  Canonical short id; pass `ModuleErrorCode.X` or a
                     raw string. Default `UNCAUGHT_EXCEPTION` for the
                     generic catch-all path.
        sender_arg:  Parameter name to extract the bus sender from. For
                     standard worker entry-fns `(recv_queue, send_queue,
                     name, config)` leave the default. For helpers that
                     take `(bus, ...)` use `sender_arg="bus"`.
        suggested_remediation: optional operator hint included in every
                     envelope (e.g. "Restart with --reload-cache").

    Returns:
        A decorator. The returned wrapper preserves the wrapped function's
        name, qualname, module, docstring, and signature
        (`functools.wraps`).

    Re-raise semantics:
        The exception is re-raised with its original traceback intact
        (bare `raise`). KeyboardInterrupt and SystemExit are NOT caught.

    Example:
        @with_error_envelope(module_name="agno_worker", subsystem="boot")
        def agno_worker_main(recv_queue, send_queue, name, config):
            agent = _init_agent(...)
            run_loop(agent, recv_queue, send_queue)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    _publish_envelope_safely(
                        fn=fn,
                        args=args,
                        kwargs=kwargs,
                        exc=exc,
                        module_name=module_name,
                        subsystem=subsystem,
                        severity=severity,
                        error_code=error_code,
                        sender_arg=sender_arg,
                        suggested_remediation=suggested_remediation,
                    )
                    raise
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                _publish_envelope_safely(
                    fn=fn,
                    args=args,
                    kwargs=kwargs,
                    exc=exc,
                    module_name=module_name,
                    subsystem=subsystem,
                    severity=severity,
                    error_code=error_code,
                    sender_arg=sender_arg,
                    suggested_remediation=suggested_remediation,
                )
                raise

        return wrapper

    return decorator


def _publish_envelope_safely(
    *,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    exc: BaseException,
    module_name: str,
    subsystem: str,
    severity: Severity,
    error_code: str | ModuleErrorCode,
    sender_arg: str,
    suggested_remediation: Optional[str],
) -> None:
    """Build + publish the envelope. Never raises — any failure inside the
    envelope-publish path is swallowed and logged so the original exception
    propagates unimpeded.

    The double-fault guard matters: a broken bus must not mask the real
    error from reaching the supervision watcher.
    """
    try:
        sender = _resolve_sender(fn, args, kwargs, sender_arg)
        if sender is None:
            # No sender available — supervision watcher (titan_hcl) will
            # catch the propagating exception and emit MODULE_CRASHED
            # with FATAL UNCAUGHT_EXCEPTION per SPEC §11.I.4. Nothing to do.
            return
        err = ModuleError.from_exception(
            exc,
            module_name=module_name,
            subsystem=subsystem,
            error_code=error_code,
            severity=severity,
            suggested_remediation=suggested_remediation,
        )
        publish_module_error(sender, err)
    except Exception as inner:
        # Double-fault: never let envelope-publish-failure mask the real exception.
        try:
            logger.error(
                "[with_error_envelope] envelope-publish failed for "
                "module=%s subsystem=%s code=%r: %s",
                module_name, subsystem, error_code, inner,
            )
        except Exception:
            pass  # logger broken — give up, propagate original exception
