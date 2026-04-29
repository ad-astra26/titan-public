"""
Regression test for Guardian module-cleanup deadlock.

On 2026-04-14 T1's asyncio event loop deadlocked for 31+ minutes inside
Guardian._cleanup_module -> multiprocessing.Queue.join_thread(), after
the 'mind' worker heartbeat-timed-out and was SIGKILL'd. py-spy showed
a QueueFeederThread stuck in os.write() on the pipe because sibling
forked children still held inherited read-end FDs, keeping the pipe
open instead of raising EPIPE.

Same bug had been intermittently force-restarting T2 via watchdog
(I-018: 9 force-restarts across Apr 11-13).

Fix: _cleanup_module must call cancel_join_thread() (documented Python
API for exactly this scenario) instead of join_thread(). Data destined
for a SIGKILL'd consumer is already lost; we lose nothing by abandoning
the flush.
"""
import inspect


def test_guardian_cleanup_uses_cancel_join_thread():
    """Static check: Guardian's cleanup chain must use cancel_join_thread()
    and must NOT call plain join_thread(). Guards against accidental revert
    of the T1-2026-04-14 / I-018 deadlock fix.

    Microkernel v2 Phase B.2.1 (2026-04-27) refactored _cleanup_module to
    dispatch into _kill_adopted_process / _kill_owned_process, then call
    _finalize_module_cleanup which holds the queue-cleanup invariant. The
    cleanup invariant is preserved by the chain — we now verify the chain
    by inspecting both functions.
    """
    from titan_plugin import guardian

    src_cleanup = inspect.getsource(guardian.Guardian._cleanup_module)
    src_finalize = inspect.getsource(guardian.Guardian._finalize_module_cleanup)

    # _cleanup_module must call _finalize_module_cleanup in every code path
    # (one call site is sufficient since both branches fall through to it)
    cleanup_code_only = "\n".join(
        line.split("#", 1)[0] for line in src_cleanup.splitlines()
    )
    assert "self._finalize_module_cleanup(info, name)" in cleanup_code_only, (
        "_cleanup_module must call _finalize_module_cleanup — otherwise the "
        "queue-cleanup invariant (cancel_join_thread + state reset) is bypassed"
    )

    # _finalize_module_cleanup must call cancel_join_thread on BOTH queues
    finalize_code_only = "\n".join(
        line.split("#", 1)[0] for line in src_finalize.splitlines()
    )
    assert "info.queue.cancel_join_thread()" in finalize_code_only, (
        "_finalize_module_cleanup must call info.queue.cancel_join_thread() — "
        "otherwise T1-2026-04-14 / I-018 deadlock returns"
    )
    assert "info.send_queue.cancel_join_thread()" in finalize_code_only, (
        "_finalize_module_cleanup must call info.send_queue.cancel_join_thread() — "
        "otherwise T1-2026-04-14 / I-018 deadlock returns"
    )

    # Must NOT call plain .join_thread() anywhere in the cleanup chain
    for src_name, code_only in (("_cleanup_module", cleanup_code_only),
                                 ("_finalize_module_cleanup", finalize_code_only)):
        residual = code_only.replace("cancel_join_thread()", "")
        assert ".join_thread()" not in residual, (
            f"{src_name} must NOT call plain .join_thread() — it deadlocks "
            "indefinitely when consumer is SIGKILL'd with phantom-FD pipes. "
            "Use cancel_join_thread() instead."
        )


if __name__ == "__main__":
    test_guardian_cleanup_uses_cancel_join_thread()
    print("OK")
