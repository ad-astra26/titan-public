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
    """Static check: guardian._cleanup_module must use cancel_join_thread()
    and must NOT call plain join_thread(). Guards against accidental revert
    of the T1-2026-04-14 / I-018 deadlock fix."""
    from titan_plugin import guardian

    src = inspect.getsource(guardian.Guardian._cleanup_module)

    # Strip line comments before checking so references to 'join_thread()'
    # in docstrings/comments don't create false positives.
    code_only = "\n".join(
        line.split("#", 1)[0] for line in src.splitlines()
    )

    # Must call cancel_join_thread on BOTH queues (info.queue, info.send_queue)
    assert "info.queue.cancel_join_thread()" in code_only, (
        "_cleanup_module must call info.queue.cancel_join_thread() — "
        "otherwise T1-2026-04-14 / I-018 deadlock returns"
    )
    assert "info.send_queue.cancel_join_thread()" in code_only, (
        "_cleanup_module must call info.send_queue.cancel_join_thread() — "
        "otherwise T1-2026-04-14 / I-018 deadlock returns"
    )

    # Must NOT call plain .join_thread() anywhere (only cancel_join_thread)
    residual = code_only.replace("cancel_join_thread()", "")
    assert ".join_thread()" not in residual, (
        "_cleanup_module must NOT call plain .join_thread() — it deadlocks "
        "indefinitely when consumer is SIGKILL'd with phantom-FD pipes. "
        "Use cancel_join_thread() instead."
    )


if __name__ == "__main__":
    test_guardian_cleanup_uses_cancel_join_thread()
    print("OK")
