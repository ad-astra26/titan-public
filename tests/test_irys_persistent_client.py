"""Phase 5 chunk 5E — persistent Irys daemon client lifecycle tests.

These tests exercise the Python wrapper against a FAKE daemon (a small
Python subprocess that speaks the same JSONL protocol). Talking to the
real `scripts/irys_upload_daemon.js` would require a funded mainnet
keypair + Node.js + live network, none of which belong in a unit suite.

The daemon class accepts a `node_binary` override (Phase 5 5E
test-affordance addition) so the fake script runs under `sys.executable`
without process-table monkey-patches.

What's verified:
  - happy path (ready handshake → request/response → graceful shutdown)
  - cache reuse (second get_daemon for same key returns same instance)
  - error propagation (status=error → IrysDaemonError)
  - mid-flight daemon death surfaces as IrysDaemonError, not hang
  - per-op timeout
  - missing daemon script raises early on start
  - concurrent requests serialize correctly through one daemon
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from titan_hcl.utils.irys_persistent_client import (
    IrysDaemon,
    IrysDaemonError,
    get_daemon,
    _DAEMONS,
)


def _fake_body(script_body: str) -> str:
    """Wrap a body fragment into a full fake-daemon Python script."""
    boilerplate = textwrap.dedent('''
        import json
        import sys
        import os

        def emit(obj):
            sys.stdout.write(json.dumps(obj) + "\\n")
            sys.stdout.flush()

        emit({"status": "ok", "ready": True, "pid": os.getpid()})
    ''').lstrip("\n") + "\n"
    return boilerplate + script_body


@pytest.fixture
def fake_daemon(tmp_path):
    """Return a builder that produces (keypair_path, repo_root, script_rel)."""
    keypair = tmp_path / "keypair.json"
    keypair.write_text("[1,2,3,4]")  # dummy bytes — fake daemon doesn't read it
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()

    def _build(body: str) -> tuple[Path, Path, str]:
        # Name the file with a .py suffix so python parses it as a module —
        # the actual daemon class doesn't care about the extension; it
        # passes `script_rel` to the spawned binary.
        script = script_dir / "fake_irys_daemon.py"
        script.write_text(_fake_body(body))
        return keypair, tmp_path, "scripts/fake_irys_daemon.py"

    return _build


@pytest.fixture(autouse=True)
def _clear_cache():
    _DAEMONS.clear()
    yield
    for d in list(_DAEMONS.values()):
        proc = d._proc  # noqa: SLF001
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
    _DAEMONS.clear()


async def _spawn(fake_daemon, body: str) -> IrysDaemon:
    keypair, repo_root, script_rel = fake_daemon(body)
    daemon = IrysDaemon(
        str(keypair), "",
        repo_root=str(repo_root),
        node_binary=sys.executable,
        script_rel=script_rel,
    )
    await daemon.start()
    return daemon


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ready_handshake_and_shutdown(fake_daemon):
    body = textwrap.dedent('''
        for line in sys.stdin:
            req = json.loads(line)
            rid = req.get("id")
            if req["op"] == "shutdown":
                emit({"id": rid, "status": "ok", "bye": True})
                sys.exit(0)
            emit({"id": rid, "status": "ok", "echo": req["op"]})
    ''')
    daemon = await _spawn(fake_daemon, body)
    assert daemon.is_alive

    resp = await daemon.request("ready")
    assert resp["status"] == "ok"

    await daemon.shutdown()
    assert not daemon.is_alive


@pytest.mark.asyncio
async def test_request_propagates_error(fake_daemon):
    body = textwrap.dedent('''
        for line in sys.stdin:
            req = json.loads(line)
            rid = req.get("id")
            emit({"id": rid, "status": "error", "message": "boom"})
    ''')
    daemon = await _spawn(fake_daemon, body)
    try:
        with pytest.raises(IrysDaemonError, match="boom"):
            await daemon.request("upload_file", path="/tmp/x")
    finally:
        await daemon._kill_and_collect()


@pytest.mark.asyncio
async def test_requests_serialized_across_concurrent_callers(fake_daemon):
    body = textwrap.dedent('''
        for line in sys.stdin:
            req = json.loads(line)
            rid = req.get("id")
            emit({"id": rid, "status": "ok", "echoed_size": req.get("size_bytes", 0)})
    ''')
    daemon = await _spawn(fake_daemon, body)
    try:
        responses = await asyncio.gather(*[
            daemon.request("price", size_bytes=i) for i in range(5)
        ])
        echoed = sorted(r["echoed_size"] for r in responses)
        assert echoed == [0, 1, 2, 3, 4]
    finally:
        await daemon._kill_and_collect()


@pytest.mark.asyncio
async def test_dead_daemon_surfaces_as_error(fake_daemon):
    body = textwrap.dedent('''
        for line in sys.stdin:
            sys.exit(7)
    ''')
    daemon = await _spawn(fake_daemon, body)
    try:
        with pytest.raises(IrysDaemonError):
            await daemon.request("balance", timeout=3.0)
    finally:
        await daemon._kill_and_collect()


@pytest.mark.asyncio
async def test_op_timeout_raises(fake_daemon):
    body = textwrap.dedent('''
        for line in sys.stdin:
            pass  # read but never respond
    ''')
    daemon = await _spawn(fake_daemon, body)
    try:
        with pytest.raises(IrysDaemonError, match="timed out"):
            await daemon.request("balance", timeout=0.5)
    finally:
        await daemon._kill_and_collect()


@pytest.mark.asyncio
async def test_ready_handshake_missing_raises(tmp_path):
    keypair = tmp_path / "keypair.json"
    keypair.write_text("[1,2,3]")
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    # Daemon that exits BEFORE emitting the ready line
    script = script_dir / "fake_dead.py"
    script.write_text("import sys\nsys.exit(99)\n")
    daemon = IrysDaemon(
        str(keypair), "",
        repo_root=str(tmp_path),
        node_binary=sys.executable,
        script_rel="scripts/fake_dead.py",
    )
    with pytest.raises(IrysDaemonError):
        await daemon.start()


@pytest.mark.asyncio
async def test_get_daemon_caches_live_instance(fake_daemon):
    body = textwrap.dedent('''
        for line in sys.stdin:
            req = json.loads(line)
            rid = req.get("id")
            if req["op"] == "shutdown":
                emit({"id": rid, "status": "ok", "bye": True})
                sys.exit(0)
            emit({"id": rid, "status": "ok"})
    ''')
    keypair, repo_root, script_rel = fake_daemon(body)
    d1 = await get_daemon(
        str(keypair), "",
        repo_root=str(repo_root),
        node_binary=sys.executable,
        script_rel=script_rel,
    )
    d2 = await get_daemon(
        str(keypair), "",
        repo_root=str(repo_root),
        node_binary=sys.executable,
        script_rel=script_rel,
    )
    assert d1 is d2
    await d1._kill_and_collect()


@pytest.mark.asyncio
async def test_missing_script_raises_irys_error(tmp_path):
    keypair = tmp_path / "keypair.json"
    keypair.write_text("[1,2,3]")
    daemon = IrysDaemon(
        str(keypair), "", repo_root=str(tmp_path),
        node_binary=sys.executable,
        script_rel="scripts/nonexistent.py",
    )
    with pytest.raises(IrysDaemonError, match="daemon script missing"):
        await daemon.start()


@pytest.mark.asyncio
async def test_upload_file_marshals_response(fake_daemon, tmp_path):
    """upload_file → daemon op echoes a structured tx_id/url/size response."""
    body = textwrap.dedent('''
        for line in sys.stdin:
            req = json.loads(line)
            rid = req.get("id")
            assert req["op"] == "upload_file"
            assert req["path"]
            emit({
                "id": rid, "status": "ok",
                "tx_id": "FAKE_TX_42", "url": "https://arweave.net/FAKE_TX_42",
                "size": 1234,
            })
    ''')
    daemon = await _spawn(fake_daemon, body)
    payload = tmp_path / "blob.bin"
    payload.write_bytes(b"x" * 1234)
    try:
        result = await daemon.upload_file(str(payload), content_type="application/zstd")
        assert result.tx_id == "FAKE_TX_42"
        assert result.url == "https://arweave.net/FAKE_TX_42"
        assert result.size_bytes == 1234
    finally:
        await daemon._kill_and_collect()
