"""
tests/integration/test_chat_endpoint_e2e.py — D-SPEC-73 test-gap remediation.

Why this file exists: D-SPEC-72 shipped with 343/343 unit tests green but
/chat returned 500 on every Titan because the AgnoProxy code path called
`bus.subscribe` via kernel_rpc — which is NOT in EXPOSED_METHODS in the
api_subprocess context. Unit tests mocked the bus; no test exercised the
real wire path. This file fills that gap.

Approach: spin up the FastAPI app via api/__init__.create_app() with:
  - a real AgnoBridgeClient backed by a Queue (no kernel_rpc mock; no real
    kernel subprocess; the bridge writes onto a Queue we drain)
  - a stub agno_worker (Python thread) that:
      • drains the Queue for CHAT_REQUEST messages
      • emits CHAT_RESPONSE back into a recv-side queue + invokes
        bridge.handle_response() (mimics _bus_listener_loop dispatch)
  - a TestClient (httpx ASGITransport) hitting POST /chat

What this catches:
  - AgnoProxy XOR validation enforced when constructed by create_app
  - send_queue → bridge dispatch path
  - bridge → AgnoProxy → /chat JSONResponse end-to-end shape
  - X-Titan-* headers from ovg_data plumbed through

What this does NOT cover (deferred to live runtime tests):
  - Real kernel_rpc Unix socket round-trip
  - Real Rust bus broker routing
  - Real agno_worker process spawn
"""
from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Optional

import pytest

from titan_hcl import bus


# ────────────────────────────────────────────────────────────────────────
# Stub agno_worker — pumps CHAT_REQUEST → CHAT_RESPONSE on a thread
# ────────────────────────────────────────────────────────────────────────

class _StubAgnoWorker:
    """Thread that drains a send_queue + dispatches synthetic CHAT_RESPONSE
    back through the bridge's handle_response. Default reply is a canned
    OK; tests can override `next_reply()` for safety-violation / error
    paths.
    """

    def __init__(self, send_queue: Queue, bridge, reply_delay_s: float = 0.05):
        self._send_queue = send_queue
        self._bridge = bridge
        self._delay = reply_delay_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="stub-agno-worker")
        self.received_requests: list = []
        self._next_payload: Optional[dict] = None

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def next_payload(self, payload: dict):
        """Set the reply payload for the NEXT CHAT_REQUEST seen."""
        self._next_payload = payload

    def _run(self):
        while not self._stop.is_set():
            try:
                msg = self._send_queue.get(timeout=0.1)
            except Empty:
                continue
            self.received_requests.append(msg)
            mt = msg.get("type")
            if mt == bus.CHAT_REQUEST:
                time.sleep(self._delay)
                payload = self._next_payload or {
                    "response": "Hello, world. I am Titan.",
                    "session_id": (msg.get("payload") or {}).get(
                        "session_id", "default"),
                    "mode": "chat",
                    "mood": "calm",
                    "ovg_data": {
                        "verified": True,
                        "block_height": 42,
                        "merkle_root": "test-root",
                        "signature": "test-sig",
                    },
                }
                self._next_payload = None
                reply = {
                    "type": bus.CHAT_RESPONSE,
                    "src": "agno_worker",
                    "dst": "api",
                    "rid": msg.get("rid"),
                    "payload": payload,
                    "ts": time.time(),
                }
                # Simulate the api_subprocess _bus_listener_loop dispatch
                self._bridge.handle_response(reply)
            elif mt == bus.CHAT_STREAM_REQUEST:
                rid = msg.get("rid")
                pid = (msg.get("payload") or {}).get("request_id")
                pieces = ["Hello", ", ", "I am ", "Titan."]
                time.sleep(self._delay)
                for i, p in enumerate(pieces):
                    chunk_msg = {
                        "type": bus.CHAT_STREAM_CHUNK,
                        "src": "agno_worker", "dst": "api",
                        "payload": {
                            "request_id": pid,
                            "chunk": p,
                            "done": False,
                        },
                        "ts": time.time(),
                    }
                    self._bridge.handle_response(chunk_msg)
                # Final done=true with ovg headers
                self._bridge.handle_response({
                    "type": bus.CHAT_STREAM_CHUNK,
                    "src": "agno_worker", "dst": "api",
                    "payload": {
                        "request_id": pid, "chunk": "", "done": True,
                        "ovg_headers": {"X-Titan-Verified": "true"},
                    },
                    "ts": time.time(),
                })


# ────────────────────────────────────────────────────────────────────────
# FastAPI fixture — builds the real app with a bridge-backed AgnoProxy
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture
def chat_e2e_app():
    """Yields (TestClient, stub_worker). Cleans up on exit."""
    from titan_hcl.api.agno_bridge_client import AgnoBridgeClient
    from titan_hcl.api.events import EventBus

    # Build the bridge against a real Queue (mp.Queue-shaped is fine)
    send_queue: Queue = Queue(maxsize=256)
    bridge = AgnoBridgeClient(send_queue=send_queue,
                              request_timeout_s=3.0, name="api")

    # Stub agno_worker drains send_queue + dispatches replies via bridge
    stub = _StubAgnoWorker(send_queue, bridge)
    stub.start()

    # Minimal `plugin` stub — create_app only touches a few attrs in the
    # paths we exercise; the chat path goes through bridge, not plugin.
    # _full_config must include [api].internal_key so X-Titan-Internal-Key
    # auth bypass works for the test client.
    _internal_key = _load_internal_key()
    if not _internal_key:
        pytest.skip("No [api].internal_key in titan_hcl/config.toml; "
                    "skipping integration test")

    class _PluginStub:
        _proxies = {}
        bus = None  # AgnoProxy install picks bridge path
        _full_config = {"api": {"internal_key": _internal_key}}

        def __getattr__(self, name):
            return None

    # Build the app
    from titan_hcl.api import create_app
    app = create_app(
        plugin=_PluginStub(),
        event_bus=EventBus(),
        config={"api": {"port": 7777, "host": "127.0.0.1"}},
        agent=None,
        titan_state=None,
        chat_bridge_bus=None,
        agno_bridge=bridge,
    )

    # Build a TestClient
    from fastapi.testclient import TestClient
    client = TestClient(app, raise_server_exceptions=False)

    try:
        yield client, stub, bridge
    finally:
        stub.stop()


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────

def test_chat_endpoint_returns_200_via_bridge(chat_e2e_app):
    """POST /chat → 200 with canned reply from stub worker.

    This is THE test that should have caught CHAT-500 before deploy.
    """
    client, stub, _ = chat_e2e_app
    headers = _internal_key_headers()
    resp = client.post("/chat",
                       json={"message": "ping", "session_id": "test-s1"},
                       headers=headers)
    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text[:500]}"
    )
    body = resp.json()
    # Standard ChatResponse shape
    assert "response" in body
    assert body["response"] == "Hello, world. I am Titan."
    assert body["session_id"] == "test-s1"
    # Verify the stub actually received the bridge-routed CHAT_REQUEST
    assert any(r.get("type") == bus.CHAT_REQUEST
               for r in stub.received_requests)


def test_chat_endpoint_propagates_ovg_headers(chat_e2e_app):
    """ovg_data on the CHAT_RESPONSE.payload → X-Titan-* response headers."""
    client, _, _ = chat_e2e_app
    headers = _internal_key_headers()
    resp = client.post("/chat",
                       json={"message": "hi", "session_id": "x-titan"},
                       headers=headers)
    assert resp.status_code == 200
    # AgnoBridgeClient._build_ovg_headers maps ovg_data → X-Titan-*
    assert resp.headers.get("X-Titan-Verified") == "true"
    assert resp.headers.get("X-Titan-Block-Height") == "42"
    assert resp.headers.get("X-Titan-Merkle-Root") == "test-root"
    assert resp.headers.get("X-Titan-Signature") == "test-sig"


def test_chat_endpoint_returns_500_on_agno_worker_error(chat_e2e_app):
    """If agno_worker returns {error: ...}, /chat returns 500 envelope."""
    client, stub, _ = chat_e2e_app
    stub.next_payload({"error": "agent_arun_failed"})
    headers = _internal_key_headers()
    resp = client.post("/chat",
                       json={"message": "buggy", "session_id": "err"},
                       headers=headers)
    assert resp.status_code == 500
    body = resp.json()
    assert body.get("error") == "agno_worker_error"
    assert "agent_arun_failed" in body.get("detail", "")


def test_chat_endpoint_no_500_on_subscribe_path(chat_e2e_app):
    """Regression guard for CHAT-500: bus.subscribe must NEVER be called by
    the bridge backend. We verify by inspecting the AgnoProxy attached to
    app.state — it must NOT have a `_reply_queue` attribute (deleted in
    D-SPEC-73 per no-shim rule)."""
    client, _, _ = chat_e2e_app
    # AgnoProxy was installed at create_app time with bridge backend.
    # The stripped _reply_queue / _ensure_reply_queue means there's no path
    # to bus.subscribe through this proxy.
    app = client.app
    proxy = getattr(app.state, "agno_proxy", None)
    assert proxy is not None, "AgnoProxy should be installed by create_app"
    assert not hasattr(proxy, "_reply_queue"), (
        "_reply_queue must be deleted per D-SPEC-73 (no shim)"
    )
    assert not hasattr(proxy, "_ensure_reply_queue"), (
        "_ensure_reply_queue method must be deleted per D-SPEC-73"
    )
    # And bridge backend was selected:
    assert proxy._bridge is not None
    assert proxy._bus is None


def test_chat_endpoint_timeout_returns_504(chat_e2e_app):
    """If stub worker is slow / dead, bridge timeout returns 504 envelope."""
    client, stub, _ = chat_e2e_app
    # Make stub effectively dead by clearing its queue with sleep > timeout
    stub.stop()
    headers = _internal_key_headers()
    resp = client.post("/chat",
                       json={"message": "no answer", "session_id": "tmo"},
                       headers=headers)
    # Bridge times out after 3.0s; chat.py returns the 504 envelope as JSON
    assert resp.status_code == 504
    body = resp.json()
    assert body.get("error") == "agno_timeout"


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _load_internal_key() -> str:
    """Read [api].internal_key from titan_hcl/config.toml.

    The /chat endpoint accepts X-Titan-Internal-Key as a Privy auth bypass
    when the header value matches `plugin._full_config["api"]["internal_key"]`
    (titan_hcl/api/auth.py:80). Returns "" if not configured (test skips).
    """
    try:
        import tomllib
        with open("titan_hcl/config.toml", "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("api", {}).get("internal_key", "") or \
               cfg.get("stealth_sage", {}).get("internal_key", "")
    except Exception:
        return ""


def _internal_key_headers() -> dict:
    """Headers for /chat auth bypass via internal-key (matches the key
    wired into the _PluginStub._full_config above)."""
    key = _load_internal_key()
    if not key:
        pytest.skip("No internal-key configured; skipping integration test")
    return {
        "X-Titan-Internal-Key": key,
        "X-Titan-User-Id": "maker",
    }
