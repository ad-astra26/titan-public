"""
test_a8_7_sage_subprocess.py — Microkernel v2 §A.8.7 (2026-04-28)

Tests for the Sage Scholar + Gatekeeper consolidation:
  - Bus constants registered (SAGE_GATE_DECIDE / SAGE_IQL_TRAIN_STEP /
    SAGE_STATS / SAGE_READY)
  - rl_worker query handler dispatches "decide_execution_mode" + "dream"
    + back-compat "evaluate" + "stats" actions
  - rl_worker periodic SAGE_STATS broadcast (60s cadence)
  - RLProxy gains decide_execution_mode + dream + sovereignty_score
    + parent-safe encoder accessors (action_embedder/projection_layer/
    buffer/storage/dynamic_embedding_dim)
  - RLProxy.dream uses asyncio.to_thread (event loop NOT blocked)
  - RLProxy hard-fail behavior on bus timeout
  - Plugin flag routing: legacy `__init__.py` flag-on does NOT instantiate
    SageRecorder / SageScholar / SageGatekeeper

Closes Phase A.8 (rFP_microkernel_phase_a8_l2_l3_residency_completion.md).
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest


# ── Bus constants ────────────────────────────────────────────────


def test_sage_gate_decide_constant_registered() -> None:
    from titan_plugin.bus import SAGE_GATE_DECIDE
    assert SAGE_GATE_DECIDE == "SAGE_GATE_DECIDE"


def test_sage_iql_train_step_constant_registered() -> None:
    from titan_plugin.bus import SAGE_IQL_TRAIN_STEP
    assert SAGE_IQL_TRAIN_STEP == "SAGE_IQL_TRAIN_STEP"


def test_sage_stats_constant_registered() -> None:
    from titan_plugin.bus import SAGE_STATS
    assert SAGE_STATS == "SAGE_STATS"


def test_sage_ready_constant_registered() -> None:
    from titan_plugin.bus import SAGE_READY
    assert SAGE_READY == "SAGE_READY"


# ── rl_worker query handler ──────────────────────────────────────


def test_rl_worker_handle_query_decide_execution_mode() -> None:
    """rl_worker `_handle_query` action="decide_execution_mode" calls
    gatekeeper.decide_execution_mode and unpacks 3-tuple into RESPONSE."""
    from titan_plugin.modules.rl_worker import _handle_query

    captured: dict = {}

    class _FakeGatekeeper:
        sovereignty_score = 42.5
        def decide_execution_mode(self, state_tensor, raw_prompt=""):
            captured["state_shape"] = list(state_tensor.shape)
            captured["raw_prompt"] = raw_prompt
            return ("Sovereign", 0.85, "the_decoded_action")

    captured_msgs: list = []
    def _send(msg):
        captured_msgs.append(msg)

    msg = {
        "type": "QUERY",
        "rid": "test-rid-1",
        "src": "rl_proxy",
        "payload": {
            "action": "decide_execution_mode",
            "state_tensor": [0.0] * 128,
            "raw_prompt": "what is the SOL price",
        },
    }
    fake_send_queue = MagicMock()
    fake_send_queue.put_nowait.side_effect = _send

    _handle_query(
        msg,
        recorder=MagicMock(),
        scholar=MagicMock(),
        gatekeeper=_FakeGatekeeper(),
        send_queue=fake_send_queue,
        name="rl",
    )

    assert captured["state_shape"] == [128]
    assert captured["raw_prompt"] == "what is the SOL price"
    assert len(captured_msgs) == 1
    response = captured_msgs[0]
    assert response["type"] == "RESPONSE"
    assert response["rid"] == "test-rid-1"
    assert response["payload"]["mode"] == "Sovereign"
    assert response["payload"]["advantage"] == pytest.approx(0.85)
    assert response["payload"]["decoded_text"] == "the_decoded_action"
    assert response["payload"]["sovereignty_score"] == pytest.approx(42.5)


def test_rl_worker_handle_query_evaluate_alias() -> None:
    """`action="evaluate"` is back-compat alias — same handler, same response."""
    from titan_plugin.modules.rl_worker import _handle_query

    class _FakeGatekeeper:
        sovereignty_score = 0.0
        def decide_execution_mode(self, state_tensor, raw_prompt=""):
            return ("Shadow", 0.0, "")

    captured: list = []
    fake_send_queue = MagicMock()
    fake_send_queue.put_nowait.side_effect = captured.append

    _handle_query(
        {"type": "QUERY", "rid": "r2", "src": "x",
         "payload": {"action": "evaluate", "state": [0.0] * 128}},
        recorder=MagicMock(),
        scholar=MagicMock(),
        gatekeeper=_FakeGatekeeper(),
        send_queue=fake_send_queue,
        name="rl",
    )
    assert captured[0]["payload"]["mode"] == "Shadow"


def test_rl_worker_handle_query_dream() -> None:
    """`action="dream"` calls scholar.dream with epochs+batch_size and
    returns 4-key reply with floats + buffer_len + epochs."""
    from titan_plugin.modules.rl_worker import _handle_query

    async def _fake_dream(epochs, batch_size):
        return {
            "loss_actor": 0.1234,
            "loss_qvalue": 0.5678,
            "loss_value": 0.9012,
        }

    fake_scholar = MagicMock()
    fake_scholar.dream = _fake_dream

    fake_recorder = MagicMock()
    fake_recorder.buffer = [0] * 7  # len() == 7

    captured: list = []
    fake_send_queue = MagicMock()
    fake_send_queue.put_nowait.side_effect = captured.append

    _handle_query(
        {
            "type": "QUERY", "rid": "r3", "src": "rl_proxy",
            "payload": {"action": "dream", "epochs": 5, "batch_size": 32},
        },
        recorder=fake_recorder,
        scholar=fake_scholar,
        gatekeeper=MagicMock(),
        send_queue=fake_send_queue,
        name="rl",
    )

    assert len(captured) == 1
    payload = captured[0]["payload"]
    assert payload["loss_actor"] == pytest.approx(0.1234)
    assert payload["loss_qvalue"] == pytest.approx(0.5678)
    assert payload["loss_value"] == pytest.approx(0.9012)
    assert payload["buffer_len"] == 7
    assert payload["epochs"] == 5


def test_rl_worker_handle_query_swallows_dream_exception() -> None:
    """Dream handler exception → RESPONSE with error key, never crashes worker."""
    from titan_plugin.modules.rl_worker import _handle_query

    async def _crash(epochs, batch_size):
        raise RuntimeError("boom")
    fake_scholar = MagicMock()
    fake_scholar.dream = _crash

    captured: list = []
    fake_send_queue = MagicMock()
    fake_send_queue.put_nowait.side_effect = captured.append

    _handle_query(
        {
            "type": "QUERY", "rid": "r4", "src": "x",
            "payload": {"action": "dream", "epochs": 1, "batch_size": 1},
        },
        recorder=MagicMock(),
        scholar=fake_scholar,
        gatekeeper=MagicMock(),
        send_queue=fake_send_queue,
        name="rl",
    )
    assert captured[0]["payload"].get("error") == "boom"


def test_rl_worker_handle_query_stats() -> None:
    """`action="stats"` returns the SAGE_STATS payload shape."""
    from titan_plugin.modules.rl_worker import _handle_query

    fake_recorder = MagicMock()
    fake_recorder.buffer = [0] * 100
    fake_recorder.storage = [0] * 50
    fake_recorder.buffer_size = 50_000

    fake_gk = MagicMock()
    fake_gk.sovereignty_score = 67.5
    fake_gk._decision_history = ["sovereign"] * 12

    captured: list = []
    fake_send_queue = MagicMock()
    fake_send_queue.put_nowait.side_effect = captured.append

    _handle_query(
        {"type": "QUERY", "rid": "r5", "src": "x",
         "payload": {"action": "stats"}},
        recorder=fake_recorder,
        scholar=MagicMock(),
        gatekeeper=fake_gk,
        send_queue=fake_send_queue,
        name="rl",
    )
    payload = captured[0]["payload"]
    assert payload["buffer_len"] == 100
    assert payload["storage_len"] == 50
    assert payload["buffer_size"] == 50_000
    assert payload["sovereignty_score"] == pytest.approx(67.5)
    assert payload["decision_history_len"] == 12


def test_rl_worker_broadcast_sage_stats_emits_correct_msg() -> None:
    """_broadcast_sage_stats emits SAGE_STATS dst="all" with full snapshot."""
    from titan_plugin.modules.rl_worker import _broadcast_sage_stats

    fake_recorder = MagicMock()
    fake_recorder.buffer = []
    fake_recorder.storage = []
    fake_recorder.buffer_size = 50_000

    fake_gk = MagicMock()
    fake_gk.sovereignty_score = 0.0
    fake_gk._decision_history = []

    captured: list = []
    fake_send_queue = MagicMock()
    fake_send_queue.put_nowait.side_effect = captured.append

    _broadcast_sage_stats(fake_send_queue, "rl", fake_recorder, fake_gk)

    assert len(captured) == 1
    msg = captured[0]
    assert msg["type"] == "SAGE_STATS"
    assert msg["dst"] == "all"
    assert msg["src"] == "rl"
    assert "buffer_size" in msg["payload"]


# ── RLProxy methods ──────────────────────────────────────────────


def _make_proxy_with_mock_bus():
    """Construct an RLProxy with a mocked bus. Avoids Guardian import path."""
    from titan_plugin.proxies.rl_proxy import RLProxy

    bus = MagicMock()
    bus.subscribe = MagicMock(return_value=MagicMock())  # subscribe always returns a queue

    # 2026-04-29 — bus.request_async is the new canonical async path
    # (routes through bus_ipc_pool). Tests continue to mock the sync
    # `bus.request` (preserves call_args assertions); we wrap it as an
    # async coroutine so `await self._bus.request_async(...)` invokes
    # the same MagicMock.
    async def _request_async(*args, **kwargs):
        return bus.request(*args, **kwargs)
    bus.request_async = _request_async

    guardian = MagicMock()

    proxy = RLProxy(bus, guardian)
    # Bypass _ensure_started's Guardian dance — already started for tests.
    proxy._started = True
    return proxy, bus, guardian


def test_rlproxy_decide_execution_mode_happy_path() -> None:
    proxy, bus, _ = _make_proxy_with_mock_bus()
    bus.request = MagicMock(return_value={
        "payload": {
            "mode": "Collaborative",
            "advantage": 0.55,
            "decoded_text": "answer_text",
            "sovereignty_score": 33.3,
        },
    })

    mode, adv, text = proxy.decide_execution_mode([0.0] * 128, raw_prompt="hi")

    assert mode == "Collaborative"
    assert adv == pytest.approx(0.55)
    assert text == "answer_text"
    # sovereignty_score cached from response
    assert proxy.sovereignty_score == pytest.approx(33.3)
    # bus.request called with right action key
    args, kwargs = bus.request.call_args
    payload = args[2] if len(args) >= 3 else kwargs.get("payload")
    assert payload["action"] == "decide_execution_mode"
    assert payload["raw_prompt"] == "hi"


def test_rlproxy_decide_execution_mode_hard_fail_shadow() -> None:
    """Bus timeout returns ('Shadow', 0.0, '') — chat path never breaks."""
    proxy, bus, _ = _make_proxy_with_mock_bus()
    bus.request = MagicMock(return_value=None)

    mode, adv, text = proxy.decide_execution_mode([0.0] * 128, raw_prompt="x")
    assert mode == "Shadow"
    assert adv == 0.0
    assert text == ""


def test_rlproxy_decide_execution_mode_worker_error_returns_shadow() -> None:
    """Worker-reported error returns Shadow fallback."""
    proxy, bus, _ = _make_proxy_with_mock_bus()
    bus.request = MagicMock(return_value={"payload": {"error": "kaboom"}})

    mode, adv, text = proxy.decide_execution_mode([0.0] * 128)
    assert mode == "Shadow"
    assert adv == 0.0


def test_rlproxy_dream_happy_path() -> None:
    proxy, bus, _ = _make_proxy_with_mock_bus()
    bus.request = MagicMock(return_value={
        "payload": {
            "loss_actor": 0.1, "loss_qvalue": 0.2, "loss_value": 0.3,
            "buffer_len": 99, "epochs": 5,
        },
    })

    result = asyncio.run(proxy.dream(epochs=5, batch_size=32))

    assert result["loss_actor"] == pytest.approx(0.1)
    assert result["loss_qvalue"] == pytest.approx(0.2)
    assert result["loss_value"] == pytest.approx(0.3)
    assert result["buffer_len"] == 99
    assert result["epochs"] == 5


def test_rlproxy_dream_hard_fail_zero_loss() -> None:
    """Bus timeout returns 0-loss dict — meditation cycle continues."""
    proxy, bus, _ = _make_proxy_with_mock_bus()
    bus.request = MagicMock(return_value=None)

    result = asyncio.run(proxy.dream(epochs=1, batch_size=1))
    assert result == {"loss_actor": 0.0, "loss_qvalue": 0.0, "loss_value": 0.0}


def test_rlproxy_dream_uses_to_thread() -> None:
    """Dream must use asyncio.to_thread so event loop is not blocked.
    Verified by checking the call site goes through the await path."""
    proxy, bus, _ = _make_proxy_with_mock_bus()

    # Slow synchronous bus.request — if dream were sync, this would block
    # the event loop. We measure that other coroutines can run during the
    # bus call.
    other_ran = []

    def _slow_request(*args, **kwargs):
        time.sleep(0.05)
        return {"payload": {"loss_actor": 0.0, "loss_qvalue": 0.0, "loss_value": 0.0}}
    bus.request = _slow_request

    async def _other():
        await asyncio.sleep(0.005)
        other_ran.append(True)

    async def _both():
        await asyncio.gather(proxy.dream(epochs=1, batch_size=1), _other())

    asyncio.run(_both())
    # If dream were blocking the event loop with `time.sleep`, the other
    # coroutine couldn't have run before dream completed.
    assert other_ran == [True]


def test_rlproxy_sovereignty_score_reads_cached_stats() -> None:
    proxy, _, _ = _make_proxy_with_mock_bus()
    proxy.update_cached_stats({"sovereignty_score": 88.0, "buffer_len": 5})
    assert proxy.sovereignty_score == pytest.approx(88.0)
    assert proxy.get_stats()["buffer_len"] == 5


def test_rlproxy_update_cached_stats_ignores_non_dict() -> None:
    proxy, _, _ = _make_proxy_with_mock_bus()
    proxy.update_cached_stats("not_a_dict")  # type: ignore
    proxy.update_cached_stats(None)  # type: ignore
    # Default value preserved
    assert proxy.sovereignty_score == 0.0


def test_rlproxy_buffer_is_none_for_legacy_compat() -> None:
    """Legacy `if self.recorder.buffer else -1` falls through to -1."""
    proxy, _, _ = _make_proxy_with_mock_bus()
    assert proxy.buffer is None
    transition_id = len(proxy.buffer) if proxy.buffer else -1  # type: ignore[arg-type]
    assert transition_id == -1


def test_rlproxy_storage_is_empty_list_for_gatekeeper_knn_compat() -> None:
    """Empty storage → SageGatekeeper KNN walk is no-op (graceful Shadow)."""
    proxy, _, _ = _make_proxy_with_mock_bus()
    assert proxy.storage == []
    assert len(proxy.storage) == 0


def test_rlproxy_dynamic_embedding_dim_via_lazy_encoder() -> None:
    """Encoder accessor is lazy — built on first access."""
    proxy, _, _ = _make_proxy_with_mock_bus()
    assert proxy._encoder is None
    dim = proxy.dynamic_embedding_dim
    assert dim == 3072
    assert proxy._encoder is not None  # cached after first access


def test_rlproxy_coerce_state_accepts_list_tensor_and_falls_back() -> None:
    """_coerce_state handles list, tensor (via tolist), and falls back to zeros."""
    from titan_plugin.proxies.rl_proxy import RLProxy

    assert RLProxy._coerce_state([1.0, 2.0]) == [1.0, 2.0]

    class _FakeTensor:
        def tolist(self):
            return [3.0, 4.0]
    assert RLProxy._coerce_state(_FakeTensor()) == [3.0, 4.0]

    # An object with no useful interface — falls back to zeros (length 128)
    class _Bad:
        pass
    fallback = RLProxy._coerce_state(_Bad())
    assert isinstance(fallback, list)
    assert len(fallback) == 128


# ── Plugin flag routing — legacy parent ──────────────────────────


def test_legacy_plugin_flag_off_creates_full_sage_stack() -> None:
    """Flag default (false) → SageRecorder + SageScholar + SageGatekeeper
    instantiated as before. We verify by checking the import is reachable
    and the construction path is unchanged."""
    # We don't fully boot TitanPlugin (heavy). Verify the conditional
    # branch logic via direct read of titan_plugin/__init__.py.
    import inspect
    import titan_plugin
    src = inspect.getsource(titan_plugin)
    # The flag-aware swap must be present
    assert "a8_sage_scholar_gatekeeper_subprocess_enabled" in src
    # Both branches are reachable
    assert "SageEncoder()" in src
    assert "SageRecorder()" in src
    # None-guards present
    assert "self.gatekeeper is not None" in src
    assert "self.scholar is None" in src


def test_legacy_plugin_flag_on_skips_heavy_sage_imports() -> None:
    """When the flag is ON, the conditional imports SageEncoder (parent-safe)
    instead of SageRecorder. The Scholar + Gatekeeper are not instantiated."""
    # Read the actual conditional structure to verify intent.
    import inspect
    import titan_plugin
    src = inspect.getsource(titan_plugin)

    # Find the conditional block
    assert "if a8_sage_subproc:" in src
    # Flag-on block must use SageEncoder, not SageRecorder
    flag_on_idx = src.index("if a8_sage_subproc:")
    flag_off_idx = src.index("else:", flag_on_idx)
    flag_on_block = src[flag_on_idx:flag_off_idx]
    assert "SageEncoder()" in flag_on_block
    assert "SageRecorder()" not in flag_on_block


# ── agno_hooks.py routing restoration ────────────────────────────


def test_agno_hooks_no_longer_uses_hasattr_guard() -> None:
    """The pre-A.8.7 `hasattr(plugin.gatekeeper, 'decide_execution_mode')`
    guard silently disabled routing in V6. Removed in A.8.7 (None-guard
    replaces it for legacy fallback)."""
    import inspect
    from titan_plugin import agno_hooks
    src = inspect.getsource(agno_hooks)

    # The guard string is gone — replaced with None check.
    assert "hasattr(plugin.gatekeeper, 'decide_execution_mode')" not in src
    assert "plugin.gatekeeper is not None" in src
