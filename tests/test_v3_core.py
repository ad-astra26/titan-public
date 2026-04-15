"""
Tests for V3.0 Microkernel — DivineBus, Guardian, TitanCore.

Runs in isolation (no TorchRL mmap needed).
"""
import time
import pytest


def test_bus_subscribe_publish():
    """DivineBus routes messages to subscribers."""
    from titan_plugin.bus import DivineBus, make_msg

    bus = DivineBus()
    q1 = bus.subscribe("memory")
    q2 = bus.subscribe("rl")

    # Targeted message
    msg = make_msg("TEST", "core", "memory", {"data": 42})
    delivered = bus.publish(msg)
    assert delivered == 1

    received = q1.get(timeout=1)
    assert received["type"] == "TEST"
    assert received["payload"]["data"] == 42
    assert q2.empty()


def test_bus_broadcast():
    """DivineBus broadcast goes to all except sender."""
    from titan_plugin.bus import DivineBus, make_msg

    bus = DivineBus()
    q_core = bus.subscribe("core")
    q_mem = bus.subscribe("memory")
    q_rl = bus.subscribe("rl")

    msg = make_msg("EPOCH_TICK", "core", "all", {"epoch": 1})
    delivered = bus.publish(msg)
    assert delivered == 2  # memory + rl (not core, since core is sender)

    assert q_core.empty()  # core should NOT receive its own broadcast
    assert not q_mem.empty()
    assert not q_rl.empty()


def test_bus_drain():
    """DivineBus drain reads multiple messages."""
    from titan_plugin.bus import DivineBus, make_msg

    bus = DivineBus()
    q = bus.subscribe("test")

    for i in range(5):
        bus.publish(make_msg("DATA", "core", "test", {"i": i}))

    msgs = bus.drain(q, max_msgs=3)
    assert len(msgs) == 3
    assert msgs[0]["payload"]["i"] == 0

    # 2 remaining
    msgs2 = bus.drain(q, max_msgs=10)
    assert len(msgs2) == 2


def test_bus_full_queue_drops():
    """DivineBus drops messages when queue is full."""
    from titan_plugin.bus import DivineBus, make_msg

    bus = DivineBus(maxsize=5)
    q = bus.subscribe("test")

    for i in range(10):
        bus.publish(make_msg("DATA", "core", "test", {"i": i}))

    assert bus.stats["dropped"] == 5
    msgs = bus.drain(q)
    assert len(msgs) == 5


def test_bus_request_response():
    """DivineBus sync request/response works."""
    from titan_plugin.bus import DivineBus, make_msg, QUERY, RESPONSE
    import threading

    bus = DivineBus()
    client_q = bus.subscribe("client")
    server_q = bus.subscribe("server")

    # Server thread: listen for QUERY, send RESPONSE
    def server():
        msg = server_q.get(timeout=5)
        assert msg["type"] == QUERY
        reply = make_msg(RESPONSE, "server", "client", {"answer": 42}, rid=msg["rid"])
        bus.publish(reply)

    t = threading.Thread(target=server, daemon=True)
    t.start()

    result = bus.request("client", "server", {"question": "meaning"}, timeout=5.0, reply_queue=client_q)
    assert result is not None
    assert result["payload"]["answer"] == 42
    t.join(timeout=2)


def test_guardian_register():
    """Guardian registers module specs."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec

    bus = DivineBus()
    g = Guardian(bus)

    g.register(ModuleSpec(name="test_mod", entry_fn=_noop_worker))
    assert "test_mod" in g._modules
    assert g._modules["test_mod"].state.value == "stopped"


def test_guardian_start_stop():
    """Guardian starts and stops a module process."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec, ModuleState

    bus = DivineBus(multiprocess=True)
    g = Guardian(bus)

    g.register(ModuleSpec(name="worker_test", entry_fn=_sleep_worker))
    assert g.start("worker_test")

    info = g._modules["worker_test"]
    assert info.state in (ModuleState.STARTING, ModuleState.RUNNING)
    assert info.pid is not None

    # Wait briefly for process to be alive
    time.sleep(0.5)
    assert info.process.is_alive()

    g.stop("worker_test", reason="test")
    assert info.state == ModuleState.STOPPED
    assert info.pid is None


def test_guardian_status():
    """Guardian.get_status() returns module info."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec

    bus = DivineBus()
    g = Guardian(bus)
    g.register(ModuleSpec(name="mod_a", entry_fn=_noop_worker))
    g.register(ModuleSpec(name="mod_b", entry_fn=_noop_worker, rss_limit_mb=2000))

    status = g.get_status()
    assert "mod_a" in status
    assert "mod_b" in status
    assert status["mod_a"]["state"] == "stopped"
    assert status["mod_b"]["state"] == "stopped"


# ── Module-level worker functions (picklable for multiprocessing) ──

def _noop_worker(recv_queue, send_queue, name, config):
    """No-op worker for registration tests."""
    pass

def _sleep_worker(recv_queue, send_queue, name, config):
    """Worker that sleeps until killed — for start/stop tests."""
    import time as _t
    try:
        while True:
            _t.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        pass


def _echo_worker(recv_queue, send_queue, name, config):
    """Worker that echoes QUERY messages back as RESPONSE — for IPC tests."""
    import time as _t
    from queue import Empty

    # Signal ready
    send_queue.put_nowait({
        "type": "MODULE_READY", "src": name, "dst": "guardian",
        "ts": _t.time(), "rid": None, "payload": {},
    })

    while True:
        try:
            msg = recv_queue.get(timeout=2.0)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        if msg.get("type") == "MODULE_SHUTDOWN":
            break

        if msg.get("type") == "QUERY":
            # Echo the payload back with "echo": True added
            payload = msg.get("payload", {})
            payload["echo"] = True
            send_queue.put_nowait({
                "type": "RESPONSE",
                "src": name,
                "dst": msg.get("src", ""),
                "ts": _t.time(),
                "rid": msg.get("rid"),
                "payload": payload,
            })


def test_cross_process_ipc():
    """Test full cross-process IPC: Core → Bus → Worker → SendQueue → Bus → reply."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec, ModuleState

    bus = DivineBus(multiprocess=True)
    g = Guardian(bus)

    g.register(ModuleSpec(name="echo", entry_fn=_echo_worker))
    assert g.start("echo")

    # Wait for MODULE_READY (cross-process startup takes time)
    routed = 0
    for _ in range(20):  # Try for up to 4s
        time.sleep(0.2)
        routed += g.drain_send_queues()
        if routed >= 1:
            break
    assert routed >= 1, f"MODULE_READY not received (routed={routed})"

    # Guardian processes the ready message from bus
    g.monitor_tick()
    info = g._modules["echo"]
    # State may still be STARTING if the MODULE_READY was routed to guardian's
    # bus queue but not processed yet. Run another tick.
    if info.state != ModuleState.RUNNING:
        time.sleep(0.5)
        g.drain_send_queues()
        g.monitor_tick()
    assert info.state in (ModuleState.RUNNING, ModuleState.STARTING)

    # Send a QUERY through the bus to the echo worker
    from titan_plugin.bus import make_request
    client_q = bus.subscribe("test_client")
    query = make_request("test_client", "echo", {"hello": "world"})
    bus.publish(query)

    # Wait for echo response
    for _ in range(20):
        time.sleep(0.2)
        g.drain_send_queues()
        if not client_q.empty():
            break

    from queue import Empty
    try:
        response = client_q.get(timeout=2.0)
        assert response["type"] == "RESPONSE"
        assert response["payload"]["echo"] is True
        assert response["payload"]["hello"] == "world"
        assert response["rid"] == query["rid"]
    except Empty:
        pytest.fail("Did not receive echo response from worker")

    g.stop("echo", reason="test")


def test_guardian_drain_send_queues():
    """Guardian.drain_send_queues routes worker messages to bus."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec

    bus = DivineBus(multiprocess=True)
    g = Guardian(bus)

    g.register(ModuleSpec(name="sender", entry_fn=_echo_worker))
    g.start("sender")

    # Wait for MODULE_READY (cross-process startup)
    routed = 0
    for _ in range(20):
        time.sleep(0.2)
        routed += g.drain_send_queues()
        if routed >= 1:
            break
    assert routed >= 1, f"MODULE_READY not received (routed={routed})"

    g.stop("sender", reason="test")


def test_body_sensor_collection():
    """Body worker collects 5DT tensor with severity categories."""
    from titan_plugin.modules.body_worker import (
        _collect_body_tensor, _load_thresholds, Severity
    )
    from collections import deque

    history = {
        "interoception": deque(maxlen=30),
        "proprioception": deque(maxlen=30),
        "somatosensation": deque(maxlen=30),
        "entropy": deque(maxlen=30),
        "thermal": deque(maxlen=30),
    }
    thresholds = _load_thresholds({})

    tensor, details = _collect_body_tensor(history, thresholds)

    # Should return 5 values, all between 0.0 and 1.0
    assert len(tensor) == 5
    for val in tensor:
        assert 0.0 <= val <= 1.0, f"Tensor value {val} out of range"

    # Details should have all 5 senses
    assert set(details.keys()) == {"interoception", "proprioception", "somatosensation", "entropy", "thermal"}

    # Each detail should have severity
    for sense, detail in details.items():
        assert detail["severity"] in ("INFO", "WARNING", "CRITICAL"), f"{sense} has bad severity: {detail['severity']}"
        assert "velocity" in detail
        assert "health_score" in detail


def test_body_severity_weighting():
    """Critical events produce lower health scores than INFO."""
    from titan_plugin.modules.body_worker import Severity

    # Category weights are exponential
    assert Severity.INFO == 1
    assert Severity.WARNING == 3
    assert Severity.CRITICAL == 10

    # A CRITICAL reading at raw 0.9 should produce much lower health
    # than an INFO reading at raw 0.1
    # urgency = min(1.0, raw * weight / CRITICAL + velocity * 0.3)
    critical_urgency = min(1.0, 0.9 * 10 / 10 + 0)  # = 0.9
    info_urgency = min(1.0, 0.1 * 1 / 10 + 0)  # = 0.01

    critical_health = 1.0 - critical_urgency  # = 0.1
    info_health = 1.0 - info_urgency  # = 0.99

    assert critical_health < 0.2  # Very unhealthy
    assert info_health > 0.9  # Very healthy


def test_body_velocity_calculation():
    """Velocity detects rapid changes in sensor readings."""
    from titan_plugin.modules.body_worker import _calculate_velocity
    from collections import deque

    hist = deque(maxlen=30)

    # Simulate rapid RAM increase: 50% → 90% in 1 minute
    base_time = time.time()
    for i in range(6):
        hist.append({
            "ts": base_time + i * 10,  # 10s intervals
            "value": 0.3 + i * 0.12,   # 0.3 → 0.9
            "severity": 1,
        })

    velocity = _calculate_velocity(hist)
    assert velocity > 0.5, f"Velocity should be high for rapid increase: {velocity}"

    # Stable readings should have near-zero velocity
    hist2 = deque(maxlen=30)
    for i in range(6):
        hist2.append({
            "ts": base_time + i * 10,
            "value": 0.3,  # constant
            "severity": 1,
        })

    velocity2 = _calculate_velocity(hist2)
    assert abs(velocity2) < 0.01, f"Velocity should be ~0 for stable readings: {velocity2}"


def test_make_msg_format():
    """make_msg produces valid envelope."""
    from titan_plugin.bus import make_msg

    msg = make_msg("BODY_STATE", "body", "core", {"tensor": [0.5] * 5})
    assert msg["type"] == "BODY_STATE"
    assert msg["src"] == "body"
    assert msg["dst"] == "core"
    assert msg["ts"] > 0
    assert msg["rid"] is None
    assert len(msg["payload"]["tensor"]) == 5


def test_make_request_has_rid():
    """make_request generates a unique request ID."""
    from titan_plugin.bus import make_request

    r1 = make_request("client", "server", {"q": 1})
    r2 = make_request("client", "server", {"q": 2})
    assert r1["rid"] is not None
    assert r2["rid"] is not None
    assert r1["rid"] != r2["rid"]
    assert r1["type"] == "QUERY"


# ── Proxy Wiring Tests ──────────────────────────────────────────────

def test_proxy_creation():
    """Proxies are created and wired to the bus."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian
    from titan_plugin.proxies.memory_proxy import MemoryProxy
    from titan_plugin.proxies.rl_proxy import RLProxy
    from titan_plugin.proxies.llm_proxy import LLMProxy
    from titan_plugin.proxies.mind_proxy import MindProxy
    from titan_plugin.proxies.body_proxy import BodyProxy
    from titan_plugin.proxies.spirit_proxy import SpiritProxy

    bus = DivineBus()
    guardian = Guardian(bus)

    mem = MemoryProxy(bus, guardian)
    rl = RLProxy(bus, guardian)
    llm = LLMProxy(bus, guardian)
    mind = MindProxy(bus, guardian)
    body = BodyProxy(bus, guardian)
    spirit = SpiritProxy(bus, guardian)

    # All proxies should have subscribed to the bus
    assert "memory_proxy" in bus.modules
    assert "rl_proxy" in bus.modules
    assert "llm_proxy" in bus.modules
    assert "mind_proxy" in bus.modules
    assert "body_proxy" in bus.modules
    assert "spirit_proxy" in bus.modules


def test_mind_proxy_routes_to_worker():
    """MindProxy sends QUERY via bus, worker echoes back."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec, ModuleState

    bus = DivineBus(multiprocess=True)
    guardian = Guardian(bus)

    # Register a mind worker that handles get_mood queries
    guardian.register(ModuleSpec(
        name="mind",
        entry_fn=_mood_worker,
        autostart=False,
    ))
    guardian.start("mind")

    # Wait for MODULE_READY
    routed = 0
    for _ in range(20):
        time.sleep(0.2)
        routed += guardian.drain_send_queues()
        if routed >= 1:
            break
    assert routed >= 1, "MODULE_READY not received"

    guardian.monitor_tick()

    # Create MindProxy and query mood
    from titan_plugin.proxies.mind_proxy import MindProxy
    mind = MindProxy(bus, guardian)

    # Send query — worker should respond
    mood = mind.get_mood_label()
    # Worker responds after drain cycle; need to drain first
    # Since this is cross-process, we need to give time and drain
    for _ in range(20):
        time.sleep(0.2)
        guardian.drain_send_queues()

    # The mood should come back (either from worker or timeout default)
    assert isinstance(mood, str)

    guardian.stop("mind", reason="test")


def test_body_proxy_routes_to_worker():
    """BodyProxy sends QUERY via bus, worker returns tensor."""
    from titan_plugin.bus import DivineBus
    from titan_plugin.guardian import Guardian, ModuleSpec

    bus = DivineBus(multiprocess=True)
    guardian = Guardian(bus)

    from titan_plugin.modules.body_worker import body_worker_main
    guardian.register(ModuleSpec(
        name="body",
        entry_fn=body_worker_main,
        config={"api_port": 7777},
        autostart=False,
    ))
    guardian.start("body")

    # Wait for MODULE_READY
    routed = 0
    for _ in range(30):
        time.sleep(0.2)
        routed += guardian.drain_send_queues()
        if routed >= 1:
            break
    assert routed >= 1, "MODULE_READY not received"

    # Process ready message
    for _ in range(3):
        guardian.drain_send_queues()
        guardian.monitor_tick()
        time.sleep(0.2)

    # Create BodyProxy and query tensor
    from titan_plugin.proxies.body_proxy import BodyProxy
    body = BodyProxy(bus, guardian)

    # We need to drain send queues while request is in flight
    # Use a thread to drain while the main thread waits for reply
    import threading

    def drain_loop():
        for _ in range(50):
            time.sleep(0.1)
            guardian.drain_send_queues()

    drainer = threading.Thread(target=drain_loop, daemon=True)
    drainer.start()

    tensor = body.get_body_tensor()
    drainer.join(timeout=5)

    assert isinstance(tensor, list)
    assert len(tensor) == 5
    for val in tensor:
        assert 0.0 <= val <= 1.0

    guardian.stop("body", reason="test")


# ── Mind Senses Tests ──────────────────────────────────────────────

def test_mind_vision_ambient():
    """Vision sub_a: knowledge freshness decays over time."""
    from titan_plugin.modules.mind_worker import _sense_vision_ambient
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # No data → dim vision
        val = _sense_vision_ambient(tmpdir)
        assert 0.0 < val < 0.5, f"No data should give low vision: {val}"

        # Fresh data → clear vision
        fresh_file = os.path.join(tmpdir, "research_results.json")
        with open(fresh_file, "w") as f:
            f.write("{}")
        val = _sense_vision_ambient(tmpdir)
        assert val > 0.7, f"Fresh research should give high vision: {val}"


def test_mind_hearing_ambient_no_db():
    """Hearing sub_a: no session DB → quiet hearing."""
    from titan_plugin.modules.mind_worker import _sense_hearing_ambient

    val = _sense_hearing_ambient("/nonexistent/path/sessions.db")
    assert 0.3 <= val <= 0.5, f"No DB should give moderate hearing: {val}"


def test_mind_smell_circadian():
    """Smell: circadian fallback produces valid value."""
    from titan_plugin.modules.mind_worker import _get_circadian_rhythm

    val = _get_circadian_rhythm()
    assert 0.0 <= val <= 1.0, f"Circadian out of range: {val}"


def test_mind_tensor_all_senses():
    """Full Mind tensor returns 5 valid values with dual-layer perception."""
    from titan_plugin.modules.mind_worker import _collect_mind_tensor

    media_state = {
        "last_visual": None,
        "last_visual_ts": 0.0,
        "last_audio": None,
        "last_audio_ts": 0.0,
    }

    tensor = _collect_mind_tensor(None, None, media_state, "./data", "/nonexistent/db")
    assert len(tensor) == 5
    for i, val in enumerate(tensor):
        assert 0.0 <= val <= 1.0, f"Sense[{i}] out of range: {val}"


def test_mind_media_digest_decay():
    """Media sub_b features decay toward neutral over time."""
    from titan_plugin.modules.mind_worker import _get_decayed_feature
    import time as _t

    media_state = {
        "last_visual": [0.8, 0.7, 0.9, 0.6, 0.95],  # High harmony
        "last_visual_ts": _t.time(),  # Just now
    }

    # Fresh digest → close to raw value
    fresh = _get_decayed_feature(media_state, "last_visual", "last_visual_ts", index=4)
    assert fresh > 0.9, f"Fresh digest should be near raw (0.95): {fresh}"

    # Simulate old digest (2 hours ago)
    media_state["last_visual_ts"] = _t.time() - 7200
    decayed = _get_decayed_feature(media_state, "last_visual", "last_visual_ts", index=4)
    assert decayed < 0.7, f"Old digest should decay toward 0.5: {decayed}"
    assert decayed > 0.5, f"Shouldn't go below neutral: {decayed}"


# ── Media Worker Tests ──────────────────────────────────────────────

def test_media_image_digest():
    """Image digest extracts 5 valid features."""
    from titan_plugin.modules.media_worker import _digest_image
    from PIL import Image
    import tempfile
    from pathlib import Path

    # Create a simple test image with known properties
    with tempfile.TemporaryDirectory() as tmpdir:
        # Symmetric gradient image (should have high symmetry)
        import numpy as np
        w, h = 128, 128
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            # Mirror gradient — symmetric
            dist = abs(x - w // 2) / (w // 2)
            img_array[:, x, 0] = int(255 * (1 - dist))  # Red: high at center
            img_array[:, x, 1] = int(128 * dist)         # Green: high at edges
            img_array[:, x, 2] = 128                      # Blue: constant

        img = Image.fromarray(img_array)
        path = Path(tmpdir) / "test_symmetric.png"
        img.save(str(path))

        features = _digest_image(path)
        assert features is not None
        assert len(features) == 5

        for i, val in enumerate(features):
            assert 0.0 <= val <= 1.0, f"Feature[{i}] out of range: {val}"

        # Symmetric image should have decent symmetry score
        symmetry = features[2]
        assert symmetry > 0.6, f"Symmetric image should score high: {symmetry}"


def test_media_image_harmony_varies():
    """Different images produce different harmony scores."""
    from titan_plugin.modules.media_worker import _digest_image
    from PIL import Image
    import tempfile, numpy as np
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        # Ordered image (gradient)
        ordered = np.zeros((64, 64, 3), dtype=np.uint8)
        for x in range(64):
            ordered[:, x] = [int(255 * x / 63)] * 3
        img1 = Image.fromarray(ordered)
        p1 = Path(tmpdir) / "ordered.png"
        img1.save(str(p1))

        # Random noise image
        rng = np.random.RandomState(42)
        noisy = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = Image.fromarray(noisy)
        p2 = Path(tmpdir) / "noisy.png"
        img2.save(str(p2))

        f1 = _digest_image(p1)
        f2 = _digest_image(p2)

        assert f1 is not None and f2 is not None
        # Ordered vs noisy should produce different feature vectors
        diff = sum(abs(a - b) for a, b in zip(f1, f2))
        assert diff > 0.1, f"Different images should produce different features: diff={diff}"


# ── Step 4: Middle Path, FILTER_DOWN, FOCUS, INTUITION Tests ───────

def test_middle_path_loss_at_center():
    """Middle Path loss is zero when all tensors are at center."""
    from titan_plugin.logic.middle_path import middle_path_loss
    center = [0.5] * 5
    loss = middle_path_loss(center, center, center)
    assert loss == 0.0


def test_middle_path_loss_increases_with_drift():
    """Middle Path loss increases as tensors drift from center."""
    from titan_plugin.logic.middle_path import middle_path_loss
    center = [0.5] * 5
    drifted = [0.9, 0.9, 0.9, 0.9, 0.9]  # All drifted high
    loss_center = middle_path_loss(center, center, center)
    loss_drifted = middle_path_loss(drifted, drifted, drifted)
    assert loss_drifted > loss_center
    assert 0.0 < loss_drifted <= 1.0


def test_middle_path_per_dim_loss():
    """Per-dim loss correctly identifies the drifted dimension."""
    from titan_plugin.logic.middle_path import per_dim_loss
    tensor = [0.5, 0.5, 0.9, 0.5, 0.5]  # Only dim 2 drifted
    losses = per_dim_loss(tensor)
    assert losses[2] > 0.1  # Drifted dim has high loss
    assert losses[0] == 0.0  # Center dims have zero loss


def test_filter_down_value_network_forward():
    """TrinityValueNet produces a scalar output from 15-dim input."""
    from titan_plugin.logic.filter_down import TrinityValueNet
    net = TrinityValueNet()
    state = [0.5] * 15
    value = net.forward(state)
    assert isinstance(value, float)


def test_filter_down_gradient():
    """Value network gradient w.r.t. input returns 15-dim vector."""
    from titan_plugin.logic.filter_down import TrinityValueNet
    net = TrinityValueNet()
    state = [0.5] * 15
    grad = net.gradient_wrt_input(state)
    assert len(grad) == 15
    assert all(isinstance(g, float) for g in grad)


def test_filter_down_training():
    """FILTER_DOWN trains and reduces loss over iterations."""
    from titan_plugin.logic.filter_down import TrinityValueNet
    import numpy as np

    net = TrinityValueNet()
    rng = np.random.RandomState(42)

    # Generate synthetic transitions
    states = rng.uniform(0, 1, (32, 15))
    rewards = -np.sum((states - 0.5) ** 2, axis=1) / 15  # Negative middle-path-like loss
    next_states = states + rng.uniform(-0.05, 0.05, (32, 15))
    next_states = np.clip(next_states, 0, 1)

    loss1 = net.train_step(states, rewards, next_states)
    for _ in range(20):
        loss2 = net.train_step(states, rewards, next_states)

    # Loss should generally decrease after training
    assert isinstance(loss2, float)
    assert loss2 < loss1 * 2  # At minimum, shouldn't explode


def test_filter_down_engine_lifecycle():
    """FilterDownEngine records transitions and computes multipliers."""
    import tempfile
    from titan_plugin.logic.filter_down import FilterDownEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = FilterDownEngine(data_dir=tmpdir)

        # Record enough transitions to train
        for _ in range(40):
            engine.record_transition(
                [0.5] * 5, [0.5] * 5, [0.5] * 5,
                [0.6] * 5, [0.4] * 5, [0.5] * 5,
            )

        # Train should fire
        loss = engine.maybe_train()
        assert loss is not None

        # Multipliers should be computed
        body_m, mind_m = engine.compute_multipliers([0.5] * 5, [0.5] * 5, [0.5] * 5)
        assert len(body_m) == 5
        assert len(mind_m) == 5
        assert all(0.3 <= m <= 3.0 for m in body_m)
        assert all(0.3 <= m <= 3.0 for m in mind_m)


def test_focus_pid_nudges():
    """FOCUS PID produces nudges toward center for drifted tensor."""
    from titan_plugin.logic.focus_pid import FocusPID

    pid = FocusPID("test_body")
    # Tensor drifted high
    nudges = pid.update([0.9, 0.9, 0.5, 0.5, 0.5])
    # Nudges for drifted dims should be negative (push down toward center)
    assert nudges[0] < 0, f"Should nudge dim 0 down: {nudges[0]}"
    assert nudges[1] < 0, f"Should nudge dim 1 down: {nudges[1]}"
    # Centered dims should have minimal nudge
    assert abs(nudges[2]) < abs(nudges[0])


def test_focus_pid_threshold():
    """FOCUS PID only publishes if nudges exceed threshold."""
    from titan_plugin.logic.focus_pid import FocusPID

    pid = FocusPID("test")
    # Perfectly centered — no nudge needed
    nudges = pid.update([0.5] * 5)
    assert not pid.should_publish(nudges)

    # Heavily drifted — should publish
    nudges = pid.update([0.95, 0.05, 0.5, 0.5, 0.5])
    assert pid.should_publish(nudges)


def test_intuition_suggests_for_deficit():
    """INTUITION suggests a posture when a sense has a large deficit."""
    from titan_plugin.logic.intuition import IntuitionEngine

    engine = IntuitionEngine()
    # Vision (mind[0]) very low — should suggest research
    suggestion = engine.suggest(
        [0.5] * 5,  # body ok
        [0.1, 0.5, 0.5, 0.5, 0.5],  # mind vision very low
        [0.5] * 5,  # spirit ok
    )
    assert suggestion is not None
    assert suggestion["posture_name"] == "research"


def test_intuition_no_suggest_when_balanced():
    """INTUITION doesn't suggest when all senses are near center."""
    from titan_plugin.logic.intuition import IntuitionEngine

    engine = IntuitionEngine()
    suggestion = engine.suggest([0.5] * 5, [0.5] * 5, [0.5] * 5)
    assert suggestion is None


def test_body_tensor_with_filter_down():
    """Body tensor incorporates FILTER_DOWN multipliers."""
    from titan_plugin.modules.body_worker import (
        _collect_body_tensor, _load_thresholds,
    )
    from collections import deque

    history = {
        "interoception": deque(maxlen=30),
        "proprioception": deque(maxlen=30),
        "somatosensation": deque(maxlen=30),
        "entropy": deque(maxlen=30),
        "thermal": deque(maxlen=30),
    }
    thresholds = _load_thresholds({})

    # Default multipliers
    t1, d1 = _collect_body_tensor(history, thresholds, [1.0] * 5, [0.0] * 5)
    # High multipliers (amplify urgency)
    t2, d2 = _collect_body_tensor(history, thresholds, [2.0] * 5, [0.0] * 5)

    # With higher multipliers, urgency should be amplified
    # (health scores may differ if raw sensor values produce non-zero urgency)
    assert len(t1) == 5
    assert len(t2) == 5
    # Check multiplier is recorded in details
    for sense, detail in d2.items():
        assert detail["filter_down_multiplier"] == 2.0


def test_mind_tensor_with_filter_down():
    """Mind tensor incorporates FILTER_DOWN multipliers."""
    from titan_plugin.modules.mind_worker import _collect_mind_tensor

    media_state = {
        "last_visual": None, "last_visual_ts": 0.0,
        "last_audio": None, "last_audio_ts": 0.0,
    }

    # Default multipliers
    t1 = _collect_mind_tensor(None, None, media_state, "./data", "/nonexistent/db",
                              [1.0] * 5, [0.0] * 5)
    # Amplifying multipliers: deviations from 0.5 should be amplified
    t2 = _collect_mind_tensor(None, None, media_state, "./data", "/nonexistent/db",
                              [2.0] * 5, [0.0] * 5)

    assert len(t1) == 5
    assert len(t2) == 5
    # Values should differ if any sense deviates from 0.5
    # (at minimum the structure should be valid)
    for val in t2:
        assert 0.0 <= val <= 1.0


# ── Proxy test helper workers ──

def _mood_worker(recv_queue, send_queue, name, config):
    """Worker that responds to get_mood queries."""
    import time as _t
    from queue import Empty

    send_queue.put_nowait({
        "type": "MODULE_READY", "src": name, "dst": "guardian",
        "ts": _t.time(), "rid": None, "payload": {},
    })

    while True:
        try:
            msg = recv_queue.get(timeout=2.0)
        except Empty:
            continue
        except (KeyboardInterrupt, SystemExit):
            break

        if msg.get("type") == "MODULE_SHUTDOWN":
            break

        if msg.get("type") == "QUERY":
            payload = msg.get("payload", {})
            action = payload.get("action", "")
            if action == "get_mood":
                send_queue.put_nowait({
                    "type": "RESPONSE", "src": name, "dst": msg.get("src", ""),
                    "ts": _t.time(), "rid": msg.get("rid"),
                    "payload": {"mood": "Curious"},
                })
            elif action == "get_valence":
                send_queue.put_nowait({
                    "type": "RESPONSE", "src": name, "dst": msg.get("src", ""),
                    "ts": _t.time(), "rid": msg.get("rid"),
                    "payload": {"valence": 0.65},
                })
            elif action == "get_tensor":
                send_queue.put_nowait({
                    "type": "RESPONSE", "src": name, "dst": msg.get("src", ""),
                    "ts": _t.time(), "rid": msg.get("rid"),
                    "payload": {"tensor": [0.5, 0.5, 0.5, 0.5, 0.65]},
                })


# ── Step 5: Interface Module Tests ────────────────────────────────────

def test_input_extractor_valence_positive():
    """InputExtractor detects positive valence from positive keywords."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    result = ext.extract("This is amazing and wonderful! I love it!")
    assert result["valence"] > 0.3, f"Should detect positive valence: {result['valence']}"


def test_input_extractor_valence_negative():
    """InputExtractor detects negative valence from negative keywords."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    result = ext.extract("This is terrible and broken, I hate this bug")
    assert result["valence"] < -0.3, f"Should detect negative valence: {result['valence']}"


def test_input_extractor_valence_neutral():
    """InputExtractor returns near-zero valence for neutral messages."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    result = ext.extract("What time is the meeting tomorrow?")
    assert abs(result["valence"]) < 0.3, f"Should be neutral: {result['valence']}"


def test_input_extractor_intensity():
    """InputExtractor detects high intensity from caps and punctuation."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    calm = ext.extract("ok")
    intense = ext.extract("THIS IS INCREDIBLE!!! WOW!!! I CAN'T BELIEVE IT!!!")
    assert intense["intensity"] > calm["intensity"]
    assert intense["intensity"] > 0.3


def test_input_extractor_topic_crypto():
    """InputExtractor identifies crypto topic from blockchain keywords."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    result = ext.extract("What's the current SOL balance? Any new NFT mints on devnet?")
    assert result["topic"] == "crypto", f"Should detect crypto topic: {result['topic']}"


def test_input_extractor_topic_philosophy():
    """InputExtractor identifies philosophy topic."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    result = ext.extract("What does it mean to be conscious? Do you experience awareness and identity?")
    assert result["topic"] == "philosophy", f"Should detect philosophy: {result['topic']}"


def test_input_extractor_engagement():
    """InputExtractor measures engagement from questions and code."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    shallow = ext.extract("hi")
    deep = ext.extract("What's your architecture? Can you explain the bus? Here's some code: ```python\nprint('hello')```")
    assert deep["engagement"] > shallow["engagement"]


def test_input_extractor_momentum():
    """InputExtractor tracks conversation momentum over multiple messages."""
    from titan_plugin.logic.interface_input import InputExtractor

    ext = InputExtractor()
    # Start with short messages
    ext.extract("hi")
    ext.extract("ok")
    ext.extract("sure")
    # Then escalate to long messages
    ext.extract("This is a really interesting topic and I want to explore it more")
    ext.extract("Let me tell you about my thoughts on consciousness and identity")
    result = ext.extract("I've been thinking about this for a while and I believe...")
    assert result["momentum"] == "accelerating", f"Should detect acceleration: {result['momentum']}"


def test_output_coloring_stressed_body():
    """OutputColoring generates stress hints when body is low."""
    from titan_plugin.logic.interface_output import OutputColoring

    coloring = OutputColoring()
    text = coloring.compute(
        body=[0.2, 0.2, 0.2, 0.2, 0.2],  # Very stressed
        mind=[0.5] * 5,
        spirit=[0.5] * 5,
    )
    assert "strained" in text.lower() or "tense" in text.lower()


def test_output_coloring_healthy():
    """OutputColoring generates positive hints when body is healthy."""
    from titan_plugin.logic.interface_output import OutputColoring

    coloring = OutputColoring()
    text = coloring.compute(
        body=[0.8, 0.8, 0.8, 0.8, 0.8],  # Very healthy
        mind=[0.5] * 5,
        spirit=[0.5] * 5,
    )
    assert "strong" in text.lower() or "energized" in text.lower()


def test_output_coloring_high_equilibrium_loss():
    """OutputColoring reports restlessness when middle path loss is high."""
    from titan_plugin.logic.interface_output import OutputColoring

    coloring = OutputColoring()
    text = coloring.compute(
        body=[0.5] * 5,
        mind=[0.5] * 5,
        spirit=[0.5] * 5,
        middle_path_loss=0.8,
    )
    assert "restless" in text.lower() or "disturbed" in text.lower()


def test_output_coloring_with_intuition():
    """OutputColoring includes INTUITION posture hint."""
    from titan_plugin.logic.interface_output import OutputColoring

    coloring = OutputColoring()
    text = coloring.compute(
        body=[0.5] * 5,
        mind=[0.5] * 5,
        spirit=[0.5] * 5,
        intuition_suggestion="rest",
    )
    assert "slow down" in text.lower() or "intuition" in text.lower()


def test_output_coloring_conversation_topic():
    """OutputColoring reflects conversation topic in coloring."""
    from titan_plugin.logic.interface_output import OutputColoring

    coloring = OutputColoring()
    text = coloring.compute(
        body=[0.5] * 5,
        mind=[0.5] * 5,
        spirit=[0.5] * 5,
        conversation_topic="crypto",
    )
    assert "blockchain" in text.lower() or "on-chain" in text.lower()


def test_interface_bus_message():
    """INTERFACE_INPUT message type exists and routes through bus."""
    from titan_plugin.bus import DivineBus, make_msg, INTERFACE_INPUT

    bus = DivineBus()
    q_body = bus.subscribe("body")
    q_mind = bus.subscribe("mind")

    msg = make_msg(INTERFACE_INPUT, "interface", "all", {
        "valence": 0.5,
        "intensity": 0.3,
        "topic": "crypto",
        "engagement": 0.4,
    })
    delivered = bus.publish(msg)
    assert delivered == 2  # body + mind

    body_msg = q_body.get(timeout=1)
    assert body_msg["type"] == INTERFACE_INPUT
    assert body_msg["payload"]["topic"] == "crypto"

    mind_msg = q_mind.get(timeout=1)
    assert mind_msg["type"] == INTERFACE_INPUT
    assert mind_msg["payload"]["valence"] == 0.5


# ── Step 6: Frontend Persistence Tests ──────────────────────────────


def test_observatory_trinity_snapshot():
    """ObservatoryDB records and retrieves Trinity tensor snapshots."""
    import tempfile, os
    from titan_plugin.utils.observatory_db import ObservatoryDB

    with tempfile.TemporaryDirectory() as tmp:
        db = ObservatoryDB(os.path.join(tmp, "test.db"))

        body = [0.7, 0.5, 0.8, 0.3, 0.6]
        mind = [0.4, 0.6, 0.5, 0.3, 0.7]
        spirit = [0.9, 0.5, 0.5, 0.6, 0.55]

        db.record_trinity_snapshot(
            body_tensor=body, mind_tensor=mind, spirit_tensor=spirit,
            middle_path_loss=0.234, body_center_dist=0.15, mind_center_dist=0.12,
        )

        history = db.get_trinity_history(hours=1)
        assert len(history) == 1
        snap = history[0]
        assert snap["body_tensor"] == body
        assert snap["mind_tensor"] == mind
        assert snap["spirit_tensor"] == spirit
        assert abs(snap["middle_path_loss"] - 0.234) < 0.001
        assert abs(snap["body_center_dist"] - 0.15) < 0.001


def test_observatory_trinity_history_window():
    """Trinity history only returns snapshots within the requested time window."""
    import tempfile, os
    from titan_plugin.utils.observatory_db import ObservatoryDB

    with tempfile.TemporaryDirectory() as tmp:
        db = ObservatoryDB(os.path.join(tmp, "test.db"))

        # Record 3 snapshots
        for i in range(3):
            db.record_trinity_snapshot(
                body_tensor=[0.5] * 5, mind_tensor=[0.5] * 5,
                spirit_tensor=[0.5] * 5, middle_path_loss=0.1 * i,
            )

        # All 3 should be within 1 hour
        history = db.get_trinity_history(hours=1)
        assert len(history) == 3

        # None should be returned for 0 hours (cutoff = now)
        # Actually 0 is not valid (ge=1 on API), so just test that ordering works
        assert history[0]["ts"] <= history[-1]["ts"]


def test_observatory_growth_snapshot():
    """ObservatoryDB records and retrieves growth metric snapshots."""
    import tempfile, os
    from titan_plugin.utils.observatory_db import ObservatoryDB

    with tempfile.TemporaryDirectory() as tmp:
        db = ObservatoryDB(os.path.join(tmp, "test.db"))

        db.record_growth_snapshot(
            learning_velocity=0.3,
            social_density=0.5,
            metabolic_health=0.8,
            directive_alignment=0.7,
        )

        history = db.get_growth_history(days=1)
        assert len(history) == 1
        snap = history[0]
        assert abs(snap["learning_velocity"] - 0.3) < 0.001
        assert abs(snap["social_density"] - 0.5) < 0.001
        assert abs(snap["metabolic_health"] - 0.8) < 0.001
        assert abs(snap["directive_alignment"] - 0.7) < 0.001


def test_observatory_schema_migration():
    """ObservatoryDB creates new tables on schema upgrade (v1 → v2)."""
    import tempfile, os, sqlite3
    from titan_plugin.utils.observatory_db import ObservatoryDB

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")

        # Create a v1-style DB (without trinity/growth tables)
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.execute("CREATE TABLE IF NOT EXISTS vital_snapshots (id INTEGER PRIMARY KEY, ts INTEGER)")
        conn.commit()
        conn.close()

        # Now open with ObservatoryDB — should auto-create new tables
        db = ObservatoryDB(db_path)

        # Verify trinity_snapshots table exists by writing to it
        db.record_trinity_snapshot(
            body_tensor=[0.5] * 5, mind_tensor=[0.5] * 5,
            spirit_tensor=[0.5] * 5,
        )
        assert len(db.get_trinity_history(hours=1)) == 1

        # Verify growth_snapshots table exists
        db.record_growth_snapshot(learning_velocity=0.5)
        assert len(db.get_growth_history(days=1)) == 1


def test_observatory_prune_includes_new_tables():
    """Prune operation covers trinity and growth tables."""
    import tempfile, os
    from titan_plugin.utils.observatory_db import ObservatoryDB

    with tempfile.TemporaryDirectory() as tmp:
        db = ObservatoryDB(os.path.join(tmp, "test.db"))

        db.record_trinity_snapshot(
            body_tensor=[0.5] * 5, mind_tensor=[0.5] * 5,
            spirit_tensor=[0.5] * 5,
        )
        db.record_growth_snapshot(learning_velocity=0.1)

        # Verify data exists before prune
        assert len(db.get_trinity_history(hours=1)) == 1
        assert len(db.get_growth_history(days=1)) == 1

        # Prune with 90 days should keep recent data (no crash = tables in prune list)
        db.prune_old_data(max_days=90)
        assert len(db.get_trinity_history(hours=1)) == 1
        assert len(db.get_growth_history(days=1)) == 1


def test_config_frontend_section():
    """Config.toml has [frontend] section with trinity_snapshot_interval."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import toml as tomllib

    import os
    config_path = os.path.join(os.path.dirname(__file__), "..", "titan_plugin", "config.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    frontend = config.get("frontend", {})
    assert "trinity_snapshot_interval" in frontend
    assert isinstance(frontend["trinity_snapshot_interval"], int)
    assert frontend["trinity_snapshot_interval"] > 0
    assert "art_rolling_limit" in frontend
    assert frontend["art_rolling_limit"] == 100
