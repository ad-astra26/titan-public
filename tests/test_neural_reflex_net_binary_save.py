"""PROFILING.md F4 — neural_reflex_net binary (msgpack/f32) persistence.

Replaces JSON float-array serialization (~27% of cognitive_worker CPU on the
save_all background thread) with msgpack + f32-LE blobs. Verifies:
  1. NeuralReflexNet weights round-trip (binary).
  2. NervousTransitionBuffer round-trip (binary).
  3. Saved file is binary (NOT JSON — no leading `{`).
  4. Dual-read: legacy JSON files still load (no migration step).
"""
import json
import os

import numpy as np

from titan_hcl.logic.neural_reflex_net import (
    NeuralReflexNet, NervousTransitionBuffer,
)


def test_reflex_net_weights_roundtrip(tmp_path):
    net = NeuralReflexNet("TEST", input_dim=8)
    net.total_updates = 123
    net.last_loss = 0.004567
    net.fire_count = 7
    p = str(tmp_path / "test_weights.json")
    net.save(p)

    # binary, not JSON
    with open(p, "rb") as f:
        assert f.read(1) != b"{"

    net2 = NeuralReflexNet("TEST", input_dim=8)
    assert net2.load(p) is True
    assert net2.total_updates == 123
    assert abs(net2.last_loss - 0.004567) < 1e-6
    assert net2.fire_count == 7
    for a in ("w1", "b1", "w2", "b2", "w3", "b3"):
        np.testing.assert_allclose(
            getattr(net2, a), getattr(net, a), atol=1e-5,
            err_msg=f"{a} mismatch after binary round-trip")


def test_buffer_roundtrip(tmp_path):
    buf = NervousTransitionBuffer(max_size=50)
    for i in range(5):
        buf.add([float(i), float(i) * 2, 0.5], urgency=0.1 * i,
                vm_baseline=0.9, reward=float(i) - 2, fired=(i % 2 == 0))
    p = str(tmp_path / "test_buffer.json")
    buf.save(p)
    with open(p, "rb") as f:
        assert f.read(1) != b"{"

    buf2 = NervousTransitionBuffer(max_size=50)
    assert buf2.load(p) is True
    assert len(buf2._observations) == 5
    np.testing.assert_allclose(buf2._observations, buf._observations, atol=1e-5)
    np.testing.assert_allclose(buf2._urgencies, buf._urgencies, atol=1e-5)
    np.testing.assert_allclose(buf2._vm_baselines, buf._vm_baselines, atol=1e-5)
    np.testing.assert_allclose(buf2._rewards, buf._rewards, atol=1e-5)
    assert buf2._fired == buf._fired
    assert all(isinstance(x, bool) for x in buf2._fired)


def test_empty_buffer_roundtrip(tmp_path):
    buf = NervousTransitionBuffer(max_size=10)
    p = str(tmp_path / "empty_buffer.json")
    buf.save(p)
    buf2 = NervousTransitionBuffer(max_size=10)
    assert buf2.load(p) is True
    assert buf2._observations == []
    assert buf2._fired == []


def test_f4b_binary_load_keeps_ndarray_rows_and_stays_usable(tmp_path):
    """F4b: binary load keeps observation rows as 1D ndarrays (no per-float tolist);
    mixed ndarray(loaded)+list(added) rows must still sample/save/json-dump cleanly."""
    buf = NervousTransitionBuffer(max_size=50)
    for i in range(4):
        buf.add([float(i), float(i) * 2, 0.5], urgency=0.1 * i,
                vm_baseline=0.9, reward=float(i), fired=(i % 2 == 0))
    p = str(tmp_path / "f4b_buffer.json")
    buf.save(p)

    buf2 = NervousTransitionBuffer(max_size=50)
    assert buf2.load(p) is True
    # rows from binary load are ndarrays (the F4b optimization), not python lists
    assert all(isinstance(r, np.ndarray) for r in buf2._observations)
    # add a fresh row (python list) → buffer is now mixed-typed
    buf2.add([9.0, 9.0, 9.0], urgency=1.0, vm_baseline=0.9, reward=1.0, fired=True)
    assert len(buf2._observations) == 5

    # sample-style stack over mixed rows works (this is what train_step does)
    stacked = np.asarray(buf2._observations, dtype=np.float64)
    assert stacked.shape == (5, 3)

    # re-save the mixed buffer round-trips
    p2 = str(tmp_path / "f4b_buffer2.json")
    buf2.save(p2)
    buf3 = NervousTransitionBuffer(max_size=50)
    assert buf3.load(p2) is True
    np.testing.assert_allclose(
        np.asarray(buf3._observations), np.asarray(buf2._observations), atol=1e-9)

    # SQLite-backup normalization (the recovery net) must json.dumps mixed rows
    obs_rows = [o.tolist() if hasattr(o, "tolist") else list(o)
                for o in buf2._observations]
    json.dumps({"observations": obs_rows})  # must not raise


def test_legacy_json_weights_still_load(tmp_path):
    """A pre-F4 JSON weights file must still load (dual-read, no migration step)."""
    net = NeuralReflexNet("LEGACY", input_dim=8)
    legacy = {
        "name": "LEGACY", "input_dim": 8, "hidden_1": net.hidden_1,
        "hidden_2": net.hidden_2, "lr": net.lr,
        "fire_threshold": net.fire_threshold, "feature_set": net._feature_set,
        "total_updates": 99, "last_loss": 0.01, "fire_count": 3,
        "w1": net.w1.tolist(), "b1": net.b1.tolist(),
        "w2": net.w2.tolist(), "b2": net.b2.tolist(),
        "w3": net.w3.tolist(), "b3": net.b3.tolist(),
    }
    p = str(tmp_path / "legacy_weights.json")
    with open(p, "w") as f:
        json.dump(legacy, f)
    net2 = NeuralReflexNet("LEGACY", input_dim=8)
    assert net2.load(p) is True
    assert net2.total_updates == 99
    np.testing.assert_allclose(net2.w1, net.w1, atol=1e-9)


def test_legacy_json_buffer_still_loads(tmp_path):
    legacy = {
        "observations": [[1.0, 2.0], [3.0, 4.0]],
        "urgencies": [0.1, 0.2], "vm_baselines": [0.9, 0.9],
        "rewards": [1.0, -1.0], "fired": [True, False],
    }
    p = str(tmp_path / "legacy_buffer.json")
    with open(p, "w") as f:
        json.dump(legacy, f)
    buf = NervousTransitionBuffer(max_size=10)
    assert buf.load(p) is True
    assert buf._observations == [[1.0, 2.0], [3.0, 4.0]]
    assert buf._fired == [True, False]
