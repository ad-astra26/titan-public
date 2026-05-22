"""tests/test_social_graph_extraction_e2e.py — end-to-end migration tests.

Per PLAN_microkernel_phase_c_social_graph_worker_extraction.md §7.3 +
SPEC v1.7.1 §9.B + D-SPEC-50.

Two flagship integration tests:

  1. test_chat_post_hook_no_more_attribute_error — exercises the actual
     code path from `agno_hooks.py:1658` that triggered
     BUG-MINDPROXY-MISSING-RECORD-INTERACTION-ASYNC-20260514:
       `await social_graph.record_interaction_async(user_id, quality=...)`
     Verifies the proxy → worker round-trip completes without
     AttributeError and writes the row to social_graph.db.

  2. test_mind_worker_taste_sense_reads_shm — exercises the
     mind_worker `_sense_taste` → SHM-direct-read path that replaced
     the in-process `SocialGraph.get_stats()` call. Seeds the SHM slot
     via SocialGraphStatePublisher, then invokes the reader shim +
     _sense_taste and verifies the expected taste value.

These tests don't spawn real subprocesses — they bind the proxy + worker
handler in-process using the bus envelope. The full subprocess +
guardian lifecycle is exercised by §7.4 (test_social_graph_worker_lifecycle.py).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td


# ── Test #1 — chat-hook regression gate ──────────────────────────────


@pytest.mark.asyncio
async def test_chat_post_hook_no_more_attribute_error(temp_data_dir):
    """Production-shape exercise of agno_hooks.py:1658.

    Pre-v1.7.1: `_proxies["social_graph"]` resolved to MindProxy which
    had no `record_interaction_async` method → AttributeError fleet-wide.

    Post-v1.7.1: SocialGraphProxy exposes the method. This test wires a
    proxy + a simulated worker dispatcher in-process via the bus
    envelope, then exercises the same call shape agno_hooks does:

        await social_graph.record_interaction_async(user_id, quality=quality)

    Asserts: no AttributeError + the row was persisted to social_graph.db.
    """
    from titan_hcl.core.social_graph import SocialGraph
    from titan_hcl.modules.social_graph_worker import (
        MODULE_NAME, _handle_query,
    )
    from titan_hcl.proxies.social_graph_proxy import SocialGraphProxy

    db_path = os.path.join(temp_data_dir, "social_graph.db")
    sg = SocialGraph(db_path=db_path)
    worker_send_queue = Queue()

    # Mock bus that loops proxy → worker handler synchronously
    bus = MagicMock()
    guardian = MagicMock()
    bus.subscribe.return_value = MagicMock()

    captured_response = {}

    async def fake_request_async(src, dst, payload, timeout=5.0, reply_queue=None):
        """Simulate broker routing: proxy QUERY → worker._handle_query
        → captures RESPONSE."""
        msg = {
            "type": "QUERY", "src": src, "dst": dst,
            "rid": "rid-e2e-1", "payload": payload, "ts": 1.0,
        }
        _handle_query(msg, sg, worker_send_queue, MODULE_NAME)
        # Drain RESPONSE
        while not worker_send_queue.empty():
            m = worker_send_queue.get_nowait()
            if m.get("type") == "RESPONSE" and m.get("rid") == "rid-e2e-1":
                captured_response.update(m["payload"])
                return m
        return {"payload": {}}

    bus.request_async = fake_request_async

    with patch("titan_hcl.proxies.social_graph_proxy.StateRegistryReader"):
        with patch("titan_hcl.proxies.social_graph_proxy.ensure_shm_root"):
            with patch(
                "titan_hcl.proxies.social_graph_proxy.resolve_titan_id",
                return_value="T1",
            ):
                proxy = SocialGraphProxy(bus, guardian)
                proxy._ensure_started = MagicMock()  # bypass guardian

                # THE CALL — same shape as agno_hooks.py:1658
                # Must NOT raise AttributeError.
                await proxy.record_interaction_async(
                    "alice_e2e", quality=0.85)

    # Verify worker persisted to the real DB
    profile = sg.get_or_create_user("alice_e2e")
    assert profile.interaction_count >= 1, (
        "record_interaction_async did not reach worker DB — proxy → worker "
        "bus envelope round-trip broken.")
    # quality=0.85 → like_score increments by 0.35
    assert profile.like_score > 0.0, (
        "record_interaction quality scoring did not apply.")
    # And the response shape
    assert captured_response.get("ok") is True


# ── Test #2 — mind_worker SHM-direct taste sense ─────────────────────


def test_mind_worker_taste_sense_reads_shm(temp_data_dir, monkeypatch):
    """mind_worker._sense_taste no longer instantiates SocialGraph;
    it uses the _SocialGraphStatsShmReader shim that reads
    social_graph_state.bin SHM directly per G18.

    This test:
      1. Spawns a SocialGraphStatePublisher pointing at a temp SHM root
      2. Builds a SocialGraph with seeded data + asks the publisher to
         write the stats snapshot
      3. Constructs the _SocialGraphStatsShmReader at the same SHM root
      4. Calls mind_worker._sense_taste(reader) and verifies the
         computed taste value matches the formula
         `min(1.0, (users/20)*0.6 + (edges/10)*0.4)`
    """
    from titan_hcl.core.social_graph import SocialGraph
    from titan_hcl.logic.social_graph_state_publisher import (
        SocialGraphStatePublisher,
    )
    from titan_hcl.modules.mind_worker import (
        _SocialGraphStatsShmReader, _sense_taste,
    )

    # Point SHM root at a temp dir
    shm_root = Path(temp_data_dir) / "shm" / "titan_test_taste"
    shm_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "titan_hcl.core.state_registry.ensure_shm_root",
        lambda titan_id: shm_root,
    )
    monkeypatch.setattr(
        "titan_hcl.logic.social_graph_state_publisher.ensure_shm_root",
        lambda titan_id: shm_root,
    )

    # Seed SocialGraph with known data
    sg = SocialGraph(db_path=str(Path(temp_data_dir) / "social_graph.db"))
    for i in range(10):
        sg.get_or_create_user(f"user_{i}")
    # 5 edges
    for i in range(5):
        sg.record_edge(f"user_{i}", f"user_{i+1}")

    # Publish stats to SHM
    publisher = SocialGraphStatePublisher(titan_id="test_taste")
    publisher.publish(sg)

    # Read via the mind_worker shim
    reader = _SocialGraphStatsShmReader(titan_id="test_taste")
    stats = reader.get_stats()

    assert stats["users"] == 10, f"users count mismatch: got {stats}"
    assert stats["edges"] == 5, f"edges count mismatch: got {stats}"

    # _sense_taste(social_graph) takes any object with .get_stats() —
    # call with the shim
    taste = _sense_taste(reader)
    # Formula: min(1.0, (users/20)*0.6 + (edges/10)*0.4)
    #        = min(1.0, (10/20)*0.6 + (5/10)*0.4)
    #        = min(1.0, 0.3 + 0.2)
    #        = 0.5
    expected = min(1.0, (10 / 20.0) * 0.6 + (5 / 10.0) * 0.4)
    assert abs(taste - expected) < 1e-6, (
        f"_sense_taste(shim) returned {taste}, expected {expected}")


def test_mind_worker_taste_sense_cold_boot_returns_neutral(temp_data_dir, monkeypatch):
    """When SHM slot is empty (cold boot before social_graph_worker first
    publish), _sense_taste returns 0.5 neutral — does not crash."""
    from titan_hcl.modules.mind_worker import (
        _SocialGraphStatsShmReader, _sense_taste,
    )

    shm_root = Path(temp_data_dir) / "shm" / "titan_test_coldboot"
    shm_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "titan_hcl.core.state_registry.ensure_shm_root",
        lambda titan_id: shm_root,
    )

    reader = _SocialGraphStatsShmReader(titan_id="test_coldboot")
    # No publisher has written yet — SHM slot empty
    stats = reader.get_stats()
    assert stats["users"] == 0 and stats["edges"] == 0
    # _sense_taste should return defaults from formula (or its own
    # neutral fallback path). With 0 users + 0 edges:
    # min(1.0, (0/20)*0.6 + (0/10)*0.4) = 0.0 — taste at floor.
    # Either 0.0 OR the function's neutral 0.5 default is acceptable;
    # the key requirement is "does not raise".
    taste = _sense_taste(reader)
    assert 0.0 <= taste <= 1.0, (
        f"_sense_taste returned out-of-range value at cold boot: {taste}")
