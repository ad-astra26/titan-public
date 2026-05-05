"""Tests for rFP_bus_payload_contracts.md (2026-05-01).

Three layers:
  - Schema validation per memory event (cache_schemas.py)
  - Registry consistency + lookup (bus_contracts.py)
  - SEND-time enforcement in _packb_safe (bus_socket.py)
"""
from __future__ import annotations

import time

import msgpack
import pytest

from titan_plugin.api import bus_contracts, cache_schemas
from titan_plugin.api.cache_schemas import (
    EVENT_SCHEMAS,
    MemoryKnowledgeGraphEvent,
    MemoryMempoolEvent,
    MemoryStatus,
    MemoryTopEvent,
    MemoryTopologyEvent,
)
from titan_plugin.core.bus_socket import (
    BusContractViolation,
    _packb_safe,
)


# ── Schema tests (cache_schemas.py) ────────────────────────────────────


class TestMemoryStatusSchema:
    def test_valid_payload(self):
        s = MemoryStatus(
            persistent_count=42,
            mempool_size=3,
            cognee_ready=True,
            memory_backend_ready=True,
            updated_at=1000.0,
        )
        assert s.persistent_count == 42
        assert s.cognee_ready is True

    def test_defaults(self):
        s = MemoryStatus()  # all defaults
        assert s.persistent_count == 0
        assert s.cognee_ready is False
        assert s.updated_at == 0.0

    def test_string_int_rejected(self):
        with pytest.raises(Exception):
            MemoryStatus(persistent_count="not_an_int")


class TestMemoryTopEvent:
    def test_required_fields(self):
        ev = MemoryTopEvent(updated_at=1.0, count=5)
        assert ev.last_id is None  # optional

    def test_with_last_id(self):
        ev = MemoryTopEvent(updated_at=1.0, count=5, last_id="abc123")
        assert ev.last_id == "abc123"

    def test_missing_required(self):
        with pytest.raises(Exception):
            MemoryTopEvent(count=5)  # missing updated_at


class TestMemoryTopologyEvent:
    def test_with_clusters(self):
        ev = MemoryTopologyEvent(
            updated_at=1.0,
            total_classified=100,
            cluster_counts={"Solana Architecture": 42, "Other": 10},
        )
        assert ev.cluster_counts["Solana Architecture"] == 42

    def test_empty_clusters(self):
        ev = MemoryTopologyEvent(updated_at=1.0, total_classified=0)
        assert ev.cluster_counts == {}


class TestMemoryKnowledgeGraphEvent:
    def test_full(self):
        ev = MemoryKnowledgeGraphEvent(
            updated_at=1.0,
            node_count=100,
            edge_count=200,
            entity_types={"Person": 30, "Topic": 50},
        )
        assert ev.node_count == 100


# ── Registry tests (bus_contracts.py) ──────────────────────────────────


class TestBusContractRegistry:
    def test_5_contracts(self):
        assert len(bus_contracts.REGISTRY) == 5

    def test_lookup_known(self):
        c = bus_contracts.get_contract("MEMORY_STATUS_UPDATED")
        assert c is not None
        assert c.schema is MemoryStatus
        assert c.max_payload_bytes == 4_000

    def test_lookup_unknown_returns_none(self):
        assert bus_contracts.get_contract("NO_SUCH_TYPE") is None

    def test_all_contracted_msg_types(self):
        types = bus_contracts.all_contracted_msg_types()
        assert "MEMORY_STATUS_UPDATED" in types
        assert "MEMORY_TOP_UPDATED" in types
        assert "MEMORY_MEMPOOL_UPDATED" in types
        assert "MEMORY_TOPOLOGY_UPDATED" in types
        assert "MEMORY_KNOWLEDGE_GRAPH_UPDATED" in types
        assert len(types) == 5

    def test_registry_schemas_match_event_schemas(self):
        for c in bus_contracts.REGISTRY:
            assert EVENT_SCHEMAS[c.msg_type] is c.schema, (
                f"REGISTRY schema for {c.msg_type} ({c.schema}) does not "
                f"match EVENT_SCHEMAS ({EVENT_SCHEMAS[c.msg_type]})"
            )

    def test_max_payload_bytes_within_default(self):
        for c in bus_contracts.REGISTRY:
            assert c.max_payload_bytes <= bus_contracts.DEFAULT_MAX_PAYLOAD_BYTES, (
                f"{c.msg_type} max_payload_bytes={c.max_payload_bytes} > "
                f"DEFAULT={bus_contracts.DEFAULT_MAX_PAYLOAD_BYTES}"
            )

    def test_all_have_producer(self):
        for c in bus_contracts.REGISTRY:
            assert c.producer_module, f"{c.msg_type} missing producer_module"

    def test_all_have_consumer(self):
        for c in bus_contracts.REGISTRY:
            assert c.consumer_modules, f"{c.msg_type} missing consumer_modules"


# ── _packb_safe SEND-time enforcement (bus_socket.py) ──────────────────


class TestPackbSafeContractEnforcement:
    def _make_msg(self, msg_type: str, payload: dict) -> dict:
        return {
            "type": msg_type,
            "src": "memory_worker",
            "dst": "all",
            "ts": time.time(),
            "rid": None,
            "payload": payload,
        }

    def test_valid_memory_status_packs(self):
        msg = self._make_msg("MEMORY_STATUS_UPDATED", {
            "persistent_count": 100,
            "mempool_size": 5,
            "cognee_ready": True,
            "memory_backend_ready": True,
            "updated_at": time.time(),
        })
        data = _packb_safe(msg)
        assert isinstance(data, bytes)
        assert len(data) > 0
        # Round-trip should succeed (strict_map_key default False here, fine)
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        assert unpacked["type"] == "MEMORY_STATUS_UPDATED"

    def test_schema_violation_raises(self):
        msg = self._make_msg("MEMORY_STATUS_UPDATED", {
            "persistent_count": "not_an_int",
        })
        with pytest.raises(BusContractViolation):
            _packb_safe(msg)

    def test_oversize_raises(self):
        # MemoryTopologyEvent allows arbitrary cluster_counts dict; pad it
        # to exceed 8000-byte limit.
        msg = self._make_msg("MEMORY_TOPOLOGY_UPDATED", {
            "updated_at": 1.0,
            "total_classified": 1,
            "cluster_counts": {f"key_{i}": i for i in range(2000)},
        })
        with pytest.raises(BusContractViolation) as exc_info:
            _packb_safe(msg)
        assert "exceeds limit" in str(exc_info.value)

    def test_unschematized_passes_through(self):
        # Unknown msg_type → no contract → no validation → packs cleanly
        msg = self._make_msg("LEGACY_UNKNOWN_TYPE", {"anything": "goes"})
        data = _packb_safe(msg)
        assert isinstance(data, bytes)

    def test_minimum_valid_top_event(self):
        msg = self._make_msg("MEMORY_TOP_UPDATED", {
            "updated_at": 1.0,
            "count": 0,
        })
        data = _packb_safe(msg)
        assert len(data) < 200  # tiny notification, well under 10KB cap

    def test_top_event_with_last_id(self):
        msg = self._make_msg("MEMORY_TOP_UPDATED", {
            "updated_at": 1.0,
            "count": 250,
            "last_id": "abc-123-def",
        })
        data = _packb_safe(msg)
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        assert unpacked["payload"]["last_id"] == "abc-123-def"


# ── BusSubscriber RECEIVE-time enforcement ─────────────────────────────


class TestBusSubscriberSchemaEnforcement:
    """Validate that BusSubscriber.handle_message validates against schemas."""

    def _make_subscriber(self):
        from titan_plugin.api.bus_subscriber import BusSubscriber
        from titan_plugin.api.cached_state import CachedState

        cs = CachedState()
        return BusSubscriber(cached_state=cs, send_queue=None), cs

    def test_valid_payload_populates_cache(self):
        sub, cs = self._make_subscriber()
        msg = {
            "type": "MEMORY_STATUS_UPDATED",
            "payload": {
                "persistent_count": 100,
                "mempool_size": 5,
                "cognee_ready": True,
                "memory_backend_ready": True,
                "updated_at": 1.0,
            },
        }
        result = sub.handle_message(msg)
        assert result is True
        assert cs.get("memory.status")["persistent_count"] == 100

    def test_invalid_payload_drops_cache_clean(self):
        sub, cs = self._make_subscriber()
        # Seed valid value first
        sub.handle_message({
            "type": "MEMORY_STATUS_UPDATED",
            "payload": {"persistent_count": 50, "mempool_size": 1,
                        "cognee_ready": True, "memory_backend_ready": True,
                        "updated_at": 1.0},
        })
        # Send malformed
        result = sub.handle_message({
            "type": "MEMORY_STATUS_UPDATED",
            "payload": {"persistent_count": "BAD"},
        })
        # Returns True (consumed) — drops the bad one + leaves cache untouched
        assert result is True
        assert cs.get("memory.status")["persistent_count"] == 50  # not corrupted

    def test_unschematized_msg_passes_through_to_handler(self):
        # OBSERVATORY_EVENT has no contract — should still dispatch normally
        sub, cs = self._make_subscriber()
        # No handler for OBSERVATORY_EVENT in BusSubscriber default handlers
        # → returns False (not handled), does NOT crash
        result = sub.handle_message({
            "type": "OBSERVATORY_EVENT",
            "payload": {"event_type": "expr", "data": {}},
        })
        assert result is False  # not in BusSubscriber._handlers


# ── End-to-end pack-then-unpack roundtrip ──────────────────────────────


class TestRoundTrip:
    def test_memory_top_roundtrip(self):
        """Valid MEMORY_TOP_UPDATED payload survives full pack→unpack cycle."""
        msg = {
            "type": "MEMORY_TOP_UPDATED",
            "src": "memory",
            "dst": "all",
            "ts": time.time(),
            "rid": None,
            "payload": {
                "updated_at": time.time(),
                "count": 200,
                "last_id": "node-abc",
            },
        }
        packed = _packb_safe(msg)
        # Default broker unpack uses strict_map_key=True per Phase B.2 spec
        unpacked = msgpack.unpackb(packed, strict_map_key=True, raw=False)
        assert unpacked["type"] == "MEMORY_TOP_UPDATED"
        assert unpacked["payload"]["count"] == 200

    def test_memory_topology_roundtrip(self):
        msg = {
            "type": "MEMORY_TOPOLOGY_UPDATED",
            "src": "memory",
            "dst": "all",
            "ts": time.time(),
            "rid": None,
            "payload": {
                "updated_at": time.time(),
                "total_classified": 100,
                "cluster_counts": {
                    "Solana Architecture": 30,
                    "Memory & Identity": 25,
                    "Other": 45,
                },
            },
        }
        packed = _packb_safe(msg)
        unpacked = msgpack.unpackb(packed, strict_map_key=True, raw=False)
        assert unpacked["payload"]["total_classified"] == 100
        assert unpacked["payload"]["cluster_counts"]["Solana Architecture"] == 30
