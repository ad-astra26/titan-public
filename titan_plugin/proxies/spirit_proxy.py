"""
Spirit Module Proxy — bridge to the supervised Spirit process.

Provides access to the 3DT+2 consciousness tensor, the unified
Trinity state (Body + Mind + Spirit), and V4 Time Awareness
components (SphereClocks, Resonance, UnifiedSpirit).
"""
import logging

from ..bus import DivineBus
from ..guardian import Guardian

logger = logging.getLogger(__name__)


class SpiritProxy:
    """
    Proxy for the Spirit module (always-on, autostart=True).
    Queries consciousness tensor, unified Trinity state, and V4 components.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("spirit_proxy", reply_only=True)

    def get_spirit_tensor(self) -> list:
        """Get the 5DT Spirit state tensor (3DT+2)."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_tensor"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("tensor", [0.5] * 5)
        return [0.5] * 5

    def get_trinity(self) -> dict:
        """Get unified Trinity state: Body, Mind, Spirit tensors + V4 data."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_trinity"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {
            "spirit_tensor": [0.5] * 5,
            "body_values": [0.5] * 5,
            "mind_values": [0.5] * 5,
            "body_center_dist": 0.0,
            "mind_center_dist": 0.0,
        }

    # ── V4 Time Awareness Queries ─────────────────────────────────

    def get_sphere_clocks(self) -> dict:
        """Get V4 SphereClockEngine state: 6 inner clock phases, radii, pulse counts."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_sphere_clock"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"error": "SphereClocks not available"}

    def get_resonance(self) -> dict:
        """Get V4 ResonanceDetector state: pair alignments, BIG/GREAT pulse counts."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_resonance"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"error": "Resonance not available"}

    def get_unified_spirit(self) -> dict:
        """Get V4 UnifiedSpirit state: 30DT tensor, velocity, stale status, focus multiplier."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_unified_spirit"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"error": "UnifiedSpirit not available"}

    def get_filter_down_status(self) -> dict:
        """Get FILTER_DOWN V4/V5 coexistence state (rFP #2 Phase 7)."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_filter_down_status"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"error": "FilterDown status not available"}

    def get_meditation_health(self) -> dict:
        """Get meditation watchdog state + tracker + overdue flag.

        Used by /v4/meditation/health and `arch_map meditation` cross-Titan
        correlation (rFP_self_healing_meditation_cadence.md I2).
        """
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_meditation_health"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"error": "Meditation health not available"}

    def get_v4_state(self) -> dict:
        """Get complete V4 Time Awareness state in a single call.

        Returns sphere clocks, resonance, unified spirit, and impulse engine
        stats combined. Uses get_trinity which already includes all V4 data.
        """
        trinity = self.get_trinity()
        return {
            "sphere_clock": trinity.get("sphere_clock", {}),
            "resonance": trinity.get("resonance", {}),
            "unified_spirit": trinity.get("unified_spirit", {}),
            "impulse_engine": trinity.get("impulse_engine", {}),
            "filter_down": trinity.get("filter_down", {}),
            "intuition": trinity.get("intuition", {}),
            "consciousness": trinity.get("consciousness", {}),
            "middle_path_loss": trinity.get("middle_path_loss"),
        }

    def get_coordinator(self) -> dict:
        """Get T3 InnerTrinityCoordinator state: topology, dreaming, nervous system."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_coordinator"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"error": "Coordinator not available"}

    def get_nervous_system(self) -> dict:
        """Get V5 Neural NervousSystem state: per-program metrics, training phase."""
        reply = self._bus.request(
            "spirit_proxy", "spirit",
            {"action": "get_nervous_system"},
            timeout=10.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {"error": "NervousSystem not available"}
