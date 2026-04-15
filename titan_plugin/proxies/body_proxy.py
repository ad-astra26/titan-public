"""
Body Module Proxy — bridge to the supervised Body process.

Provides access to the 5DT somatic tensor and urgency data
from the always-on Body module via the Divine Bus.
"""
import logging

from ..bus import DivineBus
from ..guardian import Guardian

logger = logging.getLogger(__name__)


class BodyProxy:
    """
    Proxy for the Body module (always-on, autostart=True).
    Queries the 5DT somatic tensor and urgency details.
    """

    def __init__(self, bus: DivineBus, guardian: Guardian):
        self._bus = bus
        self._guardian = guardian
        self._reply_queue = bus.subscribe("body_proxy", reply_only=True)

    def get_body_tensor(self) -> list:
        """Get the 5DT Body state tensor."""
        reply = self._bus.request(
            "body_proxy", "body",
            {"action": "get_tensor"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {}).get("tensor", [0.5] * 5)
        return [0.5] * 5

    def get_body_details(self) -> dict:
        """Get detailed sensor readings with urgency/severity breakdown."""
        reply = self._bus.request(
            "body_proxy", "body",
            {"action": "get_details"},
            timeout=5.0,
            reply_queue=self._reply_queue,
        )
        if reply:
            return reply.get("payload", {})
        return {}
