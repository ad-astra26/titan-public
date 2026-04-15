"""
titan_plugin/logic/agency/registry.py — Helper Registry (Step 7.3).

Central registry of all available helpers with capability manifests.
Each helper implements the BaseHelper protocol. The registry provides:

  - list_available() → formatted manifest string for LLM prompt injection
  - get_helper(name) → helper instance
  - status tracking: available / unavailable / degraded

The manifest format is designed to be LLM-parseable (~200 tokens total)
so the Agency Module can select the best helper for a given intent.
"""
import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseHelper(Protocol):
    """
    Protocol that all helpers must implement.

    Helpers are Titan's hands in the Outer world — each wraps a real capability
    (web search, social posting, art generation, coding, infrastructure inspection).
    """

    @property
    def name(self) -> str:
        """Unique helper name (snake_case)."""
        ...

    @property
    def description(self) -> str:
        """One-line description of what this helper does."""
        ...

    @property
    def capabilities(self) -> list[str]:
        """List of capability tags (e.g., ['search', 'scrape', 'summarize'])."""
        ...

    @property
    def resource_cost(self) -> str:
        """Resource cost: 'low' | 'medium' | 'high'."""
        ...

    @property
    def latency(self) -> str:
        """Expected latency: 'fast' | 'medium' | 'slow'."""
        ...

    @property
    def enriches(self) -> list[str]:
        """Which Trinity layers this helper enriches: ['body', 'mind', 'spirit']."""
        ...

    @property
    def requires_sandbox(self) -> bool:
        """Whether this helper needs sandboxed execution."""
        ...

    async def execute(self, params: dict) -> dict:
        """
        Run the helper with given parameters.

        Returns:
            {
                "success": bool,
                "result": str,           # Human-readable result summary
                "enrichment_data": dict,  # Trinity dimension enrichment values
                "error": str | None,      # Error message if failed
            }
        """
        ...

    def status(self) -> str:
        """
        Check helper availability.

        Returns: 'available' | 'unavailable' | 'degraded'
        """
        ...


class HelperRegistry:
    """
    Central registry for helper discovery and management.

    Helpers register themselves at boot. The Agency Module queries the registry
    to build an LLM-readable manifest of available tools.
    """

    def __init__(self):
        self._helpers: dict[str, BaseHelper] = {}
        self._status_cache: dict[str, tuple[str, float]] = {}  # name → (status, ts)
        self._status_ttl = 60.0  # Cache status for 60s

    def register(self, helper: BaseHelper) -> None:
        """Register a helper instance."""
        name = helper.name
        if name in self._helpers:
            logger.warning("[HelperRegistry] Overwriting existing helper: %s", name)
        self._helpers[name] = helper
        logger.info("[HelperRegistry] Registered: %s (%s)", name, helper.description)

    def unregister(self, name: str) -> bool:
        """Remove a helper by name. Returns True if found."""
        if name in self._helpers:
            del self._helpers[name]
            self._status_cache.pop(name, None)
            logger.info("[HelperRegistry] Unregistered: %s", name)
            return True
        return False

    def get_helper(self, name: str) -> Optional[BaseHelper]:
        """Get a helper by name, or None if not registered."""
        return self._helpers.get(name)

    def get_status(self, name: str) -> str:
        """
        Get cached status for a helper.
        Refreshes if cache expired.
        """
        import time

        cached = self._status_cache.get(name)
        if cached:
            status, ts = cached
            if time.time() - ts < self._status_ttl:
                return status

        helper = self._helpers.get(name)
        if not helper:
            return "unavailable"

        try:
            status = helper.status()
        except Exception as e:
            logger.warning("[HelperRegistry] Status check failed for %s: %s", name, e)
            status = "unavailable"

        self._status_cache[name] = (status, time.time())
        return status

    def list_helper_names(self) -> list[str]:
        """Return list of available helper names."""
        return [name for name in self._helpers
                if self.get_status(name) != "unavailable"]

    def list_available(self) -> str:
        """
        Generate a formatted manifest of available helpers for LLM consumption.

        Returns a compact text block (~200 tokens) listing each helper
        with its capabilities, cost, latency, and enrichment targets.
        """
        lines = []
        for name, helper in sorted(self._helpers.items()):
            status = self.get_status(name)
            if status == "unavailable":
                continue  # Don't show unavailable helpers to LLM

            status_mark = "" if status == "available" else " [degraded]"
            caps = ", ".join(helper.capabilities)
            enriches = ", ".join(helper.enriches)

            lines.append(
                f"- {name}: {helper.description}{status_mark}\n"
                f"  Capabilities: [{caps}]\n"
                f"  Cost: {helper.resource_cost} | Latency: {helper.latency}\n"
                f"  Enriches: [{enriches}]"
                + (f" | Sandbox: yes" if helper.requires_sandbox else "")
            )

        if not lines:
            return "No helpers available."

        return "\n".join(lines)

    def list_all_names(self) -> list[str]:
        """Return all registered helper names."""
        return list(self._helpers.keys())

    def get_all_statuses(self) -> dict[str, str]:
        """Return status for all registered helpers."""
        return {name: self.get_status(name) for name in self._helpers}

    def get_stats(self) -> dict:
        """Return registry statistics."""
        statuses = self.get_all_statuses()
        return {
            "total": len(self._helpers),
            "available": sum(1 for s in statuses.values() if s == "available"),
            "degraded": sum(1 for s in statuses.values() if s == "degraded"),
            "unavailable": sum(1 for s in statuses.values() if s == "unavailable"),
            "helpers": {
                name: {
                    "description": h.description,
                    "status": statuses.get(name, "unknown"),
                    "resource_cost": h.resource_cost,
                    "latency": h.latency,
                    "enriches": h.enriches,
                }
                for name, h in self._helpers.items()
            },
        }
