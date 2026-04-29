"""
titan_plugin/logic/spirit_state.py — SpiritState: all-seeing Spirit registry (T2).

Global view assembled from OuterState + InnerState + observables + topology.
This is what Spirit "sees" — the complete picture across all layers.

Feeds:
  - Spirit micro_enrich (enrichment quality assessment)
  - InnerTrinityCoordinator (T3)
  - TitanVM nervous system programs (T4)
  - GREAT PULSE convergence detection (T7)
"""
import time
from typing import Any, Optional


class SpiritState:
    """All-encompassing Spirit observation — reads everything, writes enrichment."""

    def __init__(self):
        # Full 30DT raw tensor (6 parts × 5 dims)
        self.full_30dt: list[float] = [0.5] * 30

        # T1: 6 parts × 5 observables
        self.observables: dict[str, dict] = {}

        # T5: Space topology
        self.topology: dict = {}

        # Spirit enrichment tracking
        self.enrichment_quality: float = 0.0
        self.micro_tick_count: int = 0

        # Assembled middle path + coherence metrics
        self.middle_path_loss: float = 0.0
        self.mean_coherence: float = 1.0

        # Metadata
        self._last_assembly_ts: float = 0.0
        self._assembly_count: int = 0

    def assemble(
        self,
        outer_snapshot: Optional[dict] = None,
        inner_snapshot: Optional[dict] = None,
        observables: Optional[dict[str, dict]] = None,
    ) -> None:
        """
        Assemble the full Spirit view from outer + inner state.

        Called every spirit publish cycle. Merges data from both registries
        into a single coherent view.
        """
        # Update 30DT from outer snapshot
        if outer_snapshot:
            body = outer_snapshot.get("body_tensor", [0.5] * 5)
            mind = outer_snapshot.get("mind_tensor", [0.5] * 5)
            spirit = outer_snapshot.get("spirit_tensor", [0.5] * 5)
            o_body = outer_snapshot.get("outer_body", [0.5] * 5)
            o_mind = outer_snapshot.get("outer_mind", [0.5] * 5)
            o_spirit = outer_snapshot.get("outer_spirit", [0.5] * 5)
            self.full_30dt = list(body) + list(mind) + list(spirit) + \
                             list(o_body) + list(o_mind) + list(o_spirit)

        # Update observables
        if observables:
            self.observables = observables
            # Compute mean coherence across all observed parts
            coherences = [
                obs.get("coherence", 1.0)
                for obs in observables.values()
            ]
            if coherences:
                self.mean_coherence = sum(coherences) / len(coherences)

        # Merge inner state data
        if inner_snapshot:
            if inner_snapshot.get("topology"):
                self.topology = inner_snapshot["topology"]

        self._last_assembly_ts = time.time()
        self._assembly_count += 1

    def get(self, key: str, default: Any = None) -> Any:
        """Generic getter for any attribute."""
        return getattr(self, key, default)

    def snapshot(self) -> dict:
        """Full Spirit view snapshot.

        Sanitizes nested dicts so the result round-trips through msgpack
        with `strict_map_key=True`. The `topology["distance_matrix"]` payload
        from `inner_coordinator` historically uses `(a, b)` tuple keys —
        msgpack serializes tuples as arrays, and arrays-as-map-keys are
        rejected by `strict_map_key=True` on unpack. This produces the
        "list is not allowed for map key" malformed-frame errors caught
        by Fix B's hex-dump diagnostic on 2026-04-28.
        """
        return {
            "full_30dt": list(self.full_30dt),
            "observables": _sanitize_dict_keys(dict(self.observables)),
            "topology": _sanitize_dict_keys(dict(self.topology)),
            "enrichment_quality": self.enrichment_quality,
            "micro_tick_count": self.micro_tick_count,
            "middle_path_loss": self.middle_path_loss,
            "mean_coherence": round(self.mean_coherence, 6),
            "assembly_count": self._assembly_count,
        }


def _sanitize_dict_keys(obj):
    """Recursively coerce all dict keys to strings.

    msgpack with `strict_map_key=True` (the broker default) accepts ONLY
    str/bytes keys — int, tuple, list, frozenset, and other non-scalar
    keys all raise on unpack. We coerce every key to its string form so
    the snapshot is always safely round-trippable.

    Tuple keys use the same shape as inner_coordinator.get_stats():
      `(a, b)` → `"a:b"` (matches existing convention for distance_matrix).
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str):
                out[k] = _sanitize_dict_keys(v)
            elif isinstance(k, tuple):
                if len(k) == 2:
                    out[f"{k[0]}:{k[1]}"] = _sanitize_dict_keys(v)
                else:
                    out[":".join(str(p) for p in k)] = _sanitize_dict_keys(v)
            else:
                # int / float / frozenset / any other type → str()
                out[str(k)] = _sanitize_dict_keys(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize_dict_keys(x) for x in obj]
    return obj
