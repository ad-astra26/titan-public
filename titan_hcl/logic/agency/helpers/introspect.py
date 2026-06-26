"""IntrospectHelper — the agency-side introspection helper (RFP_text_extraction_introspection §7.B).

Introspection is *research pointed inward*: this helper reads Titan's OWN structured
self-telemetry over a LOCK-SAFE channel (his synthesis read endpoints — the owner
process serves them, so there is no `mode=ro`-vs-writer DuckDB lock conflict), runs the
deterministic `text_oracle` over it, and returns a verifiable self-observation + a
`SELF:<aspect>` grounding target shaped so it rides the existing curiosity 3a/3b/3c
grounding path. The navel-gaze damper (INV-TX-6) is applied in `run_introspection`.

Helper contract: `async def execute(self, params: dict) -> dict` → `{success, result, ...}`
(mirrors `WebSearchHelper`). Registered as action_type "introspect" (agency-only — NOT
an OML routing action, per RFP D8).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from titan_hcl.synthesis.introspection import (
    IntrospectionDamper, run_introspection,
)

logger = logging.getLogger("agency.introspect")

# aspect → (synthesis read endpoint, default deterministic query). LOCK-SAFE: these
# are HTTP GETs against the owner process's own API, never a direct DB open.
_ASPECTS: dict[str, tuple[str, dict]] = {
    # "what skills have I formed?" — counts skills by their goal_class shape
    "skills": ("/v6/synthesis/skills",
               {"kind": "count", "pattern": r'"goal_class"\s*:\s*"(?P<g>[^"]+)"',
                "group_by": "g"}),
    # "how grounded am I?" — the one sovereignty number
    "sovereignty": ("/v6/synthesis/metrics",
                    {"kind": "fields",
                     "pattern": r'"sovereignty[^"]*"\s*:\s*(?P<sovereignty>[\d.]+)'}),
}
_DEFAULT_ASPECT = "skills"


class IntrospectHelper:
    """Reads own telemetry (lock-safe) → text_oracle extract → SELF observation."""

    name = "introspect"
    action_type = "introspect"

    def __init__(self, *, api_base: str = "http://127.0.0.1:7777",
                 internal_key: str = "", damper: Optional[IntrospectionDamper] = None):
        self._api_base = api_base.rstrip("/")
        self._internal_key = internal_key or ""
        self._damper = damper or IntrospectionDamper()

    async def _read_corpus(self, aspect: str) -> str:
        """LOCK-SAFE corpus: GET the synthesis read endpoint for `aspect`. The owner
        process serves it → no DuckDB writer-lock conflict. Returns "" on any failure
        (the faculty degrades, never crashes)."""
        ep, _q = _ASPECTS.get(aspect, _ASPECTS[_DEFAULT_ASPECT])
        url = f"{self._api_base}{ep}"
        try:
            import aiohttp
            headers = {"X-Titan-Internal-Key": self._internal_key}
            async with aiohttp.ClientSession() as s:
                async with s.get(url, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=8)) as r:
                    if r.status != 200:
                        return ""
                    body = await r.text()
                    return body or ""
        except Exception as e:  # noqa: BLE001 — a reader hiccup never breaks introspection
            logger.debug("[introspect] corpus read failed (%s): %s", aspect, e)
            return ""

    async def execute(self, params: dict) -> dict:
        """Run one introspection. params: {aspect?, query?}. Returns the helper-result
        contract; on a grounded read it stamps `_research_target` (source=introspection)
        onto helper_params so the agency's 3a curiosity-bypass grounds the SELF concept."""
        aspect = str((params or {}).get("aspect") or _DEFAULT_ASPECT).strip()
        _ep, default_q = _ASPECTS.get(aspect, _ASPECTS[_DEFAULT_ASPECT])
        query = (params or {}).get("query") or default_q

        corpus = await self._read_corpus(aspect)
        result = run_introspection(aspect, query, lambda _a: corpus, self._damper)

        # The grounded `content` (the SELF fact) = the observation + the verbatim
        # extract data, so it is substantive (clears the curiosity min-evidence gate)
        # AND carries the real, re-checkable numbers into the SELF:<aspect> concept.
        _evidence = result.observation
        if result.extract:
            _data = result.extract.get("counts") or result.extract.get("fields") or {}
            if _data:
                _evidence = f"{result.observation}\nData: {json.dumps(_data)}"

        # 3a reads helper_params["query"] as a string for the grounded payload.
        helper_params: dict[str, Any] = {"aspect": aspect,
                                         "query": f"introspect:{aspect}",
                                         "introspect_query": query,
                                         "_introspection": True}
        if result.grounded and result.research_target:
            # rides the curiosity 3a path → memory anchor → synthesis seeds SELF:<aspect>
            helper_params["_research_target"] = result.research_target
        return {
            "success": bool(result.grounded or result.observation),
            "result": _evidence if result.grounded else (
                result.observation or f"[SELF:{aspect}] {result.reason}"),
            "helper_params": helper_params,
            "introspection_grounded": bool(result.grounded),
            "introspection_reason": result.reason,
            "extract": result.extract,
        }
