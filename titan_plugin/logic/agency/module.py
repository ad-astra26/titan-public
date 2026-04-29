"""
titan_plugin/logic/agency/module.py — Agency Module (Step 7.4).

Orchestrator that listens for INTENT events on the bus, selects a helper
from the registry via lightweight LLM call, executes it asynchronously,
and publishes ACTION_RESULT with enrichment data.

This is Titan's executive function — the bridge between inner desire
(IMPULSE → INTENT) and outer action (Helper execution).

Design:
  - Non-blocking: helpers run in background asyncio tasks
  - LLM call is lightweight (~200 token prompt, JSON response)
  - Budget-capped: max LLM calls per hour (sovereignty setting)
  - Errors never crash the module — they get recorded as failed outcomes
"""
import asyncio
import json
import logging
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# LLM prompt for helper selection
_SELECTION_PROMPT = """\
You are Titan's executive function. Given an intent from your inner self, \
choose the best helper to fulfill it.

INTENT: {posture} — {reason}
Source: {layer}[{dims}], deficit={values}
Urgency: {urgency}

AVAILABLE HELPERS:
{manifest}

Respond with JSON only: {{"helper": "name", "params": {{}}, "reasoning": "..."}}
If no suitable helper exists, respond: {{"helper": "none", "reasoning": "..."}}"""

# Posture-to-reason mapping (natural language descriptions)
_POSTURE_REASONS = {
    "research": "Knowledge is stale — need to explore and learn something new",
    "socialize": "Social connection deficit — need to engage with the community",
    "create": "Creative expression needed — mood needs a boost through making",
    "rest": "Body systems stressed — need to pause and recover",
    "meditate": "Identity or purpose drifting — need to reflect and recenter",
}


class AgencyModule:
    """
    Orchestrates helper selection and execution based on INTENT events.

    Flow:
      1. Receive INTENT (enriched IMPULSE from Interface)
      2. Query HelperRegistry for available tools
      3. Call LLM to select helper + params (~200 token prompt)
      4. Execute helper async (non-blocking)
      5. Publish ACTION_RESULT with enrichment data

    The module maintains a budget counter to limit LLM calls per hour.
    """

    def __init__(
        self,
        registry=None,
        llm_fn=None,
        budget_per_hour: int = 10,
    ):
        """
        Args:
            registry: HelperRegistry instance
            llm_fn: Async callable(prompt: str) -> str for LLM inference.
                     If None, uses a no-op that returns {"helper": "none"}.
            budget_per_hour: Max LLM calls for Agency per hour
        """
        from .registry import HelperRegistry
        self._registry = registry or HelperRegistry()
        self._llm_fn = llm_fn
        self._budget_per_hour = budget_per_hour

        # State
        self._action_counter = 0
        self._llm_calls_this_hour = 0
        self._hour_start = time.time()
        self._pending_action: Optional[dict] = None
        # L3 Phase A.8.1: deque(maxlen=50) replaces manual `[-50:]` trim —
        # auto-eviction at data-structure level, no per-append slicing.
        self._history: deque = deque(maxlen=50)  # Last 50 action results
        # V5: Expression dispatch cooldowns (loaded from titan_params.toml)
        self._dispatch_cooldowns: dict[str, float] = {}
        self._last_dispatch_ts: dict[str, float] = {}
        try:
            import tomllib as _tomllib
        except ImportError:
            import tomli as _tomllib
        try:
            import os
            _params_path = os.path.join(os.path.dirname(__file__), "..", "..", "titan_params.toml")
            with open(_params_path, "rb") as f:
                _params = _tomllib.load(f)
            self._dispatch_cooldowns = _params.get("expression_dispatch_cooldowns", {})
            if self._dispatch_cooldowns:
                logger.info("[Agency] Dispatch cooldowns loaded: %s",
                            {k: f"{v}s" for k, v in self._dispatch_cooldowns.items()})
        except Exception:
            pass

    @property
    def registry(self):
        return self._registry

    async def handle_intent(self, intent: dict) -> Optional[dict]:
        """
        Process an INTENT event and execute the selected helper.

        Args:
            intent: INTENT payload from the bus (enriched IMPULSE)

        Returns:
            ACTION_RESULT payload dict, or None if no action taken.
        """
        posture = intent.get("posture", "meditate")
        urgency = intent.get("urgency", 0.0)
        impulse_id = intent.get("impulse_id", 0)
        source_layer = intent.get("source_layer", "unknown")
        source_dims = intent.get("source_dims", [])
        deficit_values = intent.get("deficit_values", [])
        trinity_snapshot = intent.get("trinity_snapshot", {})

        # Check LLM budget
        self._check_budget_reset()
        if self._llm_calls_this_hour >= self._budget_per_hour:
            logger.info("[Agency] Budget exhausted (%d/%d calls this hour)",
                        self._llm_calls_this_hour, self._budget_per_hour)
            return None

        # Get available helpers manifest
        manifest = self._registry.list_available()
        if manifest == "No helpers available.":
            logger.info("[Agency] No helpers available — skipping intent")
            return None

        # Check for learned selection from Expression Layer (bypasses LLM)
        learned = intent.get("_learned_selection")
        if learned and learned.get("helper"):
            helper_name = learned["helper"]
            params = learned.get("params", {})
            reasoning = learned.get("reasoning", "learned")
        else:
            # Select helper via LLM (or fallback)
            selection = await self._select_helper(posture, source_layer,
                                                  source_dims, deficit_values,
                                                  urgency, manifest)
            if not selection or selection.get("helper") == "none":
                logger.info("[Agency] No suitable helper for posture=%s", posture)
                return self._build_result(impulse_id, posture, None, None,
                                          "no_suitable_helper", trinity_snapshot)

            helper_name = selection["helper"]
            params = selection.get("params", {})
            reasoning = selection.get("reasoning", "")

        # Get helper instance
        helper = self._registry.get_helper(helper_name)
        if not helper:
            logger.warning("[Agency] Helper '%s' not found in registry", helper_name)
            return self._build_result(impulse_id, posture, helper_name, None,
                                      f"helper_not_found: {helper_name}", trinity_snapshot)

        # Inject Trinity context into params (all helpers can use this)
        params["trinity_snapshot"] = trinity_snapshot
        params["posture"] = posture

        # Special handling: coding_sandbox needs LLM-generated code
        if helper_name == "coding_sandbox" and not params.get("code"):
            params = await self._generate_sandbox_code(posture, source_layer,
                                                       source_dims, deficit_values)
            if not params.get("code"):
                logger.info("[Agency] Could not generate code for sandbox — skipping")
                return self._build_result(impulse_id, posture, helper_name, None,
                                          "no_code_generated", trinity_snapshot)

        # Execute helper
        logger.info("[Agency] Executing helper=%s for posture=%s (impulse #%d)",
                     helper_name, posture, impulse_id)
        try:
            result = await helper.execute(params)
        except Exception as e:
            logger.error("[Agency] Helper '%s' execution failed: %s", helper_name, e)
            result = {"success": False, "result": "", "enrichment_data": {},
                      "error": str(e)}

        return self._build_result(impulse_id, posture, helper_name, result,
                                  reasoning, trinity_snapshot)

    async def _select_helper(
        self,
        posture: str,
        source_layer: str,
        source_dims: list,
        deficit_values: list,
        urgency: float,
        manifest: str,
    ) -> Optional[dict]:
        """
        Use LLM to select the best helper for this intent.

        Falls back to rule-based selection if LLM is unavailable.
        """
        if not self._llm_fn:
            # No LLM — use rule-based fallback
            return self._rule_based_select(posture)

        reason = _POSTURE_REASONS.get(posture, "Inner state needs rebalancing")
        prompt = _SELECTION_PROMPT.format(
            posture=posture,
            reason=reason,
            layer=source_layer,
            dims=source_dims,
            values=deficit_values,
            urgency=round(urgency, 2),
            manifest=manifest,
        )

        try:
            self._llm_calls_this_hour += 1
            raw = await self._llm_fn(prompt)
            # Parse JSON from response
            return self._parse_selection(raw)
        except Exception as e:
            logger.warning("[Agency] LLM selection failed: %s — using rule-based", e)
            return self._rule_based_select(posture)

    def _rule_based_select(self, posture: str) -> Optional[dict]:
        """
        Simple rule-based helper selection when LLM is unavailable.

        Maps postures to helper names based on enrichment targets.
        """
        posture_to_helper = {
            "research": "web_search",
            "socialize": "web_search",  # social posting via SocialPressureMeter only
            "create": "art_generate",
            "rest": "infra_inspect",
            "meditate": "infra_inspect",
        }

        helper_name = posture_to_helper.get(posture)
        if helper_name and self._registry.get_helper(helper_name):
            status = self._registry.get_status(helper_name)
            if status != "unavailable":
                return {"helper": helper_name, "params": {}, "reasoning": f"rule-based: {posture}"}

        # Try any available helper (skip sandbox — it always needs LLM-generated code)
        for name in self._registry.list_all_names():
            if name == "coding_sandbox":
                continue  # Sandbox requires LLM to generate code, can't do rule-based
            status = self._registry.get_status(name)
            if status != "unavailable":
                return {"helper": name, "params": {}, "reasoning": "fallback: first available"}

        return None

    async def _generate_sandbox_code(
        self,
        posture: str,
        source_layer: str,
        source_dims: list,
        deficit_values: list,
    ) -> dict:
        """
        Use LLM to generate Python code for the coding sandbox.

        Called when coding_sandbox is selected but no code was provided.
        Returns params dict with 'code' key.
        """
        if not self._llm_fn:
            return {}

        reason = _POSTURE_REASONS.get(posture, "Inner state needs rebalancing")
        prompt = (
            f"You are Titan, a sovereign AI. Your inner state needs rebalancing.\n"
            f"Intent: {posture} — {reason}\n"
            f"Source: {source_layer}[{source_dims}], deficit={deficit_values}\n\n"
            f"Write a short Python script (max 30 lines) that does something "
            f"meaningful for this intent. Examples:\n"
            f"- research: compute statistics, analyze data patterns\n"
            f"- create: generate mathematical art (print ASCII), compute fractals\n"
            f"- meditate: calculate golden ratio sequences, prime spirals\n"
            f"- rest: compute system health metrics, uptime statistics\n\n"
            f"Only use: math, json, datetime, collections, itertools, re, statistics, "
            f"random, string, hashlib, base64, decimal, fractions.\n"
            f"Output must include print() statements.\n"
            f"Respond with ONLY the Python code, no markdown, no explanation."
        )

        try:
            self._llm_calls_this_hour += 1
            raw = await self._llm_fn(prompt, task="agency_code_gen")

            # Clean up: remove markdown code fences if present
            code = raw.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                code = "\n".join(lines).strip()

            if code and len(code) > 10:
                return {"code": code, "description": f"autonomous {posture} computation"}
        except Exception as e:
            logger.warning("[Agency] Code generation failed: %s", e)

        return {}

    @staticmethod
    def _parse_selection(raw: str) -> Optional[dict]:
        """Parse LLM JSON response into selection dict."""
        # Try direct JSON parse
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting JSON from markdown code block
        if "```" in raw:
            for block in raw.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except (json.JSONDecodeError, TypeError):
                    continue

        # Try finding JSON object in text
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _build_result(
        self,
        impulse_id: int,
        posture: str,
        helper_name: Optional[str],
        result: Optional[dict],
        reasoning: str,
        trinity_snapshot: dict,
    ) -> dict:
        """Build ACTION_RESULT payload."""
        self._action_counter += 1

        success = result.get("success", False) if result else False
        result_text = result.get("result", "") if result else ""
        enrichment = result.get("enrichment_data", {}) if result else {}
        error = result.get("error") if result else reasoning

        action_result = {
            "action_id": self._action_counter,
            "impulse_id": impulse_id,
            "posture": posture,
            "helper": helper_name,
            "success": success,
            "result": result_text[:500],  # Truncate for bus
            "enrichment_data": enrichment,
            "error": error,
            "reasoning": reasoning,
            "trinity_before": trinity_snapshot,
            "ts": time.time(),
        }

        # deque(maxlen=50) handles eviction automatically (L3 Phase A.8.1).
        self._history.append(action_result)

        return action_result

    def _check_budget_reset(self) -> None:
        """Reset hourly budget counter if the hour has passed."""
        now = time.time()
        if now - self._hour_start >= 3600:
            self._llm_calls_this_hour = 0
            self._hour_start = now

    # ── Autonomy-First Dispatch (NN-driven, no LLM) ──────────────

    async def dispatch_from_nervous_signals(
        self,
        outer_signals: list[dict],
        trinity_snapshot: dict = None,
    ) -> list[dict]:
        """
        Autonomy-first dispatch: execute helpers directly from outer
        nervous system program signals. NO LLM involved.

        Args:
            outer_signals: from NeuralNervousSystem.get_outer_dispatch_signals()
                Each: {"system": "CREATIVITY", "urgency": 0.8, "helpers": ["art_generate"]}
            trinity_snapshot: current Trinity state for context

        Returns:
            List of ACTION_RESULT dicts for each dispatched action.
        """
        results = []
        trinity_snapshot = trinity_snapshot or {}

        for signal in outer_signals:
            system = signal["system"]
            urgency = signal["urgency"]
            helpers = signal.get("helpers", [])

            if not helpers:
                continue

            # Try each configured helper in order
            dispatched = False
            for helper_name in helpers:
                helper = self._registry.get_helper(helper_name)
                if not helper:
                    continue
                status = self._registry.get_status(helper_name)
                if status == "unavailable":
                    continue

                # V5: Check dispatch cooldown (expression rate limiting)
                _cooldown = self._dispatch_cooldowns.get(helper_name, 0)
                if _cooldown > 0:
                    _last_ts = self._last_dispatch_ts.get(helper_name, 0)
                    if time.time() - _last_ts < _cooldown:
                        continue  # Still in cooldown — skip this dispatch

                # Build params from Trinity context (no LLM needed)
                params = {
                    "trinity_snapshot": trinity_snapshot,
                    "posture": system.lower(),
                    "autonomous": True,  # Flag: NN-dispatched, not LLM-dispatched
                }

                # Helper-specific default params
                if helper_name == "art_generate":
                    import hashlib, time as _time
                    params["seed_text"] = f"autonomous_{system}_{_time.time()}"
                    params["mode"] = "flow_field"
                elif helper_name == "audio_generate":
                    params["mode"] = "trinity"
                    params["duration"] = 15
                elif helper_name == "web_search":
                    # CURIOSITY needs a query — derive from recent state
                    params["query"] = "latest developments autonomous AI agents"
                    params["max_results"] = 3
                elif helper_name == "code_knowledge":
                    params["mode"] = "structure"
                    params["target"] = "titan_plugin/"

                logger.info(
                    "[Agency] AUTONOMY dispatch: %s → %s (urgency=%.3f, NO LLM)",
                    system, helper_name, urgency,
                )

                try:
                    result = await helper.execute(params)
                    action_result = self._build_result(
                        impulse_id=-1,  # -1 = NN-autonomous (not from IMPULSE loop)
                        posture=system.lower(),
                        helper_name=helper_name,
                        result=result,
                        reasoning=f"autonomy-first: {system} urgency={urgency:.3f}",
                        trinity_snapshot=trinity_snapshot,
                    )
                    results.append(action_result)
                    dispatched = True
                    self._last_dispatch_ts[helper_name] = time.time()
                    break  # Only dispatch first available helper
                except Exception as e:
                    logger.warning("[Agency] Autonomy dispatch %s→%s failed: %s",
                                   system, helper_name, e)

            if not dispatched:
                logger.debug("[Agency] No available helper for %s signal", system)

        return results

    def get_stats(self) -> dict:
        """Return agency statistics."""
        self._check_budget_reset()
        return {
            "action_count": self._action_counter,
            "llm_calls_this_hour": self._llm_calls_this_hour,
            "budget_per_hour": self._budget_per_hour,
            "budget_remaining": max(0, self._budget_per_hour - self._llm_calls_this_hour),
            "registered_helpers": self._registry.list_all_names(),
            "helper_statuses": self._registry.get_all_statuses(),
            "recent_actions": len(self._history),
        }
