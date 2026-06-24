"""
titan_hcl/logic/agency/module.py — Agency Module (Step 7.4).

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

# Helper-name → action_type mapping (rFP_trinity_130d_awakening §3.1).
# Used by `_build_result` to tag every action_result with a stable
# action_type so downstream consumers (creative_this_hour filter,
# expressive_authenticity coherence check) don't have to re-derive.
_HELPER_ACTION_TYPES = {
    "art_generate": "art",
    "audio_generate": "audio",
    "web_search": "research",
    "code_knowledge": "research",
    "coding_sandbox": "compute",
    "infra_inspect": "infra",
    "memo_inscribe": "inscription",
}


def _derive_action_type(helper_name, posture: str) -> str:
    """Map helper_name to a stable action_type for history dicts.

    Falls back to posture when helper is None or unmapped — e.g. a
    "no_suitable_helper" outcome still gets action_type='meditate' (or
    whichever posture). Returns lowercase string.
    """
    if helper_name and helper_name in _HELPER_ACTION_TYPES:
        return _HELPER_ACTION_TYPES[helper_name]
    return (posture or "unknown").lower()


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
        research_gap_provider=None,
    ):
        """
        Args:
            registry: HelperRegistry instance
            llm_fn: Async callable(prompt: str) -> str for LLM inference.
                     If None, uses a no-op that returns {"helper": "none"}.
            budget_per_hour: Max LLM calls for Agency per hour
            research_gap_provider: optional `() -> list[dict]` returning the current
                knowledge-graph research gaps (RFP_titan_research_agent §1.4 step 2b).
                When present + a gap exists, a `research` posture TARGETS the
                least-grounded concept Z instead of an open-ended deficit, so the
                outcome is verifiable (groundedness-delta). None → legacy behaviour.
        """
        from .registry import HelperRegistry
        self._registry = registry or HelperRegistry()
        self._llm_fn = llm_fn
        self._budget_per_hour = budget_per_hour
        self._research_gap_provider = research_gap_provider
        self._last_research_gap: Optional[str] = None  # anti-repeat across dispatches

        # State
        self._action_counter = 0
        self._llm_calls_this_hour = 0
        self._hour_start = time.time()
        self._pending_action: Optional[dict] = None
        # L3 Phase A.8.1: deque(maxlen=50) replaces manual `[-50:]` trim —
        # auto-eviction at data-structure level, no per-append slicing.
        # SPEC §23.8 D-SPEC-87 Phase 3.F wave 3a (2026-05-18): bumped
        # maxlen 50→500 so actions_this_day rolling-window query can see
        # a full 24h of actions for active Titans (T2 ~150 posts/day).
        # Memory cost: 500 dicts × ~500B ≈ 250KB per Titan — acceptable.
        #
        # SPEC §23.8 D-SPEC-87 Phase 3.E wave 3b (2026-05-18): persist
        # _history via RollingStateStore. Without persistence the deque
        # reset to empty on every restart, so actions_this_day stayed at
        # 0 for 24h post-cascade → outer_mind willing[10,12] frozen.
        # Restore-on-boot loads up to 500 entries ≤24h old.
        try:
            from titan_hcl.core.rolling_state_persistence import (
                RollingStateStore)
            self._history_store: "RollingStateStore | None" = RollingStateStore(
                name="agency_action_history",
                max_entries=500,
                max_age_s=24 * 3600.0,
                save_every_n=5,    # persist after every 5 actions
                save_every_s=60.0,  # or every 60s, whichever first
            )
            restored = self._history_store.load()
        except Exception as _rss_err:
            self._history_store = None
            restored = []
        self._history: deque = deque(restored, maxlen=500)
        # rFP_trinity_130d_awakening Phase 2 (SPEC §23.9 ANANDA[42] surrender_capacity).
        # Heuristic-detected retries: a new action sharing (helper, posture)
        # with a FAILED action in the prior 30s counts as a retry. Track
        # only retry outcomes here (deque of (ts, success)) so failed_retry_rate
        # is fraction-of-retries-that-failed, not fraction-of-all-actions.
        self._retry_history: deque = deque(maxlen=50)
        # Inter-action timestamps for burst_frequency (coefficient of
        # variation of deltas; high CV → bursty → low surrender_capacity).
        self._action_timestamps: deque = deque(maxlen=100)
        # Retry-detection window: same (helper, posture) within this many
        # seconds after a failure → treat as retry.
        self._retry_window_s: float = 30.0
        # V5: Expression dispatch cooldowns (loaded from titan_params.toml)
        self._dispatch_cooldowns: dict[str, float] = {}
        self._last_dispatch_ts: dict[str, float] = {}
        # RFP_config_as_shm_state §7.C/C.3b: read [expression_dispatch_cooldowns]
        # from the SHM slot (config-as-state, INV-CFG-7).
        try:
            from titan_hcl.params import get_params
            self._dispatch_cooldowns = get_params("expression_dispatch_cooldowns")
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
            params = dict(learned.get("params", {}))
            # Fix #2 (EEL-B2 / mastery §7.P9): a failure-replay revisit replays the
            # ORIGINAL research-helper input params (query/file/…, captured at enqueue
            # in intent.helper_params) so the re-run faithfully re-attempts the real
            # query/inspection — not a posture-paraphrase. `code` was deliberately not
            # captured → coding_sandbox still regenerates corrected code below.
            if isinstance(intent.get("_revisit"), dict) and intent.get("helper_params"):
                params.update(intent["helper_params"])
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

        # RFP_titan_research_agent §1.4 step 2b — TARGET the least-grounded concept Z
        # so autonomous research is VERIFIABLE (groundedness-delta), not open-ended.
        # Only fires when the gap provider is wired (agency_worker passes it only when
        # the kill-switch `research_curiosity_enabled` is on) AND this is a research
        # helper AND a gap exists. Fallback-safe: otherwise the LLM-chosen query stands.
        if (self._research_gap_provider is not None
                and _derive_action_type(helper_name, posture) == "research"):
            _gap = self._pick_research_gap()
            if _gap is not None:
                _cn = _gap.get("name") or _gap.get("concept_id")
                params["query"] = (
                    f"{_cn}: what is it, how does it work, and the key facts and "
                    f"context worth knowing — a clear, well-sourced explanation.")
                params["_research_target"] = {
                    "concept_id": _gap["concept_id"],
                    "version": int(_gap.get("version", 1) or 1),
                    "baseline_groundedness": float(_gap.get("groundedness", 0.0) or 0.0),
                    "domain_hint": _gap.get("domain_hint", "general") or "general",
                    "name": _cn,
                }
                reasoning = (reasoning or "") + f" [curiosity: ground concept '{_cn}']"

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
                                  reasoning, trinity_snapshot, helper_params=params)

    def _pick_research_gap(self) -> Optional[dict]:
        """Pick a knowledge-graph gap to ground (RFP_titan_research_agent §1.4 step
        2b): the highest-salience gap that isn't the one just researched (simple
        anti-repeat so curiosity rotates instead of fixating). None on no gaps / a
        provider error (research then falls back to the LLM-chosen topic). Soft."""
        try:
            gaps = self._research_gap_provider() or []
        except Exception as e:  # noqa: BLE001 — provider is advisory, never fatal
            logger.debug("[Agency] research_gap_provider failed: %s", e)
            return None
        if not gaps:
            return None
        for g in gaps:
            if g.get("concept_id") and g.get("concept_id") != self._last_research_gap:
                self._last_research_gap = g["concept_id"]
                return g
        g = gaps[0]
        self._last_research_gap = g.get("concept_id")
        return g

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
        correction: str = "",
    ) -> dict:
        """
        Use LLM to generate Python code for the coding sandbox.

        Called when coding_sandbox is selected but no code was provided.
        Returns params dict with 'code' key. `correction` (P8.2 solve-until-correct)
        appends the task-completion judge's concrete fix so the retry regenerates code
        that actually solves the task (not just runs clean).
        """
        if not self._llm_fn:
            return {}

        reason = _POSTURE_REASONS.get(posture, "Inner state needs rebalancing")
        _fix = (f"\nPREVIOUS ATTEMPT DID NOT SOLVE THE TASK. Fix this concretely: "
                f"{correction}\n" if correction else "")
        prompt = (
            f"You are Titan, a sovereign AI. Your inner state needs rebalancing.\n"
            f"Intent: {posture} — {reason}\n"
            f"Source: {source_layer}[{source_dims}], deficit={deficit_values}\n{_fix}\n"
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

    async def p8_rerun(self, helper_name: str, intent: dict,
                       correction: str) -> Optional[dict]:
        """P8.2 (solve-until-correct) — re-run a routing helper applying the task-
        completion judge's correction. coding_sandbox → REGENERATE code incorporating
        the correction; web_search/code_knowledge → re-query with the correction as a
        refinement. Budget-gated (counts against the agency LLM budget). Returns the
        helper result dict, or None (budget exhausted / helper missing / no code)."""
        self._check_budget_reset()
        if self._llm_calls_this_hour >= self._budget_per_hour:
            return None
        helper = self._registry.get_helper(helper_name)
        if helper is None:
            return None
        posture = intent.get("posture", "meditate")
        params = {"trinity_snapshot": intent.get("trinity_snapshot", {}),
                  "posture": posture}
        if helper_name == "coding_sandbox":
            gen = await self._generate_sandbox_code(
                posture, intent.get("source_layer", "unknown"),
                intent.get("source_dims", []), intent.get("deficit_values", []),
                correction=correction)
            if not gen.get("code"):
                return None
            params.update(gen)
        else:  # research lane — refine the query with the correction
            reason = _POSTURE_REASONS.get(posture, "")
            params["query"] = f"{posture}: {reason}. {correction}".strip()
        try:
            return await helper.execute(params)
        except Exception as e:  # noqa: BLE001
            logger.warning("[Agency] p8_rerun helper '%s' failed: %s", helper_name, e)
            return None

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
        helper_params: Optional[dict] = None,
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
            "action_type": _derive_action_type(helper_name, posture),
            "success": success,
            "result": result_text[:500],  # Truncate for bus
            "enrichment_data": enrichment,
            "error": error,
            "reasoning": reasoning,
            "trinity_before": trinity_snapshot,
            # Failure-replay fix #2 (EEL-B2/P9): the RESEARCH helper's RE-POSE input
            # fields, so a revisit faithfully replays the ORIGINAL attempt (the real
            # query/file the search/inspection used) instead of a posture-paraphrase.
            # JSON-safe subset only (excludes the big trinity_snapshot). NB `code` is
            # deliberately EXCLUDED — for coding_sandbox the revisit must REGENERATE
            # corrected code (the solve-until-correct loop), never replay the code
            # that already failed. RFP §7.P9.
            "helper_params": {k: v for k, v in (helper_params or {}).items()
                              if k in ("query", "what", "file", "page",
                                       "refinement", "topic", "url", "max_results")},
            "ts": time.time(),
        }

        # deque(maxlen=500) handles eviction automatically (D-SPEC-87 wave 3a).
        self._history.append(action_result)
        # Phase 3.E wave 3b — persist via RollingStateStore. append_and_save
        # batches per save_every_n=5 so we don't I/O-spam on every action.
        if self._history_store is not None:
            try:
                self._history_store.append_and_save(
                    action_result, list(self._history))
            except Exception:
                pass  # best-effort; persistence shouldn't block actions

        # rFP_trinity_130d_awakening Phase 2 — heuristic retry detection +
        # burst-pattern timestamp tracking (SPEC §23.9 ANANDA[42,44]).
        now_ts = action_result["ts"]
        self._action_timestamps.append(now_ts)
        # Look back through history (excluding the current append) for a
        # FAILED action with same (helper, posture) within retry_window_s.
        for prior in reversed(list(self._history)[:-1]):
            dt = now_ts - prior.get("ts", 0.0)
            if dt > self._retry_window_s:
                break
            if prior.get("success"):
                continue
            if (prior.get("helper") == helper_name
                    and prior.get("posture") == posture):
                self._retry_history.append((now_ts, success))
                break

        return action_result

    def flush(self) -> None:
        """Force-persist the action history immediately (rFP §P2 — AUDIT §C).

        `_record_*` persists via append_and_save which batches per
        save_every_n=5, so up to 4 recent actions sit only in the in-memory
        `_history` deque. On MODULE_SHUTDOWN the worker calls this to flush them
        durably before exit — otherwise a hot-reload / kill-respawn loses 1-4
        action_results (which drive helper-completion, mood delta, trinity
        enrichment). RollingStateStore.save() writes atomically (tmp+fsync+
        os.replace). SelfAssessment uses save_every_n=1 and needs no flush.
        """
        if self._history_store is not None:
            try:
                self._history_store.save(list(self._history))
            except Exception:
                pass  # best-effort

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
                    params["target"] = "titan_hcl/"

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
                        helper_params=params,  # P0/fix#2: carry query/target so a revisit faithfully re-poses this no-prompt attempt
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
        """Return agency statistics.

        Schema bridge for Trinity 130D consumers (rFP_trinity_130d_awakening §3.1):
        outer_body[3], outer_mind[3,10,12,13], outer_spirit SAT[7,9,14],
        ANANDA[30,33,42,44] read total_actions / failed_actions /
        actions_this_hour / creative_this_hour from this dict.
        """
        self._check_budget_reset()

        now = time.time()
        hist = list(self._history)
        last_hour = [h for h in hist if h.get("ts", 0) > now - 3600]
        # SPEC §23.8 D-SPEC-87 Phase 3.F wave 3a — 24h rolling-window for
        # outer_mind willing[10] action_throughput / willing[12] creative_output.
        last_day = [h for h in hist if h.get("ts", 0) > now - 86400]
        total_actions = self._action_counter
        # failed_actions: derive from full history window (deque maxlen=50 caps memory)
        failed_in_window = sum(1 for h in hist if not h.get("success"))
        # Scale to total_actions: fraction of window that failed × total
        if hist:
            failed_actions = int(round(total_actions * failed_in_window / len(hist)))
        else:
            failed_actions = 0

        return {
            # Original keys (preserved for dashboard contract — DO NOT REMOVE)
            "action_count": self._action_counter,
            "llm_calls_this_hour": self._llm_calls_this_hour,
            "budget_per_hour": self._budget_per_hour,
            "budget_remaining": max(0, self._budget_per_hour - self._llm_calls_this_hour),
            "registered_helpers": self._registry.list_all_names(),
            "helper_statuses": self._registry.get_all_statuses(),
            "recent_actions": len(self._history),
            # Phase 1 schema bridge (rFP_trinity_130d_awakening §3.1)
            "total_actions": total_actions,
            "failed_actions": failed_actions,
            "actions_this_hour": len(last_hour),
            "creative_this_hour": sum(
                1 for h in last_hour
                if h.get("action_type") in ("art", "audio", "music")
            ),
            # SPEC §23.8 D-SPEC-87 Phase 3.F wave 3a — 24h smoothing
            # counterparts consumed by outer_mind willing[10,12] (Rust
            # outer-mind-rs reads with divisor /240 = 10 × 24).
            "actions_this_day": len(last_day),
            "creative_this_day": sum(
                1 for h in last_day
                if h.get("action_type") in ("art", "audio", "music")
            ),
            # SPEC §23.9 SAT[1] expressive_authenticity — exposes the 30
            # most recent action_results with the posture + dominant hormone
            # at action time so outer_spirit_worker can compute the
            # inner-outer coherence ratio. Trinity_before is trimmed to
            # just hormone_levels to keep the broadcast small.
            "recent_actions_detail": [
                {
                    "posture": h.get("posture"),
                    "action_type": h.get("action_type"),
                    "success": h.get("success"),
                    "ts": h.get("ts"),
                    "hormones": (
                        (h.get("trinity_before") or {}).get("hormone_levels") or {}
                    ),
                }
                for h in hist[-30:]
            ],
            # Phase 2 schema bridge (SPEC §23.9 ANANDA[42] surrender_capacity).
            "failed_retry_rate": self._compute_failed_retry_rate(),
            "burst_frequency": self._compute_burst_frequency(),
        }

    def _compute_failed_retry_rate(self) -> float:
        """Fraction of retries that failed.

        SPEC §23.9 ANANDA[42] surrender_capacity input. Cold-start (no
        retries observed) → 0.0 — the SPEC-correct value: with no retries
        there is no struggle, so no contribution to gripping.
        """
        recent = list(self._retry_history)
        if not recent:
            return 0.0
        return sum(1 for _, ok in recent if not ok) / len(recent)

    def _compute_burst_frequency(self) -> float:
        """Burst score from coefficient of variation of inter-action deltas.

        High CV → bursty (many actions clustered in time, then long gaps)
        → high score (gripping pattern). Low CV → regular cadence → low
        score (releasing pattern). Cold-start (n<10 timestamps) → 0.0.
        Returned in [0, 1] via min(1.0, CV) — CV>1 saturates.

        SPEC §23.9 ANANDA[42] surrender_capacity input.
        """
        ts = list(self._action_timestamps)
        if len(ts) < 10:
            return 0.0
        deltas = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta <= 0.0:
            return 1.0
        var = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
        cv = (var ** 0.5) / mean_delta
        return min(1.0, cv)
