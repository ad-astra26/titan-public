"""
titan_plugin/logic/outer_trinity.py — V4 Time Awareness: Outer Trinity Collector.

Computes 3×5DT Outer Trinity tensors from real live data sources.
The Outer Trinity represents Titan's acting/communicating/creating surface:

  Outer Body (5DT) — V6 5DT body-felt semantics (refactored 2026-04-23 per
                      rFP_phase1_sensory_wiring.md). Each dim blends 3
                      independent rich sources with no single saturating
                      input:
    [0] interoception   — SOL balance (0.4) + block_delta (0.3) + anchor_fresh (0.3)
                          "my metabolic + blockchain-heartbeat body"
    [1] proprioception  — peer_entropy (0.5) + helper_health (0.3) + bus_module_div (0.2)
                          "how many entities around me, ecosystem width"
    [2] somatosensation — TX latency (0.4) + creation_nudge (0.3) + CPU spike rate (0.3)
                          "pressure of what I'm touching / reaching for"
    [3] entropy         — ping variance (0.4) + bus drop rate (0.3) + error rate (0.3)
                          "chaos in my world right now"
    [4] thermal         — CPU thermal (0.4) + circadian (0.3) + LLM latency EMA (0.3)
                          "my body temperature / time-of-day"

  Outer Mind (5DT) — creative/social levers:
    [0] creative_output     — Art generation rate (can influence via art_generate)
    [1] sonic_expression    — Audio generation rate (can influence via audio_generate)
    [2] memory_quality      — Cognee consolidation score (deeper conversations → richer memories)
    [3] research_depth      — Research findings quality (can influence via web_search)
    [4] social_engagement   — Interaction quality (can influence via social_post)

  Outer Lower Spirit (5DT) — meta-awareness levers:
    [0] identity_coherence  — Soul alignment (refuse identity-compromising interactions)
    [1] purpose_clarity     — Impulse→Action success rate (align with Prime Directives)
    [2] action_quality      — SelfAssessment avg score (learn from feedback)
    [3] outer_body_scalar   — mean(Outer Body[0:5])
    [4] outer_mind_scalar   — mean(Outer Mind[0:5])

All values normalized to [0.0, 1.0] where 0.5 = center (Middle Path).
Values closer to 0.5 contribute to faster sphere clock contraction (balanced).

Called from TitanCore._outer_trinity_loop() at configurable interval.
"""
import logging
import math
import os
import time
from typing import Optional
from titan_plugin.utils.silent_swallow import swallow_warn

logger = logging.getLogger(__name__)

# Expected rates for normalization (reasonable defaults)
EXPECTED_ACTIONS_PER_HOUR = 10
EXPECTED_ART_PER_DAY = 5
EXPECTED_AUDIO_PER_DAY = 3
EXPECTED_RESEARCH_PER_DAY = 5
EXPECTED_INTERACTIONS_PER_DAY = 20
MAX_LATENCY_SECONDS = 30.0  # LLM response latency ceiling


class OuterTrinityCollector:
    """
    Aggregates real live data into 3×5DT Outer Trinity tensors.

    Sources are passed as a dict from TitanCore which has access to all
    subsystem stats via proxies and direct references.
    """

    def __init__(self):
        # Rolling counters (reset at configurable windows)
        self._last_collect_ts = time.time()

        # Cache last collected values for stats/debugging
        self._last_outer_body: list[float] = [0.5] * 5
        self._last_outer_mind: list[float] = [0.5] * 5
        self._last_outer_spirit: list[float] = [0.5] * 5
        self._collect_count = 0

        # Extended-tensor health counters — exposed via get_last_tensors()
        # so warning_monitor + arch_map symmetries can detect silent failure.
        # Pre-2026-04-25 this site swallowed exceptions at logger.debug, hiding
        # 67D/132D state-vector collapse on T1 for 37+ days. See BUG-T1-
        # CONSCIOUSNESS-67D-STATE-VECTOR.
        self._extended_success_count = 0
        self._extended_failure_count = 0
        self._extended_last_failure_ts: float = 0.0
        self._extended_last_failure_msg: str = ""

    def collect(self, sources: dict) -> dict:
        """
        Collect Outer Trinity tensors from live sources.

        Args:
            sources: dict with keys:
                - agency_stats: dict from AgencyModule.get_stats()
                - assessment_stats: dict from SelfAssessment.get_stats()
                - helper_statuses: dict from HelperRegistry.get_all_statuses()
                - bus_stats: dict from DivineBus.stats
                - impulse_stats: dict from ImpulseEngine.get_stats() (via spirit proxy)
                - observatory_db: ObservatoryDB instance (for expressive counts)
                - memory_status: dict from MemoryProxy.get_memory_status()
                - soul_health: float (0.0-1.0)
                - llm_avg_latency: float (seconds)
                - uptime_seconds: float

        Returns:
            {"outer_body": [5 floats], "outer_mind": [5 floats], "outer_spirit": [5 floats]}
        """
        self._collect_count += 1

        outer_body = self._collect_outer_body(sources)
        outer_mind = self._collect_outer_mind(sources)
        outer_spirit = self._collect_outer_spirit(outer_body, outer_mind, sources)

        self._last_outer_body = outer_body
        self._last_outer_mind = outer_mind
        self._last_outer_spirit = outer_spirit
        self._last_collect_ts = time.time()

        result = {
            "outer_body": outer_body,
            "outer_mind": outer_mind,
            "outer_spirit": outer_spirit,
        }

        # OT3: Compute extended 15D mind + 45D spirit for dimensional symmetry.
        # Pattern C (WARN + counter + safe fallback) per directive_error_visibility.md
        # — silent fallback to 67D consciousness epoch is a Trinity-symmetry
        # invariant violation (see BUG-T1-CONSCIOUSNESS-67D-STATE-VECTOR for
        # the 37-day silent regression that motivated this rewrite).
        try:
            outer_mind_15d, outer_spirit_45d = self._collect_extended(
                outer_body, outer_mind, outer_spirit, sources)
            result["outer_mind_15d"] = outer_mind_15d
            result["outer_spirit_45d"] = outer_spirit_45d
            self._extended_success_count += 1
        except Exception as e:
            self._extended_failure_count += 1
            self._extended_last_failure_ts = time.time()
            self._extended_last_failure_msg = f"{type(e).__name__}: {e}"
            # Throttle WARNING log to first failure + every 100th to avoid
            # log flood while still emitting diagnostic on first occurrence.
            if (self._extended_failure_count == 1
                    or self._extended_failure_count % 100 == 0):
                logger.warning(
                    "[OuterTrinity] _collect_extended failed (count=%d, "
                    "ratio=%.3f): %s — falling back to neutral 0.5-padded "
                    "extended tensors to preserve 132D Trinity symmetry. "
                    "Investigate via arch_map symmetries --titan <T> + "
                    "warning_monitor.",
                    self._extended_failure_count,
                    self._extended_failure_count / max(1, self._collect_count),
                    self._extended_last_failure_msg,
                    exc_info=True,
                )
            # Safe fallback: 0.5-padded extended tensors. This preserves
            # 132D consciousness epoch dimensionality (and the GroundUp
            # body↔mind enrichment loop) even when the full computation
            # fails. Without this, has_outer_extended flips False and
            # the entire epoch downgrades to 67D, breaking inner↔outer
            # symmetry per fundamental_symmetry_requirements.md.
            result["outer_mind_15d"] = [0.5] * 15
            result["outer_spirit_45d"] = [0.5] * 45

        return result

    def _collect_outer_body(self, sources: dict) -> list[float]:
        """Outer Body 5DT — V6 body-felt semantics (rFP_phase1_sensory_wiring).

        Each dim is a weighted blend of 3 independent rich sources.
        No single saturating input can pin a dim — if any input is
        unavailable, it contributes 0.5 neutral and other inputs dominate.
        """
        # ── Gather inputs (all with safe defaults) ────────────────
        agency = sources.get("agency_stats") or {}
        helper_statuses = sources.get("helper_statuses") or {}
        bus = sources.get("bus_stats") or {}
        sys_stats = sources.get("system_sensor_stats") or {}
        net_stats = sources.get("network_monitor_stats") or {}
        tx_lat = sources.get("tx_latency_stats") or {}
        blk_delta = sources.get("block_delta_stats") or {}
        anchor = sources.get("anchor_state") or {}

        # ── [0] interoception: energetic body-state ──────────────
        # SOL balance (log-scaled around typical 0.5 SOL mean) +
        # block_delta normalized (chain pulse / tempo) +
        # anchor freshness (how recently we touched the chain)
        sol_balance = sources.get("sol_balance")
        if isinstance(sol_balance, (int, float)) and sol_balance >= 0:
            # 0 SOL = 0.0, 0.1 SOL = ~0.3, 0.5 SOL = 0.5, 2.0 SOL = 0.8, 10+ SOL = ~1.0
            sol_norm = _safe_clamp(math.log1p(sol_balance) / math.log1p(10.0))
        else:
            sol_norm = 0.5

        block_rate_norm = blk_delta.get("normalized", 0.5)

        # Anchor freshness: 0s=1.0, 300s=0.5, 600s=~0.33, 1800s=~0.14
        anchor_fresh = 0.5
        if anchor.get("success") and anchor.get("last_anchor_time"):
            since = time.time() - anchor.get("last_anchor_time", time.time())
            anchor_fresh = max(0.05, 1.0 / (1.0 + since / 300.0))

        interoception = _safe_clamp(
            0.4 * sol_norm + 0.3 * block_rate_norm + 0.3 * anchor_fresh
        )

        # ── [1] proprioception: body-in-network-space ────────────
        peer_entropy = net_stats.get("peer_entropy", 0.5)

        total_helpers = max(1, len(helper_statuses))
        available = sum(1 for s in helper_statuses.values() if s == "available")
        helper_health = available / total_helpers if total_helpers > 0 else 0.5

        bus_module_diversity = net_stats.get("bus_module_diversity", 0.5)

        proprioception = _safe_clamp(
            0.5 * peer_entropy + 0.3 * helper_health + 0.2 * bus_module_diversity
        )

        # ── [2] somatosensation: touch + pressure ─────────────────
        # TX latency normalized (chain congestion felt as reach-friction) +
        # current outer_body[2] as creation_nudge proxy (with decay fix
        # shipping in next commit; until then, saturates — but blend
        # limits its influence to 30%) +
        # CPU spike rate (bursts of internal load)
        tx_lat_norm = tx_lat.get("normalized", 0.5)

        current_ob2 = 0.5
        if (isinstance(self._last_outer_body, list)
                and len(self._last_outer_body) > 2):
            current_ob2 = float(self._last_outer_body[2])

        cpu_spikes = sys_stats.get("cpu_spike_rate", 0.0)

        somatosensation = _safe_clamp(
            0.4 * tx_lat_norm + 0.3 * current_ob2 + 0.3 * cpu_spikes
        )

        # ── [3] entropy: disorder / unpredictability ─────────────
        # ping variance (network jitter) + bus drop rate + action error rate
        ping_var = net_stats.get("ping_variance", 0.5)
        bus_drop_rate = net_stats.get("bus_drop_rate", 0.0)

        total_actions = agency.get("total_actions", 0)
        failed_actions = agency.get("failed_actions", 0)
        if total_actions > 0:
            error_rate = failed_actions / total_actions  # actual error rate (more = more disorder)
        else:
            error_rate = 0.0  # No actions yet → no errors recorded

        entropy = _safe_clamp(
            0.4 * ping_var + 0.3 * bus_drop_rate + 0.3 * error_rate
        )

        # ── [4] thermal: arousal / temperature ────────────────────
        # CPU thermal + circadian phase + LLM latency EMA
        cpu_thermal = sys_stats.get("cpu_thermal", 0.5)
        circadian = sys_stats.get("circadian_phase", 0.5)

        avg_latency = sources.get("llm_avg_latency", 0.0)
        if avg_latency > 0:
            # Higher latency → higher "thermal pressure" (slow LLM = heavy system)
            llm_latency_norm = _safe_clamp(avg_latency / MAX_LATENCY_SECONDS)
        else:
            llm_latency_norm = 0.5  # Neutral — producer not wired yet

        thermal = _safe_clamp(
            0.4 * cpu_thermal + 0.3 * circadian + 0.3 * llm_latency_norm
        )

        return [round(v, 4) for v in [
            interoception, proprioception, somatosensation, entropy, thermal
        ]]

    def _collect_outer_mind(self, sources: dict) -> list[float]:
        """Outer Mind 5DT — creative/social levers.

        A.8.4: prefers `art_count_100` / `audio_count_100` from sources
        (parent pre-extracts these so the worker can run without a
        observatory_db handle). Falls back to direct obs_db query when
        sources came from legacy in-parent gather without pre-extraction.
        """
        obs_db = sources.get("observatory_db")
        memory_status = sources.get("memory_status") or {}
        uptime = max(1.0, sources.get("uptime_seconds", 1.0))
        uptime_days = max(0.01, uptime / 86400.0)

        # [0] creative_output: art generation rate
        art_count = sources.get("art_count_100")
        if art_count is None:
            art_count = 0
            if obs_db:
                try:
                    archive = obs_db.get_expressive_archive(type_="art", limit=100)
                    art_count = len(archive)
                except Exception as _swallow_exc:
                    swallow_warn("[logic.outer_trinity] OuterTrinityCollector._collect_outer_mind: archive = obs_db.get_expressive_archive(type_='art', limi...", _swallow_exc,
                                 key='logic.outer_trinity.OuterTrinityCollector._collect_outer_mind.line286', throttle=100)
        art_count = int(art_count or 0)
        creative_output = _safe_clamp(min(1.0, art_count / max(1.0, EXPECTED_ART_PER_DAY * uptime_days)))

        # [1] sonic_expression: audio generation rate
        audio_count = sources.get("audio_count_100")
        if audio_count is None:
            audio_count = 0
            if obs_db:
                try:
                    archive = obs_db.get_expressive_archive(type_="audio", limit=100)
                    audio_count = len(archive)
                except Exception as _swallow_exc:
                    swallow_warn("[logic.outer_trinity] OuterTrinityCollector._collect_outer_mind: archive = obs_db.get_expressive_archive(type_='audio', li...", _swallow_exc,
                                 key='logic.outer_trinity.OuterTrinityCollector._collect_outer_mind.line296', throttle=100)
        audio_count = int(audio_count or 0)
        sonic_expression = _safe_clamp(min(1.0, audio_count / max(1.0, EXPECTED_AUDIO_PER_DAY * uptime_days)))

        # [2] memory_quality: persistent / total nodes ratio
        persistent = memory_status.get("persistent_count", 0)
        total_nodes = memory_status.get("total_nodes", 0)
        if total_nodes > 0:
            memory_quality = _safe_clamp(persistent / total_nodes)
        else:
            memory_quality = 0.5  # Neutral

        # [3] research_depth: approximated from memory quality + session count
        # Real web_search findings would come from research helper stats
        research_findings = memory_status.get("research_nodes", 0)
        research_depth = _safe_clamp(min(1.0, research_findings / max(1.0, EXPECTED_RESEARCH_PER_DAY * uptime_days)))

        # [4] social_engagement: interaction count
        interactions = memory_status.get("unique_interactors", 0)
        social_engagement = _safe_clamp(min(1.0, interactions / max(1.0, EXPECTED_INTERACTIONS_PER_DAY * uptime_days)))

        return [round(v, 4) for v in [
            creative_output, sonic_expression, memory_quality, research_depth, social_engagement
        ]]

    def _collect_outer_spirit(
        self,
        outer_body: list[float],
        outer_mind: list[float],
        sources: dict,
    ) -> list[float]:
        """Outer Lower Spirit 5DT — meta-awareness levers."""
        assessment = sources.get("assessment_stats") or {}
        impulse = sources.get("impulse_stats") or {}

        # [0] identity_coherence: soul health score
        identity_coherence = _safe_clamp(sources.get("soul_health", 0.5))

        # [1] purpose_clarity: successful impulses / total
        total_impulses = impulse.get("total_fires", 0)
        # Get successful actions from assessment
        total_assessed = assessment.get("total_assessed", 0)
        avg_score = assessment.get("average_score", 0.5)
        if total_impulses > 0 and total_assessed > 0:
            # Purpose clarity = how well impulses lead to successful actions
            purpose_clarity = _safe_clamp(avg_score)
        else:
            purpose_clarity = 0.5  # Neutral

        # [2] action_quality: rolling assessment average
        action_quality = _safe_clamp(assessment.get("average_score", 0.5))

        # [3] outer_body_scalar: mean of Outer Body
        outer_body_scalar = sum(outer_body) / len(outer_body) if outer_body else 0.5

        # [4] outer_mind_scalar: mean of Outer Mind
        outer_mind_scalar = sum(outer_mind) / len(outer_mind) if outer_mind else 0.5

        return [round(v, 4) for v in [
            identity_coherence, purpose_clarity, action_quality,
            outer_body_scalar, outer_mind_scalar,
        ]]

    def _collect_extended(self, outer_body: list, outer_mind_5d: list,
                          outer_spirit_5d: list, sources: dict) -> tuple:
        """Compute extended Outer Mind 15D + Outer Spirit 45D for symmetry.

        Returns (outer_mind_15d, outer_spirit_45d).
        """
        from titan_plugin.logic.outer_mind_tensor import collect_outer_mind_15d
        from titan_plugin.logic.outer_spirit_tensor import collect_outer_spirit_45d

        # Gather stats for extended computation
        agency = sources.get("agency_stats") or {}
        assessment = sources.get("assessment_stats") or {}
        memory_status = sources.get("memory_status") or {}
        uptime = max(1.0, sources.get("uptime_seconds", 1.0))

        # Action stats for OT1+OT2
        total_actions = agency.get("total_actions", 0)
        failed_actions = agency.get("failed_actions", 0)
        success_rate = (total_actions - failed_actions) / max(1, total_actions)
        actions_per_hour = total_actions / max(0.01, uptime / 3600.0)

        action_stats = {
            "total": total_actions,
            "success_count": total_actions - failed_actions,
            "success_rate": success_rate,
            "per_window": agency.get("actions_this_hour", 0),
            "per_hour": actions_per_hour,
            "failed_retry_rate": agency.get("failed_retry_rate", 0.0),
            "burst_frequency": agency.get("burst_frequency", 0.0),
            "error_rate": 1.0 - success_rate,
        }

        # Creative stats
        # A.8.4: prefers pre-extracted counts when present (worker mode);
        # falls back to direct obs_db query for legacy in-parent gather.
        obs_db = sources.get("observatory_db")
        art_count = sources.get("art_count_500")
        audio_count = sources.get("audio_count_500")
        if art_count is None or audio_count is None:
            if art_count is None:
                art_count = 0
            if audio_count is None:
                audio_count = 0
            if obs_db:
                try:
                    if sources.get("art_count_500") is None:
                        art_count = len(obs_db.get_expressive_archive(type_="art", limit=500))
                    if sources.get("audio_count_500") is None:
                        audio_count = len(obs_db.get_expressive_archive(type_="audio", limit=500))
                except Exception as _swallow_exc:
                    swallow_warn("[logic.outer_trinity] OuterTrinityCollector._collect_extended: art_count = len(obs_db.get_expressive_archive(type_='art'...", _swallow_exc,
                                 key='logic.outer_trinity.OuterTrinityCollector._collect_extended.line399', throttle=100)
        art_count = int(art_count or 0)
        audio_count = int(audio_count or 0)
        creative_stats = {
            "total": art_count + audio_count,
            "art_count": art_count,
            "audio_count": audio_count,
            "per_window": agency.get("creative_this_hour", 0),
            "unique_types": min(2, (1 if art_count > 0 else 0) + (1 if audio_count > 0 else 0)),
            "mean_assessment": assessment.get("average_score", 0.5),
        }

        # Guardian stats
        guardian_stats = {
            "threats_detected": agency.get("threats_detected", 0),
            "rejections": agency.get("rejections", 0),
            "severity_avg": agency.get("threat_severity_avg", 0.0),
            "rejections_per_window": agency.get("rejections_this_hour", 0),
        }

        # Social stats — enriched by Events Teacher social perception
        _sp_stats = sources.get("social_perception_stats", {})
        social_stats = {
            "interactions_per_window": memory_status.get("unique_interactors", 0),
            "sentiment_avg": _sp_stats.get("sentiment_ema", 0.5),
            "social_connection": _sp_stats.get("connection_ema", 0.0),
            "social_events_count": _sp_stats.get("events_count", 0),
            "last_contagion": _sp_stats.get("last_contagion"),
            "mean_conversation_quality": assessment.get("average_score", 0.5),
        }

        # Research stats
        research_stats = {
            "queries": memory_status.get("research_nodes", 0),
            "usage_rate": 0.5,
            "seconds_since_last": 300.0,  # Approximate
            "queries_per_window": 0,
        }

        # Assessment stats for OT1+OT2
        assessment_ext = {
            "mean_score": assessment.get("average_score", 0.5),
            "trend": assessment.get("trend", 0.0),
            "count": assessment.get("total_assessed", 0),
            "score_variance": assessment.get("score_variance", 0.3),
        }

        # Memory stats
        mem_stats = {
            "persistent_nodes": memory_status.get("persistent_count", 0),
            "growth_per_epoch": memory_status.get("growth_per_epoch", 0),
        }

        # Uptime and recovery
        uptime_ratio = min(1.0, uptime / max(1.0, uptime + 60.0))

        # Collect Outer Mind 15D
        outer_mind_15d = collect_outer_mind_15d(
            current_5d=outer_mind_5d,
            action_stats=action_stats,
            creative_stats=creative_stats,
            guardian_stats=guardian_stats,
            social_stats=social_stats,
            research_stats=research_stats,
            assessment_stats=assessment_ext,
            body_state={"values": outer_body},
            twin_state=sources.get("twin_state"),
            anchor_state=sources.get("anchor_state"),
            bus_stats=sources.get("bus_stats"),
        )

        # Collect Outer Spirit 45D
        outer_spirit_45d = collect_outer_spirit_45d(
            current_5d=outer_spirit_5d,
            outer_body=outer_body,
            outer_mind=outer_mind_15d,
            action_stats=action_stats,
            creative_stats=creative_stats,
            guardian_stats=guardian_stats,
            sovereignty_ratio=agency.get("sovereignty_ratio", 0.0),
            uptime_ratio=uptime_ratio,
            social_stats=social_stats,
            memory_stats=mem_stats,
            assessment_stats=assessment_ext,
            history=sources.get("history"),
        )

        return (
            [round(v, 4) for v in outer_mind_15d],
            [round(v, 4) for v in outer_spirit_45d],
        )

    def get_last_tensors(self) -> dict:
        """Get the most recently collected tensors."""
        return {
            "outer_body": list(self._last_outer_body),
            "outer_mind": list(self._last_outer_mind),
            "outer_spirit": list(self._last_outer_spirit),
        }

    def get_stats(self) -> dict:
        return {
            "collect_count": self._collect_count,
            "last_collect_ts": self._last_collect_ts,
            "outer_body": list(self._last_outer_body),
            "outer_mind": list(self._last_outer_mind),
            "outer_spirit": list(self._last_outer_spirit),
            # Extended-tensor health (Pattern C per directive_error_visibility.md)
            "extended_success_count": self._extended_success_count,
            "extended_failure_count": self._extended_failure_count,
            "extended_failure_ratio": (
                self._extended_failure_count / max(1, self._collect_count)
            ),
            "extended_last_failure_ts": self._extended_last_failure_ts,
            "extended_last_failure_msg": self._extended_last_failure_msg,
        }


def _safe_clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi], handling NaN/None."""
    if value is None or not isinstance(value, (int, float)):
        return 0.5
    import math
    if math.isnan(value) or math.isinf(value):
        return 0.5
    return max(lo, min(hi, float(value)))
